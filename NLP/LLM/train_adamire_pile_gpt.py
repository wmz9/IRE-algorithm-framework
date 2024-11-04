import os
import argparse
import time
import math
import random
import pickle
from contextlib import nullcontext
import data
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model_GPT import GPTConfig, GPT

# # I/O
# out_dir = 'out'
# always_save_checkpoint = True # if True, always save a checkpoint after each eval

# data
dataset = 'minipile'
data_dir = os.path.join('/mnt/share/data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
# # attempt to derive vocab_size from the dataset
# data_dir = os.path.join('/', dataset)
# meta_path = os.path.join(data_dir, 'meta.pkl')
# meta_vocab_size = None
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     meta_vocab_size = meta['vocab_size']
#     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
# print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
# if meta_vocab_size is None:
print("defaulting to vocab_size to 50304")

# model
block_size = 512
n_embd = 768
n_head = 12
n_layer = 12
bias = False
dropout = 0.0


# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16'
# compile = True # use PyTorch 2.0 to compile the model to be faster
# scale_attn_by_inverse_layer_idx = False
# # -----------------------------------------------------------------------------
# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging
# # -----------------------------------------------------------------------------


# # various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
scaler = torch.amp.GradScaler()


if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank = 0                             #ddp_rank is used in get_batch function so this has to be here also when running locally
    master_process = True
    seed_offset = 0
    # gradient_accumulation_steps *= 8 # simulate 8 gpus

# if master_process:
#     os.makedirs(out_dir, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    ix_list = []
    for jj in range(10):
        ix_list.append(torch.randint(len(data) - block_size, (batch_size,)))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix_list[ddp_rank]])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix_list[ddp_rank]])

    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

    return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, eval_iters, batch_size):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch('val', batch_size)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, min_lr, max_lr, warmup_iters, max_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)


def train(args):

    # wandb logging
    wandb_log = args.wandb_log
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name

    gradient_accumulation_steps = args.grad_micro_steps # used to simulate larger batch sizes
    batch_size = args.batch_size # if gradient_accumulation_steps > 1, this is the micro-batch size
    total_batch_size = args.total_bs
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    max_lr =args.max_lr # max learning rate
    min_lr = max_lr / 20 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    max_iters = args.max_iters # total number of training iterations
    warmup_iters = args.warmup_iters
    beta1 = args.beta1
    beta2 = args.beta2
    weight_decay = args.weight_decay
    grad_clip = args.grad_clip
    mask_interval = args.mask_interval
    beta = args.beta_Fisher
    rank = args.rank
    rank_increase = args.rank_increase
    print("rank_increase", rank_increase)
    prog = args.prog
    prog_decay = args.prog_decay
    print("prog_decay", prog_decay)
    log_interval = args.log_interval
    eval_interval = args.eval_interval
    eval_iters = args.eval_iters

    # GPT-small model
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=50304, dropout=dropout) 
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    # base_optimizer = torch.optim.AdamW
    optimizer = model.configure_optimizers(rank=rank, rank_increase=rank_increase, prog=prog, prog_decay=prog_decay, 
                                           beta=beta, betas=(beta1,beta2), weight_decay=weight_decay, eps=1e-16)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9


    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name)
        config = wandb.config 
        config.total_batch_size = total_batch_size 
        config.gradient_accumulation_steps = gradient_accumulation_steps 
        config.max_iters = max_iters  
        config.warmup_iters = warmup_iters  
        config.max_lr = max_lr
        config.min_lr = min_lr
        config.mu = prog
        config.rank = rank
        config.rank_increase = rank_increase
        config.mask_interval = mask_interval 
        config.beta1 = beta1
        config.beta2 = beta2
        config.weight_decay = weight_decay
        config.beta_Fisher = beta
        config.prog_decay = prog_decay    
        config.seed = args.seed
        config.log_interval = log_interval
        config.eval_interval = eval_interval
        config.eval_iters = eval_iters
        config.grad_clip = grad_clip

    # training loop
    X, Y = get_batch('train', batch_size) # fetch the very first batch

    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process


    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, min_lr, max_lr, warmup_iters, max_iters) if decay_lr else max_lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update the mask in AdamWIRE
        if iter_num > warmup_iters and iter_num % mask_interval == 1:
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                logits, _ = model(X, 0)
                X, Y = get_batch('train', batch_size) 
                samp_dist = torch.distributions.Categorical(logits=logits)
                y_sample = samp_dist.sample()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
                (loss / gradient_accumulation_steps).backward()
                # backward pass, with gradient scaling if training in fp16
            # step the optimizer and scaler if training in fp16
            optimizer.update_mask(iter_num - warmup_iters, max_iters - warmup_iters)
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            logits, loss = model(X, Y)
            # loss = F.cross_entropy(logits, Y.view(-1), ignore_index=-1)
            X, Y = get_batch('train', batch_size) 
            # backward pass, with gradient scaling if training in fp16
            # scaler.scale(loss).backward()
            (loss / gradient_accumulation_steps).backward()
            # step the optimizer and scaler if training in fp16
            # scaler.descent_step(optimizer, lr,max_lr)
        # gradient clip
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.descent_step(lr, max_lr)
        # scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1


        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() 
            total_param_norm = 0
            params = []
            for (name, p) in model.named_parameters():
                params.append(p)
            for p in params:
                param_norm = p.data.norm(2)
                total_param_norm += param_norm.item() ** 2
            total_param_norm = total_param_norm ** 0.5

            if iter_num % eval_interval == 0 or iter_num == max_iters:
                loss_val = estimate_loss(model, eval_iters, batch_size)
                print(f"iter {iter_num}: train loss {lossf:.4f}, val loss {loss_val:.4f}, time {dt*1000:.2f}ms")
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": lossf,
                        "val/loss": loss_val,
                        "lr": lr,
                        "param_norm": total_param_norm,
                        # "threshold": threshold
                    }, step=iter_num)
            else:
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": lossf,
                        "lr": lr,
                        "param_norm": total_param_norm,
                        # "threshold": threshold
                    }, step=iter_num)

        iter_num += 1
        local_iter_num += 1

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # termination conditions
        if iter_num > max_iters + 1:
            break

    if ddp:
        destroy_process_group()


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_log", action='store_true', help="Use Wandb Log.")
    parser.add_argument("--wandb_project", default= 'gpt_pile_std', type=str, help="Wandb project.")
    parser.add_argument("--wandb_run_name", default='moving_4_01' , type=str, help="Wandb run name.")
    parser.add_argument("--seed", default=41, type=int, help="Random seed.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size.")
    parser.add_argument("--grad_micro_steps", default=10, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--total_bs", default=300, type=int, help="Total batch size.")
    parser.add_argument("--mask_interval", default=10, type=int, help="Mask iterations.")
    parser.add_argument("--log_interval", default=50, type=int, help="Log iterations.")
    parser.add_argument("--eval_interval", default=200, type=int, help="..")
    parser.add_argument("--eval_iters", default=100, type=int, help="...")
    parser.add_argument("--max_lr", default=6e-4, type=float, help="max lr in AdamW.")
    parser.add_argument("--max_iters", default=30000, type=int, help="max iterations.")
    parser.add_argument("--warmup_iters", default=300, type=int, help="warmup iterations.")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta1 in AdamW.")
    parser.add_argument("--beta2", default=0.95, type=float, help="beta2 in AdamW.")
    parser.add_argument("--weight_decay", default=1e-1, type=float, help="weight decay in AdamW.")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="grad clip in AdamW.")
    parser.add_argument("--beta_Fisher", default=0.0, type=float, help="beta_Fisher in IRE.")
    parser.add_argument("--rank", default=0.1, type=float, help="rank in IRE.")
    parser.add_argument("--rank_increase", action='store_true', help="rank increase in IRE.")
    parser.add_argument("--prog", default=4.0, type=float, help="mu in IRE.")
    parser.add_argument("--prog_decay", action='store_true', help="prog decay in IRE.")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)


# # To run this .py (ddp)
# torchrun --standalone --nproc_per_node=4 train_adamire_web_gpt.py --batch_size=8 --grad_micro_steps=15 --total_bs=480 --rank=0.1 --prog=4.0 --prog_decay --max_lr=3e-4 --wandb_log --wandb_run_name=moving_4_01
# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --standalone --nproc_per_node=2 train_adamire_pile_gpt.py --batch_size=15 --grad_micro_steps=10 --total_bs=300 --rank=0.1 --prog=4.0 --max_lr=6e-4 --wandb_log --wandb_run_name=IRE_6e-4_4_0.1_A800