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
from modelling_llama_new import LlamaConfig, LlamaForCausalLM, build_llama_models


# # I/O
# out_dir = 'out'
# always_save_checkpoint = True # if True, always save a checkpoint after each eval

# data
dataset = 'minipile'
block_size = 512
# poor man's data loader
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
# # determine the vocab size we'll use for from-scratch training
# if meta_vocab_size is None:
print("defaulting to vocab_size to 50304")

# model
max_seq_length = block_size
d_model = 768  ## fixed due to d_{kv}
num_heads = 12  ## fixed due to d_{kv}
num_layers = 6
d_ff = d_model * 4
dropout = 0.0


# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = torch.device("cuda")
# examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
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
        logits = model(X).logits.view(-1, 50304)
        loss = F.cross_entropy(logits, Y.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, min_lr, max_lr, warmup_iters, max_iters, alpha=1.0, gamma=1.0):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    coeff = math.pow(coeff, gamma) * (1 + decay_ratio * (alpha - 1))
    coeff = min(coeff, 1)
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
    max_embed_lr =args.max_embed_lr # max learning rate
    max_head_lr =args.max_head_lr # max learning rate
    max_ln_lr =args.max_ln_lr # max learning rate
    max_qk_lr =args.max_qk_lr # max learning rate
    max_vo_lr =args.max_vo_lr # max learning rate
    max_mlp_lr =args.max_mlp_lr # max learning rate
    embed_alpha =args.embed_alpha # alpha in scheduler
    head_alpha =args.head_alpha # alpha in scheduler
    ln_alpha =args.ln_alpha # alpha in scheduler
    qk_alpha =args.qk_alpha # alpha in scheduler
    vo_alpha =args.vo_alpha # alpha in scheduler
    mlp_alpha =args.mlp_alpha # alpha in scheduler
    embed_gamma =args.embed_gamma # alpha in scheduler
    head_gamma =args.head_gamma # alpha in scheduler
    ln_gamma =args.ln_gamma # alpha in scheduler
    qk_gamma =args.qk_gamma # alpha in scheduler
    vo_gamma =args.vo_gamma # alpha in scheduler
    mlp_gamma =args.mlp_gamma # alpha in scheduler
    # min_lr = max_lr / 20 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    max_iters = args.max_iters # total number of training iterations
    warmup_iters = args.warmup_iters
    beta1 = args.beta1
    beta2 = args.beta2
    weight_decay = args.weight_decay
    grad_clip = args.grad_clip
    mask_interval = args.mask_interval
    log_interval = args.log_interval
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    eval_iters = args.eval_iters
    resume_from_checkpoint = args.resume_from_checkpoint 
    model_name = args.model_name
    
    if args.skip_init:
        import modelling_llama_new
        modelling_llama_new.skip_initialization = True

    d_input = 50304
    # d_output = d_input
    # model = Transformer(d_input, d_output, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    # model_args = dict(dim=d_model, n_layers=num_layers, n_heads=num_heads, vocab_size=d_input, multiple_of=256, 
    #                   norm_eps=1e-5, max_batch_size=32, max_seq_len=1024, device=device) 
    # gptconf = ModelArgs(**model_args)
    # model_args = dict(d_model=d_model, n_layers=num_layers, n_heads=num_heads, vocab_size=d_input, context_window=max_seq_length, dropout=dropout) 
    # model_args = LlamaConfig(vocab_size=d_input, hidden_size=d_model, num_attention_heads=num_heads, attention_dropout=dropout, num_hidden_layers=num_layers, intermediate_size=d_ff, max_position_embeddings=max_seq_length)
    # print(model_args)

    # model = LlamaForCausalLM(model_args).to(device)
    model = build_llama_models(model_name, d_input, max_seq_length, device)
    print(sum(p.numel() for p in model.parameters()))

    if resume_from_checkpoint:
        ckpt = torch.load(f'{resume_from_checkpoint}_ckpt.pt', map_location=device)
        model.load_state_dict(ckpt['model'])

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        

    embed_param = [p for name, p in model.named_parameters() if 'embed' in name]
    head_param = [p for name, p in model.named_parameters() if 'lm_head' in name]
    ln_param = [p for name, p in model.named_parameters() if 'norm' in name]
    qk_param = [p for name, p in model.named_parameters() if 'q_proj' in name or 'k_proj' in name]
    vo_param = [p for name, p in model.named_parameters() if 'v_proj' in name or 'o_proj' in name]
    mlp_param = [p for name, p in model.named_parameters() if 'mlp' in name]
    optimizer = torch.optim.AdamW([
        {'params': embed_param, 'lr': max_embed_lr, "name": "embed"},
        {'params': head_param, 'lr': max_head_lr, "name": "head"},
        {'params': ln_param, 'lr': max_ln_lr, "name": "ln"},
        {'params': qk_param, 'lr': max_qk_lr, "name": "qk"},
        {'params': vo_param, 'lr': max_vo_lr, "name": "vo"},
        {'params': mlp_param, 'lr': max_mlp_lr, "name": "mlp"},
    ], lr=max_ln_lr, betas=(beta1,beta2), weight_decay=weight_decay)
        
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    if resume_from_checkpoint:
        optimizer.load_state_dict(ckpt['optimizer'])
        iter_num = ckpt['iter_num']
        del ckpt


    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name)
        config = wandb.config 
        config.total_batch_size = total_batch_size 
        config.batch_size = batch_size
        config.gradient_accumulation_steps = gradient_accumulation_steps 
        config.max_iters = max_iters  
        config.warmup_iters = warmup_iters  
        config.max_embed_lr = max_embed_lr
        config.max_head_lr = max_head_lr
        config.max_ln_lr = max_ln_lr
        config.max_qk_lr = max_qk_lr
        config.max_vo_lr = max_vo_lr
        config.max_mlp_lr = max_mlp_lr
        config.embed_alpha = embed_alpha
        config.head_alpha = head_alpha
        config.ln_alpha = ln_alpha
        config.qk_alpha = qk_alpha
        config.vo_alpha = vo_alpha
        config.mlp_alpha = mlp_alpha
        config.embed_gamma = embed_gamma
        config.head_gamma = head_gamma
        config.ln_gamma = ln_gamma
        config.qk_gamma = qk_gamma
        config.vo_gamma = vo_gamma
        config.mlp_gamma = mlp_gamma
        config.mask_interval = mask_interval 
        config.beta1 = beta1
        config.beta2 = beta2
        config.weight_decay = weight_decay
        config.seed = args.seed
        config.log_interval = log_interval
        config.eval_interval = eval_interval
        config.save_interval = save_interval
        config.eval_iters = eval_iters
        config.grad_clip = grad_clip

    # training loop
    X, Y = get_batch('train', batch_size) # fetch the very first batch

    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    # if master_process:
    #     for i, param_group in enumerate(optimizer.param_groups):
    #         print(param_group["name"])

    max_lrs = [max_embed_lr, max_head_lr, max_ln_lr, max_qk_lr, max_vo_lr,  max_mlp_lr]
    alphas = [embed_alpha, head_alpha, ln_alpha, qk_alpha, vo_alpha,  mlp_alpha]
    gammas = [embed_gamma, head_gamma, ln_gamma, qk_gamma, vo_gamma,  mlp_gamma]
    while True:
        # determine and set the learning rate for this iteration
        lrs = []
        for max_lr, alpha, gamma, param_group in zip(max_lrs, alphas, gammas, optimizer.param_groups):
            lr = get_lr(iter_num, max_lr / 20, max_lr, warmup_iters, max_iters, alpha=alpha, gamma=gamma) if decay_lr else max_lr
            param_group['lr'] = lr
            lrs.append(lr)

        model.require_backward_grad_sync = True

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(gradient_accumulation_steps):
            # if ddp:
            #     model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            # if master_process and micro_step == 0:
            #     print("A")
            with ctx:
                logits = model(X).logits.view(-1, d_input)
                # logits = F.softmax(model(X).logits, dim=-1).view(-1, d_input)
            # if master_process and micro_step == 0:
            #     print("B")
            loss = F.cross_entropy(logits, Y.view(-1), ignore_index=-1) # more indent
            X, Y = get_batch('train', batch_size) 
            # backward pass, with gradient scaling if training in fp16
            # scaler.scale(loss).backward()
            (loss / gradient_accumulation_steps).backward()
            # step the optimizer and scaler if training in fp16
            # scaler.descent_step(optimizer, lr,max_lr)
        # gradient clip
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
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
                        "embed_lr": lrs[0],
                        "head_lr": lrs[0],
                        "ln_lr": lrs[2],
                        "qk_lr": lrs[3],
                        "vo_lr": lrs[4],
                        "mlp_lr": lrs[5],
                        "param_norm": total_param_norm,
                        # "threshold": threshold
                    }, step=iter_num)
            else:
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": lossf,
                        "embed_lr": lrs[0],
                        "head_lr": lrs[0],
                        "ln_lr": lrs[2],
                        "qk_lr": lrs[3],
                        "vo_lr": lrs[4],
                        "mlp_lr": lrs[5],
                        "param_norm": total_param_norm,
                        # "threshold": threshold
                    }, step=iter_num)
            
            if iter_num % save_interval == 0 and iter_num != 0:
                checkpoint = {
                    'model': model.module.state_dict() if ddp else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                }
                print(f"saving checkpoint")
                torch.save(checkpoint, f'{wandb_run_name}_ckpt.pt')                

        iter_num += 1
        local_iter_num += 1

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # termination conditions
        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_log", action='store_true', help="Use Wandb Log.")
    parser.add_argument("--wandb_project", default= 'llama_pile_layerwise', type=str, help="Wandb project.")
    parser.add_argument("--wandb_run_name", default='moving_4_01' , type=str, help="Wandb run name.")
    parser.add_argument("--model_name", default="0.23B", type=str)
    parser.add_argument("--seed", default=41, type=int, help="Random seed.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size.")
    parser.add_argument("--grad_micro_steps", default=10, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--total_bs", default=300, type=int, help="Total batch size.")
    parser.add_argument("--mask_interval", default=10, type=int, help="Mask iterations.")
    parser.add_argument("--log_interval", default=20, type=int, help="Log iterations.")
    parser.add_argument("--eval_interval", default=200, type=int, help="..")
    parser.add_argument("--save_interval", default=2000, type=int, help="..")
    parser.add_argument("--eval_iters", default=100, type=int, help="...")
    parser.add_argument("--max_lr", default=6e-4, type=float, help="max lr in AdamW.")
    parser.add_argument("--max_embed_lr", default=None, type=float, help="max embed lr in AdamW.")
    parser.add_argument("--max_head_lr", default=None, type=float, help="max head lr in AdamW.")
    parser.add_argument("--max_ln_lr", default=None, type=float, help="max ln lr in AdamW.")
    parser.add_argument("--max_qk_lr", default=None, type=float, help="max qk lr in AdamW.")
    parser.add_argument("--max_vo_lr", default=None, type=float, help="max vo lr in AdamW.")
    parser.add_argument("--max_mlp_lr", default=None, type=float, help="max mlp lr in AdamW.")
    parser.add_argument("--embed_alpha", default=1.0, type=float, help="embed alpha in learning rate.")
    parser.add_argument("--head_alpha", default=1.0, type=float, help="head alpha in learning rate.")
    parser.add_argument("--ln_alpha", default=1.0, type=float, help="ln alpha in learning rate.")
    parser.add_argument("--qk_alpha", default=1.0, type=float, help="qk alpha in learning rate.")
    parser.add_argument("--vo_alpha", default=1.0, type=float, help="vo alpha in learning rate.")
    parser.add_argument("--mlp_alpha", default=1.0, type=float, help="mlp alpha in learning rate.")
    parser.add_argument("--embed_gamma", default=1.0, type=float, help="embed gamma in learning rate.")
    parser.add_argument("--head_gamma", default=1.0, type=float, help="head gamma in learning rate.")
    parser.add_argument("--ln_gamma", default=1.0, type=float, help="ln gamma in learning rate.")
    parser.add_argument("--qk_gamma", default=1.0, type=float, help="qk gamma in learning rate.")
    parser.add_argument("--vo_gamma", default=1.0, type=float, help="vo gamma in learning rate.")
    parser.add_argument("--mlp_gamma", default=1.0, type=float, help="mlp gamma in learning rate.")
    parser.add_argument("--max_iters", default=30000, type=int, help="max iterations.")
    parser.add_argument("--warmup_iters", default=300, type=int, help="warmup iterations.")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta1 in AdamW.")
    parser.add_argument("--beta2", default=0.95, type=float, help="beta2 in AdamW.")
    parser.add_argument("--weight_decay", default=1e-1, type=float, help="weight decay in AdamW.")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="grad clip in AdamW.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--skip_init", action="store_true")
    args = parser.parse_args()
    args.max_embed_lr = args.max_embed_lr or args.max_lr
    args.max_head_lr = args.max_head_lr or args.max_lr
    args.max_ln_lr = args.max_ln_lr or args.max_lr
    args.max_qk_lr = args.max_qk_lr or args.max_lr
    args.max_vo_lr = args.max_vo_lr or args.max_lr
    args.max_mlp_lr = args.max_mlp_lr or args.max_lr
    setup_seed(args.seed)
    train(args)
