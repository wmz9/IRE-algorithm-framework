import argparse
import torch
from model.smooth_cross_entropy import smooth_crossentropy
from model.models import get_model
from data.dataset import CIFAR10_base,CIFAR100_base
import pickle
import numpy as np
import random
from copy import deepcopy
import time
from datetime import datetime
import os

import sys; sys.path.append("..")
from sgdire import SGDIRE
from samire import SAMIRE
#from asam import ASAM

import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def train(args):

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.dataset =='CIFAR10':
        dataset = CIFAR10_base(args.batch_size, 4, autoaugment=args.autoaugment)
        num_classes = 10
    elif args.dataset =='CIFAR100':
        dataset = CIFAR100_base(args.batch_size, 4, autoaugment=args.autoaugment)
        num_classes = 100

    # use the same batch to update IRE mask
    for batch in dataset.estimate:
        estimate_images, _ = (b.to(device) for b in batch)
        break

    
    print(f"Rank:{args.rank}  Prog:{args.prog} {args.IRE_start_epoch} random seed:{args.random_seed}")
    
    start_epoch=0
    iter_num = 0
    if args.loading_dir !=None:
    
        with open(args.loading_dir, "rb") as f:
            net_history = pickle.load(f)
            nets = net_history["net"]
            model = get_model(args.model,num_classes=num_classes).to(device)
            model.load_state_dict(nets)

        start_epoch = net_history['epoch']
        iter_num = net_history['iter']
    else:
        model = get_model(args.model,num_classes=num_classes).to(device)
    model = model.to(device)

    if args.base_optimizer=='SGD':
        base_optimizer = torch.optim.SGD
    elif args.base_optimizer =='AdamW':
        base_optimizer = torch.optim.AdamW
    if args.method == "SGD" :
        optimizer = SGDIRE(model.parameters(), base_optimizer, rank=args.rank,  beta=args.beta_Fisher,  lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.method =='AdamW':
        optimizer = SGDIRE(model.parameters(), base_optimizer, rank=args.rank,  beta=args.beta_Fisher,  lr=args.lr, weight_decay=args.weight_decay)
    elif args.method == "SAM":
        if args.base_optimizer=='SGD':
            optimizer = SAMIRE(model.parameters(), base_optimizer, rho=args.rho, rank=args.rank,  beta=args.beta_Fisher, 
                         lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else: optimizer = SAMIRE(model.parameters(), base_optimizer, rho=args.rho, rank=args.rank,  beta=args.beta_Fisher, 
                         lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    if start_epoch>0:
        optimizer.load_state_dict(net_history['optimizer'])
        scheduler.load_state_dict(net_history['scheduler'])

    best_acc = 0
    prog = 0.0
    
    print("Epoch | Train Loss | Train Acc | Test Loss | Test Acc ")
    for epoch in range(start_epoch, args.epochs):
        
        model.train()
        total_loss = 0.0
        correct = 0

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            optimizer.zero_grad()

            # update mask after args.IRE_start_epoch
            if (epoch +1) > args.IRE_start_epoch:
                if args.prog>0 and iter_num % args.mask_interval == 0:
                    with torch.enable_grad():
                        predictions = model(estimate_images)
                        samp_dist = torch.distributions.Categorical(logits=predictions)
                        y_sample = samp_dist.sample()
                        criterion = torch.nn.CrossEntropyLoss()
                        loss = criterion(predictions, y_sample)
                        loss.backward()
                        optimizer.update_mask()
            
            optimizer.zero_grad()
            # forward-backward step
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()

            if (epoch +1) > args.IRE_start_epoch:
                prog = args.prog
            if args.method == "SGD" or args.method =='AdamW':
                optimizer.descent_step(prog, (epoch +1) > args.IRE_start_epoch)

            else:
                optimizer.ascent_step()

                # second forward-backward step
                smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.descent_step(prog, (epoch +1) > args.IRE_start_epoch)

            total_loss += loss.mean().cpu().item() 
            correct += (torch.argmax(predictions.data, 1) == targets).cpu().sum()

            iter_num += 1
        train_ave_loss = total_loss/len(dataset.train)
        train_acc = correct/len(dataset.train_set)
        
        model.eval()
        total_loss = 0.0
        correct = 0

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                total_loss += loss.mean().cpu().item() 
                correct += (torch.argmax(predictions.data, 1) == targets).cpu().sum()
        test_ave_loss = total_loss/len(dataset.test)
        test_acc = correct/len(dataset.test_set)
        if test_acc > best_acc:
            best_acc = test_acc

        scheduler.step()
        if (epoch+1) % 5 == 0:
            print(f"{epoch + 1:3d}  {train_ave_loss:11.4f}  {train_acc * 100:10.2f}%  {test_ave_loss:10.4f}  {test_acc * 100:8.2f}%")
        if (epoch+1)  == args.saving_epoch:
            net_history = {}
            net_copy = deepcopy(model)
            net_history['net'] = net_copy.cpu().state_dict()
            net_history['optimizer'] = optimizer.state_dict()
            net_history['scheduler'] = scheduler.state_dict()
            net_history['epoch'] = epoch + 1
            net_history['iter'] = iter_num
            save_net_path = f"checkpoints/{args.dataset}/"
            if not os.path.exists(save_net_path):
                os.makedirs(save_net_path)
            with open(f'{save_net_path}/{args.method}_{args.model}_{epoch+1}.p', "wb") as f:
                pickle.dump(net_history, f)

    print(f"Best Accuracy: {best_acc*100 :.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='CIFAR10', type=str, help="CIFAR10 or CIFAR100.")
    parser.add_argument("--model", default='resnet56', type=str, help="Name of model architecure")
    parser.add_argument("--base_optimizer", default='SGD', type=str, help="SGD or AdamW")
    parser.add_argument("--method", default='SGD', type=str, help="SGD, SAM or ASAM")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum.")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay factor.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--loading_dir", default=None, type=str, help="loading checkpoint, if None, training from scratch.")
    parser.add_argument("--autoaugment", action='store_true', help="apply autoaugment transformation.")
    parser.add_argument("--IRE_start_epoch", default=70, type=int, help="Where to start training")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--beta_Fisher", default=0.0, type=float, help="beta_Fisher in IRE.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho for SAM and ASAM.")
    parser.add_argument("--rank", default=0.1, type=float, help="1-Sparsity")
    parser.add_argument("--prog", default=0, type=float, help="mu for IRE, don't use IRE if set to 0")
    parser.add_argument("--mask_interval", default=10, type=int, help="Mask iterations.")
    parser.add_argument("--saving_epoch", default=700, type=int, help="when to save the dir.")
    parser.add_argument("--gpu", default=0, type=int, help="Choose cirtain GPU")
    parser.add_argument("--random_seed", default=41, type=int, help="random_seed")
    args = parser.parse_args()
    assert args.dataset in ['CIFAR10', 'CIFAR100'], \
            f"Invalid data type. Please select CIFAR10 or CIFAR100"
    setup_seed(args.random_seed)
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    temp = sys.stdout
    save_path = f'./log/{args.model}-{args.method}-{args.epochs}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(f'{save_path}/{formatted_time}.log', 'w')
    sys.stdout = f
    print(args)
    train(args)
    f.close()

# python cifar_IRE_same.py --dataset=CIFAR10 --model=resnet56 --base_optimizer=SGD --method=SGD --batch_size=128 --lr=0.1 --momentum=0 --epochs=100 --IRE_start_epoch=60 --rank=0.01 --prog=0.0 --gpu=0 --random_seed=1
