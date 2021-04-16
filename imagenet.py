from __future__ import print_function

import argparse
from collections import defaultdict
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p
from models.resnet import resnet 

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='/data/imagenet/ILSVRC2012', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8). Increase this if GPU utilization is low and you have more cores available.')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                    help='test batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[31, 61],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--inspect-activations', action='store_true', help='Get statistics of activations by running the network on the validation set')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--cpu-only', action='store_true', help='ignore gpu and run on cpu only.')

# Architecture
parser.add_argument('--depth', type=int, default=50, help='Model depth.')

# recalibration type
parser.add_argument('--recalibration-type', type=str, metavar='recalibration',
                    help='recalibration type {se, srm, ge, meanrew, multise, eca} (default: None)')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
if not args.cpu_only:
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print('WARNING: GPU is not available, running on CPU instead.')
else:
    use_cuda = False

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    print("Start time: "+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    model = resnet(
                depth=args.depth,
                recalibration_type=args.recalibration_type,
            )
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    print(model)
    print('    Total params: %.4fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = set_optimizer(model, args)

    # Resume
    title = 'Resnet{}-{}'.format(args.depth, args.recalibration_type)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    elif args.pretrained:
        # Load checkpoint.
        print('==> Start from pretrained checkpoint..')
        assert os.path.isfile(args.pretrained), 'Error: no checkpoint directory found!'
        # args.checkpoint = os.path.dirname(args.pretrained)
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.',
            'Train Acc.5', 'Valid Acc.5', 'Train Time', 'Test Time'])
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.',
            'Train Acc.5', 'Valid Acc.5', 'Train Time', 'Test Time'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc, test_acc5, _ = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f (Top-1), %.2f (Top-5)' % (test_loss, test_acc, test_acc5))
        return

    if args.inspect_activations:
        print('Inspecting activations')
        get_activations(val_loader, model, use_cuda)
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, train_acc5, train_time = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_acc5, test_time = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, train_acc5,
            test_acc5, train_time, test_time])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'acc': test_acc,
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         }, is_best, checkpoint=args.checkpoint)

    logger.close()

    print('Best acc:')
    print(best_acc)
    
    print("Finish time: "+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg)
        bar.next()
        del inputs, targets, outputs, loss, prec1, prec5
    bar.finish()
    return (losses.avg, top1.avg, top5.avg, bar.elapsed/60.)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg)
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg, bar.elapsed/60.)


def get_all_layers(model):
    visualisation = {}
    def hook_fn(m, i, o):
        visualisation[m] = o

    for name, layer in model._modules.items():
        #If it is a sequential, don't register a hook on it but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        else:
            # it's a non sequential. Register a hook
            layer.register_forward_hook(hook_fn)

def get_activations(val_loader, model, use_cuda):
    model.eval()

    abs_means = defaultdict(float)
    means = defaultdict(float)
    stds = defaultdict(float)
    def save_stats(name):
        def hook(m, i, o):
            abs_means[name] += torch.mean(torch.abs(o)) #Not really necessary, as std should give similar info.
            means[name] += torch.mean(o) #, dim=(2,3)
            if not name == 'fc' and len(o.shape) == 4 and o.shape[2] > 1 and o.shape[3] > 1:
                stds[name] += torch.std(o) #, dim=(2,3)
            else:
                stds[name] += 0
        return hook
    
    for name, module in model.named_modules():
        module.register_forward_hook(save_stats(name))

    bar = Bar('Processing', max=len(val_loader))
    batch_time = AverageMeter() #Note: measured without dataloading, in contrast to all other functions in this script!
    num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        start = time.time()

        with torch.no_grad():
            # with torch.autograd.profiler.profile() as prof:
            outputs = model(inputs)

        batch_time.update(time.time() - start)

        num += 1
        bar.next()
    bar.finish()

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def set_optimizer(model, args):
    params = [{'params': [p for p in model.parameters() if not getattr(p, 'srm_param', False)]},
              {'params': [p for p in model.parameters() if getattr(p, 'srm_param', False)], #getattr: (object, name, default)
               'lr': args.lr, 'weight_decay': 0}]

    optimizer = optim.SGD(params, 
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    return optimizer

if __name__ == '__main__':
    main()
