'''
Script adapted from https://github.com/pluskid/fitting-random-labels
'''

from __future__ import print_function

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim

from cifar10_data import CIFAR10RandomLabels
from cifar10_data import CIFAR100RandomLabels
from cifar10_data import MNISTRandomLabels
from cifar10_data import FashionMNISTRandomLabels

import cmd_args
import model_mlp, model_wideresnet
import random

def get_data_loaders(args, shuffle_train=True):
  if args.data == 'cifar10':
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.data_augmentation:
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
          ])
    else:
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
  elif args.data == 'cifar100':
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.data_augmentation:
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
          ])
    else:
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        CIFAR100RandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        CIFAR100RandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
  elif args.data == 'mnist':
    #normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     #std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.data_augmentation:
      transform_train = transforms.Compose([
          transforms.RandomCrop(28, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          #normalize,
          ])
    else:
      transform_train = transforms.Compose([
          transforms.ToTensor()#,
          #normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor()#,
        #normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        MNISTRandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        MNISTRandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
  elif args.data == 'fashion_mnist':
    #normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     #std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.data_augmentation:
      transform_train = transforms.Compose([
          transforms.RandomCrop(28, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          #normalize,
          ])
    else:
      transform_train = transforms.Compose([
          transforms.ToTensor()#,
          #normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor()#,
        #normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        FashionMNISTRandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        FashionMNISTRandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
  else:
    raise Exception('Unsupported dataset: {0}'.format(args.data))


def get_model(args):
  # create model
  random.seed(args.rand_seed)
  np.random.seed(args.rand_seed)
  torch.manual_seed(args.rand_seed)
  torch.cuda.manual_seed_all(args.rand_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  if args.arch == 'wide-resnet':
    if args.data == 'cifar10' or args.data == 'cifar100':
      model = model_wideresnet.WideResNet(args.wrn_depth, args.num_classes,
                                        args.wrn_widen_factor,
                                        drop_rate=args.wrn_droprate)
    elif args.data == 'mnist' or args.data=='fashion_mnist':
      model = model_wideresnet_mnist.WideResNet(args.wrn_depth, args.num_classes,
                                        args.wrn_widen_factor,
                                        drop_rate=args.wrn_droprate)
  elif args.arch == 'mlp':
    n_units = [int(x) for x in args.mlp_spec.split('x')] # hidden dims
    n_units.append(args.num_classes)  # output dim
    if args.data == 'cifar10' or args.data == 'cifar100':
      n_units.insert(0, 32*32*3)
    elif args.data == 'mnist' or args.data=='fashion_mnist':
      n_units.insert(0, 28*28*1)        # input dim
    model = model_mlp.MLP(n_units)

  # for training on multiple GPUs.
  # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
  # model = torch.nn.DataParallel(model).cuda()
  model = model.cuda()

  return model

def get_model_params(model):
    for count, params in enumerate(model.parameters()):
        if count == 0:
            net_params = torch.flatten(params.data).cpu().numpy()
        else:
            net_params = np.append(net_params,torch.flatten(params.data).cpu().numpy())
    return net_params


def train_model(args, model, train_loader, test_loader,
                start_epoch=None, epochs=None):
    
  epochtrain_loss = []
  epochtest_loss = []
  epochparams = []

  #cudnn.benchmark = True
  random.seed(args.rand_seed)
  np.random.seed(args.rand_seed)
  torch.manual_seed(args.rand_seed)
  torch.cuda.manual_seed_all(args.rand_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # define loss function (criterion) and pptimizer
  criterion = nn.CrossEntropyLoss().cuda()
  if args.optim == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  elif args.optim == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  elif args.optim == 'adadelta':
      optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

  start_epoch = start_epoch or 1
  epochs = epochs or (args.epochs + 1)

  exp_dir = os.path.join('runs', args.exp_name)

  ############################################
  test_loss, test_acc = validate_epoch(test_loader, model, criterion, 0, args)
  train_loss, train_acc = validate_epoch(train_loader, model, criterion, 0, args)

  state = {
      'train_accuracy': train_acc,
      'train_loss': train_loss,
      'test_accuracy': test_acc,
      'test_loss': test_loss,
      'epoch': 0,
      'state_dict': model.state_dict(keep_vars=True)
  }
  opt_state = {
      'optimizer': optimizer.state_dict()
  }
  torch.save(state, exp_dir + '/model_e0.t7')
  torch.save(opt_state, exp_dir + '/opt_state_e0.t7')
  logging.info('%03d: Acc-tr: %6.2f, Acc-test: %6.2f, L-tr: %6.4f, L-test: %6.4f',
               0, train_acc, test_acc, train_loss, test_loss)
  ############################################
  
  for epoch in range(start_epoch, epochs):
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, args)

    # evaluate on validation set
    test_loss, test_acc = validate_epoch(test_loader, model, criterion, epoch, args)

    if args.eval_full_trainset:
      train_loss, train_acc = validate_epoch(train_loader, model, criterion, epoch, args)

    # ### Code for saving the gradients
    # with torch.no_grad():
    #     for count, params in enumerate(model.parameters()):
    #         if count == 0:
    #             grads = torch.flatten(params.grad.data).cpu().numpy()
    #         else:
    #             grads = np.append(grads,torch.flatten(params.grad.data).cpu().numpy())
    # np.save(exp_dir+'/gradients_e'+str(epoch)+'.npy', grads)
    # ###

    ############################################
    if epoch == args.epochs or epoch % 25 == 0 or epoch>175:
    # if epoch == 300 or epoch % args.save_epoch == 0 or epoch == 150:
    # if epoch % args.save_epoch == 0 or epoch == 150:
        state = {
            'train_accuracy': train_acc,
            'train_loss': train_loss,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'epoch': epoch,
            'state_dict': model.state_dict(keep_vars=True),
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, exp_dir + '/model_e' + str(epoch) + '.t7')
        torch.save(opt_state, exp_dir + '/opt_state_e' + str(epoch) + '.t7')
    ############################################

    logging.info('%03d: Acc-tr: %6.2f, Acc-test: %6.2f, L-tr: %6.4f, L-test: %6.4f',
                 epoch, train_acc, test_acc, train_loss, test_loss)
    
    params = [p.data.detach().cpu().numpy().flatten() for p in model.parameters()]
    nparams = np.concatenate([params[0],params[1]]) # for a 3 layer NN only need to concatenate two dimensions
    
    
    epochtest_loss.append(test_loss)
    epochtrain_loss.append(train_loss)
    epochparams.append(nparams)    

    
  return epochtrain_loss, epochtest_loss, epochparams


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
  """Train for one epoch on the training set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to train mode
  model.train()

  for i, (input, target) in enumerate(train_loader):
    target = target.cuda(non_blocking=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    losses.update(loss.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return losses.avg, top1.avg


def validate_epoch(test_loader, model, criterion, epoch, args):
  """Perform validation on the validation set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(test_loader):
    target = target.cuda(non_blocking=True)
    input = input.cuda()
    with torch.no_grad():
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

  return losses.avg, top1.avg


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    if args.epochs == 200:
        if epoch == 100:
            lr = args.learning_rate * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == 150:
            lr = args.learning_rate * (0.1**2)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == 185:
            lr = args.learning_rate * (0.1**3)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res


def setup_logging(args):
  import datetime
  exp_dir = os.path.join('runs', args.exp_name)
  if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)
  log_fn = os.path.join(exp_dir, "LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
  logging.basicConfig(filename=log_fn, filemode='w', level=logging.DEBUG)
  # also log into console
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)
  print('Logging into %s...' % exp_dir)


def main():
  args = cmd_args.parse_args()
  setup_logging(args)

  random.seed(args.rand_seed)
  np.random.seed(args.rand_seed)
  torch.manual_seed(args.rand_seed)
  torch.cuda.manual_seed_all(args.rand_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  if args.command == 'train':
    train_loader, test_loader = get_data_loaders(args, shuffle_train=True)
    model = get_model(args)
    logging.info('Number of parameters: %d', sum([p.data.nelement() for p in model.parameters()]))
    trainlossvalues, testlossvalues, paramvalues = train_model(args, model, train_loader, test_loader)
    
  torch.save(model.state_dict(),"default_model.pt")
    
  params = get_model_params(model)
  print(params.shape)
  
  np.save("trainloss.npy",trainlossvalues)
  np.save("paramvalues.npy",paramvalues)
  np.save("testloss.npy",testlossvalues)

if __name__ == '__main__':
  main()
