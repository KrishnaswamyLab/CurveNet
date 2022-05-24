'''
Script adapted from https://github.com/pluskid/fitting-random-labels
'''

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--command', default='train', choices=['train'])
parser.add_argument('--data', default='cifar10', choices=['cifar10','cifar100','mnist','fashion_mnist'])
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--data-augmentation', default=False, action='store_true')
parser.add_argument('--label-corrupt-prob', type=float, default=0.0)

parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--save_epoch', type=int, default=1, help='save every save_epoch')
parser.add_argument('--learning-rate', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0) #1e-4)
parser.add_argument('--optim', default='sgd', choices=['sgd','adam','adadelta'])

parser.add_argument('--eval-full-trainset', type=bool, default=True,
                    help='Whether to re-evaluate the full train set on a fixed model, or simply ' +
                    'report the running average of training statistics')

parser.add_argument('--arch', default='wide-resnet', choices=['wide-resnet', 'mlp'])

parser.add_argument('--wrn-depth', type=int, default=28)
parser.add_argument('--wrn-widen-factor', type=int, default=1)
parser.add_argument('--wrn-droprate', type=float, default=0.0)

parser.add_argument('--mlp-spec', default='512',
                    help='mlp spec: e.g. 512x128x512 indicates 3 hidden layers')

parser.add_argument('--name', default='', help='Experiment name')

#--------------------------------------------------------------------------
# Added
#--------------------------------------------------------------------------
parser.add_argument('--load_epoch', default = '200')
parser.add_argument('--rand_seed', default=0, type=int, help='seed for full training runs')
parser.add_argument('--step_size', default= 1.0, type=float, help='size of the step to be taken in the normalized direction')
parser.add_argument('--rt_name', default='', help='Retrain experiment name')
parser.add_argument('--retrain_label-corrupt-prob', type=float, default=0.0)

parser.add_argument('--retrain_rand_seed', default=0, type=int, help='seed for random num generator for retrain jumps')
parser.add_argument('--retrain_batch-size', type=int, default=128)
parser.add_argument('--retrain_epochs', type=int, default=50)
parser.add_argument('--retrain_save_epoch', type=int, default=1, help='save every save_epoch')
parser.add_argument('--retrain_saves_per_epoch', type=int, default=1, help='how many times to save at each epoch if retrain_save_epoch == 1')
parser.add_argument('--retrain_learning-rate', type=float, default=0.1)
parser.add_argument('--retrain_momentum', type=float, default=0.9)
parser.add_argument('--retrain_weight-decay', type=float, default=0)
parser.add_argument('--retrain_data-augmentation', default=False, action='store_true')
parser.add_argument('--retrain_valid_size', type=float, default=0.1)

# Analysis Arguments
parser.add_argument('--analysis_type', default='phate_diff_op_pers_diag')
parser.add_argument('--analysis_seeds', default = [0,1,2,3], nargs='+', type=int, help='retrain random seeds to reload for analysis')
parser.add_argument('--analysis_step_sizes', default = [0.25, 0.5, 0.75, 1.0], nargs='+', type=float, help='retrain step sizes to reload for analysis')
parser.add_argument('--analysis_num_epochs', default=40, type=int, help='how many epochs of retraining to load for analysis')
parser.add_argument('--analysis_reloads_per_epoch', default=1, type=int, help='how many times to load parameters per epoch, must be <= retrain_saves_per_epoch')
parser.add_argument('--analysis_include_optimum', default=False, action='store_true')

# Deprecated: (probably not used anywhere)
parser.add_argument('--start_jrexp_epoch', type=int, default=200, help='at what epoch to do the jump retrain experiment')
# to_do kind of an old version of experiment
parser.add_argument('--to_do', default='none', choices=['none', 'l2_dist', 'cos_dist', 'phate_diff_op'])
# ----------------------------------------

def format_experiment_name(args):
  name = args.name
  if name != '':
    name += '_'

  name += args.data + '_'
  if args.label_corrupt_prob > 0:
    name += 'corrupt%g_' % args.label_corrupt_prob

  name += args.arch
  if args.arch == 'wide-resnet':
    dropmark = '' if args.wrn_droprate == 0 else ('-dr%g' % args.wrn_droprate)
    name += '{0}-{1}{2}'.format(args.wrn_depth, args.wrn_widen_factor, dropmark)
  elif args.arch == 'mlp':
    name += args.mlp_spec

  name += '_seed{0}'.format(args.rand_seed)
  name += '_bs{0}'.format(args.batch_size)
  name += '_'+args.optim
  name += '_lr{0}_mmt{1}'.format(args.learning_rate, args.momentum)
  if args.weight_decay > 0:
    name += '_Wd{0}'.format(args.weight_decay)
  else:
    name += '_NoWd'
  if not args.data_augmentation:
    name += '_NoAug'

  return name

def format_retrain_name(args, hparams=False, seed=None, step_size=None):
  name = args.rt_name
  if name != '':
    name += '_'

  if args.retrain_label_corrupt_prob > 0:
    name += 'corrupt%g_' % args.retrain_label_corrupt_prob

  # added load_epoch
  name += 'e{0}'.format(args.load_epoch)
  # name += 'corrupt{0}_'.format(args.retrain_label_corrupt_prob)
  if not hparams:
      if seed is None and step_size is None:
          name += '_seed{0}'.format(args.retrain_rand_seed)
          name += '_step{0}'.format(args.step_size)
      else:
          name += '_seed{0}'.format(seed)
          name += '_step{0}'.format(step_size)

  name += '_bs{0}'.format(args.retrain_batch_size)
  name += '_lr{0}_mmt{1}'.format(args.retrain_learning_rate, args.retrain_momentum)
  if args.retrain_weight_decay > 0:
    name += '_Wd{0}'.format(args.retrain_weight_decay)
  else:
    name += '_NoWd'
  if not args.retrain_data_augmentation:
    name += '_NoAug'
  name+= '_valsize{0}'.format(args.retrain_valid_size)
  name+= '_se{0}'.format(args.retrain_save_epoch)
  if args.retrain_save_epoch == 1:
      name+= '_spe{0}'.format(args.retrain_saves_per_epoch)

  return name


def format_analysis_name(args):
    name = args.analysis_type
    name += '_seeds'
    for idx, seed in enumerate(args.analysis_seeds):
        if idx != 0:
            name += '-'
        name += '{0}'.format(seed)

    name += '_ssizes'
    for idx, step_size in enumerate(args.analysis_step_sizes):
        if idx != 0:
            name += '-'
        name += '{0}'.format(step_size)

    name += '_epochs{0}'.format(args.analysis_num_epochs)
    name += '_rpe{0}'.format(args.analysis_reloads_per_epoch)
    if args.analysis_include_optimum:
        name+= '_opt'
    return name


def parse_args():
  args = parser.parse_args()
  #args.exp_name = format_experiment_name(args)
  args.exp_name = args.name # Making experiment names simpler
  args.retrain_name = format_retrain_name(args)
  args.retrain_hparams = format_retrain_name(args, hparams = True)
  args.analysis_name = format_analysis_name(args)
  return args
