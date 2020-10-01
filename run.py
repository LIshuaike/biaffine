import os
import torch
import argparse
from dparser.config import Config
from dparser.utils.parallel import init_device
from dparser.utils.log import log
from dparser.cmds import Evaluate, Predict, Train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.')
    parser.add_argument('--local_rank',
                        default='-1',
                        type=int,
                        help='node rank for distributed training')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--conf',
                               '-c',
                               default='config.ini',
                               help='path to config file')
        subparser.add_argument('--model',
                               '-m',
                               default='saved/ctb7/model',
                               help='path to model file')
        subparser.add_argument('--vocab',
                               '-v',
                               default='saved/ctb7/vocab',
                               help='path to vocab file')
        subparser.add_argument('--device',
                               '-d',
                               default='-1',
                               help='ID of GPU to use')
        subparser.add_argument('--preprocess',
                               '-p',
                               action='store_true',
                               help='whether to preprocess the data first')
        subparser.add_argument('--seed',
                               '-s',
                               default=1,
                               type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads',
                               '-t',
                               default=4,
                               type=int,
                               help='max num of threads')
    args = parser.parse_args()

    log(f"Set the max num of threads to {args.threads}")
    log(f"Set the seed for generating random numbers to {args.seed}")
    log(f"Set the device with ID {args.device} visible")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    init_device(args.device, args.local_rank)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log("Override the default configs with parsed arguments")
    config = Config(args.conf)
    config.update(vars(args))
    log(config)

    log(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(config)