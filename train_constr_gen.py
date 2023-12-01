import os, sys, random
import argparse

from pathlib import Path

import numpy as np
import torch

from mdm import dataset
from mdm import constraintset
from mdm.runner import Runner
from mdm.logger import Logger

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default=None,        help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device")

    # problem & model
    parser.add_argument("--p0",             type=str,                        help="target distribution")
    parser.add_argument("--xdim",           type=int,                        help="state space dimension")
    parser.add_argument("--constraint",     type=str,   choices=["ball", "simplex", "cube"])
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--noise-sched",    type=str,   default="linear",    choices=["linear", "cosine"])

    # optimizer & loss
    parser.add_argument("--batch-size",     type=int,   default=512)
    parser.add_argument("--num-itr",        type=int,   default=50000,       help="training iteration")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.99)

    # logging
    parser.add_argument("--log-dir",        type=Path,  default=".log",      help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default=None,        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")
    parser.add_argument("--eval-itr",       type=int,   default=10000,       help="evaluation iteration")
    parser.add_argument("--save-itr",       type=int,   default=5000,        help="checkpoint iteration")

    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt / "latest.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    return opt

def main(opt):
    log = Logger(opt.log_dir, opt.global_rank)
    log.info("=======================================================")
    log.info("             Mirror Diffusion Models")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    # build constraint set & sampler
    constraint = constraintset.build(opt)
    data_sampler = dataset.build(opt, constraint)

    run = Runner(opt, log)
    run.train(opt, data_sampler, constraint)
    log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()

    opt.global_rank = 0 # single gpu
    main(opt)
