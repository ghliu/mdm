from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch

from mdm import dataset
from mdm import constraintset
from mdm.runner import Runner
from mdm.logger import Logger
from mdm.metrices import compute_metrics

import colored_traceback.always
from ipdb import set_trace as debug

REF_DIR = Path("data")

def get_ref_x0(opt, log, data_sampler):
    ref_fn = REF_DIR / f"{opt.p0}_{opt.constraint}_d{opt.xdim}_b{opt.batch_size}.pt"
    if ref_fn.exists():
        ref_x0 = torch.load(ref_fn, map_location="cpu")
        log.info(f"Loaded ref points from {ref_fn}!")
    else:
        ref_fn.parent.mkdir(exist_ok=True)
        ref_x0 = data_sampler.sample()
        torch.save(ref_x0.cpu(), ref_fn)
        log.info(f"Sampled and saved ref points to {ref_fn}!")
    return ref_x0

def build_ckpt_option(opt, log, ckpt_path):
    ckpt_path = Path(ckpt_path)
    opt_pkl_path = ckpt_path / "options.pkl"
    assert opt_pkl_path.exists()
    with open(opt_pkl_path, "rb") as f:
        ckpt_opt = pickle.load(f)
    log.info(f"Loaded options from {opt_pkl_path=}!")

    overwrite_keys = ["device", "batch_size"]
    for k in overwrite_keys:
        assert hasattr(opt, k)
        setattr(ckpt_opt, k, getattr(opt, k))

    if not hasattr(ckpt_opt, "noise_sched"): ckpt_opt.noise_sched = "linear"

    ckpt_opt.load = ckpt_path / "latest.pt"
    return ckpt_opt

@torch.no_grad()
def main(opt):
    log = Logger(".log")

    # restore ckpt
    ckpt_opt = build_ckpt_option(opt, log, opt.ckpt_dir)
    constraint = constraintset.build(ckpt_opt)
    data_sampler = dataset.build(ckpt_opt, constraint)
    run = Runner(ckpt_opt, log, save_opt=False)

    # sample reference points
    ref_x0 = get_ref_x0(ckpt_opt, log, data_sampler)
    ref_x0 = ref_x0.to(opt.device)

    # compute metrics
    metrics = defaultdict(list)
    for _ in range(opt.n_run):
        # sample predict points
        pred_x0, *_ = run.generate(ckpt_opt, constraint)

        # compute metrices
        pred_x0 = pred_x0.to(opt.device)
        m_per_run = compute_metrics(pred_x0, ref_x0, constraint=constraint)
        for k, v in m_per_run.items(): metrics[k].append(v)

    for k, v in metrics.items():
        vv = torch.stack(v)
        log.info(f"{k}: {vv.mean().item():.4f}±{vv.std().item():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir",       type=Path,  default=None)
    parser.add_argument("--batch-size",     type=int,   default=512)
    parser.add_argument("--gpu",            type=int,   default=None)
    parser.add_argument("--n-run",          type=int,   default=3)
    opt = parser.parse_args()

    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'

    main(opt)