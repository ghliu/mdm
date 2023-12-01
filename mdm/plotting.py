
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon

from .constraintset import BALL_RADIUS

from ipdb import set_trace as debug

def save_plot(fig, fn):
    fn = Path(fn)
    if not fn.exists():
        fn.parent.mkdir(exist_ok=True)

    plt.tight_layout()
    plt.savefig(fn)
    plt.close(fig)

def get_constraint(opt):
    if opt.constraint == "ball":
        return Wedge((0, 0), BALL_RADIUS, 0, 360, width=0.01, color="k", zorder=0)
    elif opt.constraint == "simplex":
        return Polygon(np.array([[0,0], [1,0], [0,1]]), closed=True, fill=False)
    elif opt.constraint == "cube":
        return Polygon(np.array([[0,0], [1,0], [1,1], [0,1]]), closed=True, fill=False)
    else:
        raise RuntimeError()

def plot_primal(opt, ax, x0):
    x0 = x0.cpu()

    constraint = get_constraint(opt)
    ax.add_artist(constraint)
    ax.scatter(x0[:, 0], x0[:, 1], s=2, color='royalblue')
    ax.set(xlim=opt.primal_lims[0], ylim=opt.primal_lims[1])

def save_snapshot(opt, fn, constraint, x0=None, y0=None):
    if opt.xdim == 2:
        save_2dsnapshot(opt, fn, constraint, x0=x0, y0=y0)
    else:
        save_ndsnapshot(opt, fn, constraint, x0=x0, y0=y0)

def save_ndsnapshot(opt, fn, constraint, x0=None, y0=None):
    assert not (x0 is None and y0 is None)

    # setup fig & ax
    proj_dims = [[0,1], [2,3], [4,5]] if opt.xdim >= 6 else [[0,1], [1,2], [2,0]]
    ncol, nrow, ax_length = len(proj_dims), 2, 2
    figsize = (ncol*ax_length, nrow*ax_length)
    fig     = plt.figure(figsize=figsize, constrained_layout=False)
    axs     = fig.subplots(nrow, ncol)

    # compute x0, y0
    if x0 is not None and y0 is None:
        y0 = constraint.to_dual(x0)
    elif x0 is None and y0 is not None:
        x0 = constraint.to_primal(y0)
    else:
        raise RuntimeError()

    # primal
    for idx, (dim0, dim1) in enumerate(proj_dims):
        plot_primal(opt, axs[0, idx], x0[:, [dim0, dim1]])

        # dual
        y0 = y0.cpu()
        axs[1, idx].scatter(y0[:, dim0], y0[:, dim1], s=2, color='salmon')
        if opt.dual_lims is not None:
            axs[1, idx].set(xlim=opt.dual_lims[0], ylim=opt.dual_lims[1])

    save_plot(fig, fn)

def save_2dsnapshot(opt, fn, constraint, x0=None, y0=None):
    assert opt.xdim == 2
    assert not (x0 is None and y0 is None)

    # setup fig & ax
    ncol, nrow, ax_length = 2, 1, 2
    figsize = (ncol*ax_length, nrow*ax_length)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    axs = fig.subplots(nrow, ncol)

    # compute x0, y0
    if x0 is not None and y0 is None:
        y0 = constraint.to_dual(x0)
    elif x0 is None and y0 is not None:
        x0 = constraint.to_primal(y0)
    else:
        raise RuntimeError()

    # primal
    plot_primal(opt, axs[0], x0)

    # dual
    y0 = y0.cpu()
    axs[1].scatter(y0[:, 0],y0[:, 1], s=2, color='salmon')
    if opt.dual_lims is not None:
        axs[1].set(xlim=opt.dual_lims[0], ylim=opt.dual_lims[1])

    # save
    save_plot(fig, fn)
