import abc
import numpy as np

import torch
import torch.distributions as td
import torch.nn.functional as F

from . import constraintset
from .constraintset import BALL_RADIUS

from ipdb import set_trace as debug

def build(opt, constraint):
    return {
        "ball": build_ball,
        "simplex": build_simplex,
        "cube": build_cube,
    }.get(opt.constraint)(opt, constraint)

def build_ball(opt, constraint):
    assert isinstance(constraint, constraintset.Ball)
    p0 = {
        "spiral": Spiral,
        "moon": Moon,
        "gmm": GMM2D,
        "gmm_nd": GMMnD,
    }.get(opt.p0)(opt, opt.batch_size, constraint)

    lim = BALL_RADIUS * 1.2
    opt.primal_lims = [[-lim, lim], [-lim, lim]]
    opt.dual_lims = p0.dual_lims
    return p0

def build_simplex(opt, constraint):
    assert isinstance(constraint, constraintset.Simplex)

    if opt.p0 == "dirichlet_nd":
        concentration = [(ii + 1) / 5 for ii in range(opt.xdim + 1)]
    else:
        concentration = {
            "dirichlet1": [2.0, 4.0, 8.0], # 3d
            "dirichlet2": [1.0, 0.1, 5.0], # 3d
            "dirichlet3": [1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0], # 7d
            "dirichlet4": [1.0, 0.5, 2.0, 0.3, 0.6, 4.0, 8.0, 8.0, 2.0], # 9d
        }.get(opt.p0)
    assert len(concentration) == opt.xdim + 1

    p0 = Dirichlet(opt, opt.batch_size, constraint, concentration)

    opt.primal_lims = [[-0.2, 1.2], [-0.2, 1.2]]
    opt.dual_lims = None

    return p0

def build_cube(opt, constraint):
    assert isinstance(constraint, constraintset.Cube)
    opt.primal_lims = [[0., 1.], [0., 1.]]
    opt.dual_lims = None
    return CubeGMMnD(opt, opt.batch_size, constraint)

class ConstraintSampler(metaclass=abc.ABCMeta):
    def __init__(self, opt, batch_size, constraint):
        self.opt = opt
        self.batch_size = batch_size
        self.constraint = constraint
        self.dim = constraint.dim

    @abc.abstractmethod
    def _sample(self, batch):
        raise NotImplementedError()

    def is_feasible(self, x):
        return self.constraint.is_feasible(x)[0]

    def sample(self):
        n, batch = 0, self.batch_size
        xs = torch.empty(batch, self.dim)
        while n < batch:
            x = self._sample(batch)
            feasible = self.is_feasible(x)
            x = x[feasible,...]

            n_feasible = x.shape[0]
            xs[n:n+n_feasible,...] = x[0:min(n_feasible,batch-n),...]
            n += n_feasible
        assert self.constraint.is_feasible(xs)[0].all()
        return xs

###################################################################################
###########################   Ball Constraint Sampler   ###########################
###################################################################################

class Moon(ConstraintSampler):
    def __init__(self, opt, batch_size, constraint):
        assert opt.xdim == 2
        assert isinstance(constraint, constraintset.Ball)
        super(Moon, self).__init__(opt, batch_size, constraint)
        self.moon_radius = BALL_RADIUS * 0.8
        self.sigma = BALL_RADIUS / 40.
        self.dual_lims = [[-3., 3.], [-3., 3.]] # plotting

    def _sample(self, batch):
        n = batch
        x = np.linspace(0.25*np.pi, 0.75*np.pi, n // 2)
        u = np.stack([np.cos(x), np.sin(x)], axis=1) * self.moon_radius
        u += self.sigma * np.random.normal(size=u.shape)
        v = np.stack([np.cos(x), -np.sin(x)], axis=1) * self.moon_radius
        v += self.sigma * np.random.normal(size=v.shape)
        x = np.concatenate([u, v], axis=0)
        return torch.Tensor(x)

class Spiral(ConstraintSampler):
    def __init__(self, opt, batch_size, constraint):
        assert opt.xdim == 2
        assert isinstance(constraint, constraintset.Ball)
        super(Spiral, self).__init__(opt, batch_size, constraint)
        self.moon_radius = BALL_RADIUS * 0.8
        self.sigma = BALL_RADIUS / 40.
        self.dual_lims = [[-18., 18.], [-30., 10.]] # plotting

    def _sample(self, batch):
        n = batch
        r = np.linspace(0, 0.95, n)
        t = np.linspace(0, 1000, n)
        x = r*np.cos(np.radians(t)) + 0.025*np.random.randn(n)
        y = r*np.sin(np.radians(t)) + 0.025*np.random.randn(n)

        return torch.Tensor(np.stack([x, y], axis=1))

class GMM2D(ConstraintSampler):
    def __init__(self, opt, batch_size, constraint):
        assert isinstance(constraint, constraintset.Ball)
        super(GMM2D, self).__init__(opt, batch_size, constraint)
        radius = BALL_RADIUS * 0.8
        sigma = BALL_RADIUS / 20.
        self.dual_lims = [[-8., 8.], [-8., 8.]] # plotting

        # build mu's and sigma's
        num = 8
        arc = 2 * np.pi / num
        xs = [np.cos(arc * idx) * radius for idx in range(num)]
        ys = [np.sin(arc * idx) * radius for idx in range(num)]
        means = [[x, y] for x,y in zip(xs,ys)]
        var = sigma * torch.ones(num,2)

        mix = td.Categorical(torch.ones(num,))
        comp = td.Independent(td.Normal(
            torch.Tensor(means),
            var),
            1
        )
        self.distribution = td.MixtureSameFamily(mix, comp)

    def _sample(self, batch):
        return self.distribution.sample([batch])

class GMMnD(ConstraintSampler):
    def __init__(self, opt, batch_size, constraint):
        assert isinstance(constraint, constraintset.Ball)
        assert opt.xdim > 2
        super(GMMnD, self).__init__(opt, batch_size, constraint)

        sigma = BALL_RADIUS / 20.
        self.dual_lims = [[-75., 300.], [-75., 300.]] # plotting

        # build mu's and sigma's
        means = F.one_hot(torch.arange(opt.xdim), num_classes=opt.xdim).float()
        var = sigma * torch.ones(opt.xdim, opt.xdim)

        mix = td.Categorical(torch.ones(opt.xdim,))
        comp = td.Independent(td.Normal(
            torch.Tensor(means),
            var),
            1
        )
        self.distribution = td.MixtureSameFamily(mix, comp)

    def _sample(self, batch):
        return self.distribution.sample([batch])

###################################################################################
###########################   Cube Constraint Sampler   ###########################
###################################################################################

class CubeGMMnD(ConstraintSampler):
    def __init__(self, opt, batch_size, constraint):
        assert isinstance(constraint, constraintset.Cube)
        super(CubeGMMnD, self).__init__(opt, batch_size, constraint)

        means = F.one_hot(torch.arange(opt.xdim), num_classes=opt.xdim).float()
        var = 0.2 * torch.ones(opt.xdim, opt.xdim)

        mix = td.Categorical(torch.ones(opt.xdim,))
        comp = td.Independent(td.Normal(
            torch.Tensor(means),
            var),
            1
        )
        self.distribution = td.MixtureSameFamily(mix, comp)

    def reflect(self, x):
        # https://github.com/louaaron/Reflected-Diffusion/blob/master/cube.py#L34
        xm2 = x % 2
        xm2[xm2 > 1] = 2 - xm2[xm2 > 1]
        return xm2

    def _sample(self, batch):
        sample = self.distribution.sample([batch])
        return self.reflect(sample)

##################################################################################
#########################   Simplex Constraint Sampler   #########################
##################################################################################

class Dirichlet(ConstraintSampler):
    def __init__(self, opt, batch_size, constraint, concentration):
        assert isinstance(constraint, constraintset.Simplex)
        super(Dirichlet, self).__init__(opt, batch_size, constraint)
        self.distribution = td.dirichlet.Dirichlet(torch.Tensor(concentration))

    def _sample(self, batch):
        return self.distribution.sample([batch])[:, :-1] # dump last dim
