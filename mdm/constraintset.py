import abc
import numpy as np
import torch

from ipdb import set_trace as debug

BALL_RADIUS=1.

def build(opt):
    if 'polytope' in opt.constraint:
        return PolyTope(opt)
    elif 'ball' in opt.constraint:
        return Ball(opt.xdim, beta=0.5)
    elif 'simplex' in opt.constraint:
        return Simplex(opt.xdim)
    elif 'cube' in opt.constraint:
        return Cube(opt.xdim)
    else:
        raise RuntimeError()

class ConstraintSet(metaclass=abc.ABCMeta):
    def __init__(self,dim):
        self.dim = dim

    @abc.abstractmethod
    def residual(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def is_feasible(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def grad_phi(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def grad_phi_dual(self, y):
        raise NotImplementedError()

    def to_dual(self, x):
        return self.grad_phi(x)

    def to_primal(self, y):
        return self.grad_phi_dual(y)

class Cube(ConstraintSet):
    def __init__(self, xdim):
        super(Cube, self).__init__(xdim)
        self.eps = 1e-4

    def project(self, x):
        return torch.clamp(x, min=0, max=1)

    def is_feasible(self, x):
        cond1 = (x > 0).all(dim=-1)
        cond2 = (x < 1).all(dim=-1)
        assert cond1.shape == cond2.shape == (x.shape[0],)
        return torch.logical_and(cond1, cond2), None

    def residual(self, x):
        debug()
        pass

    def grad_phi(self, x):
        assert self.is_feasible(x)[0].all()
        x = torch.clamp(x, min=self.eps, max=1.-self.eps) # [0,1]
        return torch.atanh(x*2-1) # -> [-1,1] -> R

    def grad_phi_dual(self, y):
        x = torch.tanh(y) # R -> [-1,1]
        x = (x+1.)/2. # -> [0,1]
        assert self.is_feasible(x)[0].all()
        return x

class PolyTope(ConstraintSet):
    """ A^T x >= b with log-barrier """
    def __init__(self, data_dim, A, b):
        super(PolyTope, self).__init__(data_dim)
        assert A.ndim == 2 and b.ndim
        assert A.shape[0] == data_dim and A.shape[1] == b.shape[0]
        self.A = A # (xdim, m)
        self.b = b # (m,)
        self.m = A.shape[1] # num of constraints

    def residual(self, x):
        assert x.ndim == 2 and x.shape[1] == self.dim
        # (batch, xdim) @ (xdim, m) = (batch, m)
        return x @ self.A - self.b

    def is_feasible(self, x):
        batch = x.shape[0]

        residual = self.residual(x)
        feasible = (residual >= 0).all(dim=1)
        assert feasible.ndim == 1 and feasible.shape[0] == batch
        return feasible, residual

    def phi(self, x):
        """ phi(x) = - sum_i^m log(a_i^T x - b_i) """
        feasible, residual = self.is_feasible(x)
        assert feasible.all()
        return - torch.log(residual).sum(dim=1)

    def grad_phi(self, x):
        """ \nabla phi(x) = - sum_i^m a_i/(a_i^T x - b_i) """
        feasible, residual = self.is_feasible(x)
        assert feasible.all()
        # (batch, m) @ (m, xdim) = (batch, xdim)
        return - (1/residual) @ self.A.t()

class Simplex(ConstraintSet):
    def __init__(self, xdim):
        super(Simplex, self).__init__(xdim)

        self.xdim = xdim
        self.n_cnstnt = xdim + 1

        A = np.concatenate([-np.eye(xdim), np.ones((1,xdim))], axis=0)
        B = np.zeros((xdim+1,1)); B[-1] = 1
        assert A.shape == (xdim+1, xdim) and B.shape == (xdim+1, 1)

        self.A = torch.from_numpy(A).float().cuda() # m, m-1
        self.B = torch.from_numpy(B).float().cuda() # m

    def project(self, x):
        x[x < 0] = 0
        x[x.sum(dim=1) > 1.] = x[x.sum(dim=1) > 1.] / x[x.sum(dim=1) > 1.].sum(dim=1, keepdim=True)
        return x

    def violation_mtx(self, x):
        A, B = self.A.to(x.device), self.B.to(x.device)
        residual = B.t() - x @ A.t()
        return (residual < 0).long()

    def residual(self, x):
        assert x.ndim == 2 and x.shape[1] == self.dim, 'Get x.shape={}'.format(x.shape)
        return 1. - x.sum(dim=1) # (batch,)

    def residual_dual(self, y):
        assert y.ndim == 2 and y.shape[1] == self.dim, 'Get y.shape={}'.format(x.shape)
        res_dual = 1. + y.exp().sum(dim=1)
        assert not (torch.isinf(res_dual).any() or torch.isnan(res_dual).any())
        return res_dual

    def is_feasible(self, x):
        batch = x.shape[0]

        residual = self.residual(x)
        feasible = (residual >= -1e-5) * (x >= -1e-5).all(dim=1)
        # feasible = (residual >= 0)
        assert feasible.ndim == 1 and feasible.shape[0] == batch
        return feasible, residual

    def phi(self, x):
        feasible, residual = self.is_feasible(x)
        assert feasible.all()
        return (x * torch.log(x)).sum(dim=1) + residual * torch.log(residual)

    def grad_phi(self, x):
        feasible, residual = self.is_feasible(x)
        assert feasible.all()
        return torch.log(x) - torch.log(residual)[:,None]

    def grad_phi_dual(self, y):
        residual = self.residual_dual(y) # (batch,)
        return torch.exp(y) / residual[:,None]

    def hess_phi_dual(self, y):
        residual = self.residual_dual(y)    # (batch,)
        Y = torch.exp(y) / residual[:,None] # (batch, xdim)
        D = torch.diag_embed(Y)             # (batch, xdim, xdim)
        # (batch, xdim, 1) @ (batch, 1, xdim) = (batch, xdim, xdim)
        return D - Y[...,None] @ Y[:,None,...]

class Ball(ConstraintSet):
    """ |x|^2 <= R^2 with log-barrier """
    def __init__(self, dim, beta=0.5):
        super(Ball, self).__init__(dim)
        self.beta = beta
        self.radius = BALL_RADIUS

    def project(self, x):
        infeasible = ~self.is_feasible(x)[0]
        x[infeasible] = x[infeasible] / x[infeasible].norm(dim=1,keepdim=True)
        return x

    def residual(self, x):
        assert x.ndim == 2 and x.shape[1] == self.dim, 'Get x.shape={}'.format(x.shape)
        return self.radius**2 - x.norm(dim=-1)**2 # (batch,)

    def is_feasible(self, x):
        batch = x.shape[0]

        residual = self.residual(x)
        feasible = residual >= 1e-5 # nonzero for numerical stability
        assert feasible.ndim == 1 and feasible.shape[0] == batch
        return feasible, residual

    def phi(self, x, check=True):
        """  """
        feasible, residual = self.is_feasible(x)
        if check: assert feasible.all()
        return - self.beta * torch.log(residual)

    def grad_phi(self, x, check=True):
        """ \nabla phi(x) = 2 * beta * x / (1-x.norm()**2) """
        feasible, residual = self.is_feasible(x)
        if check: assert feasible.all()
        return (2*self.beta/residual)[:,None] * x

    def grad_phi_dual(self, y):
        """ \nabla phi*(y) = y / (sqrt{y.norm()**2 + beta**2} + beta)  """
        beta, norm_y = self.beta, y.norm(dim=-1, keepdim=True)**2
        return self.radius**2 / (torch.sqrt(self.radius**2 * norm_y + beta**2) + beta) * y

    def hess_phi_dual(self, y):
        beta = self.beta
        ynorm_beta = torch.sqrt(self.radius*y.norm(dim=-1)**2 + beta**2) # (batch,)

        s1 = self.radius/(ynorm_beta + beta)
        s2 = s1/ynorm_beta
        s1, s2 = s1[:,None,None], s2[:,None,None] # (batch, 1, 1)

        I = batch_eye_like(y) # (batch, xdim, xdim)
        yyT = y[...,None] @ y[:,None,...] # (batch, xdim, xdim)
        return s1 * (I - s2 * yyT)

def batch_eye_like(x):
    """ given x.shape = [batch, xdim], return batch eye with shape [batch, xdim, xdim] """
    batch, dim, device = x.shape[0], x.shape[1], x.device
    return torch.eye(dim, device=device).reshape(1,dim,dim).repeat(batch,1,1)

