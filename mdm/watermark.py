from pathlib import Path
from tqdm import tqdm
import torch

def gramschmidt(V):
    """" modified Gram-Schmidt: https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process """
    n, k = V.shape
    U = torch.zeros_like(V, device=V.device)
    U[:,0] = V[:,0] / V[:,0].norm(keepdim=True)
    pbar = tqdm(range(1, k), desc="gramschmidt")
    for i in pbar:
        U[:,i] = V[:,i]
        for j in range(i):
            U[:,i] = U[:,i] - (U[:,j] * U[:,i]).sum() * U[:,j]
        U[:,i] = U[:,i] / U[:,i].norm(keepdim=True)

    return U

def create_orthonormal_basis(dim, N):
    assert N > 1
    V = torch.randn(dim, N)
    U = gramschmidt(V)
    assert U.shape == (dim, N)

    # sanity check
    for i in range(U.shape[1]):
        for j in range(i, U.shape[1]):
            if i != j:
                assert (U[:,i] * U[:,j]).sum().abs() <= 1e-4
    return U

def satisfy_constraint(A, x, upper, lower):
    coeff = x @ A
    cond1 = (lower <= coeff).all(dim=1)
    cond2 = (coeff <= upper).all(dim=1)
    return torch.logical_and(cond1, cond2)

def s_inv(dot_ay, b, c):
    rescale = torch.tanh(dot_ay) # output: (-1,1)
    assert torch.logical_and(-1. <= rescale, rescale <= 1.).all()

    dot_ax = (rescale + 1.) * (b - c) / 2. + c
    assert torch.logical_and(c <= dot_ax, dot_ax <= b).all()
    return dot_ax

def s(dot_ax, b, c):
    rescale = (dot_ax - c) / (b - c) * 2. - 1.
    assert torch.logical_and(-1. < rescale, rescale < 1.).all()
    return torch.atanh(rescale)

def project(dot_ax, upper, lower, random_proj=True):
    """
        dot_ax: (B, N)
        ===
        dot_ax_proj: (B, N)
    """
    (B, N), device = dot_ax.shape, dot_ax.device

    if random_proj: # random projection
        violate_l_mask = (dot_ax < lower).long()
        violate_u_mask = (dot_ax > upper).long()

        def rand_range(l, h):
            assert h > l
            return torch.rand(B, N, device=device) * (h - l) + l

        dot_ax_proj = (1-violate_l_mask) * dot_ax + violate_l_mask * rand_range(lower, 0.)
        dot_ax_proj = (1-violate_u_mask) * dot_ax_proj + violate_u_mask * rand_range(0., upper)
    else:
        dot_ax_proj = torch.clamp(dot_ax, lower, upper)

    assert dot_ax.shape == dot_ax_proj.shape == (B, N)
    return dot_ax_proj

def to_dual(A, x, b, c, eps, random_proj=True):
    """
        A: (D, N)
        x: (B, D)
        b: upper bound
        c: lower bound
        ===
        y: (B, D)
    """
    assert A.shape[0] == x.shape[1]
    (D, N), B = A.shape, x.shape[0]

    dot_ax = x @ A
    dot_ax_proj = project(dot_ax, b - eps, c + eps, random_proj=random_proj)
    assert dot_ax.shape == dot_ax_proj.shape == (B, N)
    assert torch.logical_and(c < dot_ax_proj, dot_ax_proj < b).all()

    coeff = s(dot_ax_proj, b, c)
    assert coeff.shape == (B, N)

    y = x + (coeff - dot_ax) @ A.t()
    assert y.shape == (B, D)
    return y

def to_primal(A, y, b, c):
    """
        A: (D, N)
        y: (B, D)
        b: upper bound
        c: lower bound
        ===
        x: (B, D)
    """
    assert A.shape[0] == y.shape[1]
    (D, N), B = A.shape, y.shape[0]

    dot_ay = y @ A
    coeff = s_inv(dot_ay, b, c)
    assert dot_ay.shape == coeff.shape == (B, N)

    x = y + (coeff - dot_ay) @ A.t()
    assert x.shape == (B, D)
    return x

def build(opt):
    if 'imagewatermark' in opt.constraint:
        return ImageWatermark(opt.p0, opt.device)
    else:
        raise RuntimeError()

def get_default_upper_lower_bounds(name):
    bound = {
        "ffhq": 1.05,
        "afhqv2": 0.9,
    }.get(name)
    return bound, -bound

class ImageWatermark:

    dim = 3*64*64
    n_constraint = 100
    eps = 1e-4 # if dim < 4000 else 5e-3

    def __init__(self, dataset_name, orthn_basis=None, upper=None, lower=None, device=torch.device('cuda')):
        if orthn_basis is None:
            # compute orthonormal basis
            ortho_fn = Path(f"data/orthobasis_3x64x64.pt")
            if ortho_fn.exists():
                orthn_basis = torch.load(ortho_fn)
                print(f"Loaded existing orthonormal basis from {ortho_fn}!")
            else:
                print(f"Create orthonormal basis for dim=3x64x64 (this can take a while) ...")
                orthn_basis = create_orthonormal_basis(self.dim, self.n_constraint)
                torch.save(orthn_basis, ortho_fn)
                print(f"Saved new orthonormal basis to {ortho_fn}!")

        if upper is None or lower is None:
            upper, lower = get_default_upper_lower_bounds(dataset_name)
            print(f"Loaded default bounds for {dataset_name}: {upper=}, {lower=}!")

        self.A = orthn_basis.to(device)
        self.device = device
        self.upper = upper
        self.lower = lower

    def vectorize(self, x, device):
        return x.reshape(x.shape[0], -1).to(device)

    def is_feasible(self, x):
        vec_x = self.vectorize(x, self.device)
        return satisfy_constraint(self.A, vec_x, self.upper, self.lower)

    def detect(self, x): return self.is_feasible(x)

    def to_dual(self, x):
        vec_x = self.vectorize(x, self.device)
        vec_y = to_dual(self.A, vec_x, self.upper, self.lower, self.eps)
        return vec_y.reshape_as(x).to(x.device)

    def to_primal(self, y):
        vec_y = self.vectorize(y, self.device)
        vec_x = to_primal(self.A, vec_y, self.upper, self.lower)
        return vec_x.reshape_as(y).to(y.device)

    def __call__(self, x):
        dual_imgs = self.to_dual(x)
        return self.to_primal(dual_imgs)