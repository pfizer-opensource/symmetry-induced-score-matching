""" 
Code for diffusion noise schedulers and generalized score matching
"""

import torch
from torch import Tensor
import numpy as np

def get_diffusion_coefficients(T, kind="linear-time", alpha_clip_min=0.001):
    if kind == "linear-time":
        t = np.linspace(1e-3, 1.0, T + 2)
        alphas_cumprod = 1.0 - t
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1 - alphas
        betas = np.clip(betas, 0.0, 1.0 - alpha_clip_min)
        betas = torch.from_numpy(betas).float().squeeze()
    elif kind == "cosine":
        s = 0.008
        steps = T + 2
        x = torch.linspace(0, T, steps)
        alphas_cumprod = (
                torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5)
                ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = alphas.clip(min=alpha_clip_min)
        betas = 1 - alphas
        betas = torch.clip(betas, 0.0, 1.0 - alpha_clip_min).float()
    elif kind == "ddpm":
        betas = torch.linspace(0.1 / T, 20 / T, T + 1)
    return betas


class SO3Prior:
    def __init__(self, std_r=1, std_theta=1):
        self.std_r = std_r
        self.std_theta = std_theta
        self.log_r_d = torch.distributions.Normal(0, 1)
        self.theta_d = torch.distributions.Normal(0, 1)
        
    def sample(self, n, device="cpu"):
        r = torch.exp(self.log_r_d.sample((n,)) * self.std_r)
        theta = self.theta_d.sample((n,)) * self.std_theta
        phi = self.theta_d.sample((n,)) * self.std_theta
        return torch.stack([r, theta, phi], dim=-1).to(device)
    


def get_so3_cartesian_axes(theta, axis: str):
    if theta.ndim == 0:
        theta = theta.unsqueeze(0)
    assert theta.ndim == 1
    n = len(theta)
    R = torch.zeros((n, 3, 3), 
                    device=theta.device
                   )
    assert axis in ["x", "y", "z"]
    if axis == "x":
        R[:, 0, 0] = 1.0
        R[:, 1, 1] = torch.cos(theta)
        R[:, 2, 2] = torch.cos(theta)
        R[:, 1, 2] = -torch.sin(theta)
        R[:, 2, 1] = torch.sin(theta)
    elif axis == "y":
        R[:, 1, 1] = 1.0
        R[:, 0, 0] = torch.cos(theta)
        R[:, 2, 2] = torch.cos(theta)
        R[:, 2, 0] = -torch.sin(theta)
        R[:, 0, 2] = torch.sin(theta)
    else:
        R[:, 2, 2] = 1.0
        R[:, 0, 0] = torch.cos(theta)
        R[:, 1, 1] = torch.cos(theta)
        R[:, 1, 0] = torch.sin(theta)
        R[:, 0, 1] = -torch.sin(theta)
    return R

class GeneralizedScoreMatching3Nsphere(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        I = torch.eye(3).float().unsqueeze(0)
        Ax = torch.tensor([[0, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]]).float().unsqueeze(0)
        Ay = torch.tensor([[0, 0, 1],
                           [0, 0, 0],
                           [-1, 0, 0]]).float().unsqueeze(0)
        Az = torch.tensor([[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 0]]).float().unsqueeze(0)

        ex = torch.tensor([1, 0, 0]).float()
        ey = torch.tensor([0, 1, 0]).float()
        ez = torch.tensor([0, 0, 1]).float()

        Ix = torch.einsum("i,j -> ij", ex, ex).unsqueeze(0)
        Iy = torch.einsum("i,j -> ij", ey, ey).unsqueeze(0)
        Iz = torch.einsum("i,j -> ij", ez, ez).unsqueeze(0)
        self.register_buffer("Ax", Ax)
        self.register_buffer("Ay", Ay)
        self.register_buffer("Az", Az)
        self.register_buffer("ex", ex.unsqueeze(0))
        self.register_buffer("ey", ey.unsqueeze(0))
        self.register_buffer("ez", ez.unsqueeze(0))
        self.register_buffer("I", I)
        self.register_buffer("Ix", Ix)
        self.register_buffer("Iy", Iy)
        self.register_buffer("Iz", Iz)
        self.register_buffer("Ax2", Ax @ Ax)
        self.register_buffer("Ay2", Ay @ Ay)
        self.register_buffer("Az2", Az @ Az)

        self.prior = SO3Prior()

    def get_so3_cartesian_axes(self, theta, axis):
        return get_so3_cartesian_axes(theta, axis)
    
    def to_cartesian(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(dim=0)
        r, theta, phi = input.chunk(3, dim=1)
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.concat([x, y, z], dim=1)

    def to_spherical(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(dim=0)
        x, y, z = input.chunk(3, dim=1)
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.atan2((x**2 + y**2)**0.5, z)
        phi = torch.atan2(y, x)
        return torch.concat([r, theta, phi], dim=1)

    def _check_assert_shape(self, x: Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        else:
            assert x.ndim == 2
        return x

    def lie_bracket(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Computes lie bracket for matrices A and B assuming the commutator is [A, B] = AB - BA
        """
        return A @ B - B @ A

    def vector_to_skew_matrix(self, x: Tensor):
        """_summary_
        Computes the skew-form of a vector.
        I.e. transforming a vector into a lie algebra element of SO(3) by left multiplication with the basis matrices.
        """
        x = self._check_assert_shape(x)
        skew = (x[:, 0].unsqueeze(-1).unsqueeze(-1) * self.Ax \
            + x[:, 1].unsqueeze(-1).unsqueeze(-1) * self.Ay
            + x[:, 2].unsqueeze(1).unsqueeze(-1) * self.Az)
        return skew

    def rodrigues_formula(self, k: Tensor, theta: Tensor):
        """
        Returns the full rotation matrix for a rotation around a vector k.
        """
        k = self._check_assert_shape(k)
        K = self.vector_to_skew_matrix(k)

        R = self.I + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K

        return R