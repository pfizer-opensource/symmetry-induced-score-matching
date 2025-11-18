"""
Code for toy datasets in 2d and 3d
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import pandas as pd
import torch.distributions as D
import random


def concentric_potential(x, r, var=2.0):
    d = x.pow(2).sum(-1).sqrt()
    energy = 1 / (2 * math.pi) * torch.exp(-1.0 / (2.0 * var) * (d - r).pow(2))
    return energy


def generate_line_distribution(n=5, length=3, nsamples=10000):

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Initialize an empty list to store the samples
    samples = []

    # Generate samples along each line
    for angle in angles:
        for r in np.linspace(0.1, length, nsamples):
            rd = np.random.randn() * 0.05
            x = r * np.cos(angle + rd)
            y = r * np.sin(angle + rd)
            samples.append((x, y))

    return np.array(samples)


def generate_concentric_circle_dataset(
    nsamples, random_seed: int = 42, r0=4.0, r1=8.0, var=0.5, both=True
):

    sampled = []
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    a = -r1 - 2
    b = r1 + 2
    cnt = 0
    while cnt < nsamples:
        samples = (a - b) * torch.rand((nsamples, 2)) + b
        pdf0 = concentric_potential(samples, r=r0, var=var)
        pdf1 = concentric_potential(samples, r=r1, var=var)
        if both:
            pdf = pdf0 + pdf1
        else:
            pdf = pdf0
        dataset = pd.DataFrame()
        dataset["x1"] = samples[:, 0]
        dataset["x2"] = samples[:, 1]
        dataset["energy"] = pdf
        dataset["select"] = dataset["energy"].apply(lambda x: x > 0.10)
        selected = dataset.loc[dataset.select][["x1", "x2"]].values
        cnt += len(selected)
        sampled.extend(selected)

    sampled = np.stack(sampled, axis=0)
    return sampled[:nsamples]


def get_2d_rotation(theta):
    R = torch.Tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    return R


def generate_c4_gmm_dataset(nsamples, r=3, vars=0.2, random_seed: int = 42):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    theta0 = 90 * (math.pi / 180)
    theta1 = 180 * (math.pi / 180)
    theta2 = 270 * (math.pi / 180)
    R0 = get_2d_rotation(theta0)
    R1 = get_2d_rotation(theta1)
    R2 = get_2d_rotation(theta2)

    mean = torch.Tensor([r, 0])
    m0 = (R0 @ mean.T).T
    m1 = (R1 @ mean.T).T
    m2 = (R2 @ mean.T).T

    mixture_means = torch.stack([mean, m0, m1, m2])
    mixing = torch.ones((4,)) / 4.0
    variance = torch.Tensor([vars, vars])
    vars = torch.stack([variance] * 4)
    mixing_cat = D.Categorical(mixing)
    comp = D.Independent(D.Normal(mixture_means, vars), 1)
    gmm = D.MixtureSameFamily(mixing_cat, comp)
    samples = gmm.sample((nsamples,))
    return samples


def get_2d_rotation(theta):

    return torch.Tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )


def generate_gmm_dataset(nsamples, ncomponents=4, r=3, var=0.2, random_seed: int = 42):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    angles = np.linspace(0, 2 * math.pi, ncomponents, endpoint=False)

    mixture_means = []
    for theta in angles:
        R = get_2d_rotation(theta)
        mean = torch.Tensor([r, 0])
        rotated_mean = (R @ mean.T).T
        mixture_means.append(rotated_mean)

    mixture_means = torch.stack(mixture_means)

    mixing = torch.ones((ncomponents,)) / ncomponents

    variance = torch.Tensor([var, var])
    vars = torch.stack([variance] * ncomponents)

    mixing_cat = D.Categorical(mixing)
    comp = D.Independent(D.Normal(mixture_means, vars), 1)
    gmm = D.MixtureSameFamily(mixing_cat, comp)

    samples = gmm.sample((nsamples,))

    return samples


# Example usage:
nsamples = 1000
ncomponents = 6  # You can set this to any number of components
samples = generate_gmm_dataset(nsamples, ncomponents=ncomponents)


def my_gmm_sample(self, sample_shape=torch.Size()):
    with torch.no_grad():
        sample_len = len(sample_shape)
        batch_len = len(self.batch_shape)
        gather_dim = sample_len + batch_len
        es = self.event_shape
        # mixture samples [n, B]
        mix_sample = self.mixture_distribution.sample(sample_shape)
        mix_shape = mix_sample.shape

        # component samples [n, B, k, E]
        comp_samples = self.component_distribution.sample(sample_shape)

        # Gather along the k dimension
        mix_sample_r = mix_sample.reshape(mix_shape + torch.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(
            torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
        )

        samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
        return samples.squeeze(gather_dim), mix_sample


def generate_mog_2d(nsamples, random_seed: int = 42):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device("cpu")

    mean0 = torch.tensor([-6.5, -6.0], device=device)
    mean1 = torch.tensor([-2.5, 2.5], device=device)
    mean2 = torch.tensor([-0.5, -0.1], device=device)
    mean3 = torch.tensor([5.5, -5.5], device=device)
    mean4 = torch.tensor([6.0, 6.0], device=device)
    mean5 = torch.tensor([-10.0, 10.0], device=device)
    mean = torch.stack([mean0, mean1, mean2, mean3, mean4, mean5])
    vars = torch.rand(6, 2, device=device) + (1.0 - 0.5)
    pi = torch.randn(6, device=device).softmax(dim=0)
    mix = D.Categorical(pi)
    comp = D.Independent(D.Normal(mean, vars), 1)
    gmm = D.MixtureSameFamily(mix, comp)
    selected_samples, classes = my_gmm_sample(gmm, (nsamples,))
    selected_samples, classes = selected_samples.cpu(), classes.cpu()
    return selected_samples, gmm


def generate_vectors(length1, length2, angle):
    # Convert the angle from degrees to radians
    theta_delta = torch.randn(1) * 10
    theta = math.radians(angle + theta_delta)
    length1_delta = torch.randn(1) / 2
    length2_delta = torch.randn(1) / 2

    # Add a random rotation to the angle
    rand_angle = torch.tensor(math.radians(random.uniform(-180, 180))).squeeze()
    rot_matrix = torch.tensor(
        [
            [torch.cos(rand_angle), -torch.sin(rand_angle)],
            [torch.sin(rand_angle), torch.cos(rand_angle)],
        ]
    )

    # Compute the x and y components of the first vector
    x1 = (length1 + length1_delta).abs() * math.cos(theta)
    y1 = (length1 + length1_delta).abs() * math.sin(theta)

    # Compute the x and y components of the second vector
    x2 = (length2 + length2_delta).abs()
    y2 = 0.0

    v1 = torch.tensor([x1, y1])
    v2 = torch.tensor([x2, y2])

    v1rot = torch.matmul(rot_matrix, v1)
    v2rot = torch.matmul(rot_matrix, v2)
    # Return the two vectors as tuples
    return (
        torch.stack([v1, v2], dim=0),
        torch.stack([v1rot, v2rot], dim=0),
        rot_matrix,
        rand_angle,
    )


def generate_2particles_2d_dataset(
    length1, length2, angle, num_samples, random_seed: int = 42
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    dataset = []
    for _ in range(num_samples):
        vs, Rvs, R, phi = generate_vectors(
            length1=length1, length2=length2, angle=angle
        )
        dataset.append(Rvs)
    dataset = torch.stack(dataset, dim=0)
    return dataset


def generate_samples_from_3d_torus(
    nsamples, R: float = 3.0, r: float = 1.0, random_seed: int = 42
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    i = 0
    select = []
    while len(select) <= nsamples:
        u, v, w = np.split(np.random.rand(3 * nsamples, 3), 3, axis=-1)
        theta = u * 2.0 * np.pi
        phi = v * 2.0 * np.pi

        s = (R + np.cos(theta)) / (R + r)

        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        xyz = np.hstack([x, y, z])
        accept = np.where((w <= s).squeeze())[0]
        if i == 0:
            select = xyz[accept]
        else:
            select = np.concatenate([select, xyz[accept]], axis=0)
        i += 1
    select = select[:nsamples, :]
    return select


def generate_samples_from_3d_sphere(
    nsamples, r1: float = 1.0, r2: float = 2.0, random_seed: int = 42, both=True
):
    torch.manual_seed(random_seed)

    if both:
        x1 = torch.randn(nsamples // 2, 3)
        x1 = x1 / x1.norm(p=2, dim=1).unsqueeze(-1)
        x1 = x1 * r1

        x2 = torch.randn(nsamples // 2, 3)
        x2 = x2 / x2.norm(p=2, dim=1).unsqueeze(-1)
        x2 = x2 * r2
    else:
        x1 = torch.randn(nsamples, 3)
        x1 = x1 / x1.norm(p=2, dim=1).unsqueeze(-1)
        x1 = x1 * r1
        x2 = None

    return x1, x2


def generate_mog_3d(num_samples, random_seed: int = 42):
    torch.manual_seed(random_seed)
    device = torch.device("cpu")

    ngauss = 6
    means = torch.randn((ngauss, 3)) * 3.0
    # print(means)
    vars = torch.rand(ngauss, 3, device=device) * 0.5
    pi = torch.randn(ngauss, device=device).softmax(dim=0)
    mix = D.Categorical(pi)
    comp = D.Independent(D.Normal(means, vars), 1)
    gmm = D.MixtureSameFamily(mix, comp)
    selected_samples = gmm.sample((num_samples,))
    return selected_samples, gmm


def generate_samples_from_mobius_strip(
    nsamples, width: float = 1.0, random_seed: int = 42
):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Generate random values for u and v
    u = np.random.uniform(0, 2 * np.pi, nsamples)
    v = np.random.uniform(-width, width, nsamples)

    # Parametric equations for the Möbius strip
    x = (1 + (v / 2) * np.cos(u / 2)) * np.cos(u)
    y = (1 + (v / 2) * np.cos(u / 2)) * np.sin(u)
    z = (v / 2) * np.sin(u / 2)

    # Stack the coordinates
    mobius_strip_samples = np.stack([x, y, z], axis=1)

    return mobius_strip_samples
def generate_mog_4d(num_samples, random_seed: int = 42):
    torch.manual_seed(random_seed)
    device = torch.device("cpu")

    ngauss = 6
    means = torch.randn((ngauss, 4)) * 3.
    vars = torch.rand(ngauss, 4, device=device) * 0.5
    pi = torch.randn(ngauss, device=device).softmax(dim=0)
    mix = D.Categorical(pi)
    comp = D.Independent(D.Normal(means, vars), 1)
    gmm = D.MixtureSameFamily(mix, comp)
    selected_samples = gmm.sample((num_samples,))
    return selected_samples, gmm
