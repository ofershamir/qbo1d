from numpy import pi as PI
import torch

from . import utils

class WaveSpectrum(torch.nn.Module):
    def __init__(self, solver, As=None, cs=None, ks=None, Gsa=0):
        super().__init__()

        self.train(False)

        self.rho = utils.get_rho(solver.z)
        self.alpha = utils.get_alpha(solver.z)

        if As is None:
            As = torch.tensor([6e-4, -6e-4]) / self.rho[0]
        if cs is None:
            cs = torch.tensor([32, -32])
        if ks is None:
            ks = (1 * 2 * PI / 4e7) * torch.ones(2)
            
        self.As = As
        self.cs = cs
        self.ks = ks

        self.z = solver.z
        self.current_time = solver.current_time
        self.D1 = solver.D1

        self.g_func = lambda c, k, u : (utils.NBV * self.alpha /
        (k * ((c - u) ** 2)))

        self.F_func = lambda A, g : (A * torch.exp(-torch.hstack((
            torch.zeros(1),
            torch.cumulative_trapezoid(g, dx=solver.dz)
            ))))

        self.G_func = lambda z, t : (Gsa * 2 * (z - 28e3) * 1e-3 * 2 * PI /
        180 / 86400 * torch.sin(2 * PI / 180 / 86400 * t))

    def forward(self, u):
        Ftot = torch.zeros(u.shape)
        for A, c, k in zip(self.As, self.cs, self.ks):
            g = self.g_func(c, k, u)
            F = self.F_func(A, g)
            Ftot += F

        G = torch.zeros(self.z.shape)
        idx = (28e3 <= self.z) & (self.z <= 35e3)
        G[idx] = self.G_func(self.z[idx], self.current_time)

        return torch.matmul(self.D1, Ftot) * self.rho[0] / self.rho - G

