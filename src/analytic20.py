import numpy as np
import torch
import utils

class WaveSpectrum(torch.nn.Module):
    def __init__(self, solver,
        sfe=3.7e-3, sfv=1e-8, cwe=32, cwv=225, corr=0.75):
        super(WaveSpectrum, self).__init__()

        self.train(False)

        self.rho = utils.get_rho(solver.z)
        self.alpha = utils.get_alpha(solver.z)

        self.z = solver.z
        self.current_time = solver.current_time
        self.D1 = solver.D1

        self.g_func = lambda c, k, u : (utils.NBV * self.alpha /
        (k * ((c - u) ** 2)))

        self.F_func = lambda A, g : (A * torch.exp(-torch.hstack((
            torch.zeros(1),
            torch.cumulative_trapezoid(g, dx=solver.dz)
            ))))

        self.ks = 2 * 2 * np.pi / 4e7 * torch.ones(20)
        self.cs = torch.hstack([torch.arange(-100., 0., 10.),
        torch.arange(10., 110., 10.)])

        # desired mean and variances of the bivariate log-normal distribution
        # (total wave flux [Pa], spectral width [m s^{-1}])
        es = torch.tensor([sfe, cwe])
        vs = torch.tensor([sfv, cwv])

        # resulting means and variances of the corresponding normal distribution
        self.mu = - 0.5 * torch.log(vs / es**4 + 1 / es**2)
        variances = torch.log(es**2) - 2 * self.mu

        self.sigma = torch.tensor([[variances[0],
        corr*(variances[0]*variances[1])**0.5],
        [corr*(variances[0]*variances[1])**0.5,
        variances[1]]])

        # choosing seed for reproducibility
        torch.manual_seed(21*9+8)

    def As(self):
        normal_dist = (
        torch.distributions.multivariate_normal.MultivariateNormal(self.mu,
        self.sigma))
        normal_samp = normal_dist.sample()
        lognormal_samp = torch.exp(normal_samp)

        sf = lognormal_samp[0]
        cw = lognormal_samp[1]

        amps = torch.sign(self.cs) * torch.exp(-np.log(2) * (self.cs / cw)**2)
        amps *= sf / torch.sum(torch.abs(amps)) / 0.1006

        return amps

    def forward(self, u):
        Ftot = torch.zeros(u.shape)
        for A, c, k in zip(self.As(), self.cs, self.ks):
            g = self.g_func(c, k, u)
            F = self.F_func(A, g)
            Ftot += F

        return torch.matmul(self.D1, Ftot) * self.rho[0] / self.rho

