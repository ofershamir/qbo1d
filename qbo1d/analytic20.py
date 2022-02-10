import numpy as np
import torch

from . import utils

class WaveSpectrum(torch.nn.Module):
    """A ModelClass for setting up the analytic 2 wave source spectrum.

    The control source spectrum consists of 20 waves with equal wavenumbers 2
    and equally spaced phase speeds in :math:`[-100, -10]`, :math:`[10, 100]`.
    The amplitudes depend on the phase speeds as in (17) of AD99 with
    stochastically varying magnitude (total flux at source level) and spectrum
    width sampled from a 5 parameter distribution (for the means, variances
    , and correlation). Here, the magnitude and width are first drawn from a
    bivariate normal distribution with the specified correlation and then mapped
    to bivariate log-normal distribution with the specified means and variances.

    Parameters
    ----------
    solver : ADSolver
        A solver instance holding the grid and differentiation matrix
    sfe : float, optional
        Total source flux mean, by default 3.7e-3
    sfv : float, optional
        Total source flux variance, by default 1e-8
    cwe : float, optional
        Spectrum width mean, by default 32
    cwv : float, optional
        Spectrum width variance, by default 225
    corr : float, optional
        Correlation in the underlying normal distribution, by default 0.75

    Attributes
    ----------
    g_func : func
        An interface for keeping track of the function g in the analytic forcing
    F_func : func
        An interface for keeping track of the function F in the analytic forcing
    """

    def __init__(self, solver,
        sfe=3.7e-3, sfv=1e-8, cwe=32, cwv=225, corr=0.75):
        super().__init__()

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
        """Draws the wave amplitudes from a bivariate log-normal distribution.

        Returns
        -------
        tensor
            An instance of wave amplitudes to be used in self.forward method
        """
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
        """An interface for calculating the source term as a function of u. By
        default, torch.nn.Module uses the forward method.

        Parameters
        ----------
        u : tensor
            Zonal wind profile

        Returns
        -------
        tensor
            Source term as a function of u
        """

        Ftot = torch.zeros(u.shape)
        for A, c, k in zip(self.As(), self.cs, self.ks):
            g = self.g_func(c, k, u)
            F = self.F_func(A, g)
            Ftot += F

        return torch.matmul(self.D1, Ftot) * self.rho[0] / self.rho

