import numpy as np
import torch

from . import utils


def sample_sf_cw(n, sfe, sfv, cwe, cwv, corr, seed):
    """Draw a squence of source fluxes and spectral widths.

    The control spectrum consists of 20 waves with equal wavenumbers 2
    and equally spaced phase speeds in :math:`[-100, -10]`, :math:`[10, 100]`.
    The amplitudes depend on the phase speeds as in (17) of AD99 with
    stochastically varying magnitude (total flux at source level) and spectrum
    width sampled from a 5-parameter distribution (for the means, variances
    , and correlation). Here, the magnitude and width are first drawn from a
    bivariate normal distribution with the specified correlation and then mapped
    to bivariate log-normal distribution with the specified means and variances.

    Parameters
    ----------
    n : int
        Number of realizations
    sfe : float
        Total source flux mean
    sfv : float
        Total source flux variance
    cwe : float
        Spectrum width mean
    cwv : float
        Spectrum width variance
    corr : float
        Correlation in the underlying normal distribution
    seed : int
        A seed for the pseudorandom number generator

    Returns
    -------
    (float, float)
        Source fluxes, spectral widths
    """

    # means and variances of the bivariate log-normal distribution
    es = torch.tensor([sfe, cwe])
    vs = torch.tensor([sfv, cwv])

    # resulting means and variances of the corresponding normal distribution
    mu = - 0.5 * torch.log(vs / es**4 + 1 / es**2)
    variances = torch.log(es**2) - 2 * mu

    # covariance matrix
    sigma = torch.tensor([[variances[0],
    corr*(variances[0]*variances[1])**0.5],
    [corr*(variances[0]*variances[1])**0.5,
    variances[1]]])

    # choose seed for reproducibility
    torch.manual_seed(seed)

    # draw from normal distribution
    normal_dist = (
    torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma))
    normal_samples = normal_dist.sample((n,))

    # map to log-normal distribution
    lognormal_samples = torch.exp(normal_samples)

    # surface fluxes and spectral widths
    sf = lognormal_samples[:, 0]
    cw = lognormal_samples[:, 1]

    return sf, cw


class WaveSpectrum(torch.nn.Module):
    """A ModelClass for setting up the stochastic source term.

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
    seed : int, optional
        A seed for the pseudorandom number generator, by default 197

    Attributes
    ----------
    g_func : func
        An interface for keeping track of the function g in the analytic forcing
    F_func : func
        An interface for keeping track of the function F in the analytic forcing
    sf : tensor
        A realization of surface fluxes for each timestep
    cw : tensor
        A realization of spectral width for each timestep
    ks : tensor
        Wavenumbers
    cs : tensor
        Phase speeds
    As : tensor
        Wave amplitudes
    """

    def __init__(self, solver,
        sfe=3.7e-3, sfv=1e-8, cwe=32, cwv=225, corr=0.75, seed=int(21*9+8)):
        super().__init__()

        self.train(False)

        self._z = solver.z
        self._nlev = solver.nlev
        self._nsteps, = solver.time.shape
        self._D1 = solver.D1

        self._rho = utils.get_rho(self._z)
        self._alpha = utils.get_alpha(self._z)

        self._current_step = 0

        # keep track of source
        self.s = torch.zeros((self._nsteps, self._nlev))

        # surface fluxes and spectral widths
        self.sf, self.cw = sample_sf_cw(n=self._nsteps,
        sfe=sfe, sfv=sfv, cwe=cwe, cwv=cwv, corr=corr, seed=seed)

        # wavenumbers and phase speeds
        self.ks = 2 * 2 * np.pi / 4e7 * torch.ones(20)
        self.cs = torch.hstack([torch.arange(-100., 0., 10.),
        torch.arange(10., 110., 10.)])

        # wave amplitudes
        self.As = torch.empty((self._nsteps, 20))
        for i in range(self._nsteps):
            self.As[i, :] = torch.exp(-np.log(2) * (self.cs / self.cw[i])**2)
            self.As[i, :] *= torch.sign(self.cs)
            self.As[i, :] *= self.sf[i] / self.As[i, :].abs().sum() / 0.1006

        self.g_func = lambda c, k, u : (utils.NBV * self._alpha /
        (k * ((c - u) ** 2)))

        self.F_func = lambda A, g : (A * torch.exp(-torch.hstack((
            torch.zeros(1),
            torch.cumulative_trapezoid(g, dx=solver.dz)
            ))))


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
            Source term as a function of the zonal wind u
        """

        Ftot = torch.zeros(u.shape)
        for A, c, k in zip(self.As[self._current_step, :], self.cs, self.ks):
            g = self.g_func(c, k, u)
            F = self.F_func(A, g)
            Ftot += F

        s = torch.matmul(self._D1, Ftot) * self._rho[0] / self._rho
        self.s[self._current_step, :] = s

        self._current_step += 1

        return s

