import torch

import utils

class ADSolver:
    """
    Class to solve a one-dimensional advection-diffusion equation with a
    source term.
    """

    def __init__(self, z_min=17e3, z_max=35e3, dz=250, t_min=0,
    t_max=360*12*86400, dt=86400, w=1e-5, kappa=3e-1, initial_condition=None):
        """
        """

        self.dz = dz
        self.dt = dt

        self.z = torch.arange(z_min, z_max + 1, self.dz)
        self.nlev, = self.z.shape

        self.time = torch.arange(t_min, t_max + 1, self.dt)

        if initial_condition is None:
            self.initial_condition = (lambda z:
            -14/81e6 * (z - z_min) * (z - z_max))

        D1 = torch.zeros((self.nlev, self.nlev))
        for i in range(1, self.nlev -1):
            D1[i, i + 1] = 1
            D1[i, i - 1] = -1
        D1 /= 2 * self.dz

        D2 = torch.zeros((self.nlev, self.nlev))
        for i in range(1, self.nlev - 1):
            D2[i, [i - 1, i + 1]] = 1
            D2[i, i] = -2
        D2 /= self.dz ** 2

        self.D = 0.5 * self.dt * (w * D1 - kappa * D2)

        # LHS
        B = torch.eye(self.nlev) + self.D

        Q, self.R = torch.linalg.qr(B)
        self.QT = Q.T

    def solve(self, nsteps=None, source_func=None):
        """
        Integrates the model for a given number of steps. Arguments are:
            n_steps : number of time steps to take in the integration.

        Returns an array of shape (n_steps, m), where m is the number of
        levels, whose entries are the zonal wind profiles at each time.
        """

        if nsteps is None:
            nsteps, = self.time.shape

        if source_func is None:
            source_func = utils.make_source_func(self)

        u = torch.zeros((nsteps, self.nlev))
        u[0] = self.initial_condition(self.z)

        for n in range(nsteps - 1):

            source = source_func(u[n])
            source[0] = source[-1] = 0

            # RHS multiplied by QT on the left
            b = torch.matmul(self.QT, (
                torch.matmul(torch.eye(self.nlev) - self.D, u[n]) -
                self.dt * source
            )).reshape(-1, 1)

            u[n + 1] = torch.triangular_solve(b, self.R).solution.flatten()

        return u
