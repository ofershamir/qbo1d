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

        self.z_min = z_min
        self.z_max = z_max
        self.dz = dz
        self.t_min = t_min
        self.t_max = t_max
        self.dt = dt
        self.w = w
        self.kappa = kappa

        self.z = torch.arange(z_min, z_max + 1, self.dz)
        self.nlev, = self.z.shape

        self.time = torch.arange(t_min, t_max + 1, self.dt)

        self.initial_condition = initial_condition
        if self.initial_condition is None:
            self.initial_condition = (lambda z:
            -14/81e6 * (z - z_min) * (z - z_max))

        self.D1 = torch.zeros((self.nlev, self.nlev))
        for i in range(1, self.nlev -1):
            self.D1[i, i + 1] = 1
            self.D1[i, i - 1] = -1
        self.D1 /= 2 * self.dz

        D2 = torch.zeros((self.nlev, self.nlev))
        for i in range(1, self.nlev - 1):
            D2[i, [i - 1, i + 1]] = 1
            D2[i, i] = -2
        D2 /= self.dz ** 2

        self.D = self.dt * (w * self.D1 - kappa * D2)

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

        # a single forward Euler step
        source = source_func(u[0])
        u[1] = (torch.matmul(torch.eye(self.nlev) - self.D, u[0]) -
        self.dt * source)

        for n in range(1, nsteps - 2):

            source = source_func(u[n])

            # RHS multiplied by QT on the left
            b = torch.matmul(self.QT, (
                torch.matmul(torch.eye(self.nlev) - self.D, u[n - 1]) -
                2 * self.dt * source
            )).reshape(-1, 1)

            u[n + 1] = torch.triangular_solve(b, self.R).solution.flatten()

        return u
