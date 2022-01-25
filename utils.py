from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import torch

GRAV = 9.8
P = 101325
R = 287.04
T = 204  # corresponding to the brunt-vaisala frequency below
NBV = 2.16e-2  # brunt-vaisala frequency
MU = 1e-6  # wave dissipation rate [s^{-1}]


def get_alpha(z):
    """Calculates alpha as defined in (9) of Holton and Lindzen."""

    alpha = torch.zeros(z.shape)
    idx = (17e3 <= z) & (z <= 30e3)

    alpha[idx] = (1 / 21) + (2 / 21) * (z[idx] - 17e3) / 13e3
    alpha[~idx] = 1 / 7

    return alpha / (60 * 60 * 24)


def get_rho(z):
    """Calculates the density profile."""

    return (P / (R * T)) * torch.exp(-(GRAV * z) / (R * T))


def make_source_func(solver, As=None, cs=None, ks=None):
    """
    """

    rho = get_rho(solver.z)
    alpha = get_alpha(solver.z)

    if As is None:
        As = torch.tensor([6e-4, -6e-4]) / rho[0]
    if cs is None:
        cs = torch.tensor([32, -32])
    if ks is None:
        ks = (1 * 2 * np.pi / 4e7) * torch.ones(2)

    if isinstance(As, torch.Tensor):
        As_copy = As
        As = lambda: As_copy
    if isinstance(cs, torch.Tensor):
        cs_copy = cs
        cs = lambda: cs_copy
    if isinstance(ks, torch.Tensor):
        ks_copy = ks
        ks = lambda: ks_copy

    def source_func(u):
        dFdz = torch.zeros(u.shape)
        As_now, cs_now, ks_now = As(), cs(), ks()

        for A, c, k in zip(As_now, cs_now, ks_now):
            g = NBV * alpha / (k * ((c - u) ** 2))
            F = A * torch.exp(-torch.hstack((
                torch.zeros(1),
                torch.cumulative_trapezoid(g, dx=solver.dz)
            )))
            dFdz -= F * g

        return dFdz * rho[0] / rho

    return source_func


def estimate_period(time, z, u, height=25e3, spinup=0, bw_filter=False):
    """
    Returns the estimated period in months.
    """

    fs = 86400 / (time[1] - time[0]).item()
    u = u[abs(time - spinup).argmin():, abs(z - height).argmin()]

    if bw_filter:
        sos = signal.butter(9, 1 / 120, output='sos', fs=fs)
        u = signal.sosfilt(sos, u - u.mean())

    amps = np.fft.fft(u)
    freqs = np.fft.fftfreq(amps.shape[0])

    return abs(1 / freqs[abs(amps).argmax()]) / 30


def estimate_amplitude(time, z, u, height=25e3, spinup=0, bw_filter=False):
    """
    Returns the estimated amplitude in m s^{-1}.
    """

    fs = 86400 / (time[1] - time[0]).item()
    u = u[abs(time - spinup).argmin():, abs(z - height).argmin()]

    if bw_filter:
        sos = signal.butter(9, 1 / 120, output='sos', fs=fs)
        u = torch.tensor(signal.sosfilt(sos, u - u.mean()))

    return torch.std(u)


def simple_display(time, z, u, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

    years = time / (360 * 86400)
    kms = z / 1000
    cmax = abs(u).max()

    ax.contourf(
        years, kms, u,
        vmin=-cmax, vmax=cmax,
        cmap='RdBu_r',
        levels=21
    )

    ax.set_xlabel('year')
    ax.set_ylabel('z (km)')

    xticks = np.linspace(years.min(), years.max(), 7)
    yticks = np.linspace(kms.min(), kms.max(), 7)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(which='both', left=True, right=True, bottom=True, top=True)

