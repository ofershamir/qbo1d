from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import torch

from . import deterministic_forcing

#: Physical constants
GRAV = 9.8  #: Earth's gravitational acceleration [:math:`\mathrm{m \, s^{-2}}`]
P = 101325  #: Reference pressure [Pa]
R = 287.04  #: Gas const for dry air [:math:`\mathrm{J \, Kg^{-1} \, K^{-1}}`]
T = 204  #: Reference temperature [K], corresponding to the Brunt-Vaisala frequency
NBV = 2.16e-2  #: Brunt-Vaisala frequency [:math:`\mathrm{s^{-1}}`]


def get_alpha(z):
    """Calculates the waves dissipation profile, :math:`\\alpha(z)` as defined
    in (9) of Holton and Lindzen.

    Parameters
    ----------
    z : tensor
        Height [m]

    Returns
    -------
    tensor
        :math:`\\alpha (z)` [:math:`\mathrm{s^{-1}}`]
    """

    alpha = torch.zeros(z.shape)
    idx = (17e3 <= z) & (z <= 30e3)

    alpha[idx] = (1 / 21) + (2 / 21) * (z[idx] - 17e3) / 13e3
    alpha[~idx] = 1 / 7

    return alpha / (60 * 60 * 24)


def get_rho(z):
    """Calculates the density profile for an isothermal atmosphere.

    Parameters
    ----------
    z : tensor
        Height [m]

    Returns
    -------
    tensor
        :math:`\\rho (z)` [:math:`\mathrm{Kg \, m^{-3}}`]
    """

    return (P / (R * T)) * torch.exp(-(GRAV * z) / (R * T))


def make_source_func(solver, As=None, cs=None, ks=None, Gsa=0):
    """A wrraper for setting up the source function. At present the solver class
    assumes that the source depend depend explicitly only on the unknown u.

    Parameters
    ----------
    solver : ADSolver
        A solver instance holding the grid and differentiation matrix
    As : tensor/function, optional
        Wave amplitudes [Pa], by default None
    cs : tensor/function, optional
        Phase speeds [:math:`\mathrm{m \, s^{-1}}`], by default None
    ks : tensor/function, optional
        Wavenumbers [:math:`\mathrm{m^{-1}}`], by default None
    Gsa : float, optional
        Amplitude of semi-annual oscillation [:math:`\mathrm{m \, s^{-2}}`], by default 0

    Returns
    -------
    function
        Source term as a function of u
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

    g_func = lambda c, k, u : NBV * alpha / (k * ((c - u) ** 2))
    F_func = lambda A, g : (A * torch.exp(-torch.hstack((
        torch.zeros(1),
        torch.cumulative_trapezoid(g, dx=solver.dz)
        ))))
    G_func = lambda z, t : (Gsa * 2 * (z - 28e3) * 1e-3 * 2 * np.pi / 180 /
        86400 * torch.sin(2 * np.pi / 180 / 86400 * t))

    def source_func(u):
        Ftot = torch.zeros(u.shape)
        As_now, cs_now, ks_now = As(), cs(), ks()

        for A, c, k in zip(As_now, cs_now, ks_now):
            g = g_func(c, k, u)
            F = F_func(A, g)
            Ftot += F

        G = torch.zeros(solver.z.shape)
        idx = (28e3 <= solver.z) & (solver.z <= 35e3)
        G[idx] = G_func(solver.z[idx], solver.current_time)

        return torch.matmul(solver.D1, Ftot) * rho[0] / rho - G

    return source_func, g_func, F_func


def load_model(solver, ModelClass=None, path_to_state_dict=None):
    """Utility for loading a Pytorch model specifying the source term as a
    function of the zonal wind. By default, the utility loads the analytic
    2-wave example, treating the analytic source term as a non-trainable dummy
    neural network.

    Parameters
    ----------
    solver : ADSolver
        A solver instance holding the grid and differentiation matrix
    ModelClass : torch.nn.Module, optional
        The model class, by default None
    path_to_state_dict : str, optional
        Path to saved state_dict corresponding to ModelClass, by default None

    Returns
    -------
    ModelClass
        A ModelClass instance in eval mode
    """

    if ModelClass is None:
        ModelClass = deterministic_forcing.WaveSpectrum

    if path_to_state_dict is None:
        path_to_state_dict = 'models/deterministic_forcing.pth'

    model = ModelClass(solver)
    model.load_state_dict(torch.load(path_to_state_dict))
    model.eval()
    return model

def _process_signal(time, z, u, height, spinup, bw_filter, pad):
    """Applies various signal processing options to a zonal wind field and
    returns the signal and returns the processed signal along with the periods
    and amplitudes of its FFT. Should only be called by other functions in
    this file.
    
    Parameters
    ----------
    time : tensor
        Time [:math:`\mathrm{s}`]
    z : tensor
        Height [m]
    u : tensor
        Zonal wind [:math:`\mathrm{m \, s^{-1}}`]
    height : float
        Height for estimating the period [:math:`\mathrm{m}`]
    spinup : float
        Spinup time to exclude from the estimation [:math:`\mathrm{s}`]
    bw_filter : bool
        Flag for invoking a Butterworth filter
    pad : bool
        Flag for zero-padding the signal before Fourier-transforming.
        
    Returns
    -------
    tensor
        u, possibly with Butterworth filtering applied
    tensor
        periods of FFT modes in months
    tensor
        complex amplitudes of FFT modes
    
    """
    fs = 86400 / (time[1] - time[0]).item()
    u = u[abs(time - spinup).argmin():, abs(z - height).argmin()]
    
    if pad:
        n_fft = int(2.5e6)
    else:
        n_fft = u.shape[0]
        
    if bw_filter:
        sos = signal.butter(9, 1 / 120, output='sos', fs=fs)
        u = torch.tensor(signal.sosfilt(sos, u - u.mean()))
        
    amps = torch.fft.fft(u, n=n_fft)
    periods = 1 / torch.fft.fftfreq(amps.shape[0])
    
    idx = 0 <= periods
    if pad:
        idx = idx & (periods <= u.shape[0])
        
    return u, periods[idx] / 30, amps[idx]
        

def estimate_period(time, z, u, height=25e3, spinup=0, bw_filter=False):
    """Returns the estimated QBO period in months using the dominant (maximal)
    Fourier mode.

    Parameters
    ----------
    time : tensor
        Time [:math:`\mathrm{s}`]
    z : tensor
        Height [m]
    u : tensor
        Zonal wind [:math:`\mathrm{m \, s^{-1}}`]
    height : float, optional
        Height for estimating the period [:math:`\mathrm{m}`], by default 25e3
    spinup : float, optional
        Spinup time to exclude from the estimation [:math:`\mathrm{s}`], by default 0
    bw_filter : bool, optional
        Flag for invoking a Butterworth filter, by default False

    Returns
    -------
    float
        QBO period [months]
    """
    
    _, periods, amps = _process_signal(
        time, z, u, 
        height, spinup, 
        bw_filter, pad=True
    )
    
    return periods[abs(amps).argmax()]


def estimate_amplitude(time, z, u, height=25e3, spinup=0, bw_filter=False):
    """Returns the estimated QBO amplitude in m s^{-1} using the standard
    deviation.

    Parameters
    ----------
    time : tensor
        Time [:math:`\mathrm{s}`]
    z : tensor
        Height [m]
    u : tensor
        Zonal wind [:math:`\mathrm{m \, s^{-1}}`]
    height : float, optional
        Height for estimating the period [:math:`\mathrm{m}`], by default 25e3
    spinup : float, optional
        Spinup time to exclude from the estimation [:math:`\mathrm{s}`], by default 0
    bw_filter : bool, optional
        Flag for invoking a Butterworth filter, by default False

    Returns
    -------
    float
        QBO amplitude [:math:`\mathrm{m \, s^{-1}}`]
    """
    
    u, _, _ = _process_signal(
        time, z, u,
        height, spinup,
        bw_filter, pad=False
    )

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


def simple_periodogram(time, z, u, height=25e3, spinup=0, ax=None):
    _, periods, amps = _process_signal(
        time, z, u,
        height, spinup,
        bw_filter=False, pad=True
    )
    print(periods.shape)
    idx = periods <= 100
    periods, amps = periods[idx], amps[idx]
    
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

    ax.plot(periods, torch.abs(amps), marker='.')

    ax.set_xlim(0, 100)
    ax.set_xlabel(r'$\tau$ (months)')
    ax.set_ylabel(r'$|\hat{u}|^{2}$')

    ax.grid()
