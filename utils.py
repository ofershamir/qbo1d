from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy import signal
import torch

GRAV = 9.8
P = 101325
R = 287.04
T = 204  # corresponding to the brunt-vaisala frequency below
NBV = 2.16e-2  # brunt-vaisala frequency
MU = 1e-6  # wave dissipation rate [s^{-1}]

def make_source_func(solver, As=None, cs=None, ks=None):
    """
    """

    rho = P / (R * T) * torch.exp(-(GRAV * solver.z) / (R * T))

    if As is None:
        As = torch.tensor([4e-3, -4e-3])
    if cs is None:
        cs = torch.tensor([30, -30])
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
            g = NBV * MU / (k * ((c - u) ** 2))
            F = rho[0] * A * torch.exp(-torch.hstack((
                torch.zeros(1),
                torch.cumulative_trapezoid(g, dx=solver.dz)
            )))
            dFdz -= F * g

        return dFdz / rho

    return source_func

def estimate_period(time, z, u, height=25e3):
    """
    Returns the estimated period in months.
    """
    
    fs = 86400 / (time[1] - time[0]).item()
    u = u[:, abs(z - height).argmin()]
    
    sos = signal.butter(9, 1 / 120, output='sos', fs=fs)
    u = signal.sosfilt(sos, u - u.mean())
    
    amps = np.fft.fft(u)
    freqs = np.fft.fftfreq(amps.shape[0])
    
    return abs(1 / freqs[abs(amps).argmax()]) / 30

def estimate_amplitude(u):
    """An estimate of the QBO amplitude in terms of the standard deviation at a given vertical level.
    """

    amp = torch.std(u)

    return amp

def ax_pos_inch_to_absolute(fig_size, ax_pos_inch):
    ax_pos_absolute = []
    ax_pos_absolute.append(ax_pos_inch[0]/fig_size[0])
    ax_pos_absolute.append(ax_pos_inch[1]/fig_size[1])
    ax_pos_absolute.append(ax_pos_inch[2]/fig_size[0])
    ax_pos_absolute.append(ax_pos_inch[3]/fig_size[1])

    return ax_pos_absolute

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
    

def display(time, z, u, amp25=None, amp20=None, tau25=None):
    """Plots u to show the presence (or absence) of the QBO.
    """

    fig_size = (06.90, 02.20 + 01.50)
    fig = plt.figure(figsize=fig_size)

    ax = []

    ax.append(fig.add_axes(ax_pos_inch_to_absolute(fig_size, [00.75, 01.25, 06.00, 02.00])))

    cmin = -np.max(np.abs(u))
    cmax = np.max(np.abs(u))

    xmin = 84.
    xmax = 96.
    ymin = 17.
    ymax = 35.

    ax[0].set_xlim(left=84.)
    ax[0].set_xlim(right=96.)
    ax[0].set_ylim(bottom=17.)
    ax[0].set_ylim(top=35.)

    h = []

    h.append(ax[0].contourf(time/86400/360, z/1000, u,
                            21, cmap="RdYlBu_r", vmin=cmin, vmax=cmax))

    ax[0].axhline(25., xmin=0, xmax=1, color='white', linestyle='dashed', linewidth=1.)
    ax[0].axhline(20., xmin=0, xmax=1, color='white', linestyle='dashed', linewidth=1.)

    ax[0].set_ylabel('Km', fontsize=10)

    ax[0].set_xlabel('model year', fontsize=10)

    xticks_list = np.arange(xmin, xmax+1, 1)
    ax[0].set_xticks(xticks_list)

    yticks_list = np.arange(ymin, ymax+2, 2)
    ax[0].set_yticks(yticks_list)

    xticklabels_list = list(xticks_list)
    xticklabels_list = [ '%.0f' % elem for elem in xticklabels_list ]
    ax[0].set_xticklabels(xticklabels_list, fontsize=10)

    ax[0].xaxis.set_minor_locator(MultipleLocator(1.))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1.))

    ax[0].tick_params(which='both', left=True, right=True, bottom=True, top=True)
    ax[0].tick_params(which='both', labelbottom=True)

    if amp25 is not None:
        ax[0].text(95.50, 25, r'$\sigma_{25}$ = ' '%.1f' %amp25 + r'$\mathrm{m s^{-1}}$',
                    horizontalalignment='right', verticalalignment='bottom', color='black')

    if amp20 is not None:
        ax[0].text(95.50, 20, r'$\sigma_{20}$ = ' '%.1f' %amp20 + r'$\mathrm{m s^{-1}}$',
                    horizontalalignment='right', verticalalignment='bottom', color='black')

    if tau25 is not None:
        ax[0].text(84.50, 25, r'$\tau_{25}$ = ' '%.0f' %tau25 + 'months',
                    horizontalalignment='left', verticalalignment='bottom', color='black')

    # # colorbars
    cbar_ax0 = fig.add_axes(ax_pos_inch_to_absolute(fig_size, [01.00, 00.50, 05.50, 00.10]))
    ax[0].figure.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r"), cax=cbar_ax0, format='% 2.0f',
                        boundaries=np.linspace(cmin, cmax, 21), orientation='horizontal',
                        label=r'$\mathrm{m s^{-1}}$')

    return fig, ax
