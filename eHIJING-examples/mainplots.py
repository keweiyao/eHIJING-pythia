#!/usr/bin/env python3 
import os, subprocess
import logging
from pathlib import Path

import h5py
import hsluv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import glob
from scipy.interpolate import interp1d, interp2d
from matplotlib.gridspec import GridSpec
from scipy.integrate import quad

fontsmall, fontnormal, fontlarge = 5, 6, 7
offblack = '#262626'
aspect = 1/1.618
resolution = 72.27
textwidth = 260/resolution
textheight = 200/resolution
fullwidth = 300/resolution
fullheight = 200/resolution

plt.rcdefaults()
plt.rcParams.update({
    'font.family': 'DejaVu Sans-Serif',
    'font.sans-serif': ['Lato'],
    'mathtext.fontset': 'custom',
    'mathtext.default': 'it',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic:medium',
    'mathtext.cal': 'sans',
    'font.size': fontnormal,
    'legend.fontsize': fontsmall,
    'axes.labelsize': fontnormal,
    'axes.titlesize': fontnormal,
    'xtick.labelsize': fontsmall,
    'ytick.labelsize': fontsmall,
    'lines.linewidth': .8,
    'lines.markersize': 3,
    'lines.markeredgewidth': 0,
    'patch.linewidth': .5,
    'axes.linewidth': .4,
    'xtick.major.width': .4,
    'ytick.major.width': .4,
    'xtick.minor.width': .4,
    'ytick.minor.width': .4,
    'xtick.major.size': 1.2,
    'ytick.major.size': 1.2,
    'xtick.minor.size': .8,
    'ytick.minor.size': .8,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
    'axes.formatter.limits': (-5, 5),
    'axes.spines.top': True,
    'axes.spines.right': True,
    'ytick.right': True,
    'axes.labelpad': 3,
    'text.color': offblack,
    'axes.edgecolor': offblack,
    'axes.labelcolor': offblack,
    'xtick.color': offblack,
    'ytick.color': offblack,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.frameon': False,
    'image.cmap': 'Blues',
    'image.interpolation': 'none',
    'pdf.fonttype': 42
})
cm1, cm2 = plt.cm.Blues(.8), plt.cm.Reds(.8)
cb,co,cg,cr = plt.cm.Blues(.6), \
    plt.cm.Oranges(.6), plt.cm.Greens(.6), plt.cm.Reds(.6)
offblack = '#262626'
gray = '0.8'


plotdir = Path('plots')
plotdir.mkdir(exist_ok=True)

plot_functions = {}


def plot(f):
    """
    Plot function decorator.  Calls the function, does several generic tasks,
    and saves the figure as the function name.

    """
    def wrapper(*args, **kwargs):
        logging.info('generating plot: %s', f.__name__)
        f(*args, **kwargs)

        fig = plt.gcf()

        if not fig.get_tight_layout():
            set_tight(fig)

        plotfile = plotdir / '{}.png'.format(f.__name__)
        fig.savefig(str(plotfile), dpi=300)
        logging.info('wrote %s', plotfile)
        plt.close(fig)

    plot_functions[f.__name__] = wrapper

    return wrapper


def set_tight(fig=None, **kwargs):
    """
    Set tight_layout with a better default pad.

    """
    if fig is None:
        fig = plt.gcf()

    kwargs.setdefault('pad', .01)
    fig.set_tight_layout(kwargs)


def auto_ticks(ax, axis='both', minor=False, **kwargs):
    """
    Convenient interface to matplotlib.ticker locators.

    """
    axis_list = []

    if axis in {'x', 'both'}:
        axis_list.append(ax.xaxis)
    if axis in {'y', 'both'}:
        axis_list.append(ax.yaxis)

    for axis in axis_list:
        axis.get_major_locator().set_params(**kwargs)
        if minor:
            axis.set_minor_locator(ticker.AutoMinorLocator(minor))

def darken(rgb, amount=.3):
    """
    Darken a color by the given amount in HSLuv space.

    """
    h, s, l = hsluv.rgb_to_hsluv(rgb)
    return hsluv.hsluv_to_rgb((h, s, (1 - amount)*l))


def obs_color_hsluv(obs, nPDF):
    """
    Return a nice color for the given observable in HSLuv space.
    Use obs_color() to obtain an RGB color.

    """
    if obs == 'RAA':
        return 250, 90, 55

    if 'V2' in obs:
        return 250, 90, 55

    if obs == 'qhat':
        return 250, 90, 55

    if obs == 'posterior':
        return 250, 90, 55

    raise ValueError('unknown observable: {} {}'.format(obs, subobs))

@plot
def inclusive_hadronization(Q=1):

    from scipy.special import beta
    def pythia_frag(x):
        m = 0.2
        return 1/x*(1-x)**0.68 * np.exp(-0.98*m**2/x)
    def g2pi(x): 
        return 0.115 * x**1.4 * (1-x)**8 
    def v2pi(x): 
        return 0.546 * x**(-1.1) * (1-x)**1.28
    def convolve(z,yv,yg,a):
        fv = interp1d(z,yv,fill_value="extrapolate")
        fg = interp1d(z,yg,fill_value="extrapolate")
        ynew = np.zeros_like(z)
        for i, iz in enumerate(z):
            def dy(x):
                return v2pi(x)/x*fv(iz/x) + g2pi(x)/x*fg(iz/x)
            def dy2(x):
                return pythia_frag(x)/x*fv(iz/x) + g2pi(x)/x*fg(iz/x)
            ynew[i] = quad(dy,iz,1-1e-3)[0] if a else quad(dy2,iz,1-1e-3)[0]
        return ynew
    fig, ax = plt.subplots(1,1,figsize=(.65*textwidth,.65*textwidth), sharey=True,sharex=True)
    bins = np.linspace(1e-3,1-1e-3,41)
    db = bins[1:]-bins[:-1]
    zm = (bins[1:]+bins[:-1])/2.
    dd=False
    pid, z = np.loadtxt("Partons/ed.dat").T
    cut = (pid==2) | (pid==-1)
    Yv0 = np.histogram(z[cut],bins=bins,density=dd)[0]/db/1e5
    cut = (pid==21)
    Yg0 = np.histogram(z[cut],bins=bins,density=dd)[0]/db/1e5
    Y0 = [convolve(zm,Yv0,Yg0,True),
          convolve(zm,Yv0,Yg0,False)]

    for N,color,A in zip(['He','Ne','Kr','Xe'],[cr,cg,cb,co],[4,20,84,131]):
        pid, z = np.loadtxt("Partons/e{}.dat".format(N)).T
        cut = (pid==2) | (pid==-1)
        Yv = np.histogram(z[cut],bins=bins,density=dd)[0]/db/1e5
        cut = (pid==21)
        Yg = np.histogram(z[cut],bins=bins,density=dd)[0]/db/1e5
        Y1 = [convolve(zm,Yv,Yg,True),
              convolve(zm,Yv,Yg,False)]
        ax.fill_between(zm, Y1[0]/Y0[0], Y1[1]/Y0[1], color=color, label=r"${{}}^{{{:d}}}${:s}".format(A,N), alpha=.4)

    ax.set_xlim(.0,1)
    ax.plot([.0,1],[1,1],'k-',lw=.3)

    for N,color,A in zip(['He','Ne','Kr','Xe'],[cr,cg,cb,co],[4,20,84,131]):
        x,xl,xh,y,ystat,_,ysys,_ = np.loadtxt("ExpData/{}.dat".format(N)).T
        ax.errorbar(x,y,yerr=ystat,fmt='o', color=darken(color), label=r'HERMES $Q>1.0$ GeV, ${{}}^{{{:d}}}${:s}'.format(A,N))
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')
    
    ax.legend(ncol=2,loc='lower left')
    ax.set_ylim(.2,1.2)

    ax.set_xlabel(r"$z_h=E_h/\nu$ in target frame")
    ax.set_ylabel(r"$D_{eA}/D_{ed}: \pi^{\pm}$")
    ax.annotate(r"$Q>{:1.1f}$ GeV, $\nu>7$ GeV".format(Q),
             xy=(.01,.85),xycoords="axes fraction")
    ax.annotate(r"Fixed $A$, $E_e = 27.6$ GeV, $W>2.0$ GeV, $y<0.85$",
             xy=(.01,.93),xycoords="axes fraction")
    ax.set_title("Changing $D_q^{\pi}(z)$")

    ax.annotate(r"$\hat{q}_{0,q} = \frac{C_F}{C_A}\hat{q}_{0,g} = 0.027$ GeV${}^2$/fm",
             xy=(.01,.3),xycoords="axes fraction")


if __name__ == '__main__':
    import argparse

    choices = list(plot_functions)

    def arg_to_plot(arg):
        arg = Path(arg).stem
        if arg not in choices:
            raise argparse.ArgumentTypeError(arg)
        return arg

    parser = argparse.ArgumentParser(description='generate plots')
    parser.add_argument(
        'plots', nargs='*', type=arg_to_plot, metavar='PLOT',
        help='{} (default: all)'.format(', '.join(choices).join('{}'))
    )
    args = parser.parse_args()

    if args.plots:
        for p in args.plots:
            plot_functions[p]()
    else:
        for f in plot_functions.values():
            f()
