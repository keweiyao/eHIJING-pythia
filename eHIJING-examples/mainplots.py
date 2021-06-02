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
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
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
def pT_dist():
    fig, axes = plt.subplots(4,4,figsize=(1.2*textwidth,1.2*textwidth), sharey=False,sharex=True)
    name = {211:"$\pi^+$",
           -211:"$\pi^-$",
            321:"$K^+$",
           -321:"$K^-$"}
    
    b = np.linspace(0,1,10)
    x = (b[:-1]+b[1:])/2.
    dx = b[1:]-b[:-1]
    Ne = 1e5
    pid, z, pT = np.loadtxt("baseline/1-2-cutRA.dat").T
    for row, zbin in zip(axes,[[.2,.3],[.3,.4],[.4,.6],[.6,.8]]):
        for ax,iid in zip(row,[211,-211,321,-321]):
            cut = (pid==iid) & (zbin[0]<z) & (z<zbin[1])
            Y = np.histogram((pT)[cut],bins=b)[0]/dx/Ne/(zbin[1]-zbin[0])
            ax.plot(x, Y, color=cr, lw=1,alpha=.8, label='Eyeball fits')
            ax.set_xlim(0,1)
            if ax.is_last_row():
                ax.set_xlabel(r"$p_T$ [GeV${}$]")
            if ax.is_first_col():
                ax.set_ylabel(r"$dN/dp_T/dz$")
                ax.legend(loc="lower right")
            ax.set_ylim(ymin=0)


    for row, zbin in zip(axes,[[.2,.3],[.3,.4],[.4,.6],[.6,.8]]):
        for ax,iid,n in zip(row,[211,-211,321,-321],['pi+','pi-','K+','K-']):
            x,y,ystat,_,ysys,_ = np.loadtxt("ExpData/pT/ed-{}-z-{}-{}.dat".format(n,*zbin)).T
            ax.errorbar(x,y,yerr=ystat,fmt='.', color='k'
            #, label=r'HERMES {}'.format(name[iid])
            )
            xb = [0.]+list((x[1:]+x[:-1])/2.) + [1.]
            for il,ih,yl,yh in zip(xb[:-1],xb[1:],y-ysys,y+ysys):
                ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
            ax.legend(loc="upper right")

    for i in [0,1]:
        axes[0,i].set_ylim(0,5)
        axes[1,i].set_ylim(0,3)
        axes[2,i].set_ylim(0,1.5)
        axes[3,i].set_ylim(0,.5)
    for i in [2,3]:
        axes[0,i].set_ylim(0,.7)
        axes[1,i].set_ylim(0,.5)
        axes[2,i].set_ylim(0,.3)
        axes[3,i].set_ylim(0,.15)
    set_tight(fig,pad=0.4)






@plot
def z_dist():
    fig, axes = plt.subplots(2,4,figsize=(1.2*textwidth,.6*textwidth), sharex=True)
    name = {211:"$\pi^+$",
           -211:"$\pi^-$",
            321:"$K^+$",
           -321:"$K^-$"}
    

    for ax,iid,n in zip(axes[0],[211,-211,321,-321],['pi+','pi-','K+','K-']):
        x,y,ystat,_,ysys,_ = np.loadtxt("ExpData/z/ed-{}.dat".format(n)).T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k'
           , label=r'HERMES {}'.format(name[iid])
           )
        xb = [0.1]+list((x[1:]+x[:-1])/2.) + [.9]
        for il,ih,yl,yh in zip(xb[:-1],xb[1:],y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')  

    #xb = np.linspace(0,1,10)

    Ne = 1e5
    pid, z, pT = np.loadtxt("baseline/1-2-cutRA.dat").T
    for ax,axr,iid,n in zip(axes[0],axes[1],[211,-211,321,-321],['pi+','pi-','K+','K-']):
        x,y,ystat,_,ysys,_ = np.loadtxt("ExpData/z/ed-{}.dat".format(n)).T
        xb = [0.1]+list((x[1:]+x[:-1])/2.) + [.9]
        xb = np.array(xb)
        x = (xb[:-1]+xb[1:])/2.
        dx  = xb[1:]-xb[:-1]
        cut = (pid==iid)
        Y = np.histogram(z[cut],bins=xb)[0]/dx/Ne
        ax.plot(x, Y, color=cr, lw=1,alpha=.8,label='Eyeball fits')
        ax.set_xlim(0.,1.)
        if axr.is_last_row():
            ax.set_xlabel(r"$z$")
        if ax.is_first_col():
            ax.set_ylabel(r"$dN/dz$")
        if ax.is_first_col():
            axr.set_ylabel("Exp./MC.")
        ax.semilogy()
        ax.set_ylim(5e-4,1e1)

        x,y,ystat,_,ysys,_ = np.loadtxt("ExpData/z/ed-{}.dat".format(n)).T
        y/=Y
        ystat/=Y
        ysys /= Y
        axr.errorbar(x,y,yerr=ystat,fmt='.', color=cr,
           )
        xb = [0.]+list((x[1:]+x[:-1])/2.) + [1.]
        for il,ih,yl,yh in zip(xb[:-1],xb[1:],y-ysys,y+ysys):
            axr.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=cr,facecolor='none')
        axr.plot([0,1],[1,1],'k--')
        axr.set_ylim(0,3)
        ax.annotate(n, xy=(.6,.8), xycoords="axes fraction")



    #for i in [0,1]:
    #    axes[i].set_ylim(1e-2,5)
    #for i in [2,3]:
    #    axes[i].set_ylim(1e-2,1)
    set_tight(fig,pad=0.4)
   



   

def RA_z(iid):
    sid = {211:'pi+', -211:'pi-', 111:'pi0',
           321:'K+',  -321:'K-',
           2212:'p',  -2212:'pbar'}[iid]
    ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
           321:'K^+',  -321:'K^-',
           2212:'p',  -2212:"\\bar{p}"}[iid]
    fig, ax = plt.subplots(1,1,figsize=(.55*textwidth,.5*textwidth), sharey=True,sharex=True)
    bins = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9, 1])
    db = bins[1:]-bins[:-1]
    zm = (bins[1:]+bins[:-1])/2.
    pid, z, pT = np.loadtxt("test-el/1-2-cutRA.dat", usecols=[0,1,2]).T
    cut = pid==iid
    Y0 = np.histogram(z[cut],bins=bins)[0]/db/4e4
    Q = 1
    for N,color,Z,A in zip(['He','Ne','Kr','Xe'],[cr,cg,cb,co,'k'],
                           [2,10,36,54],[4,20,84,131]):
        pid, z, pT = np.loadtxt("test-el/{}-{}-cutRA.dat".format(Z,A), usecols=[0,1,2]).T
        cut = pid==iid
        Y = np.histogram(z[cut],bins=bins)[0]/db/4e4
        ax.plot(zm, Y/Y0, color=color, label=r"$e$-HIJING", alpha=.7)
    ax.set_xlim(.0,1.5)
    ax.plot([.0,1.5],[1,1],'k-',lw=.3)

    for N,color,A in zip(['He','Ne','Kr','Xe'],[cr,cg,cb,co],[4,20,84,131]):
        x,xl,xh,y,ystat,_,ysys,_ = \
       np.loadtxt("Exp/HERMES/SIDIS/RA_z/e{}-{}.dat".format(N,sid), 
                 skiprows=12,
                 delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color=darken(color), 
                 label=r'HERMES ${{}}^{{{:d}}}${:s}'.format(A,N))
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')
    
    ax.legend(ncol=2,loc='lower left')
    ax.set_ylim(0,1.5)
    ax.set_xlabel(r"$z_h=E_h/\nu$ in target frame")
    ax.set_ylabel(r"$R_{{A}}$")
    ax.annotate(r"${:s}$".format(ssid),xy=(.07,.9),xycoords="axes fraction")
    set_tight(fig, pad=.2)

@plot
def Rpi0():
    RA_z(111)

@plot
def Rpi_p():
    RA_z(211)
@plot
def Rpi_m():
    RA_z(-211)
@plot
def RK_p():
    RA_z(321)
@plot
def RK_m():
    RA_z(-321)
@plot
def Rp():
    RA_z(2212)
@plot
def Rpbar():
    RA_z(-2212)

def RA_pT(iid):
    sid = {211:'pi+', -211:'pi-', 111:'pi0',
           321:'K+',  -321:'K-',
           2212:'p',  -2212:'pbar'}[iid]
    ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
           321:'K^+',  -321:'K^-',
           2212:'p',  -2212:"\\bar{p}"}[iid]
    fig, ax = plt.subplots(1,1,figsize=(.55*textwidth,.5*textwidth), sharey=False,sharex=True)

    for N,color,A in zip(['He','Ne','Kr','Xe'],[cr,cg,cb,co],[4,20,84,131]):
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_pT2/e{}-{}.dat".format(N,sid), 
            #skiprows=12,
            delimiter=',').T
        print(x,y)
        ax.errorbar(x,y,yerr=ystat,fmt='.', color=darken(color), 
                 label=r'HERMES ${{}}^{{{:d}}}${:s}'.format(A,N))
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')
    
    b = np.linspace(0,1.4,10)
    x = (b[:-1]+b[1:])/2.
    dx = b[1:]-b[:-1]
    Ne = 4e5
    zmin = 0.2
    pid, z, pT = np.loadtxt("test-el/1-2-cutRA.dat", usecols=[0,1,2]).T
    cut = (pid == iid) & (z>zmin)
    Y0 = np.histogram(pT[cut],bins=b)[0]/dx/Ne
    for N,color,(Z,A) in zip(['He','Ne','Kr','Xe'],
                                [cr,cg,cb,co,'k'],
                                [(2,4),(10,20),(36,84),(54,131)]):
        pid, z, pT = np.loadtxt("test-el/{}-{}-cutRA.dat".format(Z,A), usecols=[0,1,2]).T
        cut = (pid == iid) & (z>zmin)
        Y = np.histogram(pT[cut],bins=b)[0]/dx/Ne
        ax.plot(x**2, Y/Y0, color=color, lw=1,alpha=.8)
    ax.plot([1e-2,10],[1,1],'k-', lw=.5)
    ax.set_xlim(0,2)
    ax.set_ylim(.4,2.1)
    ax.set_xlabel(r"$p_T^2$ [GeV${}^2$]")
    ax.set_ylabel(r"$R_{A}$")
    ax.legend(loc="upper left")
    ax.annotate(r"${:s}$".format(ssid),xy=(.6,.9),xycoords="axes fraction")
    set_tight(fig,pad=0.2)

@plot
def RApT_pi0():
    RA_pT(111)
@plot
def RApT_pi_p():
    RA_pT(211)
@plot
def RApT_pi_m():
    RA_pT(-211)
@plot
def RApT_K_p():
    RA_pT(321)
@plot
def RApT_K_m():
    RA_pT(-321)
@plot
def RApT_p():
    RA_pT(2212)
@plot
def RApT_pbar():
    RA_pT(-2212)


@plot
def RA_parton():
    def v2pi(x): 
        if x>1 or x<0:
            return 0;
        return 0.546 * x**(-1.1) * (1-x)**1.28 # pi
    def g2pi(x): 
        if x>1 or x<0:
            return 0;
        return 0.115 * x**1.4 * (1-x)**8 # pi
    def convolve(z,Q,G):
        fv = interp1d(z,Q,fill_value=0, bounds_error=False)
        fg = interp1d(z,G,fill_value=0, bounds_error=False)
        def frag(x,zi):
            return  v2pi(x)*fv(zi/x)/x + g2pi(x)*fg(zi/x)/x
        return np.array([quad(frag,iz,1.5,args=(iz,))[0] for iz in z])

    fig, ax = plt.subplots(1,1,figsize=(.55*textwidth,.5*textwidth), sharey=True,sharex=True)
    bins = np.linspace(0.0,1.5,51)
    db = bins[1:]-bins[:-1]
    zm = (bins[1:]+bins[:-1])/2.
    pid, z, pT = np.loadtxt("test/1-2-cutRA.dat").T
    print(z.max())
    Yg0 = np.histogram(z[pid==21],bins=bins)[0]/db/4e4
    Yq0 = np.histogram(z[np.abs(pid)<3],bins=bins)[0]/db/4e4
    Y0 = convolve(zm, Yq0, Yg0)
    for N,color,Z,A in zip(['He','Ne','Kr','Xe'],[cr,cg,cb,co,'k'],
                           [2,10,36,54],[4,20,84,131]):
        pid, z, pT = np.loadtxt("test/{}-{}-cutRA.dat".format(Z,A)).T
        print(z.max())
        cut = pid==21
        Yg = np.histogram(z[cut],bins=bins)[0]/db/4e4

        cut = np.abs(pid)<3
        Yq = np.histogram(z[cut],bins=bins)[0]/db/4e4

        Y = convolve(zm,Yq, Yg)
        ax.plot(zm, Y/Y0, '-', color=color, alpha=.7)

    ax.set_xlim(.0,1)
    ax.plot([.0,1],[1,1],'k-',lw=.3)

    for N,color,A in zip(['He','Ne','Kr','Xe'],[cr,cg,cb,co],[4,20,84,131]):
        x,xl,xh,y,ystat,_,ysys,_ = \
       np.loadtxt("Exp/HERMES/SIDIS/RA_z/e{}-{}.dat".format(N,'pi+'), 
                 skiprows=12,
                 delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color=darken(color), 
                 label=r'HERMES ${{}}^{{{:d}}}${:s}'.format(A,'pi+'))
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')
    
    ax.legend(ncol=2,loc='lower left')
    ax.set_ylim(0,1.2)
    ax.set_xlabel(r"$z_h=E_h/\nu$ in target frame")
    ax.set_ylabel(r"$R_{{A}}$")
    set_tight(fig, pad=.2)



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
