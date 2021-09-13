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
import matplotlib.gridspec as gridspec
fontsmall, fontnormal, fontlarge = 4.5, 6, 7
offblack = '#262626'
aspect = 1/1.618
resolution = 72.27
textwidth = 270/resolution
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
cb,co,cg,cr,ck = plt.cm.Blues(.6), \
    plt.cm.Oranges(.6), plt.cm.Greens(.6), plt.cm.Reds(.6),\
    plt.cm.copper(.1)
offblack = '#262626'
gray = '0.8'


plotdir = Path('plots')
plotdir.mkdir(exist_ok=True)

plot_functions = {}

def set_prelim(ax, xy=(.45,.05)):
    ax.annotate("Preliminary", xy=xy, alpha=.5, xycoords="axes fraction")

def plot(f):
    """
    Plot function decorator.  Calls the function, does several generic tasks,
    and saves the figure as the function name.

    """
    def wrapper(*args, **kwargs):
        logging.info('generating plot: %s', f.__name__)
        f(*args, **kwargs)

        fig = plt.gcf()

        #if not fig.get_tight_layout():
        #    set_tight(fig)
        os.makedirs(plotdir /f.__name__.split('_')[0], exist_ok=True)
        plotfile = plotdir /f.__name__.split('_')[0]/ '{}.png'.format(f.__name__.split('_')[1])
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

def darken(rgb, amount=.15):
    """
    Darken a color by the given amount in HSLuv space.

    """
    h, s, l = hsluv.rgb_to_hsluv(rgb)
    return hsluv.hsluv_to_rgb((h, s, (1 - amount)*l))


NCZA = ['He','Ne','Kr','Xe'],[cb,cg,cr,ck],[2,10,36,54],[4,20,84,131]



Ks = ['0d5','1','2']
entry = ['id','z','pT','nu','Q2']
"""f = h5py.File("Production.h5",'w')
for group in ['Generalized','Collinear']:
    g = f.create_group(group)
    for K in Ks:
        folder = "Run/Production/{}/K{}/".format(group, K)
        g2 = g.create_group(K)
        for N, Z, A in zip(['d', 'He', 'Ne', 'Kr', 'Xe'],
                           [ 1,   2,    10,   36,   54],
                           [ 2,   4,    20,   84,   131]):
            print(group, K, N)
            data = np.loadtxt(folder+"/{}-{}-cutRA.dat".format(Z,A)).T
            g3 = g2.create_group(N)
            for name, ds in zip(entry, data):
                g3.create_dataset(name, data=ds)"""
print("loading")
f0 = h5py.File("Production-pTbroadening.h5",'r')
sid = {211:'pi+', -211:'pi-', 111:'pi0',
   321:'K+',  -321:'K-',
   2212:'p',  -2212:'pbar'}
ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
   321:'K^+',  -321:'K^-',
   2212:'p',  -2212:"\\bar{p}"}

def RA_z(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth*1.1,.3*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):       
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_z/e{}-{}.dat".format(N,sid[iid]),
                 skiprows=12,
                 delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',
                 label=r'HERMES')
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array(list(xl)+[xh[-1]])
    db = bins[1:]-bins[:-1]
    pid, z, pT, nu, Q2 = [np.concatenate([f0['Generalized/{}/d/{}'.format(K,it)][()] for K in Ks])
                          for it in entry]
                        
    cut = (pid==iid) & (nu>6)
    Y0 = np.histogram(z[cut],bins=bins)[0]/db/3
    for ax,N,_,Z,A in zip(axes,*NCZA):
        """Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Collinear/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = ( pid==iid ) & (nu>6)
            Y.append(np.histogram(z[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cr, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cr, alpha=.4, label='Higher-twist')"""

        Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Generalized/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = ( pid==iid ) & (nu>6)
            Y.append(np.histogram(z[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cb, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cb, alpha=.4, label='Generalized')
        ax.set_xlim(.1,1.05)
        ax.plot([.0,1.5],[1,1],'k-',lw=.3)
        if ax.is_first_col():
            ax.legend(loc='lower left')
            ax.set_ylabel(r"$R_{{A}}$")
        ax.annotate(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]),xy=(.07,.86),xycoords="axes fraction")
        ax.set_ylim(.2,1.3)
        ax.set_xlabel(r"$z_h$")
    set_prelim(ax)
    set_tight(fig)


def RA_pT(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth*1.1,.3*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):       
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_pT2/e{}-{}.dat".format(N,sid[iid]),
            #skiprows=12,
            delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',
                 label=r'HERMES')
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array(list(xl)+[xh[-1]])
    db = bins[1:]-bins[:-1]
    pid, z, pT, nu, Q2 = [np.concatenate([f0['Generalized/{}/d/{}'.format(K,it)][()] for K in Ks])
                          for it in entry]
                        
    cut = (pid==iid) & (nu>6) & (z>.2)
    Y0 = np.histogram((pT**2)[cut],bins=bins)[0]/db/3
    for ax,N,_,Z,A in zip(axes,*NCZA):
        """Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Collinear/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = ( pid==iid ) & (nu>6) & (z>.2)
            Y.append(np.histogram((pT**2)[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cr, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cr, alpha=.4, label='Higher-twist')
"""
        Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Generalized/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = ( pid==iid ) & (nu>6) & (z>.2)
            Y.append(np.histogram((pT**2)[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cb, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cb, alpha=.4, label='Generalized')
        ax.plot([.0,2],[1,1],'k-',lw=.3)
        if ax.is_first_col():
            #ax.legend(loc=(.07,.5))
            ax.set_ylabel(r"$R_{{A}}$")
        ax.annotate(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]),xy=(.07,.9),xycoords="axes fraction")
        ax.set_xlim(0,2.2)
        ax.set_ylim(.5,1.5)
        ax.set_xlabel(r"$p_T^2$ [GeV${}^2$]")
    set_prelim(ax)
    set_tight(fig)


def RA_nu(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth,.31*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):       
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_nu/e{}-{}.dat".format(N,sid[iid]),
            #skiprows=12,
            delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',
                 label=r'HERMES')
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array(list(xl)+[xh[-1]])
    db = bins[1:]-bins[:-1]
    pid, z, pT, nu, Q2 = [np.concatenate([f0['Generalized/{}/d/{}'.format(K,it)][()] for K in Ks])
                          for it in entry]
                        
    cut = (pid==iid) & (z>.2)
    Y0 = np.histogram(nu[cut],bins=bins)[0]/db/3
    for ax,N,_,Z,A in zip(axes,*NCZA):
        """Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Collinear/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = (pid==iid) & (z>.2)
            Y.append(np.histogram(nu[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cr, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cr, alpha=.4, label='Collinear')
"""
        Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Generalized/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = (pid==iid) & (z>.2)
            Y.append(np.histogram(nu[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cb, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cb, alpha=.4, label='Generalized')
        ax.plot([2, 25],[1,1],'k-',lw=.3)
        if ax.is_first_col():
            ax.legend(loc='lower left')
            ax.set_ylabel(r"$R_{{A}}$")
        ax.annotate(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]),xy=(.07,.9),xycoords="axes fraction")
        ax.set_ylim(0,1.4)
        ax.set_xlim(2,25)
        ax.set_xlabel(r"$\nu$ [GeV]")
    set_prelim(ax)
    set_tight(fig)


def RA_Q2(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth,.31*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):       
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_Q2/e{}-{}.dat".format(N,sid[iid]),
            #skiprows=12,
            delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',
                 label=r'HERMES')
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array(list(xl)+[xh[-1]])
    db = bins[1:]-bins[:-1]
    pid, z, pT, nu, Q2 = [np.concatenate([f0['Generalized/{}/d/{}'.format(K,it)][()] for K in Ks])
                          for it in entry]
                        
    cut = (pid==iid) & (z>.2)& (nu>6)
    Y0 = np.histogram(Q2[cut],bins=bins)[0]/db/3
    for ax,N,_,Z,A in zip(axes,*NCZA):
        """Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Collinear/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = (pid==iid) & (z>.2) & (nu>6)
            Y.append(np.histogram(Q2[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cr, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cr, alpha=.4, label='Collinear')
"""
        Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Generalized/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = (pid==iid) & (z>.2)& (nu>6)
            Y.append(np.histogram(Q2[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cb, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cb, alpha=.4, label='Generalized')
        ax.plot([1,25],[1,1],'k-',lw=.3)
        if ax.is_first_col():
            ax.legend(loc='lower left')
            ax.set_ylabel(r"$R_{{A}}$")
        ax.annotate(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]),xy=(.07,.9),xycoords="axes fraction")
        ax.set_ylim(0,1.4)
        ax.set_xlim(1,25)
        ax.semilogx()
        ax.set_xlabel(r"$Q^2$ [GeV${}^2$]")
    set_prelim(ax)
    set_tight(fig)
@plot
def Rz_pi0():
    RA_z(111)
@plot
def Rz_pip():
    RA_z(211)
@plot
def Rz_pim():
    RA_z(-211)
@plot
def Rz_Kp():
    RA_z(321)
@plot
def Rz_Km():
    RA_z(-321)
@plot
def Rz_p():
    RA_z(2212)
@plot
def Rz_pbar():
    RA_z(-2212)


@plot
def RpT_pi0():
    RA_pT(111)
@plot
def RpT_pip():
    RA_pT(211)
@plot
def RpT_pim():
    RA_pT(-211)
@plot
def RpT_Kp():
    RA_pT(321)
@plot
def RpT_Km():
    RA_pT(-321)
@plot
def RpT_p():
    RA_pT(2212)
@plot
def RpT_pbar():
    RA_pT(-2212)
@plot
def RQ2_pi0():
    RA_Q2(111)
@plot
def RQ2_pip():
    RA_Q2(211)
@plot
def RQ2_pim():
    RA_Q2(-211)
@plot
def RQ2_Kp():
    RA_Q2(321)
@plot
def RQ2_Km():
    RA_Q2(-321)
@plot
def RQ2_p():
    RA_Q2(2212)
@plot
def RQ2_pbar():
    RA_Q2(-2212)

@plot
def Rnu_pi0():
    RA_nu(111)
@plot
def Rnu_pip():
    RA_nu(211)
@plot
def Rnu_pim():
    RA_nu(-211)
@plot
def Rnu_Kp():
    RA_nu(321)
@plot
def Rnu_Km():
    RA_nu(-321)
@plot
def Rnu_p():
    RA_nu(2212)
@plot
def Rnu_pbar():
    RA_nu(-2212)




@plot 
def Other_Qs():
    Tamin = 0.05/5.076**2
    Tamax = 2.8/5.076**2
    Ta = np.linspace(Tamin, Tamax, 21)
    lnQ2x = np.linspace(np.log(1), np.log(1e3), 21)
    Qs2_data = np.loadtxt("Run/Tables/Qs.dat").reshape(21,21)
    from scipy.interpolate import interp1d
    
    fig, ax = plt.subplots(1,1, figsize=(textwidth*.4, textwidth*.4))
    x = np.exp(np.linspace(-4.1,0,21))
    for ita, c in zip([-1,5],[cb,cr]):
        ta = Ta[ita]
        Qs2 = interp1d(lnQ2x, Qs2_data[ita])
        y1 = Qs2( np.log(4/x))
        y2 = Qs2( np.log(16/x))
        ax.plot(x, y1, '-', color=c, label='$Q=2$ GeV')
        ax.plot(x, y2, '--', color=c, label='$Q=4$ GeV') 
        ax.fill_between(x, y1, y2, color=c, alpha=.3)
    ax.legend()
    ax.set_ylabel(r"$Q_S^2$ [GeV${}^2$]")
    ax.set_xlabel(r"$x_q$")
    ax.set_xlim(1e-2,1)
    ax.set_ylim(0,.57)
    ax.annotate(r"$T_A={:1.2f}$ [fm${{}}^{{-2}}$]".format(Ta[-1]*5.076**2),
               xy=(.1, .88), xycoords="axes fraction", color=cb, fontsize=5)
    ax.annotate(r"$T_A={:1.2f}$ [fm${{}}^{{-2}}$]".format(Ta[5]*5.076**2),
               xy=(.1, .75), xycoords="axes fraction", color=cr, fontsize=5)
    ax.semilogx()


@plot 
def Other_qhat():
    Tamin = 0.05/5.076**2
    Tamax = 2.8/5.076**2
    Ta = np.linspace(Tamin, Tamax, 21)
    lnQ2x = np.linspace(np.log(1), np.log(1e3), 21)
    Qs2_data = np.loadtxt("Run/Tables/Qs.dat").reshape(21,21)
    from scipy.interpolate import interp1d
    rho0 = 0.17/5.076**3
    fig, ax = plt.subplots(1,1, figsize=(textwidth*.4, textwidth*.4))
    x = np.exp(np.linspace(-4.1,0,21))
    for ita, c in zip([-1,5],[cb,cr]):
        ta = Ta[ita]
        Qs2 = interp1d(lnQ2x, Qs2_data[ita])
        y1 = Qs2( np.log(4/x))/ta*rho0 * 5.076 * 4/9
        y2 = Qs2( np.log(16/x))/ta*rho0 * 5.076 * 4/9
        ax.plot(x, y1, '-', color=c, label='$Q=2$ GeV')
        ax.plot(x, y2, '--', color=c, label='$Q=4$ GeV') 
        ax.fill_between(x, y1, y2, color=c, alpha=.3)
    ax.legend()
    ax.set_ylabel(r"$\hat{q}_F$ [GeV${}^2$/fm]")
    ax.set_xlabel(r"$x_q$")
    ax.set_xlim(1e-2,1)
    ax.set_ylim(0,.03)
    ax.set_yticks([0,.01,.02, .03])
    ax.annotate(r"$T_A={:1.2f}$ [fm${{}}^{{-2}}$]".format(Ta[-1]*5.076**2),
               xy=(.1, .88), xycoords="axes fraction", color=cb, fontsize=5)
    ax.annotate(r"$T_A={:1.2f}$ [fm${{}}^{{-2}}$]".format(Ta[5]*5.076**2),
               xy=(.1, .75), xycoords="axes fraction", color=cr, fontsize=5)
    ax.semilogx()

@plot
def Other_xG():
    #fig, ax = plt.subplots(1,1, )
    x = np.exp(np.linspace(-4.6,0,21))
    kT = np.exp(np.linspace(-2.3,2.3,21))
    xG = np.loadtxt("/home/weiyao/Research/TMDlib/tmdlib-2.2.01/examples-c++/xG.dat")
    #for ikT, ixG in zip(kT, xG.T):
    #    ax.plot(x+ikT, ixG+ikT, 'r-')



    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    from mpl_toolkits.mplot3d import Axes3D  
    fig = plt.figure(figsize=(textwidth*.5, textwidth*.5)) 
    ax = fig.add_subplot(projection='3d',
                      )



    X, Y = np.meshgrid(np.log(x), np.log(kT))
    X = X.T
    Y = Y.T
    # Plot the surface.
    surf = ax.plot_surface(X, Y, xG, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


    #ax.semilogx()
    ax.set_xlim(np.log(.01), 0)
    ax.set_ylim(-2.3,2.3)
    ax.set_ylabel(r"$k_T$ [GeV]")
    ax.set_title(r"$xG(x, k_T, \mu=5 {\rm~GeV})$")
    ax.set_xlabel(r"$x$")
    ax.view_init(30,50)
    #ax.xaxis.set_scale('log')
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xticks([np.log(0.01), np.log(.1), np.log(1)])
    ax.set_xticklabels([r"$10^{-2}$", r"$10^{-1}$",r"$10^{0}$"])
    ax.set_yticks([np.log(0.1), np.log(1), np.log(10)])
    ax.set_yticklabels([r"$10^{-1}$", r"$10^{0}$",r"$10^{1}$"])
    set_tight(fig, pad=0.5, rect=[.05,.12,1,1])

@plot
def Flavor_RA():
    fig, axes = plt.subplots(2,3,figsize=(textwidth*.8,.6*textwidth), sharey=True,sharex=True)
    axes = axes.flatten()
    for ax, iid in zip(axes, [211,321,2212,111, -321, -2212]):
        N,_,Z,A = [it[-1] for it in NCZA]
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_z/e{}-{}.dat".format(N,sid[iid]),
                 skiprows=12,
                 delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',
                 label=r'HERMES')
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
        bins = np.array(list(xl)+[xh[-1]])
        db = bins[1:]-bins[:-1]
        pid, z, pT, nu, Q2 = [np.concatenate([f0['Generalized/{}/d/{}'.format(K,it)][()] for K in Ks])
                          for it in entry]
                        
        cut = (pid==iid) & (nu>6)
        Y0 = np.histogram(z[cut],bins=bins)[0]/db/3
        """Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Collinear/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = ( pid==iid ) & (nu>6) 
            Y.append(np.histogram(z[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cr, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cr, alpha=.4, label='Higher-twist')"""

        Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Generalized/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = ( pid==iid ) & (nu>6) 
            Y.append(np.histogram(z[cut],bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cb, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cb, alpha=.4, label='Generalized')
        ax.set_xlim(.1,1.1)
        ax.plot([.0,1.5],[1,1],'k-',lw=.3)
        if ax.is_first_col():
            
            ax.set_ylabel(r"$R_{{A}}$")
        ax.annotate(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]),xy=(.25,.86),xycoords="axes fraction")
        ax.set_ylim(0,1.25)
        if ax.is_last_row():
            ax.set_xlabel(r"$z_h$")
    axes[2].legend(loc='lower left')
    set_prelim(ax, xy=(.2,.1))

    #set_tight(fig)
    plt.subplots_adjust(wspace=0, hspace=0, top=.99, right=.99, left=.1)


@plot
def zdiff_RApT():
    fig, axes = plt.subplots(1,3,figsize=(textwidth*.8,.35*textwidth), sharey=True,sharex=True)
    axes = axes.flatten()
    iid = 211
    N,_,Z,A = [it[-1] for it in NCZA]   
    for ax,zl,zh in zip(axes, [.2,.4,.7],[.4,.7,1.0]):
        x,xl,xh,y,ystat,_,ysys,_ = \
        np.loadtxt("Exp/HERMES/SIDIS/RA_z_pT/{}-pi-{}-{}.dat".format(N,zl,zh)
                  ).T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',
                 label=r'HERMES')
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
        bins = np.array(list(xl)+[xh[-1]])
        db = bins[1:]-bins[:-1]
        pid, z, pT, nu, Q2 = [np.concatenate([f0['Generalized/{}/d/{}'.format(K,it)][()] for K in Ks])
                          for it in entry]
                        
        cut = (pid==iid) & (nu>6) & (zl<z) & (z<zh)
        Y0 = np.histogram(pT[cut]**2,bins=bins)[0]/db/3
        """Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Collinear/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = ( pid==iid ) & (nu>6) & (zl<z) & (z<zh)
            Y.append(np.histogram(pT[cut]**2,bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cr, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cr, alpha=.4, label='Higher-twist')"""

        Y = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['Generalized/{}/{}/{}'.format(K,N,it)][()]
                          for it in entry]
            cut = ( pid==iid ) & (nu>6) & (zl<z) & (z<zh)
            Y.append(np.histogram(pT[cut]**2,bins=bins)[0]/db)
        Y = np.array(Y)/Y0
        ax.plot(x, Y[1], color=cb, alpha=.7)
        ax.fill_between(x, Y[0], Y[2], 
                        color=cb, alpha=.4, label='Generalized')
        ax.set_xlim(0,1.9)
        ax.plot([.0,2],[1,1],'k-',lw=.3)
        if ax.is_first_col():
            
            ax.set_ylabel(r"$R_{{A}}$")
        ax.annotate(r"${}<z_h<{}$".format(zl,zh),xy=(.1,.86),xycoords="axes fraction")
        ax.set_ylim(0,2)
        if ax.is_last_row():
            ax.set_xlabel(r"$p_T^2$ [GeV${}^2$]")
    axes[0].legend(loc='lower right')
    set_prelim(ax, xy=(.2,.1))
    plt.suptitle(r"$e+{{\rm {:s}}}\rightarrow \pi^{{\pm}}+\cdots$".format(N))

    plt.subplots_adjust(wspace=0, hspace=0, top=.9, right=.99, left=.1, bottom=.2)

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
        for it, f in zip(plot_functions.keys(), plot_functions.values()):
            print("Generating ", it+".png")
            f()
