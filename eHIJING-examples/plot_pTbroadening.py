#!/usr/bin/env python3
import os, subprocess
import logging
from pathlib import Path
import sys
import h5py
import hsluv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import glob
from scipy.interpolate import interp1d, interp2d
from matplotlib.gridspec import GridSpec
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

def set_prelim(ax, xy=(.1,.05)):
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

labels = [r'Gen',r'H-T']
Models  = ['Generalized','Collinear']
ModelColors = [cr, cb]
scaled = False
Ks = ['0d5','1','2']
ModelK = Ks
iKs = [0.5, 1,2]
entry = ['id','z','pT','nu','Q2']
rerun = True
outfile = "Production-Frag.h5"
if rerun:
    f = h5py.File(outfile,'w')
    for group in Models:
        if group in f:
            del f[group]
        g = f.create_group(group)
        for K in Ks:
            folder = "Run/Production/Frag/{}/{}/".format(group, K)
            if not os.path.exists(folder):
                continue
            g2 = g.create_group(K)
            for N, Z, A in zip(['d', *NCZA[0]],
                               [ 1,   2,    10,   36,   54],
                               [ 2,   4,    20,   84,   131]):
                print(group, K, N)
                data = np.loadtxt(folder+"/{}-{}-cutRA.dat".format(Z,A)).T
                g3 = g2.create_group(N)
                for name, ds in zip(entry, data):
                    g3.create_dataset(name, data=ds)
print("loading")
f0 = h5py.File(outfile,'r')
sid = {211:'pi+', -211:'pi-', 111:'pi0',
   321:'K+',  -321:'K-',
   2212:'p',  -2212:'pbar'}
ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
   321:'K^+',  -321:'K^-',
   2212:'p',  -2212:"\\bar{p}"}

def DpT2_x(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth,.35*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):
        x,y,ystat,ysys =  np.loadtxt("pTbroadening-ExpData/pT2_vs_x/e{}-{}.dat".format(N,sid[iid])).T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',label=r'HERMES')
        bins = np.array([x[0]*1.5-x[1]*.5, *((x[1:]+x[:-1])/2.), x[-1]*1.5-x[-2]*.5])
        xl = bins[:-1]
        xh = bins[1:]
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.exp(np.linspace(np.log(.01),np.log(1.0),5))
    db = bins[1:]-bins[:-1]
    x = (bins[1:]+bins[:-1])/2.
    for model, color, label in zip(Models, ModelColors, labels):
        YK0 = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
            pT2 = pT**2
            xB = Q2/2./0.938/nu
            cut = (pid==iid) & (nu<23) & (z>.2)
            binv = xB[cut]
            wv = pT2[cut]
            X1 = np.histogram(binv, weights=wv, bins=bins)[0] \
                 / np.histogram(binv, bins=bins)[0]
            X2 = np.histogram(binv, weights=wv, bins=bins)[0] \
                 / np.histogram(binv, bins=bins)[0]
            Neff = np.histogram(binv, bins=bins)[0]
            dY0 = np.sqrt((X2-X1**2)/Neff)
            Y0 = np.copy(X1)
            YK0.append(Y0)
        for ax,N,_,Z,A in zip(axes,*NCZA):
            YK1 = []
            for K in Ks:
                pid, z, pT, nu, Q2 = [f0['{}/{}/{}/{}'.format(model,K,N,it)][()] for it in entry]
                pT2 = pT**2
                xB = Q2/2./nu/.938
                cut = (pid==iid) & (nu<23) & (z>.2)
                binv = xB[cut]
                wv = pT2[cut]
                X1 = np.histogram(binv, weights=wv, bins=bins)[0] \
                     / np.histogram(binv, bins=bins)[0]
                X2 = np.histogram(binv, weights=wv, bins=bins)[0] \
                     / np.histogram(binv, bins=bins)[0]
                Neff = np.histogram(binv, bins=bins)[0]
                dY = np.sqrt((X2-X1**2)/Neff)
                Y = np.copy(X1)
                YK1.append(Y)
            #Y = np.array(Y) - Y0
            #dY = np.sqrt(dY**2+dY0**2)
            #ax.(x, Y, yerr=dY, color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.fill_between(x, YK1[0]-YK0[0], YK1[2]-YK0[2],
                            color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK1[1]-YK0[1], color=color)
            ax.set_xlim(.02,1.0)
            ax.plot([.02,1.0],[0,0],'k-',lw=.3)
            if ax.is_first_col():
                ax.legend(loc=(.1,.6))
                ax.set_ylabel(r"$\Delta\langle p_T^2\rangle$ [GeV${}^2$]" if not scaled \
                         else r"$\Delta\langle p_T^2\rangle / K$")
            ax.set_title(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]))
            ax.set_ylim(-.02,.1)
            ax.set_xlabel(r"$x_B$")
            ax.semilogx()
    set_prelim(ax)
    plt.subplots_adjust(wspace=0, left=.12, right=.97, bottom=.2, top=.88)


def DpT2_z(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth,.35*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):
        x,y,ystat,ysys =  np.loadtxt("pTbroadening-ExpData/pT2_vs_z/e{}-{}.dat".format(N,sid[iid])).T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',label=r'HERMES')
        bins = np.array([x[0]*1.5-x[1]*.5, *((x[1:]+x[:-1])/2.), x[-1]*1.5-x[-2]*.5])
        xl = bins[:-1]
        xh = bins[1:]
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array([0.2,.3,.4, 0.5, 0.7, .9, 1.1])
    print(bins)
    db = bins[1:]-bins[:-1]
    x = (bins[1:]+bins[:-1])/2.
    for model, color, label in zip(Models, ModelColors, labels):
        YK0 = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
            pT2 = pT**2
            xB = Q2/2./0.938/nu
            cut = (pid==iid)
            binv = z[cut]
            wv = pT2[cut]
            X1 = np.histogram(binv, weights=wv, bins=bins)[0] \
                 / np.histogram(binv, bins=bins)[0]
            X2 = np.histogram(binv, weights=wv, bins=bins)[0] \
                 / np.histogram(binv, bins=bins)[0]
            Neff = np.histogram(binv, bins=bins)[0]
            dY0 = np.sqrt((X2-X1**2)/Neff)
            Y0 = np.copy(X1)
            YK0.append(Y0)
        for ax,N,_,Z,A in zip(axes,*NCZA):
            YK1 = []
            for K in Ks:
                pid, z, pT, nu, Q2 = [f0['{}/{}/{}/{}'.format(model,K,N,it)][()] for it in entry]
                pT2 = pT**2
                xB = Q2/2./nu/.938
                cut = (pid==iid)
                binv = z[cut]
                wv = pT2[cut]
                X1 = np.histogram(binv, weights=wv, bins=bins)[0] \
                     / np.histogram(binv, bins=bins)[0]
                X2 = np.histogram(binv, weights=wv, bins=bins)[0] \
                     / np.histogram(binv, bins=bins)[0]
                Neff = np.histogram(binv, bins=bins)[0]
                dY = np.sqrt((X2-X1**2)/Neff)
                Y = np.copy(X1)
                YK1.append(Y)
            #Y = np.array(Y) - Y0
            #dY = np.sqrt(dY**2+dY0**2)
            #ax.(x, Y, yerr=dY, color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.fill_between(x, YK1[0]-YK0[0], YK1[2]-YK0[2],
                            color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK1[1]-YK0[1], color=color)
            ax.set_xlim(.1,1.05)
            ax.plot([.1,1.05],[0,0],'k-',lw=.3)
            if ax.is_first_col():
                ax.legend(loc=(.1,.6))
                ax.set_ylabel(r"$\Delta\langle p_T^2\rangle$ [GeV${}^2$]" if not scaled \
                         else r"$\Delta\langle p_T^2\rangle / K$")
            ax.set_title(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]))
            ax.set_ylim(-.02,.08)
            ax.set_xlabel(r"$z_h$")
    set_prelim(ax)
    plt.subplots_adjust(wspace=0, left=.12, right=.97, bottom=.2, top=.88)


def DpT2_nu(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth*1,.35*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):
        x,y,ystat,ysys =  np.loadtxt("pTbroadening-ExpData/pT2_vs_nu/e{}-{}.dat".format(N,sid[iid])).T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',label=r'HERMES')
        bins = np.array([x[0]*1.5-x[1]*.5, *((x[1:]+x[:-1])/2.), x[-1]*1.5-x[-2]*.5])
        xl = bins[:-1]
        xh = bins[1:]
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array([x[0]*1.5-x[1]*.5, *((x[1:]+x[:-1])/2.), x[-1]*1.5-x[-2]*.5])
    db = bins[1:]-bins[:-1]
    x = (bins[1:]+bins[:-1])/2.
    for model, color, label in zip(Models, ModelColors, labels):
        YK0 = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
            pT2 = pT**2
            xB = Q2/2./0.938/nu
            cut = (pid==iid) & (z>.2)
            binv = nu[cut]
            wv = pT2[cut]
            X1 = np.histogram(binv, weights=wv, bins=bins)[0] \
                 / np.histogram(binv, bins=bins)[0]
            X2 = np.histogram(binv, weights=wv, bins=bins)[0] \
                 / np.histogram(binv, bins=bins)[0]
            Neff = np.histogram(binv, bins=bins)[0]
            dY0 = np.sqrt((X2-X1**2)/Neff)
            Y0 = np.copy(X1)
            YK0.append(Y0)
        for ax,N,_,Z,A in zip(axes,*NCZA):
            YK1 = []
            for K in Ks:
                pid, z, pT, nu, Q2 = [f0['{}/{}/{}/{}'.format(model,K,N,it)][()] for it in entry]
                pT2 = pT**2
                xB = Q2/2./nu/.938
                cut = (pid==iid) & (z>.2)
                binv = nu[cut]
                wv = pT2[cut]
                X1 = np.histogram(binv, weights=wv, bins=bins)[0] \
                     / np.histogram(binv, bins=bins)[0]
                X2 = np.histogram(binv, weights=wv, bins=bins)[0] \
                     / np.histogram(binv, bins=bins)[0]
                Neff = np.histogram(binv, bins=bins)[0]
                dY = np.sqrt((X2-X1**2)/Neff)
                Y = np.copy(X1)
                YK1.append(Y)
            #Y = np.array(Y) - Y0
            #dY = np.sqrt(dY**2+dY0**2)
            #ax.(x, Y, yerr=dY, color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.fill_between(x, YK1[0]-YK0[0], YK1[2]-YK0[2],
                            color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK1[1]-YK0[1], color=color)
            ax.set_xlim(0,25)
            ax.plot([0,25],[0,0],'k-',lw=.3)
            if ax.is_first_col():
                ax.legend(loc=(.1,.6))
                ax.set_ylabel(r"$\Delta\langle p_T^2\rangle$ [GeV${}^2$]" if not scaled \
                         else r"$\Delta\langle p_T^2\rangle / K$")
            ax.set_title(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]))
            ax.set_ylim(-.02,.06)
            ax.set_xlabel(r"$\nu$ [GeV]")
    set_prelim(ax)
    plt.subplots_adjust(wspace=0, left=.12, right=.97, bottom=.2, top=.88)

def DpT2_Q2(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth*1,.35*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):
        x,y,ystat,ysys =  np.loadtxt("pTbroadening-ExpData/pT2_vs_Q2/e{}-{}.dat".format(N,sid[iid])).T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',label=r'HERMES')
        bins = np.array([x[0]*1.5-x[1]*.5, *((x[1:]+x[:-1])/2.), x[-1]*1.5-x[-2]*.5])
        xl = bins[:-1]
        xh = bins[1:]
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array([x[0]*1.5-x[1]*.5, *((x[1:]+x[:-1])/2.), x[-1]*1.5-x[-2]*.5])
    db = bins[1:]-bins[:-1]
    x = (bins[1:]+bins[:-1])/2.
    for model, color, label in zip(Models, ModelColors, labels):
        YK0 = []
        for K in Ks:
            pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
            pT2 = pT**2
            xB = Q2/2./0.938/nu
            cut = (pid==iid) & (z>.2)
            binv = Q2[cut]
            wv = pT2[cut]
            X1 = np.histogram(binv, weights=wv, bins=bins)[0] \
                 / np.histogram(binv, bins=bins)[0]
            X2 = np.histogram(binv, weights=wv, bins=bins)[0] \
                 / np.histogram(binv, bins=bins)[0]
            Neff = np.histogram(binv, bins=bins)[0]
            dY0 = np.sqrt((X2-X1**2)/Neff)
            Y0 = np.copy(X1)
            YK0.append(Y0)
        for ax,N,_,Z,A in zip(axes,*NCZA):
            YK1 = []
            for K in Ks:
                pid, z, pT, nu, Q2 = [f0['{}/{}/{}/{}'.format(model,K,N,it)][()] for it in entry]
                pT2 = pT**2
                xB = Q2/2./nu/.938
                cut = (pid==iid) & (z>.2)
                binv = Q2[cut]
                wv = pT2[cut]
                X1 = np.histogram(binv, weights=wv, bins=bins)[0] \
                     / np.histogram(binv, bins=bins)[0]
                X2 = np.histogram(binv, weights=wv, bins=bins)[0] \
                     / np.histogram(binv, bins=bins)[0]
                Neff = np.histogram(binv, bins=bins)[0]
                dY = np.sqrt((X2-X1**2)/Neff)
                Y = np.copy(X1)
                YK1.append(Y)
            #Y = np.array(Y) - Y0
            #dY = np.sqrt(dY**2+dY0**2)
            #ax.(x, Y, yerr=dY, color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.fill_between(x, YK1[0]-YK0[0], YK1[2]-YK0[2],
                            color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK1[1]-YK0[1], color=color)
            ax.set_xlim(.5,20)
            ax.plot([.5,20],[0,0],'k-',lw=.3)
            if ax.is_first_col():
                ax.legend(loc=(.1,.6))
                ax.set_ylabel(r"$\Delta\langle p_T^2\rangle$ [GeV${}^2$]" if not scaled \
                         else r"$\Delta\langle p_T^2\rangle / K$")
            ax.set_title(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]))
            ax.set_ylim(-.02,.08)
            ax.set_xlabel(r"$Q^2$ [GeV${}^2$]")
            ax.semilogx()
    set_prelim(ax)
    plt.subplots_adjust(wspace=0, left=.12, right=.97, bottom=.2, top=.88)

def RA_z(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth,.35*textwidth), sharey=True,sharex=True)
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
    for K, model, color, label in zip(ModelK, Models, ModelColors, labels):
        YK0 = []
        for K in ModelK:
            # e+d
            pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
            cut = (pid==iid) & (nu>6)
            Y0 = np.histogram(z[cut],bins=bins)[0]/db
            YK0.append(Y0)
        # e+A
        for ax,N,_,Z,A in zip(axes,*NCZA):
            YK1 = []
            for K in ModelK:
                pid, z, pT, nu, Q2 = [f0['{}/{}/{}/{}'.format(model,K,N,it)][()] for it in entry]
                cut = ( pid==iid ) & (nu>6)
                Y = np.histogram(z[cut],bins=bins)[0]/db
                YK1.append(Y)

            ax.fill_between(x, YK1[0]/YK0[0], YK1[2]/YK0[2], color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK1[1]/YK0[1], color=color)
            ax.set_xlim(.1,1.05)
            ax.plot([.0,1.5],[1,1],'k-',lw=.3)
            if ax.is_first_col():
                ax.legend(loc='best')
                ax.set_ylabel(r"$R_{{A}}$")
            ax.set_title(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]))
            ax.set_ylim(.2,1.3)
            ax.set_xlabel(r"$z_h$")
    set_prelim(ax)
    plt.subplots_adjust(wspace=0, left=.08, right=.99, bottom=.2, top=.88)


def RA_pT(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth,.35*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_pT2/e{}-{}.dat".format(N,sid[iid]),
                 delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',
                 label=r'HERMES')
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array(list(xl)+[xh[-1]])
    db = bins[1:]-bins[:-1]
    for K, model, color, label in zip(ModelK, Models, ModelColors, labels):
        YK0 = []
        for K in ModelK:
            # e+d
            pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
            cut = (pid==iid) & (nu>6) & (z>.2)
            Y0 = np.histogram((pT**2)[cut],bins=bins)[0]/db
            YK0.append(Y0)
        # e+A
        for ax,N,_,Z,A in zip(axes,*NCZA):
            YK1 = []
            for K in ModelK:
                pid, z, pT, nu, Q2 = [f0['{}/{}/{}/{}'.format(model,K,N,it)][()] for it in entry]
                cut = ( pid==iid ) & (nu>6) & (z>.2)
                Y = np.histogram((pT**2)[cut],bins=bins)[0]/db
                YK1.append(Y)

            ax.fill_between(x, YK1[0]/YK0[0], YK1[2]/YK0[2], color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK1[1]/YK0[1], color=color)
            ax.set_xlim(0,2.2)
            ax.plot([.0,2.2],[1,1],'k-',lw=.3)
            if ax.is_first_col():
                ax.legend(loc='best')
                ax.set_ylabel(r"$R_{{A}}$")
            ax.set_title(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]))
            ax.set_xlim(0,2.2)
            ax.set_ylim(.5,2.0)
            ax.set_xlabel(r"$p_T^2$ [GeV${}^2$]")
    set_prelim(ax)
    plt.subplots_adjust(wspace=0, left=.08, right=.99, bottom=.2, top=.88)

def RA_nu(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth,.35*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_nu/e{}-{}.dat".format(N,sid[iid]),
                 delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',
                 label=r'HERMES')
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array(list(xl)+[xh[-1]])
    db = bins[1:]-bins[:-1]
    for K, model, color, label in zip(ModelK, Models, ModelColors, labels):
        YK0 = []
        for K in ModelK:
            # e+d
            pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
            cut = (pid==iid) & (z>.2)
            Y0 = np.histogram(nu[cut],bins=bins)[0]/db
            YK0.append(Y0)
        # e+A
        for ax,N,_,Z,A in zip(axes,*NCZA):
            YK1 = []
            for K in ModelK:
                pid, z, pT, nu, Q2 = [f0['{}/{}/{}/{}'.format(model,K,N,it)][()] for it in entry]
                cut = ( pid==iid )  & (z>.2)
                Y = np.histogram(nu[cut],bins=bins)[0]/db
                YK1.append(Y)

            ax.fill_between(x, YK1[0]/YK0[0], YK1[2]/YK0[2], color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK1[1]/YK0[1], color=color)
            ax.set_xlim(0,25)
            ax.plot([.0,25],[1,1],'k-',lw=.3)
            if ax.is_first_col():
                ax.legend(loc='best')
                ax.set_ylabel(r"$R_{{A}}$")
            ax.set_title(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]))
            ax.set_xlim(0,25)
            ax.set_ylim(.2,1.4)
            ax.set_xlabel(r"$\nu$ [GeV]")
    set_prelim(ax)
    plt.subplots_adjust(wspace=0, left=.08, right=.99, bottom=.2, top=.88)



def RA_Q2(iid):
    fig, axes = plt.subplots(1,4,figsize=(textwidth, .35*textwidth), sharey=True,sharex=True)
    for ax,N,_,Z,A in zip(axes,*NCZA):
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_Q2/e{}-{}.dat".format(N,sid[iid]),
                 delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k',
                 label=r'HERMES')
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')
    bins = np.array(list(xl)+[xh[-1]])
    db = bins[1:]-bins[:-1]
    for K, model, color, label in zip(ModelK, Models, ModelColors, labels):
        YK0 = []
        for K in ModelK:
            # e+d
            pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
            cut = (pid==iid) & (z>.2)
            Y0 = np.histogram(Q2[cut],bins=bins)[0]/db
            YK0.append(Y0)
        # e+A
        for ax,N,_,Z,A in zip(axes,*NCZA):
            YK1 = []
            for K in ModelK:
                pid, z, pT, nu, Q2 = [f0['{}/{}/{}/{}'.format(model,K,N,it)][()] for it in entry]
                cut = ( pid==iid )  & (z>.2) & (np.logical_not(np.isnan(z)))
                Y = np.histogram(Q2[cut],bins=bins)[0]/db
                YK1.append(Y)

            ax.fill_between(x, YK1[0]/YK0[0], YK1[2]/YK0[2], color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK1[1]/YK0[1], color=color)
            ax.set_xlim(0,16)
            ax.plot([.0,16],[1,1],'k-',lw=.3)
            if ax.is_first_col():
                ax.legend(loc='best')
                ax.set_ylabel(r"$R_{{A}}$")
            ax.set_title(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]))
            ax.set_xlim(0,16)
            ax.set_ylim(.2,1.4)
            ax.set_xlabel(r"$Q^2$ [GeV [${}^2$]]")
    set_prelim(ax)
    plt.subplots_adjust(wspace=0, left=.08, right=.99, bottom=.2, top=.88)

@plot
def Flavor_RA():
    fig, axes = plt.subplots(3,2,figsize=(textwidth*.6,.7*textwidth), sharex=True)
    axes = axes.flatten()
    ylims = [(.3,1.2),(.3,1.2),(.3,1.3),(.3,1.3),(0,1.4),(0,1.4)]
    for ax, iid, ylim in zip(axes, [211,111,321,-321,2212,-2212],
                             ylims):
        ax.set_ylim(*ylim)
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
        print(db)
        N = 'Xe'
        for model, color, label in zip(Models, ModelColors, labels):
            YK1 = []
            for K in ModelK:
                pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
                cut = (pid==iid) & (nu>6)  & (np.logical_not(np.isnan(z)))
                Y0 = np.histogram(z[cut],bins=bins)[0]/db
                pid, z, pT, nu, Q2 = [f0['{}/{}/{}/{}'.format(model,K,N,it)][()]
                            for it in entry]
                cut = ( pid==iid ) & (nu>6)  & (np.logical_not(np.isnan(z)))
                Y = np.histogram(z[cut],bins=bins)[0]/db
                Y = np.array(Y)/Y0
                YK1.append(Y)
            ax.fill_between(x, YK1[0], YK1[2], color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK1[1], color=color)
            if ax.is_last_col():
                ax.set_yticks([])
            ax.set_xlim(.05,1.15)
            ax.set_xticks([0.2,0.4,.6,.8,1.])
            ax.plot([.0,1.5],[1,1],'k-',lw=.3)
            if ax.is_first_col():
                ax.set_ylabel(r"$R_{{A}}$")
            ax.annotate(r"$e+{{\rm {:s}}}\rightarrow {:s}+\cdots$".format(N,ssid[iid]),xy=(.25,.86),xycoords="axes fraction")

            if ax.is_last_row():
                ax.set_xlabel(r"$z_h$")
    axes[4].legend(loc='lower left')
    set_prelim(ax, xy=(.1,.05))

    #set_tight(fig)
    plt.subplots_adjust(wspace=0, hspace=0, top=.99, right=.98, left=.15,bottom=.1)


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

        for model, color, label in zip(Models, ModelColors, labels):
            YK = []
            for K in ModelK:
                # e+d
                pid, z, pT, nu, Q2 = [f0['{}/{}/d/{}'.format(model,K,it)][()] for it in entry]
                cut = ( pid==iid ) & (nu>6) & (zl<z) & (z<zh)
                Y0 = np.histogram(pT[cut]**2,bins=bins)[0]/db
                # e+A
                #N = 'Xe'
                pid, z, pT, nu, Q2 = [f0['{}/{}/Xe/{}'.format(model,K,it)][()]
                              for it in entry]
                cut = ( pid==iid ) & (nu>6) & (zl<z) & (z<zh)
                Y = np.histogram(pT[cut]**2,bins=bins)[0]/db
                Y = np.array(Y)/Y0
                YK.append(Y)

            ax.fill_between(x, YK[0], YK[2], color=color, alpha=.7, label=label if ax.is_first_col() else '')
            ax.plot(x, YK[1], color=color)
            ax.set_xlim(0,1.9)
            ax.plot([.0,2],[1,1],'k-',lw=.3)
            if ax.is_first_col():

                ax.set_ylabel(r"$R_{{A}}$")
            ax.annotate(r"${}<z_h<{}$".format(zl,zh),xy=(.1,.86),xycoords="axes fraction")
            ax.set_ylim(0,2)
            if ax.is_last_row():
                ax.set_xlabel(r"$p_T^2$ [GeV${}^2$]")
    axes[0].legend(loc='lower right')
    plt.suptitle(r"$e+{{\rm {:s}}}\rightarrow \pi^{{\pm}}+\cdots$".format(N))

    set_prelim(ax)
    plt.subplots_adjust(wspace=0, left=.1, right=.99, bottom=.25, top=.9)


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
"""

@plot
def DpT2x_pip():
    DpT2_x(211)
@plot
def DpT2x_pim():
    DpT2_x(-211)
@plot
def DpT2x_Kp():
    DpT2_x(321)

@plot
def DpT2z_pip():
    DpT2_z(211)
@plot
def DpT2z_pim():
    DpT2_z(-211)
@plot
def DpT2z_Kp():
    DpT2_z(321)


@plot
def DpT2Q2_pip():
    DpT2_Q2(211)
@plot
def DpT2Q2_pim():
    DpT2_Q2(-211)
@plot
def DpT2Q2_Kp():
    DpT2_Q2(321)


@plot
def DpT2nu_pip():
    DpT2_nu(211)
@plot
def DpT2nu_pim():
    DpT2_nu(-211)
@plot
def DpT2nu_Kp():
    DpT2_nu(321)
"""



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
