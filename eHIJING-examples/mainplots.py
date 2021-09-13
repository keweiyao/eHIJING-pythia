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
textwidth = 200/resolution
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
folder = "Run/Production/Generalized/K0d5/"

print("loading")
#samples = {it: np.loadtxt(folder+"/{}-{}-cutRA.dat".format(Z,A)).T
#       for it, c, Z, A in zip(*NCZA)}
#samples['d'] = np.loadtxt(folder+"/1-2-cutRA.dat").T
print("done")

def AvgKinematics(xname):
    AllLabels = [r"$\langle \nu\rangle$", r"$\langle z\rangle$", r"$\langle Q^2\rangle$ [GeV${}^2$]",r"$\langle p_T^2\rangle$ [GeV${}^2$]"]
    if xname=='nu':
        idx = 0
        idy = (1,2,3)
    if xname=='z':
        idx = 1
        idy = (0,2,3)
    if xname=='Q2':
        idx = 2
        idy = (0,1,3)
    if xname=='pt2':
        idx = 3
        idy = (0,1,2)
    ylabels = [AllLabels[i] for i in idy]
    xlabel = AllLabels[idx]
    fig, axes = plt.subplots(1,3,figsize=(textwidth,.35*textwidth), sharex=True)
    xl, xh, x, y1, y2, y3 = data = np.loadtxt("Exp/HERMES/SIDIS/AvgKinmatics/{}-binned.dat".format(xname)).T
    if xname in ['Q2','pt2']:
        axes[0].semilogx()
    for ax, y, ylabel in zip(axes, [y1,y2,y3], ylabels):
        ax.errorbar(x, y ,xerr=[x-xl,xh-x],fmt='k.')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(np.max([xl.min(),.01]), xh.max())
        ax.set_ylim(ymin=0, ymax=y.max()*1.5)
    pid, z, pT, nu, Q2 = np.loadtxt(folder+"/36-84-cutRA.dat").T
    zmin = .1 if xname=='z' else .2
    numin = 4. if xname=='nu' else 6.
    cut = ( (pid==211) | (pid==-211) | (pid==111)) & (z>zmin) & (nu>numin)

    z = z[cut]
    nu = nu[cut]
    Q2 = Q2[cut]
    pt2 = pT[cut]**2
    data = [nu, z, Q2, pt2]
    for ax, iy in zip(axes, idy):
        X = []
        Y = []
        for il, ih in zip(xl, xh):
            cut = (il<data[idx])&(data[idx]<ih)
            X.append(data[idx][cut].mean())
            Y.append(data[iy][cut].mean())
        ax.plot(X, Y,'ro')

    set_tight(fig,pad=0.4)

@plot
def dihadron_fig():
    fig, axes = plt.subplots(1,3,figsize=(textwidth,textwidth*.45), sharex=True,sharey=True)
    # exp:
    for Z, A, name, ax in zip([7,36,54],[14,84,131],['N','Kr','Xe'], axes):
        _, _, X, Y, Ystat = np.loadtxt("Exp/HERMES/Dihadron/R12_{}.dat".format(name)).T
        ax.errorbar(X,Y,yerr=Ystat,fmt='k.', label="HERMES")
    # model:
    b = np.linspace(0, .5, 7)
    x = (b[:-1]+b[1:])/2.
    dx = b[1:]-b[:-1]
    Ks = ['0d5','1','2']
    for model,color,label in zip(['Generalized','Collinear'],[cr,cb],['Gen.','H-T']):
        YK0 = []
        for K in Ks:
            z1, z2 = np.loadtxt("Run/Production/Dihadron/{}/{}/1-2-cutRA.dat".format(model,K)).T
            dN0, _ = np.histogram(z2[z1>.5], bins=b)
            N0 = np.sum(z1>.5)
            YK0.append(dN0/N0)
        for Z, A, name, ax in zip([7,36,54],[14,84,131],['N','Kr','Xe'], axes):
            YK1 = []
            for K in Ks:
                z1, z2 = np.loadtxt("Run/Production/Dihadron/{}/{}/{}-{}-cutRA.dat".format(model,K,Z,A)).T
                dN0, _ = np.histogram(z2[z1>.5], bins=b)
                N0 = np.sum(z1>.5)
                YK1.append(dN0/N0)
            ax.fill_between(x, YK1[0]/YK0[0], YK1[2]/YK0[2], color=color, label=label, alpha=.7)
            ax.plot(x, YK1[1]/YK0[1], color=color)
            ax.set_xlabel(r"$z_2$")
            if ax.is_first_col():
                ax.set_ylabel(r"$R_{2h}(z_2)$")
            ax.plot([0,.5], [1,1], 'k-', lw=.5)
    ax.set_xlim(0,.5)
    ax.set_ylim(0.7,1.4)
    axes[0].legend()
    ax.annotate(r"$z_1>0.5$"+"\n"+"Excluding $+-$", xy=(.25,.7), xycoords="axes fraction")
    plt.subplots_adjust(wspace=0, left=.12, right=.98, bottom=.2, top=.86)
#@plot
def AvgKinematics_nu():
    AvgKinematics('nu')
#@plot
def AvgKinematics_z():
    AvgKinematics('z')
#@plot
def AvgKinematics_Q2():
    AvgKinematics('Q2')
#@plot
def AvgKinematics_pt2():
    AvgKinematics('pt2')

@plot
def sudakov_fig():
    fig, ax = plt.subplots(1,1,figsize=(textwidth*.5,textwidth*.5), sharex=True,sharey=True)
    Neve = 1e4
    for Z, A, name in zip([1,2,4,7,10,20, 36,54,82],[2,4,8,14,20,40,84,131,208],['d','He','Be','N','Ne','Ca', 'Kr','Xe','Pb']):
        nc,nh,ns = np.loadtxt("Run/Sudakov/{}-{}-stat.dat".format(Z,A)).T
        nrad = nh+ns
        P = ((nc==0)).sum()/Neve
        ax.plot(A**.3333, P, 'k.', label='No collision (quark)' if A==2 else '')
        P = ((nc>0) & (nrad==0) ).sum()/Neve
        ax.plot(A**.3333, P, '.', color=cr, label='Collision w/o radiation' if A==2 else '')
        P = ((nc>0) & (nrad>0)).sum()/Neve
        ax.plot(A**.3333, P, '.', color=cb, label='Collision w/ radiation' if A==2 else '')
    A = np.linspace(1,300,100)
    ax.plot(A**.3333, np.exp(-A**.3333*.16),'k--', label=r"$\exp(-0.16 A^{1/3})$")
    ax.legend(loc='best', framealpha=.5)
    ax.set_xlabel(r"$A^{1/3}$")
    ax.set_ylabel("Prob.")
    ax.set_ylim(0,1.3)
    ax.set_xlim(1,7)
    ax.set_xticks([1,3,5,7])
    set_tight(fig)

# page 10 why differ at low multiplicuty
@plot
def ed_dNdzdpTpi():
    fig, axes = plt.subplots(1,4,figsize=(1.3*textwidth,textwidth*.4), sharex=True,sharey=True)

    b = np.linspace(0,1,10)
    x = (b[:-1]+b[1:])/2.
    dx = b[1:]-b[:-1]
    Ne = 1e6
    axes[0].semilogy()
    pid, z, pT, nu, Q2 = np.loadtxt("Run/Production/ed/1-2-StopM0d0.dat").T
    for ax, zbin in zip(axes,[[.2,.3],[.3,.4],[.4,.6],[.6,.8]]):
        ax.annotate(r"${}<z_h<{}$".format(*zbin), xy=(.1, .9), xycoords="axes fraction")
        for line, color, iid in zip(['-','--'],[cr, cb],[211,-211]):
            cut = (pid==iid) & (zbin[0]<z) & (z<zbin[1])

            Y = np.histogram((pT)[cut],bins=b)[0]/dx/Ne/(zbin[1]-zbin[0])
            ax.plot(x, Y, line, color=color, lw=1, alpha=.8)
            ax.set_xlim(0,1)
            if ax.is_last_row():
                ax.set_xlabel(r"$p_T$ [GeV${}$]")
            if ax.is_first_col():
                ax.set_ylabel(r"$dN/dp_T/dz_h$")
            ax.set_ylim(ymin=0)


    for ax, zbin in zip(axes,[[.2,.3],[.3,.4],[.4,.6],[.6,.8]]):
        for color,iid,n in zip([cr,cb],[211,-211],['pi+','pi-']):
            x,y,ystat,_,ysys,_ = np.loadtxt("ExpData/pT/ed-{}-z-{}-{}.dat".format(n,*zbin)).T
            ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
                       321:'K^+',  -321:'K^-',
                       2212:'p',  -2212:"\\bar{p}"}[iid]
            ax.errorbar(x,y,yerr=ystat, fmt='.', color=darken(color), label=r'HERMES ${}$'.format(ssid)
            )
            xb = [0.]+list((x[1:]+x[:-1])/2.) + [1.]
            for il,ih,yl,yh in zip(xb[:-1],xb[1:],y-ysys,y+ysys):
                ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')
    axes[0].legend(loc=(.03, .1))

    ax.set_ylim(1e-2,1e1)
    ax.set_xlim(0.0,1.1)
    ax.set_xticks([0.,.3,.6,.9])
    plt.subplots_adjust(wspace=0, left=.1,right=.99,bottom=.25, top=.96)


@plot
def ed_dNdzdpTK():
    fig, axes = plt.subplots(1,4,figsize=(1.3*textwidth,textwidth*.4), sharex=True,sharey=True)

    b = np.linspace(0,1,10)
    x = (b[:-1]+b[1:])/2.
    dx = b[1:]-b[:-1]
    Ne = 1e6
    axes[0].semilogy()
    pid, z, pT, nu, Q2 = np.loadtxt("Run/Production/ed/1-2-StopM0d0.dat").T
    for ax, zbin in zip(axes,[[.2,.3],[.3,.4],[.4,.6],[.6,.8]]):
        ax.annotate(r"${}<z_h<{}$".format(*zbin), xy=(.1, .9), xycoords="axes fraction")
        for line, color, iid in zip(['-','--'],[cr, cb],[321,-321]):
            cut = (pid==iid) & (zbin[0]<z) & (z<zbin[1])

            Y = np.histogram((pT)[cut],bins=b)[0]/dx/Ne/(zbin[1]-zbin[0])
            ax.plot(x, Y, line, color=color, lw=1, alpha=.8)
            ax.set_xlim(0,1)
            if ax.is_last_row():
                ax.set_xlabel(r"$p_T$ [GeV${}$]")
            if ax.is_first_col():
                ax.set_ylabel(r"$dN/dp_T/dz_h$")
            ax.set_ylim(ymin=0)


    for ax, zbin in zip(axes,[[.2,.3],[.3,.4],[.4,.6],[.6,.8]]):
        for color,iid,n in zip([cr,cb],[321,-321],['K+','K-']):
            x,y,ystat,_,ysys,_ = np.loadtxt("ExpData/pT/ed-{}-z-{}-{}.dat".format(n,*zbin)).T
            ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
                       321:'K^+',  -321:'K^-',
                       2212:'p',  -2212:"\\bar{p}"}[iid]
            ax.errorbar(x,y,yerr=ystat, fmt='.', color=darken(color), label=r'HERMES ${}$'.format(ssid)
            )
            xb = [0.]+list((x[1:]+x[:-1])/2.) + [1.]
            for il,ih,yl,yh in zip(xb[:-1],xb[1:],y-ysys,y+ysys):
                ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')
    axes[0].legend(loc=(.03, .1))

    ax.set_ylim(1e-3,1e1)
    ax.set_xlim(0.0,1.1)
    ax.set_xticks([0.,.3,.6,.9])
    plt.subplots_adjust(wspace=0, left=.1,right=.99,bottom=.25, top=.96)




@plot
def ed_dNdz():
    fig = plt.figure(figsize=(1.3*textwidth,.5*textwidth))
    spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)
    axes = np.array([
    [fig.add_subplot(spec[:2, i]) for i in range(4)],
    [fig.add_subplot(spec[2, i]) for i in range(4)]
    ])
    name = {211:"$\pi^+$",
           -211:"$\pi^-$",
            321:"$K^+$",
           -321:"$K^-$"}


    for ax,iid,n in zip(axes[0],[211,-211,321,-321],['pi+','pi-','K+','K-']):
        x,y,ystat,_,ysys,_ = np.loadtxt("ExpData/z/ed-{}.dat".format(n)).T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color='k'
           , label=r'HERMES'
           )
        xb = [0.1]+list((x[1:]+x[:-1])/2.) + [.9]
        for il,ih,yl,yh in zip(xb[:-1],xb[1:],y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor='k',facecolor='none')

    Ne = 1e6
    for stopM, ss, color, label in zip([1.0, 0.0], ('1d0','0d0'), [cr, cb], [r'$M_{\rm stop}=1.0$ GeV', r'$M_{\rm stop}=0$']):
        pid, z, pT, nu, Q2 = np.loadtxt("Run/Production/ed/1-2-StopM{}.dat".format(ss)).T
        for ax,axr,iid in zip(axes[0],axes[1],[211,-211,321,-321]):

            sid = {211:'pi+', -211:'pi-', 111:'pi0',
                       321:'K+',  -321:'K-',
                       2212:'p',  -2212:'pbar'}[iid]
            ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
                       321:'K^+',  -321:'K^-',
                       2212:'p',  -2212:"\\bar{p}"}[iid]
            x,y,ystat,_,ysys,_ = np.loadtxt("ExpData/z/ed-{}.dat".format(sid)).T
            xb = [0.1]+list((x[1:]+x[:-1])/2.) + [.9]
            xb = np.array(xb)
            x = (xb[:-1]+xb[1:])/2.
            dx  = xb[1:]-xb[:-1]
            cut = (pid==iid)
            Y = np.histogram(z[cut],bins=xb)[0]/dx/Ne
            ax.plot(x, Y, color=color, lw=1,alpha=.8,label=label)
            ax.set_xlim(0.05,1.05)
            axr.set_xlim(0.05,1.05)
            axr.set_xlabel(r"$z_h$")
            if ax.is_first_col():
                ax.set_ylabel(r"$dN/dz_h$")
            if ax.is_first_col():
                axr.set_ylabel("Exp./MC.")
            ax.semilogy()
            ax.set_ylim(5e-5,1e1)

            x,y,ystat,_,ysys,_ = np.loadtxt("ExpData/z/ed-{}.dat".format(sid)).T
            y/=Y
            ystat/=Y
            ysys /= Y
            axr.errorbar(x,y,yerr=ystat,fmt='.', color=color,label=label)
            xb = [0.]+list((x[1:]+x[:-1])/2.) + [1.]
            for il,ih,yl,yh in zip(xb[:-1],xb[1:],y-ysys,y+ysys):
                axr.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=color,facecolor='none')
            axr.plot([0,1.05],[1,1],'k--', lw=.6)
            axr.set_ylim(0,2.3)
            ax.annotate(r"${}$".format(ssid), xy=(.6,.8), xycoords="axes fraction")
            if ax.is_first_col():
                pass
                #ax.legend()
            else:
                ax.set_yticklabels([])
            ax.set_xticklabels([])
            if axr.is_first_col():
                pass
                #axr.legend()
            else:
                axr.set_yticklabels([])
    axes[0,0].legend()
    #for i in [0,1]:
    #    axes[i].set_ylim(1e-2,5)
    #for i in [2,3]:
    #    axes[i].set_ylim(1e-2,1)
    plt.subplots_adjust(wspace=.0, hspace=.0, left=.1, right=.97, bottom=.2, top=.97)

def RA_z(iid):
    sid = {211:'pi+', -211:'pi-', 111:'pi0',
           321:'K+',  -321:'K-',
           2212:'p',  -2212:'pbar'}[iid]
    ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
           321:'K^+',  -321:'K^-',
           2212:'p',  -2212:"\\bar{p}"}[iid]
    fig, ax = plt.subplots(1,1,figsize=(.55*textwidth,.5*textwidth), sharey=True,sharex=True)
    for N,color,Z,A in zip(*NCZA):
        x,xl,xh,y,ystat,_,ysys,_ = \
       np.loadtxt("Exp/HERMES/SIDIS/RA_z/e{}-{}.dat".format(N,sid),
                 skiprows=12,
                 delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color=darken(color),
                 label=r'HERMES ${{}}^{{{:d}}}${:s}'.format(A,N))
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')

    bins = np.array(list(xl)+[xh[-1]])
    db = bins[1:]-bins[:-1]
    zm = x#(bins[1:]+bins[:-1])/2.
    pid, z, pT, nu, Q2 = samples['d']#np.loadtxt(folder+"/1-2-cutRA.dat").T
    cut = (pid==iid) & (nu>6)
    Y0 = np.histogram(z[cut],bins=bins)[0]/db
    for N,color,Z,A in zip(*NCZA):
        pid, z, pT, nu, Q2 = samples[N]
        #np.loadtxt(folder+"/{}-{}-cutRA.dat".format(Z,A)).T
        cut = ( pid==iid ) & (nu>6)
        Y = np.histogram(z[cut],bins=bins)[0]/db
        ax.plot(zm, Y/Y0, color=color, alpha=.7)
    ax.set_xlim(.0,1.05)
    ax.plot([.0,1.5],[1,1],'k-',lw=.3)


    ax.legend(loc='lower left')
    ax.set_ylim(0,1.5)
    ax.set_xlabel(r"$z_h=E_h/\nu$ in target frame")
    ax.set_ylabel(r"$R_{{A}}$")
    ax.annotate(r"${:s}$".format(ssid),xy=(.07,.9),xycoords="axes fraction")
    set_tight(fig, pad=.2)


def RA_pT(iid):
    sid = {211:'pi+', -211:'pi-', 111:'pi0',
           321:'K+',  -321:'K-',
           2212:'p',  -2212:'pbar'}[iid]
    ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
           321:'K^+',  -321:'K^-',
           2212:'p',  -2212:"\\bar{p}"}[iid]
    fig, ax = plt.subplots(1,1,figsize=(.55*textwidth,.5*textwidth), sharey=False,sharex=True)

    for N,color,Z,A in zip(*NCZA):
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_pT2/e{}-{}.dat".format(N,sid),
            #skiprows=12,
            delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color=darken(color),
                 label=r'HERMES ${{}}^{{{:d}}}${:s}'.format(A,N))
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')

    b = np.array(list(xl)+[xh[-1]])
    x = x
    dx = b[1:]-b[:-1]
    zmin = 0.2
    pid, z, pT, nu, Q2 = samples['d']#np.loadtxt(folder+"/1-2-cutRA.dat").T
    cut = (pid == iid) & (z>zmin) & (nu>6)
    Y0 = np.histogram((pT**2)[cut],bins=b)[0]/dx
    for N,color,Z,A in zip(*NCZA):
        pid, z, pT, nu, Q2 = samples[N]#np.loadtxt(folder+"/{}-{}-cutRA.dat".format(Z,A)).T
        cut = (pid == iid) & (z>zmin)& (nu>6)
        Y = np.histogram((pT**2)[cut],bins=b)[0]/dx
        ax.plot(x, Y/Y0, color=color, lw=1,alpha=.8)
    ax.plot([1e-2,10],[1,1],'k-', lw=.5)
    ax.set_xlim(0,2)
    ax.set_ylim(.4,2.1)
    ax.set_xlabel(r"$p_T^2$ [GeV${}^2$]")
    ax.set_ylabel(r"$R_{A}$")
    ax.legend(loc="upper left")
    ax.annotate(r"${:s}$".format(ssid),xy=(.6,.9),xycoords="axes fraction")
    set_tight(fig,pad=0.2)


def RA_nu(iid):
    sid = {211:'pi+', -211:'pi-', 111:'pi0',
           321:'K+',  -321:'K-',
           2212:'p',  -2212:'pbar'}[iid]
    ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
           321:'K^+',  -321:'K^-',
           2212:'p',  -2212:"\\bar{p}"}[iid]
    fig, ax = plt.subplots(1,1,figsize=(.55*textwidth,.5*textwidth), sharey=False,sharex=True)

    for N,color,Z,A in zip(*NCZA):
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_nu/e{}-{}.dat".format(N,sid),
            #skiprows=12,
            delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color=darken(color),
                 label=r'HERMES ${{}}^{{{:d}}}${:s}'.format(A,N))
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')

    b = np.array(list(xl)+[xh[-1]])
    x = x
    dx = b[1:]-b[:-1]
    zmin = 0.2
    pid, z, pT, nu, Q2 = samples['d']#np.loadtxt(folder+"/1-2-cutRA.dat").T
    cut = (pid == iid) & (z>zmin)
    Y0 = np.histogram(nu[cut],bins=b)[0]/dx
    for N,color,Z, A in zip(*NCZA):
        pid, z, pT, nu, Q2 = samples[N]#np.loadtxt(folder+"/{}-{}-cutRA.dat".format(Z,A)).T
        cut = (pid == iid) & (z>zmin)
        Y = np.histogram(nu[cut],bins=b)[0]/dx
        ax.plot(x, Y/Y0, color=color, lw=1,alpha=.8)
    ax.plot([0,25],[1,1],'k-', lw=.5)
    ax.set_xlim(0,25)
    ax.set_ylim(0,1.5)
    ax.set_xlabel(r"$\nu$ [GeV]")
    ax.set_ylabel(r"$R_{A}$")
    ax.legend(loc="lower right")
    ax.annotate(r"${:s}$".format(ssid),xy=(.6,.9),xycoords="axes fraction")
    set_tight(fig,pad=0.2)

def RA_Q2(iid):
    sid = {211:'pi+', -211:'pi-', 111:'pi0',
           321:'K+',  -321:'K-',
           2212:'p',  -2212:'pbar'}[iid]
    ssid = {211:'\pi^+', -211:'\pi^-', 111:'\pi^0',
           321:'K^+',  -321:'K^-',
           2212:'p',  -2212:"\\bar{p}"}[iid]
    fig, ax = plt.subplots(1,1,figsize=(.55*textwidth,.5*textwidth), sharey=False,sharex=True)

    for N,color,Z,A in zip(*NCZA):
        x,xl,xh,y,ystat,_,ysys,_ = \
            np.loadtxt("Exp/HERMES/SIDIS/RA_Q2/e{}-{}.dat".format(N,sid),
            #skiprows=12,
            delimiter=',').T
        ax.errorbar(x,y,yerr=ystat,fmt='.', color=darken(color),
                 label=r'HERMES ${{}}^{{{:d}}}${:s}'.format(A,N))
        for il,ih,yl,yh in zip(xl,xh,y-ysys,y+ysys):
            ax.fill_between([il,ih],[yl,yl],[yh,yh],
                            edgecolor=darken(color),facecolor='none')

    b = np.array(list(xl)+[xh[-1]])
    x = x#(b[:-1]+b[1:])/2.
    dx = b[1:]-b[:-1]
    zmin = 0.2
    pid, z, pT, nu, Q2 =samples['d']# np.loadtxt(folder+"/1-2-cutRA.dat").T
    cut = (pid == iid) & (z>zmin) & (nu>6)
    Y0 = np.histogram(Q2[cut],bins=b)[0]/dx
    for N,color,Z,A in zip(*NCZA):
        pid, z, pT, nu, Q2 = samples[N]#np.loadtxt(folder+"/{}-{}-cutRA.dat".format(Z,A)).T
        cut = (pid == iid) & (z>zmin) & (nu>6)
        Y = np.histogram(Q2[cut],bins=b)[0]/dx

        ax.plot(x, Y/Y0, color=color, lw=1,alpha=.8)
    ax.plot([0.5,25],[1,1],'k-', lw=.5)
    ax.set_xlim(.5,25)
    ax.set_ylim(0,1.5)
    ax.semilogx()
    ax.set_xlabel(r"$Q^2$ [GeV${}^2$]")
    ax.set_ylabel(r"$R_{A}$")
    ax.legend(loc="lower left")
    ax.annotate(r"${:s}$".format(ssid),xy=(.6,.9),xycoords="axes fraction")
    set_tight(fig,pad=0.2)
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
