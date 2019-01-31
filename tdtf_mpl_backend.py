import matplotlib.pyplot as plt
import matplotlib as mpl
# from general_plotting_prettyness import *
from collections import namedtuple
from math import atan2, asin, log10, log, pi, sqrt
import cmath
import math
import mpl_toolkits.mplot3d.axes3d as p3

png_folder = r"/home/gray/wk/metaid/sea_sysid_and_synthesis/rawplots/"
pdf_folder = r"/home/gray/wk/metaid/sea_sysid_and_synthesis/rawplots/"

BodeExp = namedtuple("BodeExp",["omega","mag","phase"])
ORI = namedtuple("ORI",["omega","real","imag"])

dB = lambda m: 20.0 * log10(m) if m > 0 else -200
deg = lambda o: ang(o*180.0/pi)

ang = lambda o: o if o < 0 else o - 360
safe_asin = lambda e: asin(e) if e < 1 and e > -1 else 360 * np.sign(e)
sat_ang = lambda a: 0 if a > 0 else -360 if a < -360 else a
nsig = 2.0
def gen_amps(data):
    return list(set([data[k]["experiment.u[0].amp"][0] for k in list(data.keys()) if len(data[k]["experiment.u[0].amp"])>0]))

from mpl_toolkits.mplot3d import proj3d
import numpy as np
 
def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    # -0.0001 added for numerical stability as suggested in:
    # http://stackoverflow.com/questions/23840756
    return np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,a,b],
                        [0,0,-0.0001,zback]])
 
# Later in your plotting code ...
proj3d.persp_transformation = orthogonal_proj

def getRandomBoarderlineContraction(n):
    # this samples over all diagonalizable matrices with unit length singular
    # values
    A = np.array([[np.random.normal() + complex(0, 1.) * np.random.normal()
                   for j in range(0, n)] for i in range(0, n)])
    return cast_to_boarderline_contraction(A)


def cast_to_boarderline_contraction(A):
    u, s, v = np.linalg.svd(A)
    sprime = [1.0 if abs(si)<1e-6 else (si / abs(si)) for si in s]
    assert np.linalg.norm(A - u.dot(np.diagflat(s)).dot(v)) < 1e-12
    unity_A2 = u.dot(np.diagflat(sprime)).dot(v)
    return unity_A2

class Log3DNyquistPlot(object):
    ''' A nice-looking log-amplitude nyquist plot class, similar to BodePlot'''
    def __init__(self,min_amp):
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

        # exit()
        self.min_amp = min_amp
        self.max_amp = 0.0
        self.min_omega = float('inf')
        self.max_omega = 0.0
        self.ax.set_xlabel(r"Real Part")
        self.ax.set_ylabel(r"Imaginary Part")
        self.ax.set_zlabel(r"Log of Frequency")
        self.ax.yaxis.set_visible(0)
        self.plot_name="3D Log-Nyquist Plot"
        # plt.locator_params(nbins=10)

    def title(self,title):
        self.ax.set_title(title)


    def save(self, name):
        self.fig.savefig(pdf_folder+name+".pdf", 
            facecolor='w', edgecolor='w',
            pad_inches=0.01)
        self.fig.savefig(png_folder+name+".png", 
            facecolor='w', edgecolor='w', dpi=400,
            pad_inches=0.01)

    def add_data(self, data, yind,  alpha=1.0):
        experiment_keys=list(data.keys())
        amps = gen_amps(data)
        amp_indexed_exps={}
        for e in experiment_keys:
            if len(data[e]['experiment.u[0].amp'])==0:
                continue
            amp = data[e]['experiment.u[0].amp'][0]
            if amp not in amp_indexed_exps:
                amp_indexed_exps[amp]=[] # zip(omega, real, imag) format
            omegas = data[e]['experiment.omega']
            reals = data[e]['y[%d].real'%yind]/amp
            imags = data[e]['y[%d].imag'%yind]/amp

            amp_indexed_exps[amp].extend(list(zip(omegas, reals, imags)))

        for amp in sorted(amp_indexed_exps.keys()):
            col=color_lambda(amp)
            self.add_ori_scatter(amp_indexed_exps[amp], color_lambda(amp), alpha=alpha)
        self.title("y[%d] %s, %d dB"%(yind, self.plot_name, dB(self.min_amp)))
        self.setup_tics()
        # self.fig.tight_layout()

    def dBscale(self, r,i):
        dat = sqrt(r**2+i**2)
        dB_value = dB(dat)-dB(self.min_amp) if dB(dat)-dB(self.min_amp)>0 else 0
        scale = dB_value/dat if dat>0 else 0.0
        return scale

    def setup_tics(self):
        # self.fig.tight_layout()
        M1 = (dB(self.max_amp)-dB(self.min_amp))*np.sqrt(2.0)
        M2 = 0.0
        O1 = np.log10(self.max_omega)
        O2 = np.log10(self.min_omega)
        Xb = (M1)*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten()
        Yb = (M1)*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten()
        Zb = 0.5*(O1-O2)*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(O1+O2)
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            # print xb, yb, zb
            self.ax.plot([xb], [yb], [zb], 'w')
        self.ax.set_aspect('equal')
        self.setup_negative_one_point()
        self.add_ori_line([(self.min_omega, -1.0, 0.0)],marker='o', ms=1, color='#AAAAAA')


        num_ticks = int((dB(1.0)-dB(self.min_amp))/20)
        angles = 12
        # exit()
        amps=[pow(10,-n) for n in range(num_ticks)]
        for a in amps:
            self.draw_circle((0,0),a,lw=.5, color="#AAAAAA")

        self.draw_circle((0,0),1,lw=.5, color="#666666")

        radius = -dB(self.min_amp)
        # print(radius)
        degree_sign= '\N{DEGREE SIGN}'
        # exit()
        for q in np.linspace(0,2*np.pi,angles+1)[:-1]:
            scale=1.2
            self.add_ori_line([(self.min_omega, 0.0, 0.0),(self.min_omega, np.cos(q), np.sin(q))], mirror=False, lw=0.5, color='#AAAAAA')
            self.add_ori_line([(self.min_omega, np.cos(q), np.sin(q)),(self.min_omega, scale*np.cos(q), scale*np.sin(q))], mirror=False, lw=1.0, color='#111111')
            self.ax.text(scale*radius*np.cos(q),scale*radius*np.sin(q), np.log10(self.min_omega), ("%d"+degree_sign)%int(180.0*q /np.pi+0.5))
        scale=10
        q = np.pi/4.
        self.add_ori_line([(self.min_omega, 0.0, 0.0),(self.min_omega, np.cos(q), np.sin(q))], mirror=False, lw=0.5, color='#AAAAAA')
        self.add_ori_line([(self.min_omega,np.cos(q), np.sin(q)),(self.min_omega, scale*np.cos(q), scale*np.sin(q))], mirror=False, lw=1.0, color='#AAAAAA')
        
        sweep_range = np.linspace(-0.05+q,0.05+q, 100)
        for a in amps+[10]:
            self.add_ori_line([(self.min_omega, a*np.cos(s), a*np.sin(s)) for s in sweep_range],mirror=False, lw=1.0, color='#111111')
            self.ax.text((dB(a)+radius)*np.cos(q),(dB(a)+radius)*np.sin(q), np.log10(self.min_omega), "%d dB"%(dB(a)) )

        self.add_ori_line([(self.min_omega, 0.0, 0.0), (self.max_omega,0,0)],mirror=False,lw=.5, color="#666666")

    def plot_SISO(self, tf, omega_lims = None, num_omegas=1000, num_markers=10, **kwargs):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)
        Z = [tf(w) for w in omegas]
        ori = [(w, z.real, z.imag) for w,z in zip(omegas, Z)]
        ori_2 = [(self.min_omega, r, i) for o,r,i in ori]
        self.add_ori_line(ori,**kwargs)
        if "alpha" in kwargs:
            kwargs["alpha"]*=0.5
        self.add_ori_line(ori_2,**kwargs)

    def plot_MIMO(self, tf, colors, omega_lims = None, num_omegas=1000, **kwargs):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)
        Z = [tf(w) for w in omegas]
        for i in range(tf(1).shape[0]):
            for j in range(tf(1).shape[1]):
                ori = [(w, z[i,j].real, z[i,j].imag) for w,z in zip(omegas, Z)]
                self.add_ori_line(ori, color=colors[i][j], **kwargs)

    def plot_PEJ_MIMO(self, P_tf,E_tf, J_tf, colors, omega_lims = None, num_omegas=100, num_thetas=12, **kwargs):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)
        thetas = np.linspace(0.0, np.pi*2, num_thetas+1)[:-1]
        P = [P_tf(w) for w in omegas]
        E = [E_tf(w) for w in omegas]
        J = [J_tf(w) for w in omegas]
        for i in range(P_tf(1).shape[0]):
            for j in range(P_tf(1).shape[1]):
                ori = [(w, z[i,j].real, z[i,j].imag) for w,z in zip(omegas, P)]
                self.add_ori_line(ori, color=colors[i][j], **kwargs)
                deltas =  [max(np.absolute(Eo[i,:]))*max(np.absolute(Jo[:,j])) for Eo, Jo in zip(E, J)]
                for th in thetas:
                    q = complex(np.cos(th), np.sin(th))
                    ori = [(w, z[i,j].real+q.real*delta, z[i,j].imag+q.imag*delta) for w,z, delta in zip(omegas, P, deltas)]
                    self.add_ori_line(ori, color=colors[i][j], **kwargs)

    def plot_LFT_MIMO(self, G_tf, N, colors, omega_lims = None, num_omegas=100, num_thetas=12, **kwargs):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)
        for i in range(G_tf(1).shape[0]-N):
            for j in range(G_tf(1).shape[1]-N):
                ori = [(w, G_tf(w)[i,j].real, G_tf(w)[i,j].imag) for w in omegas]
                self.add_ori_line(ori, color=colors[i][j], **kwargs)
        for delta in [getRandomBoarderlineContraction(N) for i in range(20)]:
            Gs = [G_tf(w) for w in omegas]
            Ts = [G[:-N,:-N]+G[:-N,-N:].dot(delta).dot(np.linalg.solve(np.eye(N)-delta.dot(G[-N:,-N:]), G[-N:,:-N])) for G in Gs]
            # print("hi")
            for i in range(G_tf(1).shape[0]-N):
                for j in range(G_tf(1).shape[1]-N):
                    ori = [(w, z[i,j].real, z[i,j].imag) for w,z in zip(omegas, Ts)]
                    self.add_ori_line(ori, color=colors[i][j], **kwargs)

    def plot_SSu_MIMO(self, G, N, colors, omega_lims = None, num_omegas=100, num_thetas=12, **kwargs):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)

        ((A, B, Bq),(C, _, _,),(Cq, Dq, _)) = G

        for i in range(G_tf(1).shape[0]-N):
            for j in range(G_tf(1).shape[1]-N):
                ori = [(w, G_tf(w)[i,j].real, G_tf(w)[i,j].imag) for w in omegas]
                self.add_ori_line(ori, color=colors[i][j], **kwargs)
        for delta in [getRandomBoarderlineContraction(N) for i in range(20)]:
            Gs = [G_tf(w) for w in omegas]
            Ts = [G[:-N,:-N]+G[:-N,-N:].dot(delta).dot(np.linalg.solve(np.eye(N)-delta.dot(G[-N:,-N:]), G[-N:,:-N])) for G in Gs]
            # print("hi")
            for i in range(G_tf(1).shape[0]-N):
                for j in range(G_tf(1).shape[1]-N):
                    ori = [(w, z[i,j].real, z[i,j].imag) for w,z in zip(omegas, Ts)]
                    self.add_ori_line(ori, color=colors[i][j], **kwargs)

    def plot_USF_MIMO(self, G, N, colors, omega_lims = None, num_omegas=100, num_thetas=12, **kwargs):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)

        ((A, B0, B1),(C0, _, D01,),(C1, D10, D11)) = G

        sImA = lambda w: np.eye(A.shape[0])*complex(0,w)-A
        G00 = lambda w: C0.dot(np.linalg.solve(sImA(w),B0))
        G10 = lambda w: C1.dot(np.linalg.solve(sImA(w),B0))+D10
        G11 = lambda w: C1.dot(np.linalg.solve(sImA(w),B1))+D11
        G01 = lambda w: C0.dot(np.linalg.solve(sImA(w),B1))+D01



        for i in range(G_tf(1).shape[0]-N):
            for j in range(G_tf(1).shape[1]-N):
                ori = [(w, G_tf(w)[i,j].real, G_tf(w)[i,j].imag) for w in omegas]
                self.add_ori_line(ori, color=colors[i][j], **kwargs)
        for delta in [getRandomBoarderlineContraction(N) for i in range(20)]:
            Gs = [G_tf(w) for w in omegas]
            Ts = [G[:-N,:-N]+G[:-N,-N:].dot(delta).dot(np.linalg.solve(np.eye(N)-delta.dot(G[-N:,-N:]), G[-N:,:-N])) for G in Gs]
            # print("hi")
            for i in range(G_tf(1).shape[0]-N):
                for j in range(G_tf(1).shape[1]-N):
                    ori = [(w, z[i,j].real, z[i,j].imag) for w,z in zip(omegas, Ts)]
                    self.add_ori_line(ori, color=colors[i][j], **kwargs)
                

    def plot_robust_SISO(self, tf1, tf2, omega_lims = None, num_omegas=15,  **kwargs):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)
        Z1 = [tf1(w) for w in omegas]
        Z2 = [tf2(w) for w in omegas]
        ori1 = [(w, z.real, z.imag) for w,z in zip(omegas, Z1)]
        ori2p = [(w, z.real, z.imag) for w,z in zip(omegas, Z2)]
        ori1proj = [(self.min_omega, r, i) for o,r,i in ori1]
        self.add_robust_ori_line(ori1, ori2p, **kwargs)
        if "alpha" in kwargs:
            kwargs["alpha"]*=0.5
        self.add_robust_ori_line(ori1proj, ori2p, **kwargs)

    def add_robust_ori_line(self, oriA, oriDelta, **kwargs):
        # self.add_ori_line(oriA, **kwargs)
        thetas=np.linspace(0,np.pi*2, 50)
        for (o, r, i), (o1, rD, iD) in zip(oriA, oriDelta):
            dat=[[],[],[]]
            for theta in thetas:
                dat[0].append(np.log10(o))
                rp=r+np.cos(theta)*abs(complex(rD,iD))
                ip=i+np.sin(theta)*abs(complex(rD,iD))
                dat[1].append(rp*self.dBscale(rp, ip))
                dat[2].append(ip*self.dBscale(rp, ip))
            self.ax.plot(dat[1],dat[2],dat[0], **kwargs)

    def add_robust_ori_circles(self, oriA, oriDelta, **kwargs):
        return self.add_robust_ori_line(oriA, oriDelta, **kwargs)

    def add_ori_line(self, ori_zip, mirror=True, shadow=False, **kwargs):
        if len(ori_zip)==0:
            return
        re_scaled_reals = [ r*self.dBscale(r,i) for w,r,i in ori_zip]
        re_scaled_imags = [ i*self.dBscale(r,i) for w,r,i in ori_zip]
        negative__imags = [-i for i in re_scaled_imags]
        log_omegas = [np.log10(w) for w,r,i in ori_zip]
        self.max_amp = max(self.max_amp, max([sqrt(r**2+i**2) for w,r,i in ori_zip]))
        self.min_omega = min(self.min_omega, min([w for w,r,i in ori_zip]))
        self.max_omega = max(self.max_omega, max([w for w,r,i in ori_zip]))
        min_omegas = [np.log10(self.min_omega) for w,r,i in ori_zip]
            
        self.ax.plot(re_scaled_reals, re_scaled_imags, log_omegas, **kwargs)
        if mirror:
            self.ax.plot(re_scaled_reals, negative__imags, log_omegas, **kwargs)
        if shadow:
            self.ax.plot(re_scaled_reals, re_scaled_imags, min_omegas, **kwargs)

    def draw_circle(self, center, radius, **kwargs):
        thetas=np.linspace(0,2*pi,1000)

        re = center[0] + radius * np.cos(thetas)
        im = center[1] + radius * np.sin(thetas)
        omega=[self.min_omega for x in re]
        self.add_ori_line(list(zip(omega,re,im)), mirror=False, **kwargs)

    def setup_negative_one_point(self, lw=1.0, ls=':'):
        amps=[0.01,0.25,0.5,1.0]
        for a in amps:
            self.draw_circle((-1,0),a,ls=ls,lw=lw, color='#888888')
        # self.add_ori_line([(self.min_omega, -1,0)],ls=":",ms=4)
    def show(self):
        plt.show()

def bode_axs():
    # bode plot, shown with 2 sigma uncertainty.
    fig, axs = plt.subplots(2, sharex=True)
    plt.locator_params(nbins=10)
    # axs[0].set_yscale('log')
    # axs[1].set_xscale('log')
    # axs[0].set_xscale('log')
    return fig, axs
default_color_lambda = lambda amp: custom_cm(amp/8.0+0.2)

class BodePlot(object):
    ''' A nice-looking bode plot class, with hand made x-tic labels'''
    def __init__(self,**kwargs):
        self.fig, self.axs = plt.subplots(2, sharex=True,**kwargs)
        self.min_omega = float('inf')
        self.max_omega = 0.0
        self.axs[1].set_xlabel(r"$\omega$ rad/s")
        self.axs[1].set_ylabel(r"$\angle{H(s)}$ deg")
        self.axs[1].set_yticks([0,-90,-180, -270, -360])
        self.axs[1].set_ylim([-360,0])
        self.axs[0].set_ylabel(r"$\|{H(s)}\|$ dB")
        # plt.locator_params(nbins=10)

    def setup_x_labels(self):
        low_log = int(log10(self.min_omega)-1.5)
        high_log = int(log10(self.max_omega)+1.5)
        self.axs[1].set_xticks(range(low_log,high_log))
        self.axs[1].set_xticklabels(["$10^{%d}$"%i for i in range(low_log,high_log)])

    def add_data(self, data, yind, color_lambda=default_color_lambda, sigma=5e-3):
        experiment_keys=list(data.keys())

        amps = gen_amps(data)
        amp_indexed_exps={}
        for e in experiment_keys:
            if len(data[e]['experiment.u[0].amp'])==0:
                continue
            amp = data[e]['experiment.u[0].amp'][0]
            if amp not in amp_indexed_exps:
                amp_indexed_exps[amp]=[] # zip(omega, real, imag) format
            omegas = data[e]['experiment.omega']
            reals = data[e]['y[%d].real'%yind]/amp
            imags = data[e]['y[%d].imag'%yind]/amp

            amp_indexed_exps[amp].extend(list(zip(omegas, reals, imags)))

        for amp in sorted(amp_indexed_exps.keys()):
            col=color_lambda(amp)
            self.add_ori_scatter(amp_indexed_exps[amp], color_lambda(amp), sigma)
        self.title("y[%d] bode plot"%yind)
        self.setup_tics()
        self.fig.tight_layout()

    def add_ori4_scatter(self, ori4, **kwargs):
        ori3 = [(o, r, i) for o,r,i,d in ori4]
        self.add_ori_scatter(ori3, **kwargs)

    def add_ori_scatter(self, ori_zip, sigma=0.01, decimate_shadow=0,**kwargs):
        log10_omegas=[log10(w)+np.random.normal(0.0,sigma) for w,r,i in ori_zip]
        mags_dB = [dB(sqrt(r**2+i**2)) for w,r,i in ori_zip]
        phase_deg = [deg(cmath.phase(complex(r,i))) for w,r,i in ori_zip]
        self.axs[0].plot(log10_omegas, mags_dB, ',', **kwargs)
        self.axs[1].plot(log10_omegas, phase_deg, ',', **kwargs)
        self.min_omega = min(self.min_omega, min([w for w,r,i in ori_zip]))
        self.max_omega = max(self.max_omega, max([w for w,r,i in ori_zip]))

    def plot_tf(self, tf, color = 'k', omega_lims = None, num_omegas=1000):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)
        Z = [tf(w) for w in omegas]
        n = tf(0.0).shape[0]
        # print(tf(0.0).shape)
        # print(tf(0.0)[0,0])
        for i in range(0,n):
            ori = [(w, z[i,0].real, z[i,0].imag) for w,z in zip(omegas, Z)]
            self.add_ori_line(ori, color=color)

    def add_ori4_surf(self, ori4, lower_dB_cap=-80, **kwargs):
        """ """

        log10_omegas=[log10(w) for w, r, i, _ in ori4]
        max_dB = []
        min_dB = []
        ppatches = [([], [], [],),]
        def update_patch(patch, w, lo, hi):
            patch[0].append(log10(w))
            patch[1].append(lo)
            patch[2].append(hi)
        def new_patch():
            ppatches.append(([], [], []))
        def swap_patches():
            main = tuple(ppatches[-2])
            aux = tuple(ppatches[-1])
            ppatches[-1] = main
            ppatches[-2] = aux
        ppatch_state = 0 
        """ uncertainty ppatch_state:
        0 --- contained
        1 --- over
        2 --- under
        3 --- dominant
        """

        def last_w(w):
            if len(ppatches[-1][0])>0:
                return pow(10, ppatches[-1][0][-1])
            else:
                return w

        def inrange(phi):
            assert np.isfinite(phi)
            if phi>0:
                return phi-360
            if phi<-360:
                return phi+360

        old_ppatch_state = -1
        for w, r, i, d in ori4:
            if not (old_ppatch_state == ppatch_state):
                # print(ppatch_state)
                old_ppatch_state = ppatch_state
        
            max_dB.append(dB(sqrt(r**2+i**2)+d))
            nom_phase = deg(cmath.phase(complex(r,i)))
            if r**2+i**2<d**2:
                # if ppatch_state == 1 or ppatch_state == 2:
                #     # update_patch(ppatches[-2], w, -360, 0)
                # Maximum phase_adjust.
                if ppatch_state == 1:
                    update_patch(ppatches[-2], w, nom_phase-90, 0) # main
                    update_patch(ppatches[-1], w, -360, nom_phase+90-360) # aux
                if ppatch_state == 2:
                    update_patch(ppatches[-1], w, nom_phase-90, 0) # main
                    update_patch(ppatches[-2], w, -360, nom_phase+90-360) # aux
                ppatch_state = 3
                update_patch(ppatches[-1], w, -360, 0)
                min_dB.append(lower_dB_cap)
            else:
                min_dB.append(max(lower_dB_cap, dB(sqrt(r**2+i**2)-d)))

                phase_adjust = 180.0/pi*asin(d/np.sqrt(r**2+i**2))

                if phase_adjust+nom_phase>0:
                    # we are now in state 1: ppatches[-2] main (top), ppatches[-1] on bottom
                    if ppatch_state == 0:
                        # introduce new patch on the bottom
                        w0 = last_w(w)
                        new_patch()
                        update_patch(ppatches[-1], w0, -360, -360)
                    elif ppatch_state == 1:
                        # no transition needed
                        pass
                    elif ppatch_state == 2:
                        # switch patches and continue
                        swap_patches()
                    elif ppatch_state == 3:
                        # find introduce new patch. Maximum phase_adjust.
                        w0 = last_w(w)
                        new_patch()
                        update_patch(ppatches[-2], w0, nom_phase-90, 0) # main
                        update_patch(ppatches[-1], w0, -360, nom_phase+90-360) # aux

                    # finalize state update
                    ppatch_state = 1

                    # apply update to both patches
                    update_patch(ppatches[-2], w, nom_phase-phase_adjust, 0) # main
                    update_patch(ppatches[-1], w, -360, nom_phase+phase_adjust-360) # aux

                elif nom_phase-phase_adjust<-360:
                    # we are now in state 2: main ppatches[-2] on bottom, aux on top ppatches[-1]
                    if ppatch_state == 0:
                        # introduce aux patch on top
                        w0 = last_w(w)
                        new_patch()
                        update_patch(ppatches[-1], w0, 0, 0)
                    elif ppatch_state == 1:
                        # swap top and bottom patches
                        swap_patches()
                    elif ppatch_state == 2:
                        # no state update needed
                        pass
                    elif ppatch_state == 3:
                        # find introduce new patch. maximum phase_adjust.
                        w0 = last_w(w)
                        new_patch()
                        update_patch(ppatches[-2], w0, -360, nom_phase+90) # main
                        update_patch(ppatches[-1], w0, nom_phase-90+360, 0) # aux
                    # finalize state update
                    ppatch_state = 2

                    # apply update to both patchs
                    update_patch(ppatches[-2], w, -360, nom_phase+phase_adjust) # main
                    update_patch(ppatches[-1], w, nom_phase-phase_adjust+360, 0) # aux


                else:
                    # we are now in state 0
                    if ppatch_state == 0:
                        # no update needed
                        pass
                    if ppatch_state == 1 or ppatch_state == 2:
                        # swap patches and proceed with the main patch
                        swap_patches()
                    if ppatch_state == 3:
                        # add data point for maximum phase adjust
                        w0 = last_w(w)
                        update_patch(ppatches[-1], w0, sat_ang(nom_phase-90), sat_ang(nom_phase+90))
                    # finalize state update
                    ppatch_state = 0

                    # apply update to single patch
                    update_patch(ppatches[-1], w, nom_phase-phase_adjust, nom_phase+phase_adjust)

        # print(ppatches)
        for tmplog10omegas, low_phases, high_phases in ppatches:
            assert len(tmplog10omegas) == len(low_phases)
            assert len(tmplog10omegas) == len(high_phases)
            # self.axs[1].plot(tmplog10omegas, low_phases, 'm')
            # self.axs[1].plot(tmplog10omegas, high_phases, 'b')

            self.axs[1].fill_between(tmplog10omegas, low_phases, high_phases, linewidths=0, **kwargs)

        # self.axs[0].plot(log10_omegas, min_dB, 'm')
        # self.axs[0].plot(log10_omegas, max_dB, 'b')
        ret = self.axs[0].fill_between(log10_omegas, min_dB, max_dB, linewidths=0, **kwargs)
        self.axs[0].set_ylim(bottom=lower_dB_cap)

        # mags_dB = [dB(sqrt(r**2+i**2)+d) for w, r, i, d in ori4]
        # mags_dB = [dB(max(min_disp_mag, sqrt(r**2+i**2)-d)) for w, r, i, d in ori4]
        # phase_deg = [deg(cmath.phase(complex(r,i))) for w, r, i, d in ori4]
        # self.axs[0].plot(log10_omegas, mags_dB, **kwargs)
        # self.axs[1].plot(log10_omegas, phase_deg, **kwargs)
        self.min_omega = min(self.min_omega, min([w for w,r,i,_ in ori4]))
        self.max_omega = max(self.max_omega, max([w for w,r,i,_ in ori4]))
        self.setup_x_labels()
        return ret

        # self.axs[0].plot(x, y1, x, y2, color='black')
        # self.axs[0].fill_between(x, y1, y2, where=y2>y1, facecolor='green')

    def add_ori4_line(self, ori4, **kwargs):
        """ conversion method. """
        ori3 = [(o, r, i) for o, r, i, _ in ori4]
        self.add_ori_line(ori3, **kwargs)

    def plot_SISO(self, tf, color = 'k', omega_lims = None, num_omegas=1000, **kwargs):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)
        Z = [tf(w) for w in omegas]
        ori = [(w, z.real, z.imag) for w,z in zip(omegas, Z)]
        self.add_ori_line(ori, color=color, **kwargs)

    def plot_MIMO(self, tf, colors, omega_lims = None, num_omegas=1000, **kwargs):
        if omega_lims==None:
            omega_lims = [self.min_omega, self.max_omega]
        omegas = np.logspace(log10(omega_lims[0]), log10(omega_lims[1]), num_omegas)
        Z = [tf(w) for w in omegas]
        for i in range(tf(0).shape[0]):
            for j in range(tf(0).shape[1]):
                ori = [(w, z[i,j].real, z[i,j].imag) for w,z in zip(omegas, Z)]
                self.add_ori_line(ori, color=colors[i,j], **kwargs)

    def add_ori_line(self, ori_zip, mirror=False, shadow=False, **kwargs):
        log10_omegas=[log10(w) for w,r,i in ori_zip]
        mags_dB = [dB(sqrt(r**2+i**2)) for w,r,i in ori_zip]
        phase_deg = [deg(cmath.phase(complex(r,i))) for w,r,i in ori_zip]
        phase_parts = [([],[])]
        def append_to_pp(phase_part, w, phase):
            phase_part[0].append(log10(w))
            phase_part[1].append(phase)
        sector = -1
        for w, r, i in ori_zip:
            nominal_phase = deg(cmath.phase(complex(r,i)))
            new_sector = 1 if nominal_phase<-310 else 2 if nominal_phase>-50 else 0
            if not(sector==new_sector) and not (sector==0 or new_sector==0):
                if sector==1 and new_sector==2:
                    append_to_pp(phase_parts[-1], w, -360)
                if sector==2 and new_sector==1:
                    append_to_pp(phase_parts[-1], w, 0)
                phase_parts.append(([],[]))
            sector=new_sector
            append_to_pp(phase_parts[-1], w, nominal_phase)
        mag_line, = self.axs[0].plot(log10_omegas, mags_dB, **kwargs)
        # self.axs[1].plot(log10_omegas, phase_deg, **kwargs)
        kwargs=dict(kwargs)
        kwargs["color"]=mag_line.get_color()
        for ws, ps in phase_parts:
            self.axs[1].plot(ws, ps, **kwargs)  
        self.min_omega = min(self.min_omega, min([w for w,r,i in ori_zip]))
        self.max_omega = max(self.max_omega, max([w for w,r,i in ori_zip]))

    def setup_negative_one_point(self):
        pass

    def show(self):
        # plt.tight_layout()
        plt.show()


    def title(self,title):
        self.axs[0].set_title(title)

    def setup_tics(self):
        round_it = lambda x: (x*10)/10
        tics=np.linspace(log10(self.min_omega), log10(self.max_omega)+0.01, 6)
        tics_exponent = [math.floor(t) for t in tics]
        tics_mantissa = [pow(10.0,t-math.floor(t)) for t in tics]
        tics_mantissa = [math.floor(m*100)/100.0 for m in tics_mantissa]
        tics_final = [log10(m) + e for m,e in zip(tics_mantissa, tics_exponent)]
        tic_s = ['$%.2f \\times 10^{%d}$'%(m,e) for m,e in zip(tics_mantissa, tics_exponent)]
        self.axs[1].set_xticks(tics_final)
        self.axs[1].set_xticklabels(tic_s)
        self.axs[1].set_xlim([tics_final[0],tics_final[-1]])

    def add_center_line(self, data, yind, color_lambda=default_color_lambda, alpha=1.0):
        experiment_keys=list(data.keys())
        amps = gen_amps(data)
        amp_indexed_exps={}
        for e in experiment_keys:
            amp = data[e]['experiment.u[0].amp'][0]
            if amp not in amp_indexed_exps:
                amp_indexed_exps[amp]=[] # zip(omega, real, imag) format
            num = len(data[e]['experiment.u[0].amp'])
            omega = (sum(data[e]['experiment.omega'])/num)
            real = (sum(data[e]['y[%d].real'%yind]/amp)/num)
            imag = (sum(data[e]['y[%d].imag'%yind]/amp)/num)

            amp_indexed_exps[amp].append((omega, real, imag))

        for amp in sorted(amp_indexed_exps.keys()):
            amp_indexed_exps[amp].sort(key=lambda x: x[0]) # sort by omega
            col=color_lambda(amp)
            self.add_ori_line(amp_indexed_exps[amp], color=color_lambda(amp),lw=4)
            # self.add_ori_freq_markers(amp_indexed_exps[amp], color=color_lambda(amp))

    def save(self, name):
        plt.tight_layout()
        self.fig.savefig(pdf_folder+name+".pdf", 
            facecolor='w', edgecolor='w',
            pad_inches=0.01)
        self.fig.savefig(png_folder+name+".png", 
            facecolor='w', edgecolor='w', dpi=400,
            pad_inches=0.01)

def main():
    nyq = Log3DNyquistPlot(1e-5)
    nyq.min_omega=.1
    nyq.max_omega=10
    nyq.setup_tics()
    nyq.setup_negative_one_point()
    nyq.show()

if __name__ == '__main__':
    main()