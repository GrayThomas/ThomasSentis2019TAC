"""
A Bode plot class which handles uncertainty using area plots.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import namedtuple
from math import atan2, asin, log10, log, pi, sqrt
import cmath
import math
import mpl_toolkits.mplot3d.axes3d as p3

png_folder = r"../results/"
pdf_folder = r"../results/"

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

import numpy as np

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
        # fix for CodeOcean: save figures instead of showing them interactively.
        self.fig.savefig("../results/output.pdf", 
            facecolor='w', edgecolor='w',
            pad_inches=0.01)
        self.fig.savefig("../results/output.png", 
            facecolor='w', edgecolor='w', dpi=400,
            pad_inches=0.01)


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
