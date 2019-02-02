"""
Demonstration of robust QIP modelling with OBFs.

Gray Cortright Thomas

This is the main file of the demo. In the main function it regenerates the simulation result from the paper ThomasSentis2019TAC.

"""

import numpy as np
from gQIP import GeneralizedQuadricInclusionProgram, GQIPLS, expand_to_fit
from collections import namedtuple
import time
import matplotlib.pyplot as plt
from uncertain_bode import BodePlot
import control_LMIs as clmi
import control as ctrl
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as pyplt


import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.25

high_hasharg="a000"

# Setup sizing information. QIP can be applied to systems with more inputs and
# outputs than this simple example demonstrates.
EXAMPLE_SYSTEM_POLES = 4
OBF_POLES = 6
OBF_OUTPUTS = 6
SYSTEM_OUTPUTS = 1
SYSTEM_INPUTS = 1


class ExampleSystem(object):
    def __init__(self, new_ibs_exp=None):
        self.nx = EXAMPLE_SYSTEM_POLES
        self.nu = SYSTEM_INPUTS 
        self.ny = SYSTEM_OUTPUTS

        self.A = np.array([[0,1,0,0],[-25,-.5,0,0],[0,0,0,1],[0,0,-.25,-.25]])
        self.B = np.array([[0],[25],[0],[.25]])
        self.C = np.array([[.1, 0,1,0]])
        self.D = np.array([[0.0]])

        self.time_delay = 0.1
        self.Sigma_half_y = np.eye(self.ny) * 2e-3
        self.Sigma_half_p = np.eye(self.nx) * 0e-16
        self.Sigma_half_u = np.eye(self.nu) * 4e-1

    def set_dr_1(self, dr):
        w = np.sqrt(-self.A[1,0])
        self.A[1,1] = -2*dr*w

    def set_dr_2(self, dr):
        w = np.sqrt(-self.A[3,2])
        self.A[3,3] = -2*dr*w

    def frequency_domain_experiment(self, omega, u_phasor):
        assert u_phasor.shape==(self.nu,)
        scale_noise = np.sqrt(abs(omega) / (4. * np.pi))
        # scale_noise = 1
        measurement_noise = scale_noise * (np.dot(self.Sigma_half_y, np.random.randn(
            self.ny,) + np.random.randn(self.ny,) * complex(0., 1.)))
        process_noise =     scale_noise * (np.dot(self.Sigma_half_p, np.random.randn(
            self.nx,) + np.random.randn(self.nx,) * complex(0., 1.)))
        input_noise =       scale_noise * (np.dot(self.Sigma_half_u, np.random.randn(
            self.nu,) + np.random.randn(self.nu,) * complex(0., 1.)))

        x_phasor = np.linalg.solve(
            complex(0, omega) *np.eye(self.nx) - self.A,
            (self.B).dot(u_phasor + input_noise) +
            process_noise)

        y_phasor = (
            self.C.dot(x_phasor) +
            self.D.dot(
                u_phasor +
                input_noise) +
            measurement_noise)

        y_phasor[0] *= np.exp(self.time_delay * complex(0, -omega))

        return y_phasor


    def ori4(self, omega, input_selector, output_selector, eps=1e-2):
        # returns a four tuple representing a transfer function at one frequency

        system_00 = output_selector.dot(
            np.exp(self.time_delay * complex(0, -omega))*(self.D+self.C.dot(
                np.linalg.solve(
                    complex(0, omega) * np.eye(self.nx) - self.A,
                    self.B))).dot(input_selector))

        return (omega, system_00.real, system_00.imag, 0.0)


    def cg_average_experiment(self, omega, u_phasor, N=10):
        # simulates N experiments and calculates condition group statistics.
        y_phasors = [
            self.frequency_domain_experiment(
                omega, u_phasor) for i in range(N)]
        avg_y_phasor = sum(y_phasors) * (1. / N)
        cov_y_phasor_real = sum([np.outer((y_phasor - avg_y_phasor).real,
                                          (y_phasor - avg_y_phasor).real) for y_phasor in y_phasors]) * (1. / (N - 1) / N)
        cov_y_phasor_imag = sum([np.outer((y_phasor - avg_y_phasor).imag,
                                          (y_phasor - avg_y_phasor).imag) for y_phasor in y_phasors]) * (1. / (N - 1) / N)
        return avg_y_phasor, (cov_y_phasor_real, cov_y_phasor_imag), y_phasors

    def multi_cg_experiment(self, omegas, u_phasor, N=10):
        # multi condition group experiment
        data = []
        for omega in omegas:
            avg_y_phasor, cov_y_phasor, y_phasors = self.cg_average_experiment(
                omega, u_phasor, N=N)
            data.append(
                MCGData(
                    omega,
                    u_phasor,
                    avg_y_phasor,
                    cov_y_phasor,
                    y_phasors))
        return data

class UncertainBasisFunctionSystem(clmi.StateSpace):
    def __init__(self, new_ibs_exp=None):
        # These linear systems approximate the poles of the system. Some are
        # just high frequency low pass filters. Some are intentionally similar
        # to the resonant poles of a series elastic actuator.
        lp_0 = clmi.StateSpace(np.array([[-pow(10,1.5)]]), np.array([[1]]), np.array([[1]]), np.array([[0]]))
        lp_1 = clmi.StateSpace(np.array([[-pow(10,1.0)]]), np.array([[1]]), np.array([[1]]), np.array([[0]]))
        # Adjust these pole locations and dampings to see the influence of basis function quality on the QIP result.
        sys1 = clmi.StateSpace(np.array([[0, 1], [-pow(10,.7*2), -.2*pow(10,.7)]]), np.array([[0],[1]]), np.array([[1, 0]]), np.array([[0]]))
        sys2 = clmi.StateSpace(np.array([[0, 1], [ -pow(10,-.32*2), -.8*pow(10,-.32)]]), np.array([[0],[1]]), np.array([[1, 0]]), np.array([[0]]))

        # Assemble the system from component blocks
        ((in0,), (out0,)) = lp_0.tags
        lp_0 = lp_0.inline(out0, lp_1)
        lp_0 = lp_0.inline(lp_1.tags[1][0], sys1)
        lp_0 = lp_0.inline(lp_1.tags[1][0], sys2)
        lp_0 = lp_0.cleaned_inputs([in0])

        # The gramian is used to ortho-normalize the system
        Wc = ctrl.gram(lp_0,'c')
        sig, U = (np.linalg.eigh(Wc))
        # this assertion explains the result of the eigh function
        assert np.linalg.norm(U.dot(np.diagflat(sig).dot(U.T))-Wc)<1e-7
        C = np.diagflat([1.0/np.sqrt(si) for si in sig]).dot(U.T)
        super().__init__(lp_0.A, lp_0.B, C, np.zeros((C.shape[0],1)))

        # The grammian is used to confirm that the basis is orthonormal in H2.
        Wc = ctrl.gram(self, 'c')
        assert np.linalg.norm(U.T.dot(Wc).dot(U)-np.diagflat(sig))<1e-7
        assert np.linalg.norm(Wc-ctrl.gram(lp_0, 'c'))<1e-7
        print("should be I", self.C.dot(Wc).dot(self.C.T))

    def apply_learned_model_result(self, qip_a, qip_b, qip_c):
        """ Uses regressor definition to reset learned matrices. """
        self.qip_a = qip_a
        self.qip_b = qip_b
        self.qip_c = qip_c

    def obf(self, omega):
        # orthonormal basis function
        return self.D+self.C.dot(np.linalg.solve(complex(0, omega) * np.eye(self.states) - self.A, self.B))


    def ori4(self, omega, input_selector, output_selector, eps=1e-2):
        # ori4 is a data type which lists 4-tuples of omega (angular
        # frequency), real (component of transfer function), imaginary, and
        # uncertainty

        x = self.obf(omega).dot(input_selector)

        sAx = output_selector.dot(self.qip_a.dot(x))
        Cx = self.qip_c.dot(x)
        sB = output_selector.dot(self.qip_b)

        return (omega, sAx.real, sAx.imag,
                eps*np.linalg.norm(sAx)
                +np.linalg.norm(sB)*np.linalg.norm(Cx))

    def get_gQIP_extractor(self):
        def extractor(data):

            new_data = []
            for omega, u_phasor, avg_y_phasor, cov_y_phasor, raw_yphasors in data:
                cov_y_phasor_real, cov_y_phasor_imag = cov_y_phasor
                data_pair = []
                s = complex(0, omega)
                x = np.array(self.obf(omega).dot(u_phasor)).reshape((-1,))
                y = avg_y_phasor

                data_pair.append((x.real, y.real, x.real, cov_y_phasor_real))
                data_pair.append((x.imag, y.imag, x.imag, cov_y_phasor_imag))
                new_data.append(data_pair)
            return new_data
        return extractor
    def get_OLS_raw_extractor(self):
        def extractor(data):

            new_data = []
            for omega, u_phasor, avg_y_phasor, cov_y_phasor, raw_yphasors in data:
                s = complex(0, omega)
                x = np.array(self.obf(omega).dot(u_phasor)).reshape((-1,))
                for y in raw_yphasors:
                    data_pair = []
                    data_pair.append((x.real, y.real))
                    data_pair.append((x.imag, y.imag))
                    new_data.append(data_pair)
            return new_data
        return extractor

MCGData = namedtuple("MCGData", ['omega','u_phasor','avg_y_phasor','cov_y_phasor', 'raw_yphasors'])

def latex_print(mat):
    print('\\\\ \n'.join([' & '.join(["%.3f"%a for a in row]) for row in np.array(mat) ]))
    
def phasor_data_2_ori4(phasor_data, n_sigma=5):
    # MCGData = namedtuple("MCGData", ['omega','u_phasor','avg_y_phasor','cov_y_phasor', 'raw_yphasors'])
    n_sigma_ori4s = []
    raw_scatter_ori4s = []
    for mcg_datum in phasor_data:
        omega = mcg_datum.omega
        u = mcg_datum.u_phasor
        tf_nom = mcg_datum.avg_y_phasor/mcg_datum.u_phasor
        sigma = np.sqrt((mcg_datum.cov_y_phasor[0]+mcg_datum.cov_y_phasor[1])/(mcg_datum.u_phasor.conjugate().T.dot(mcg_datum.u_phasor)).real)
        n_sigma_ori4s.append((omega, tf_nom.real, tf_nom.imag, n_sigma*sigma))
        for y in mcg_datum.raw_yphasors:
            raw_scatter_ori4s.append((omega, (y/u).real, (y/u).imag, 0))
    return n_sigma_ori4s, raw_scatter_ori4s

def ori4_system_response(system, omegas):
    rows = []
    for out in range(system.outputs):
        row = []
        for inz in range(system.inputs):
            ori4=[]
            for omega in omegas:
                s = complex(0, omega)
                z = system.C[out,:].dot(np.linalg.solve(s*np.eye(system.states)-system.A, system.B[:,inz]))+ system.D[out,inz]
                ori4.append((omega, z.real, z.imag, 0.0))
            row.append(ori4)
        rows.append(row)
    return rows

def main():
    example_system = ExampleSystem()
    ubfs_system = UncertainBasisFunctionSystem()  
    ubfs_ls_sys = UncertainBasisFunctionSystem()   

    np.random.seed(1337)

    plt = BodePlot()
    fine_omegas = np.logspace(-2, 2, 5000)
    plt.max_omega = 1e2
    plt.min_omega = 1e-2
    plt.setup_negative_one_point()


    ## GRAY true system plot
    example_ori4s_1 = [example_system.ori4(omega, np.ones((1,)), np.ones((1,)), eps=3e-7) for omega in fine_omegas]

    print("generating data")
    t0 = time.time()

    phasor_data = example_system.multi_cg_experiment(np.logspace(-2, 2, 100), u_phasor=np.array([complex(1.,0.)]))
    example_system.set_dr_2(0.35)
    example_ori4s_2 = [example_system.ori4(omega, np.ones((1,)), np.ones((1,)), eps=3e-7) for omega in fine_omegas]
    phasor_data.extend(example_system.multi_cg_experiment(np.logspace(-2, 2, 100), u_phasor=np.array([complex(1.,0.)])))
    example_system.set_dr_1(0.25)
    example_ori4s_3 = [example_system.ori4(omega, np.ones((1,)), np.ones((1,)), eps=3e-7) for omega in fine_omegas]
    phasor_data.extend(example_system.multi_cg_experiment(np.logspace(-2, 2, 100), u_phasor=np.array([complex(1.,0.)])))

    n_sigma_ori4, raw_scatter_ori4 = phasor_data_2_ori4(phasor_data)
    plt.add_ori4_line(raw_scatter_ori4, mirror=False, color=(0.5, 0.5, 0.5), zorder=-1,linestyle='none', ms=2.5, marker=".")
    plt.add_ori4_line(example_ori4s_1, mirror=False, color=(0.3, 0.3, 0.3))
    plt.add_ori4_line(example_ori4s_2, mirror=False, color=(0.3, 0.3, 0.3))
    plt.add_ori4_line(example_ori4s_3, mirror=False, color=(0.3, 0.3, 0.3))
    
    qip_data = ubfs_system.get_gQIP_extractor()(phasor_data)
    ols_data = ubfs_system.get_OLS_raw_extractor()(phasor_data)
    print("data generated in %.4f seconds" % (time.time() - t0))

    # COLORFUL OBFs plot
    # ori4s = ori4_system_response(ubfs_system, fine_omegas)
    # for row in ori4s:
    #     plt.add_ori4_line(row[0], linestyle="-", lw=2)


    ## Solve QIP
    print("learning model")
    t0 = time.time()
    gqip = GeneralizedQuadricInclusionProgram(ubfs_system.states, SYSTEM_OUTPUTS, ubfs_system.states, uplim=1e4, alpha=3)
    gqip.set_data(qip_data)
    args = dict( solver="CVXOPT", verbose=True,
                    abstol=1e-9, reltol=1e-9, feastol = 1e-9, 
                    refinement=5, kktsolver="ldl2", # {"ldl2", 'chol2', "chol", "robust"} robust=LDL
                    max_iters=400)
    (qip_a, qip_b, qip_c), width = gqip.solve(**args)
    print("learned model in %.4f seconds" % (time.time() - t0))

    qip_alpha=1.5
    ## Solve OLS
    gls = GQIPLS(ubfs_system.states, SYSTEM_OUTPUTS, ubfs_system.states, uplim=1e4, alpha=qip_alpha)
    gls.set_data(qip_data)
    args = dict()
    (ls_a, ls_b, ls_c), width = gls.solve(**args)

    ## Expand models
    ratio_ls = 1.0
    for i in range(10):
        (ls_a, ls_b, ls_c), max_ratio_ls = expand_to_fit((ls_a, ls_b, ls_c), qip_data, qip_alpha)
        ratio_ls*=max_ratio_ls
    print("scaled B least squares by a total of", ratio_ls)
    ratio_qip = 1.0
    for i in range(10):
        (qip_a, qip_b, qip_c), max_ratio_qip = expand_to_fit((qip_a, qip_b, qip_c), qip_data, qip_alpha)
        ratio_qip*=max_ratio_qip
    print("scaled B qip by a total of", ratio_qip)

    ## Plot QIP Model
    ubfs_system.apply_learned_model_result(qip_a, qip_b, qip_c)
    learned_ori4 = [ubfs_system.ori4(omega, np.ones((1,1)), np.ones((1,1)), eps=1e-7) for omega in fine_omegas]
    plt.add_ori4_surf((learned_ori4), zorder=4, lw=.25, facecolor="none", edgecolor="red", alpha=1.0, hatch='+++')
    plt.add_ori4_line((learned_ori4), color="red", zorder=2, linestyle="-.", lw=2.0)

    ## Plot LS Model
    ubfs_ls_sys.apply_learned_model_result(ls_a, ls_b, ls_c)
    learned_ori4 = [ubfs_ls_sys.ori4(omega, np.ones((1,1)), np.ones((1,1)), eps=1e-7) for omega in fine_omegas]
    args = dict(facecolor="gray", alpha=.5, edgecolor="none")
    obj2 = plt.add_ori4_surf((learned_ori4), zorder=4, lw=.25, facecolor="none", edgecolor="blue", alpha=1.0, hatch='///')
    plt.add_ori4_line((learned_ori4), color="blue", zorder=2, linestyle="-.", lw=2.0)

    fontP = FontProperties()
    fontP.set_size('small')
    leg = plt.axs[0].legend([
        "measurements",
        "true system, cfg. 1","true system, cfg. 2","true system, cfg. 3", 
        'qip nominal', "ls nominal",'qip fit', "ls fit"], 
        bbox_to_anchor=(1.0,1.00), prop=fontP)
    pyplt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
