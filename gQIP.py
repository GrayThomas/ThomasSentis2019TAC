#/* Gray Thomas, UT Austin---NSTRF NNX15AQ33H---Spring 2018 */
"""

A QIP implementation which is slightly more general than the one in the paper
(Written a year after the first journal paper submission).

Note that the demo only uses the features of the original QIP.

The generality introduced here is that y=Ax+B\DeltaCz (and z is not necessarily x).

"""
import numpy as np
import cvxpy as cvx # CVXOPT needs to work
import sys
import traceback
from collections import namedtuple
from cvx_utils import Variable, Semidef, trace, log_det

gQIPDatum = namedtuple("gQIPDatum", ["x", "y", "z", "covY"])
QuadricInclusion = namedtuple("QuadricInclusion", ["A", "B", "C"])

class UpperBoundException(Exception):
    """ Trivial Class for Handling Upper Bounds in the QIP"""
    pass

class GQIPLS(object):
    """ Least Squares version of QIP (uses same interface) """
    def __init__(self, nx, ny, nz, uplim=1e4, alpha=3, 
        extraction_eigenvalue_zero_tolerance= 1e-6, 
        extraction_assertion_zero_tolerance = 1e-5):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        assert ny==1 # Only implemented for this
        assert nx==nz # only implemented for this
        self.alpha = alpha
        self.XwinvX = np.zeros((self.nx, self.nx))
        self.XwinvY = np.zeros((self.nx, self.ny))
        self.YwinvY = np.zeros((self.ny, self.ny))
        self.l_tol = extraction_eigenvalue_zero_tolerance

    def check_data_sizes(self, data):
        x, y, z, Cov_yy = data[0][0]
        for datum_list in data:
            for x, y, z, Cov_yy in datum_list:
                assert(x.shape == (self.nx,))
                assert(y.shape == (self.ny,))
                assert(z.shape == (self.nz,))

                # assert(Cov_xx.shape == (nx, nx))
                # assert(Cov_xy.shape == (nx, ny))
                assert(Cov_yy.shape == (self.ny, self.ny))

    def set_data_raw(self, rawdata):
        # This assumes the data is homoscedastic
        self.raw_data = rawdata
        self.n_dat = len(rawdata)*2
        for complex_list in rawdata:
            allowance_side, error_side = 0, 0
            for x, y in complex_list:
                x = x.reshape((-1,1))
                y = y.reshape((-1,1))
                self.XwinvX += x.dot(x.T)
                self.XwinvY += x.dot(y.T)
                self.YwinvY += y.dot(y.T)

    def solve_raw(self, **kwargs):
        # attempt an OLS solution (homoscedastic)
        A = np.linalg.solve(self.XwinvX, self.XwinvY).T

        self.residual_Sigma_yy = 1.0/(self.n_dat-self.nx)*(A.dot(self.XwinvX).dot(A.T)-2*A.dot(self.XwinvY)+self.YwinvY)
        Sigma_A = np.linalg.inv(self.XwinvX)*self.residual_Sigma_yy

        l_C, U_C = np.linalg.eigh(Sigma_A)
        sqrt_l_C = [np.sqrt(l) if l>self.l_tol**2 else self.l_tol for l in l_C]
        C = np.diagflat(sqrt_l_C).dot(U_C.T)
        norm_C = np.linalg.norm(C, "fro")
        C = C/norm_C
        B = np.array([[norm_C]])

        return QuadricInclusion(A, B, C), None

    def set_data(self, data):
        # Like QIP, this uses local covariance learning (a heteroscedastic model)
        self.data = [[gQIPDatum(*pre_datum) for pre_datum in pre_datum_list] for pre_datum_list in data]
        self.check_data_sizes(self.data)
        self.n_dat = len(self.data)*2

        for g_qip_datum_list in self.data:
            allowance_side, error_side = 0, 0
            for x, y, z, Cov_yy in g_qip_datum_list:
                x = x.reshape((-1,1))
                y = y.reshape((-1,1))
                Cov_yy = Cov_yy.real
                self.XwinvX += x.dot(x.T)/Cov_yy[0,0]
                self.XwinvY += x.dot(y.T)/Cov_yy[0,0]
                self.YwinvY += y.dot(y.T)/Cov_yy[0,0]

    def solve(self, **kwargs):
        # attempt a solution using QIP style data (heteroscedastic)
        A = np.linalg.solve(self.XwinvX, self.XwinvY).T

        self.residual_Sigma_yy = 1.0/(self.n_dat-self.nx)*(A.dot(self.XwinvX).dot(A.T)-2*A.dot(self.XwinvY)+self.YwinvY)
        Sigma_A = np.linalg.inv(self.XwinvX)*self.residual_Sigma_yy

        l_C, U_C = np.linalg.eigh(Sigma_A)
        sqrt_l_C = [np.sqrt(l) if l>self.l_tol**2 else self.l_tol for l in l_C]
        C = np.diagflat(sqrt_l_C).dot(U_C.T)

        norm_C = np.linalg.norm(C, "fro")
        C = C/norm_C
        B = np.array([[norm_C]])

        return QuadricInclusion(A, B, C), None

def expand_to_fit(model, data, alpha):
    (A, B, C) = model
    normC = np.linalg.norm(C,'fro')
    C = C/normC
    B = B*normC
    X_B = np.linalg.inv(B.dot(B.T))
    max_ratio = 1.0
    print("max_ratio set to 1.0")
    for g_qip_datum_list in data:
        allowance_side, error_side = 0, 0
        for x, y, z, Cov_yy in g_qip_datum_list:
            error_side += np.linalg.norm(np.linalg.solve(B, y-A.dot(x)))**2
            allowance_side += np.linalg.norm(C.dot(x))**2 + alpha * trace(Cov_yy*X_B)
        ratio = error_side/allowance_side
        if ratio>max_ratio:
            max_ratio=ratio
            print("error %.2e, allowance %.2e, ratio %.2f" %(error_side, allowance_side, max_ratio))
    
    return QuadricInclusion(A, B*np.sqrt(max_ratio), C), max_ratio

# generalized quadric inclusion program
class GeneralizedQuadricInclusionProgram(object):
    def __init__(self, nx, ny, nz, uplim=1e4, alpha=3, 
        extraction_eigenvalue_zero_tolerance= 1e-6, 
        extraction_assertion_zero_tolerance = 1e-5):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.uplim = uplim
        self.alpha = alpha
        self.l_tol = extraction_eigenvalue_zero_tolerance
        self.a_tol = extraction_assertion_zero_tolerance
        self.setup_cvx_variables()
        self.set_model_constraints()

    def setup_cvx_variables(self):
        self.Qprime = Semidef(self.nx+self.ny)*1e0
        self.X_B = self.Qprime[0:self.ny, 0:self.ny]
        self.X_A = -self.Qprime[0:self.ny, self.ny:]
        self.X_AA = self.Qprime[self.ny:, self.ny:]
        self.X_C = Semidef(self.nz)*1e-0

    def check_data_sizes(self, data):
        x, y, z, Cov_yy = data[0][0]
        for datum_list in data:
            for x, y, z, Cov_yy in datum_list:
                assert(x.shape == (self.nx,))
                assert(y.shape == (self.ny,))
                assert(z.shape == (self.nz,))

                assert(Cov_yy.shape == (self.ny, self.ny))

    def set_data(self, data):
        self.data = [[gQIPDatum(*pre_datum) for pre_datum in pre_datum_list] for pre_datum_list in data]
        self.check_data_sizes(self.data)
        self.dcons = []
        
        for g_qip_datum_list in self.data:
            allowance_side, error_side = 0, 0
            for x, y, z, Cov_yy in g_qip_datum_list:
                Cov_yy = Cov_yy.real
                xi = cvx.Constant(np.hstack([y, x]))
                allowance_side += cvx.quad_form(z, self.X_C)+self.alpha*trace(Cov_yy*self.X_B) 
                error_side += cvx.quad_form(xi, self.Qprime)
            self.dcons.append(allowance_side >= error_side) 

    def set_model_constraints(self, X_C_norm=None, X_C_n_val=1):
        if X_C_norm==None:
            X_C_norm = np.eye(self.nz)
        self.mcons=[]
        self.mcons.append(self.X_B - np.eye(self.ny)*1e-6 == Semidef(self.ny)*1e1)
        self.upperbound_con = Semidef(self.nx+self.ny)*self.uplim + trace(self.Qprime) == self.uplim*np.eye(self.nx+self.ny)
        self.mcons.append(self.upperbound_con)
        self.trace_con = (trace(self.X_C*X_C_norm) == X_C_n_val)
        self.mcons.append(self.trace_con)


    def default_solve(self, max_solver_attempts = 2, universal_tol=1e-9):
        for solver_attempt in range(max_solver_attempts):
            try:
                return self.solve(solver="CVXOPT", verbose=True,
                    abstol=universal_tol, reltol=universal_tol, feastol = universal_tol, 
                    refinement=[5,40][solver_attempt], kktsolver="ldl2", # {"ldl2", 'chol2', "chol", "robust"} robust=LDL
                    max_iters=400)
                # There are only two options in CVX which can solve SD + EXP cone
                # problems: SCS and CVXOPT. However SCS never finds optimal
                # solutions, perhaps due to the unusual sub-problem we face in
                # uncertainty shape identification. The performance of CVXOPT
                # depends heavily on the refinement parameter, but it is unclear
                # what this actually does or why it is so important.
            except (cvx.error.SolverError) as e:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                if solver_attempt==max_solver_attempts-1:
                    print("it failed")
                    raise cvx.error.SolverError()
            except AssertionError as e:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                print("Assertion error (in the model checking)")
                raise e
            else:
                break

    def solve(self, **kwargs):
        # atempt a solution. raises cvx.error.SolverError 
        self.prob = cvx.Problem(cvx.Maximize(log_det(self.X_B)), self.mcons+self.dcons)
        prob_result = self.prob.solve(**kwargs)
        if np.linalg.norm(self.upperbound_con.dual_value)>1e-7:
            print(self.extract_model())
            raise UpperBoundException()
        return self.extract_model(), prob_result

    def extract_model(self):
        # Checks that the SS-DD is linear inclusion equivalent and finds this inclusion
        X_B = np.matrix(self.X_B.value)
        X_A = np.matrix(self.X_A.value)
        X_AA = np.matrix(self.X_AA.value)
        X_C = np.matrix(self.X_C.value)

        # Check equation (11) from the paper
        error = X_AA - X_A.T.dot(np.linalg.solve(X_B, X_A))
        print("(11) error:", np.linalg.norm(error))
        assert (np.linalg.norm(error)<self.a_tol)


        l_C, U_C = np.linalg.eigh(X_C)
        l_B, U_B = np.linalg.eigh(X_B)
        sqrt_l_C = [np.sqrt(l) if l>self.l_tol**2 else self.l_tol for l in l_C]
        inv_sqrt_l_B = [1./np.sqrt(l) if l>self.l_tol**2 else self.l_tol for l in l_B]
        assert (np.linalg.norm(X_B-U_B.dot(np.diagflat(l_B)).dot(U_B.T),"fro")<1e-7)
        assert (np.linalg.norm(X_C-U_C.dot(np.diagflat(l_C)).dot(U_C.T),"fro")<1e-7)
        C = np.diagflat(sqrt_l_C).dot(U_C.H)
        B = U_B.dot(np.diagflat(inv_sqrt_l_B))
        A = B.dot(B.T).dot(X_A)

        # Check SS-DD definitional equations (13) from the paper
        assert (np.linalg.norm(X_B - np.linalg.inv(B.dot(B.T)))<self.a_tol)
        assert (np.linalg.norm(X_A - np.linalg.solve(B.dot(B.T),A))<self.a_tol)
        assert (np.linalg.norm(X_AA - A.T.dot(np.linalg.solve(B.dot(B.T),A)))<self.a_tol)
        assert (np.linalg.norm(X_C - C.T.dot(C))<self.a_tol)

        return QuadricInclusion(A, B, C)
