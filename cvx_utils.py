"""
Just some utilities to work around the most recent updates to cvxpy.
"""
import cvxpy as cvx

def Semidef(nx):
    return cvx.Variable((nx, nx), PSD=True)

def Variable(*args):
	return cvx.Variable(tuple(args))

def log_det(X):
	if X.shape[0]==1:
		return cvx.log(X)
	return cvx.log_det(X)

def trace(X):
	if X.shape==(1,1):
		return X
	else:
		return cvx.trace(X)