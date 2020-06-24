import numpy as np
import numpy.random as rand
import scipy.fftpack as fft

# load recording
signal = np.loadtxt("piano.txt").astype("float32")

# define parameters
samprate = 44.1e3
duration = 1/8
N = int(duration*samprate)
M = 300
t = np.linspace(0, duration, N)

# extract short portion of recording
sig_start = 40000
x = signal[sig_start:sig_start + N]

# simulate compressive measurements
yi = rand.randint(0, N, M)
yi = np.sort(yi)
y = x[yi]

# L1 optimization using CVX ECOS
import cvxpy as cvx
xhat_cvx = cvx.Variable(N)
objective = cvx.Minimize(cvx.Norm(xhat_cvx, 1))
constraints = [A*xhat_cvx == y]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True, solver="ECOS")
x_cvx = np.array(xhat_cvx.value)
x_cvx = np.squeeze(x_cvx)
x_cvx = fft.dct(x_cvx, norm="ortho", axis=0)

# L1-regularized L2 optimization using LASSO
from sklearn.linear_model import Lasso, LassoCV
lasso = LassoCV(cv=10, random_state=0, verbose=True, n_jobs=-1)
lasso.fit(A, y)
x_lasso = fft.idct(lasso.coef_)