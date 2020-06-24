import numpy as np
import numpy.random as rand
import scipy.fftpack as fft
import cv2 as cv

X_orig = cv.resize(
	cv.imread("escher_relativity.jpg", 0),
	(1600, 976)
)
ny, nx = X_orig.shape
sec = 16
k = int(X_orig.size * 0.5)
ri = rand.choice(X_orig.size, k, replace=False)
Xm = 255 * np.ones(X_orig.shape)
Xm.T.flat[ri] = X_orig.T.flat[ri]

n = np.arange(sec)
slices = np.zeros((sec, sec), object)
recover = np.zeros((sec, sec), object)
for j in n:
	for i in n:
		slices[j,i] = X_orig[
			ny//sec*j:ny//sec*(j+1),
			nx//sec*i:nx//sec*(i+1)
		]
		
ny, nx = slices[j, i].shape
ex = 0.5
k = np.round(nx * ny * ex).astype(int)
ri = rand.choice(nx * ny, k, replace=False)
A = np.kron(
	fft.idct(np.identity(nx), norm='ortho', axis=0),
	fft.idct(np.identity(ny), norm='ortho', axis=0)
)
A = A[ri,:]

for j in trange(len(slices)):
	for i in trange(len(slices)):
		b = slices[j,i].T.flat[ri]
		vx = cvx.Variable(nx * ny)
		objective = cvx.Minimize(cvx.norm(vx, 1))
		constraints = [A*vx == b]
		prob = cvx.Problem(objective, constraints)
		result = prob.solve()
		Xat2 = np.array(vx.value)
		Xat2 = np.squeeze(Xat2)
		
		Xat = Xat2.reshape(nx, ny).T
		Xa = idct2(Xat)
		recover[j,i] = Xa
		
recovery = np.zeros(len(recover), object)
for i in range(len(recover)):
	recovery[i] = np.concatenate(recover[i,:], axis=1)
recovery = recovery.T
recovery = np.concatenate(recovery[:], axis=0)