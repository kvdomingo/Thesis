import numpy as np
import numpy.random as rand
import scipy.fftpack as fft
import cv2 as cv

def encrypt(filename, compression, key1, key2, key3):
	if isinstance(filename, str):
		X = cv.imread(filename, 0)
	if isinstance(filename, np.ndarray):
		X = filename.copy()
	N = len(X)
	M = int(compression * N)
	
	lamda1 = [key1]
	lamda2 = [key2]
	for i in range(1, 2*N):
		lamda1.append(key3*lamda1[i-1]*(1 - lamda1[i-1]))
		lamda2.append(key3*lamda2[i-1]*(1 - lamda2[i-1]))

	s1 = np.array(lamda1[N:])
	s2 = np.array(lamda2[N:])
	n = np.arange(N)
	sorty1 = np.array([s1, n]).T
	sorty2 = np.array([s2, n]).T
	sorty1 = sorty1[sorty1[:, 0].argsort()]
	sorty2 = sorty2[sorty2[:, 0].argsort()]
	l1 = sorty1.T[1].astype(np.uint8)
	l2 = sorty2.T[1].astype(np.uint8)
	
	H = np.linalg.hadamard(2**(np.round(np.log2(N)).astype(int)))
	Phi1 = H[l1[:M], :N]
	Phi2 = H[l2[:M], :N]
	
	Psi = fft.dct(np.identity(N))
	beta1 = Phi1.dot(Psi.T.dot(X))
	beta = Psi.T.dot(X.T.dot(Psi))
	beta2 = Psi.T.dot(beta1.T)
	Y = Phi2.dot(beta.dot(Phi1.T))
	return (Y, compression)
	
def decrypt(signal_in, decompress, key1, key2, key3):
	M = len(signal_in)
	N = int(M/decompress)
	lamda1 = [key1]
	lamda2 = [key2]
	for i in range(1, 2*N):
		lamda1.append(key3*lamda1[i-1]*(1 - lamda1[i-1]))
		lamda2.append(key3*lamda2[i-1]*(1 - lamda2[i-1]))
		
	s1 = np.array(lamda1[N:])
	s2 = np.array(lamda2[N:])
	n = np.arange(N)
	sorty1 = np.array([s1, n]).T
	sorty2 = np.array([s2, n]).T
	sorty1 = sorty1[sorty1[:, 0].argsort()]
	sorty2 = sorty2[sorty2[:, 0].argsort()]
	l1 = sorty1.T[1].astype(np.uint8)
	l2 = sorty2.T[1].astype(np.uint8)
	
	H = np.linalg.hadamard(2**(np.round(np.log2(N)).astype(int)))
	Phi1 = H[l1[:M], :N]
	Phi2 = H[l2[:M], :N]
	
	y1 = SL0(Phi2, signal_in)
	y2 = SL0(Phi1, y1.T)
	Y = idct2(ydir2)
	return (Y, decompress)