import numpy as np
import numpy.random as rand
import tensorflow as tf
import scipy.fftpack as fft

# define L0 norm estimator function
def Fsigma(x_i, sigma):
	x = tf.abs(x_i)
	return x_i * tf.exp(-x**2/(2*sigma**2))

def SL0(A, b, Fsigma=Fsigma, sigma_min=1e-12, mu_0=2, L=3, sdf=0.5):
	A = tf.convert_to_tensor(A, dtype=tf.float32)
	b = tf.convert_to_tensor(b, dtype=tf.float32)
	A_plus = tf.linalg.pinv(A)
	x = tf.linalg.matmul(A_plus, b)
	sigma = 2 * np.max(np.abs(x.numpy()))
	while sigma > sigma_min:
		for i in range(L):
			delta = Fsigma(x, sigma)
			x -= mu_0 * delta
			x -= tf.linalg.matmul(A_plus, tf.linalg.matmul(A, x) - b)
		sigma *= sdf
	return x.numpy()