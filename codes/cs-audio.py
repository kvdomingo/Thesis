import numpy as np
import numpy.random as rand
import scipy.fftpack as fft
import scipy.io.wavfile as wav
import IPython.display as disp

class compsenseFromFile:
	def __init__(self, filename, downsample=False, downrate=None):
		self.rate, self.data = wav.read(filename)
		self.filename = filename
		self.name = filename[:-4]
		if len(self.data.shape) > 1 and self.data.shape[1] > 1:
			self.data = self.data.mean(axis=1)
		self.N = len(self.data)
		self.dur = self.N/self.rate
		self.t = np.linspace(0, self.dur, self.N)
		self.coef = fft.fft(self.data)
		self.coefshift = fft.fftshift(self.coef)
		if downsample:
			self.downrate = downrate
			self.Nd = int(downrate * self.dur)
			nd = np.round(np.linspace(0, self.N-1, self.Nd)).astype(int)
			self.data = self.data[nd]
			self.t = self.t[nd]
			self.coef = fft.fft(self.data)
			self.coefshift = fft.fftshift(self.coef)
			self.rate = downrate
		else:
			self.Nd = self.N
			self.t = np.linspace(0, self.dur, self.Nd)
			
	def getDominantFrequency(coef, rate):
		return np.argmax(abs(coef))/len(coef)*rate
	
	def sampleCompressive(self, mode, rate):
		self.subrate = rate
		self.M = int(self.subrate*self.dur)
	
		if mode == "random":
			m = np.sort(rd.randint(0, self.Nd, self.M))
			elif mode == "subnyquist":
			m = np.round(np.linspace(0, self.Nd-1, self.M)).astype(int)
		else:
			raise ValueError("Specified mode is invalid")
		
		self.y = self.data[m]
		self.tm = self.t[m]
		self.compressedCoef = fft.fft(self.y)
		self.compressedCoefShift = fft.fftshift(self.compressedCoef)
		self.m = m

	def recovery(self, method, **method_kwargs):
		self.method = method
		d = fft.dct(np.identity(self.Nd))
		A = d[self.m]
		
		if method == "lasso":
			prob = skl.Lasso(**method_kwargs)
		elif method == "lassocv":
			prob = skl.LassoCV(**method_kwargs)
		elif method == "omp":
			prob = skl.OrthogonalMatchingPursuit(**method_kwargs)
		elif method == "sl0":
			prob = SL0(A, self.y)
			self.recoveredCoef = prob
			return
		else:
			raise ValueError("Specified method is invalid")
		
		prob.fit(A, self.y)
		self.recoveredCoef = prob.coef_
	
	def main(self, mode, rate, method, **method_kwargs):
		self.sampleCompressive(mode, rate)
		self.recovery(method, **method_kwargs)