class SpeechCS:
	def __init__(
		self,
		filename,
		rate=None,
		downsample=False,
		downrate=None
	):
		source_type = type(filename)
		if source_type == str:
			if filename.endswith(".wav"):
				self.data, self.rate = sf.read(filename)
			elif filename.endswith(".csv"):
				self.data = np.genfromtxt(filename, delimiter=",")
				if rate is not None:
					self.rate = rate
				else:
					self.rate = int(input("Enter sample rate: "))
			else:
				return "Unsupported file type"
		elif source_type == np.ndarray:
			self.data = filename
			if rate is not None:
				self.rate = rate
			else:
				self.rate = int(input("Enter sample rate: "))
		
		self.dur = self.data.size/self.rate
		self.name = filename[:-4]
		self.filename = filename
		
		if len(self.data.shape) > 1 and self.data.shape[1] > 1:
			self.data = self.data.mean(axis=1)
		
		if downsample:
			self.rate = downrate
			downsize = int(self.dur * downrate)
			down_idx = np.round(np.linspace(
				0,
				self.data.size-1,
				downsize
			)).astype(int)
			self.data = self.data[down_idx]
		
		self.t = np.linspace(0, self.dur, self.data.size)
	
	def sampleCompressive(
		self,
		comp_ratio,
		seglen,
		percent_overlap,
		window="hann"
	):
		hop_size = int(seglen * percent_overlap)
		starts = np.arange(
			0,
			self.data.size,
			seglen - hop_size,
			dtype=int
		)
		starts = starts[starts + seglen < self.data.size]
		w = sig.get_window(window, seglen + 1)[:-1]
		comp_size = int(seglen * comp_ratio)
		comp_rate = int(self.rate * comp_ratio)
		xhat = np.zeros_like(self.data, complex)
		wsum = np.zeros_like(xhat)
		rand_idx = np.zeros((len(starts), comp_size), int)
		for i in range(len(rand_idx)):
			rand_idx[i] = rd.choice(
				seglen,
				size=comp_size,
				replace=False
			)
		spars_basis = fft.dct(np.identity(seglen))
		for i,n in enumerate(starts):
			x = w * self.data[n : n + seglen]
			y = x[rand_idx[i]]
			A = spars_basis[rand_idx[i]]
			
			if i == 0:
				prob = skl.LassoCV(
					cv=10,
					random_state=0,
					n_jobs=3
				)
				prob.fit(A, y)
				alpha = prob.alpha_
			
			prob = skl.Lasso(alpha=alpha)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				prob.fit(A, y)
			
			xhat[n : n + seglen] += w * fft.idct(prob.coef_)
			wsum[n : n + seglen] += w**2
		
		self.seglen = seglen
		self.comp_ratio = comp_ratio
		self.percent_overlap = percent_overlap
		self.hop_size = hop_size
		self.starts = starts
		self.w = w
		self.xhat = xhat