import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from scipy.fftpack import idct


class qFool():
	def __init__(self, im_orig, net, device, im_target=None, method='non-targeted', tolerance=1e-2,
		epsilon=0.7, max_queries=1000, init_estimation=100, omega0=3*1e-2, delta=10, subspacedim=None):
		self.image = self.transform_input(im_orig)
		self.net = net
		self.net.eval()
		self.device = device

		self.method = method
		self.tolerance = tolerance
		self.epsilon = epsilon
		self.max_queries = max_queries
		self.init_estimation = init_estimation
		self.omega0 = omega0
		self.phi0 = -1
		self.delta = delta

		self.input_shape, self.input_dim, self.input_label = self.process_input_info()
		self.lb_tensor, self.ub_tensor = self.bounds()

		self.subspacedim = subspacedim
		if not self.subspacedim:
			self.subspacedim = self.input_shape[1]

		if method=='targeted':
			self.target = self.transform_input(im_target)
			self.target_label = self.query(self.target)

		self.n = []
		self.separatorline = "-"*50
		self.labels = open(os.path.join(os.getcwd(), '../data','synset_words.txt'), 'r').read().split('\n')


	def transform_input(self, im_orig):
		resizescale, imagesize, mean, std = self.transform_constants()

		# first resize the non-uniform image by "resizescale" to make it uniform
		# next center crop the uniform image by "imagesize" to only keep an important proportion of the image
		# next transform the input to a Tensor, which makes its color range to [0,1]
		# then normalize it by removing the mean and dividing by std
		image = transforms.Compose([
			transforms.Resize(resizescale),
			transforms.CenterCrop(imagesize),
			transforms.ToTensor(),
			transforms.Normalize(mean = mean, std = std)])(im_orig)
		return image


	def inverse_transform(self, x_tensor):
		resizescale, imagesize, mean, std = self.transform_constants()

		xinv = transforms.Compose([transforms.Lambda(lambda x: self.clip_tensor(x)),
			transforms.Normalize(mean=[0 for _ in mean], std=[1/s for s in std]),
			transforms.Normalize(mean=[-m for m in mean], std=[1 for _ in std]),
			transforms.ToPILImage(),
			transforms.CenterCrop(imagesize)])(x_tensor)
		return xinv


	def transform_constants(self):
		# for explanation on these constants, specially "mean" and "std",
		# see "https://pytorch.org/vision/stable/models.html"
		# they are specifically used in "transforms"
		resizescale = 256
		imagesize = 224    # see page 6 of the paper. the alternative is 299
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		return resizescale, imagesize, mean, std


	def process_input_info(self):
		input_shape = self.image.numpy().shape
		input_dim = 1
		for s in input_shape:
			input_dim *= s
		input_label = self.query(self.image)
		return input_shape, input_dim, input_label


	def run(self):
		if self.method == 'non-targeted':
			r, n, label_orig, label_pert, pert_image = self.non_targeted()
		elif self.method == 'targeted':
			r, n, label_orig, label_pert, pert_image = self.targeted()

		str_label_orig = self.labels[label_orig.item()].split(',')[0]
		str_label_pert = self.labels[label_pert.item()].split(',')[0]
		print("Original label = ", str_label_orig)
		print("Perturbed label = ", str_label_pert)

		xadv = self.inverse_transform(pert_image)
		xpertinv = self.inverse_transform(r)

		# plt.figure()
		# plt.imshow(xadv)
		# plt.title(str_label_pert)
		# plt.figure()
		# plt.imshow(xpertinv)
		# plt.show()

		mse = torch.linalg.norm(r) / self.input_dim

		return xadv, xpertinv, mse, str_label_orig, str_label_pert


	def non_targeted(self):
		# line 1
		loop_i = 0

		# line 2
		P = []
		P0, num_calls1 = self.search_P0()
		P0, r, num_calls2 = self.bisect(self.image, P0)
		P.append(P0)

		print(self.separatorline)
		print("starting the loop to find x_adv ...")

		# line 3 and line 14
		x_adv, xi = [], []
		while np.sum(self.n) < self.max_queries:
			print(f"	#iteration: {loop_i}")

			# equation 4
			omega = self.omega0
			phi = self.phi0

			# line 4
			self.n.append(0)

			# line 5
			x_adv.append(P[loop_i])
			x_adv_prime = P[loop_i]

			# line 6 and line 12
			z_dot_eta = 0
			xi.append(0)
			while self.termination_condition(P[loop_i], self.n[loop_i], x_adv_prime, x_adv[loop_i]):
				
				# line 7
				z_dot_eta_new, rho, num_calls3 = self.estimate_gradient(P[loop_i], omega, phi)
				z_dot_eta += z_dot_eta_new
				print(f"     	#queries until now: n[{loop_i}]={self.n[loop_i]+num_calls3}")

				# line 8
				x_adv_prime = x_adv[loop_i]

				# line 9
				self.n[loop_i] += num_calls3		# n[loop_i] += self.init_estimation

				# line 10
				xi[loop_i] = self.normalize(z_dot_eta)

				# line 11
				x_adv[loop_i], num_calls4 = self.search_boundary_in_direction(self.image, xi[loop_i], r)
				x_adv[loop_i], r, num_calls5 = self.bisect(self.image, x_adv[loop_i])

				# equation 4
				omega, phi = self.update_omega(omega, phi, rho)

			print(f"	#queries: n[{loop_i}]={self.n[loop_i]}")

			# line 13
			P.append(x_adv[loop_i])
			loop_i += 1

		print("found x_adv!")

		# line 15
		pert_image = P[-1]
		r_tot = pert_image - self.image
		f_pert = self.query(pert_image)

		return r_tot, np.sum(self.n), self.input_label, f_pert, pert_image	# following deepfool's interface


	def targeted(self):
		# line 1
		loop_i = 0

		# line 2
		v = []
		v0 = self.normalize(self.target-self.image)
		v.append(v0)

		# line 3
		P = []
		P0, r, num_calls1 = self.bisect(self.image, self.target)
		P.append(P0)

		print(self.separatorline)
		print("starting the loop to find x_adv ...")

		# line 4 and line 17
		x_adv, xi, Q = [], [], []
		while np.sum(self.n) < self.max_queries:
			print(f"	#iteration: {loop_i}")

			# equation 4
			omega = self.omega0
			phi = self.phi0

			# line 5
			self.n.append(0)

			# line 6
			x_adv.append(P[loop_i])
			x_adv_prime = P[loop_i]

			# line 7 and line 15
			z_dot_eta = 0
			xi.append(0)
			Q.append(0)
			v.append(0)
			while self.termination_condition(P[loop_i], self.n[loop_i], x_adv_prime, x_adv[loop_i]):
				
				# line 8
				z_dot_eta_new, rho, num_calls2 = self.estimate_gradient(P[loop_i], omega, phi)
				z_dot_eta += z_dot_eta_new
				print(f"     	#queries until now: n[{loop_i}]={self.n[loop_i]+num_calls2}")

				# line 9
				x_adv_prime = x_adv[loop_i]

				# line 10
				self.n[loop_i] += num_calls2		# n[loop_i] += self.init_estimation

				# line 11
				xi[loop_i] = self.normalize(z_dot_eta)

				# line 12
				Q[loop_i], num_calls3 = self.search_boundary_in_direction(P[loop_i], xi[loop_i], self.delta)

				# line 13
				v[loop_i+1] = self.normalize(Q[loop_i]-self.image)

				# line 14
				x_adv[loop_i], num_calls3 = self.search_boundary_in_direction(self.image, v[loop_i+1], r)
				x_adv[loop_i], r, num_calls4 = self.bisect(self.image, x_adv[loop_i])

				# equation 4
				omega, phi = self.update_omega(omega, phi, rho)

			print(f"	#queries: n[{loop_i}]={self.n[loop_i]}")

			# line 16
			P.append(x_adv[loop_i])
			loop_i += 1

		print("found x_adv!")

		# line 15
		pert_image = P[-1]
		r_tot = pert_image - self.image
		f_pert = self.query(pert_image)

		return r_tot, np.sum(self.n), self.input_label, f_pert, pert_image	# following deepfool's interface


	def search_P0(self):
		P0 = deepcopy(self.image)
		sigma0, k = self.P0_constants()
		sigma = sigma0
		num_calls = 0

		print(self.separatorline)
		print("searching for the boundary using random noise ...")

		while not self.is_adversarial(P0) and num_calls < k:
			rj = self.create_perturbation() * sigma
			P0 = self.add_noise(self.image, rj)
			num_calls += 1
			sigma = sigma0 * (num_calls+1)

			print(f"	sigma_{num_calls} = {sigma}", end='')
			print(f"	pert norm = {torch.linalg.norm(rj)}")
			# print(f"	pert mse = {1/self.input_dim*torch.linalg.norm(rj)}")

		if num_calls == k:
			print("cannot find P0!")
		else:
			print("found P0!")
		return P0, num_calls


	def P0_constants(self):
		sigma0 = 0.02
		k = 100
		return sigma0, k


	def termination_condition(self, Pi, ni, x_adv_prime, x_adv_i):
		cond1 = np.sum(self.n) < self.max_queries

		# note that "ni == 0" is redundant, since "x_adv_prime == Pi" includes that
		if torch.linalg.norm(x_adv_prime-Pi) == 0 or ni == 0:
			return cond1

		newrate = torch.linalg.norm(x_adv_i-Pi) / (ni + self.init_estimation)
		prevrate = torch.linalg.norm(x_adv_prime-Pi) / ni

		print(f"newrate/prevrate = {newrate/prevrate}")
		cond2 = newrate <= self.epsilon * prevrate
		return cond1 and cond2


	def estimate_gradient(self, Pi, omega, phi):
		z_dot_eta, cnt, num_calls = 0, 0, 0

		for _ in range(self.init_estimation):
			eta = self.create_perturbation()
			eta = self.normalize(eta) * omega
			P_eta = self.add_noise(Pi, eta)
			if self.is_adversarial(P_eta):
				z = 1
				cnt += 1
			else:
				z = -1
			z_dot_eta += z * eta
			num_calls += 1
		
		rho = 0.5 - cnt / self.init_estimation
		print(f"rho = {rho}")
		# num_calls = self.init_estimation
		return z_dot_eta, rho, num_calls


	def update_omega(self, omega, phi, rho):
		if rho > 0:
			new_phi = -phi
		else:
			new_phi = phi
		new_omega = omega * (1+phi*rho)
		return new_omega, new_phi


	def bisect(self, image, adv_image):
		x = deepcopy(image)
		x_tilde = deepcopy(adv_image)
		num_calls = 0

		print("bisecting to get closer to boundary ...")
		while torch.linalg.norm(x - x_tilde) > self.tolerance:
			x_mid = (x + x_tilde)/2
			print(f"	norm from image = {torch.linalg.norm(x_mid - image)}, ", end='')
			print(f"norm from adv = {torch.linalg.norm(x_mid - adv_image)}")
			# print(f"	mse from image = {1/self.input_dim*torch.linalg.norm(x_mid - image)}, ", end='')
			# print(f"mse from adv = {1/self.input_dim*torch.linalg.norm(x_mid - adv_image)}")

			if self.is_adversarial(x_mid):
				x_tilde = x_mid
			else:
				x = x_mid
			num_calls += 1
		print("bisection done!")
		return x_tilde, torch.linalg.norm(x_tilde - image), num_calls


	def create_perturbation(self):
		pert_tensor = torch.zeros_like(self.image)

		# added after implementing subspaces
		subspace_shape = (self.input_shape[0], self.subspacedim, self.subspacedim)
		pert_tensor[:, :self.subspacedim, :self.subspacedim] = torch.randn(size=subspace_shape)

		if self.subspacedim < self.input_shape[1]:
			pert_tensor = torch.from_numpy(idct(idct(pert_tensor.numpy(), axis=2, norm='ortho'),
				axis=1, norm='ortho'))
			
		return pert_tensor


	def search_boundary_in_direction(self, image, xi, r):
		x_adv = self.add_noise(image, xi*r)
		num_calls = 1
		while not self.is_adversarial(x_adv):
			num_calls += 1
			x_adv = self.add_noise(image, xi*num_calls*r)
		return x_adv, num_calls


	def add_noise(self, image, pert):
		x_tilde = image + pert
		return self.clip_tensor(x_tilde)


	def normalize(self, vec):
		return vec / torch.linalg.norm(vec)


	def query(self, image):
		x = deepcopy(image)
		if len(x) == 3:
			x = x[None, :]
		output = self.net(x)
		top1_label = output.max(1, keepdim=True)[1]
		return top1_label


	def is_adversarial(self, x):
		top1_label = self.query(x)
		if self.method == 'non-targeted':
			return not top1_label.item() == self.input_label.item()
		elif self.method == 'targeted':
			return top1_label.item() == self.target_label.item()


	def bounds(self):
		zeros = transforms.ToPILImage()(torch.zeros_like(self.image))
		fulls = transforms.ToPILImage()(torch.ones_like(self.image))

		lb_tensor = self.transform_input(zeros)
		ub_tensor = self.transform_input(fulls)
		return lb_tensor, ub_tensor


	def clip_tensor(self, x):
		xc = torch.maximum(x, self.lb_tensor)
		xc = torch.minimum(xc, self.ub_tensor)
		# xc = torch.maximum(x, torch.zeros_like(self.image))
		# xc = torch.minimum(xc, 255*torch.ones_like(self.image))
		return xc