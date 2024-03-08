import numpy as np
from matplotlib import pyplot as plt
# import src.Output as outp

# Used as the parent optimizer's sub_opt class for Grad_CG
class Grad_CG():
	def __init__(self,opt_parent):
		self.opt_parent = opt_parent
		self.objective_handler = opt_parent.objective_handler
		self.min_delta_prec = opt_parent.min_delta_prec
		self.max_delta_prec = opt_parent.max_delta_prec
		self.debug_num = 0
		self.step_num = 0
		self.N_coeffs = opt_parent.N_coeffs
		self.init_opt_params(opt_parent.opt_config)
		self.cg_max_count = 69
		self.cg_count = 0
		self.alpha = 0
		self.is_steepest = True
		self.limp_steps = 0
		self.coeffs = opt_parent.coeffs
		self.descent_vec = np.copy(self.coeffs)
		self.prev_descent_vec = np.copy(self.coeffs)
		self.mimp = (self.limp + self.uimp) / 2
		self.improvement = 0
		# self.prev_improvement = 0
		self.objective = 0
		self.update_objective()
		self.prev_objective = self.objective
		self.gradient = opt_parent.gradient
		self.prev_gradient = np.copy(self.gradient)
		self.prev_coeffs = opt_parent.prev_coeffs
		self.coeff_deltas = opt_parent.coeff_deltas
		self.lbounds = np.copy(opt_parent.lbounds)
		self.dalpha = (opt_parent.min_delta_prec + 
			opt_parent.max_delta_prec)/2

	#Initializes class variables from opt_config
	def init_opt_params(self,opt_config):
		for param in opt_config:
			setattr(self,param,opt_config[param])
		n_rows = self.n_steps//self.output_interval + 1
		self.data = np.zeros((n_rows,8))
		self.coeff_trajs = np.zeros((
			n_rows,self.N_coeffs + 1))

	#returns the Fletcher beta for CG descent
	def update_beta(self):
		return (np.dot(self.gradient,self.gradient)/
				np.dot(self.prev_gradient,self.prev_gradient))
	
	# Updates the self.descent_vec using the gradient for 
	# Gradient or CG descent
	def update_descent_vec(self):
		self.opt_parent.update_gradient()
		constrained_descent = False
		if (not self.is_steepest and self.step_num != 0 and 
			self.cg_count != self.cg_max_count):
			beta = self.update_beta()
			if beta < 0:
				self.descent_vec = np.copy(self.gradient)
				self.cg_count = 0
			else:
				self.descent_vec = (beta * self.prev_descent_vec +
					self.gradient)
				for j in range(self.N_coeffs):
					if (self.prev_coeffs[j] == self.lbounds[j] and
						self.descent_vec[j] > 0):
						self.descent_vec[j] = 0
						constrained_descent = True
				if constrained_descent:
					self.cg_count = self.cg_max_count
				else:
					self.cg_count += 1
		else:
			self.descent_vec = np.copy(self.gradient)
			self.cg_count = 0

	# Returns the objective's improvement from the previous step
	def calc_improvement(self):
		return ((self.prev_objective - self.objective)/
			self.prev_objective)

	# Updates self.objective using self.coeffs
	def update_objective(self):
		self.opt_parent.update_objective()
		self.objective = self.opt_parent.objective

	# Used to analyze improvement vs alpha for debugging
	def debug_plot(self):
		N = 140
		minexp = -15
		factor=0.067
		alphas = np.zeros(N)
		improvs = np.zeros(N)
		for j in range(N):
			self.alpha = 10 ** (j*factor+minexp) 
			alphas[j] = self.alpha
			improvs[j] = self.step_alpha()
		self.alpha = 10 ** (199*0.1-15)
		self.step_alpha()
		plt.plot(alphas,improvs,label='for.')
		# plt.xlim([1E-6,7E0])
		# plt.ylim([-1E-3,1.1E-3])
		# plt.xscale('log')
		for j in range(N):
			self.alpha = -10 ** (j*factor+minexp) 
			alphas[j] = -self.alpha
			improvs[j] = self.step_alpha()
		plt.plot(alphas,improvs,label='rev.')
		plt.ylabel('Improvement')
		plt.xlabel('alpha')
		plt.legend()
		plt.savefig('I_vs_a.png')

	# Updates self.coeffs by stepping along self.descent_vec using
	# stepsize self.alpha
	def step_alpha(self):
		alpha_descent_vec = self.alpha*self.descent_vec
		np.subtract(self.prev_coeffs,alpha_descent_vec,
			out=self.coeffs
		)
		self.update_objective()
		return self.calc_improvement()

	# Performs a linesearch along self.descent_vec to select
	# stepsize self.alpha using the target improvment range
	# [limp,uimp]
	# TODO: potentially improve dalpha determination
	def search_alpha(self):
		dalpha = self.get_dalpha()
		if self.objective - self.prev_objective > 0:
			self.dalpha = self.min_delta_prec
			self.alpha = -self.min_delta_prec
			self.step_alpha()
		elif self.is_steepest:
			self.search_alpha_SD(dalpha)
		else:
			self.search_alpha_CG(dalpha)
		self.validate_bounds_alpha()


	def validate_bounds_alpha(self):
		lower_violated_indices = np.argwhere(
			self.coeffs < self.lbounds)
		if(len(lower_violated_indices) > 0):
			min_alpha = float('inf')
			for js in lower_violated_indices:
				for j in js:
					# if self.step_num == self.debug_num:
					# 	print(j,self.prev_coeffs[j],self.coeffs[j])
					self.alpha = ((self.prev_coeffs[j] - self.lbounds[j])/
						self.descent_vec[j])
					if self.alpha < min_alpha:
						min_alpha = self.alpha
						min_index = j
			self.alpha = min_alpha
			self.improvement = self.step_alpha()
			self.coeffs[min_index] = self.lbounds[min_index]
			# self.limp_steps = 0
			# print("Constraining coefficient {:d}".format(j))
			# quit()
		if self.is_steepest and self.improvement < self.limp:
			self.limp_steps += 1
			if self.limp_steps > self.max_limp_steps:
				self.is_steepest = False
				print('Switching to conjugate gradient after step',self.step_num)
		else:
			self.limp_steps = 0

	def search_alpha_SD(self,dalpha):
		# if self.step_num == self.debug_num:
			# self.debug_plot()
			# quit()
		a_0 = 0
		I_0 = 0
		max_reset_iter = 8
		reset_iter = 0
		newt_iter = 0
		max_newt_iter = 1
		max_dI_iter = 10
		dI_iter = 0
		limp = self.limp
		uimp = self.uimp
		mimp = self.mimp
		dalpha0 = dalpha
		while True:
			self.alpha = a_0 + dalpha
			dI_da = (self.step_alpha() - I_0)/dalpha
			# if self.step_num == self.debug_num:
			# 	print(dalpha,dI_da,reset_iter)
				# quit()
			if (reset_iter == 0 and newt_iter == 0):
				while (dI_da <= 0 and dI_iter < max_dI_iter):
					dalpha /= 10
					dI_da = (self.step_alpha() - I_0)/dalpha
					dI_iter += 1
				if (dI_iter == max_dI_iter):
					dI_iter = 0
					dalpha = dalpha0
					while (dI_da <= 0 and dI_iter < max_dI_iter):
						dalpha *= 10
						dI_da = (self.step_alpha() - I_0)/dalpha
						dI_iter += 1
					if (dI_iter == max_dI_iter):
						self.alpha = dalpha0
						I_0 = self.step_alpha()
						break
			self.alpha = a_0 + (mimp - I_0)/dI_da
			I_0 = self.step_alpha()
			a_0 = self.alpha
			# if self.step_num == self.debug_num:
			# 	print(a_0,I_0,dI_da,dalpha)
			# 	quit()
			if I_0 > limp and I_0 < uimp:
				break
			if (a_0 <= 0 or I_0 <= 0 or 
				newt_iter == max_newt_iter):
				newt_iter = 0
				if reset_iter == max_reset_iter:
					self.alpha = dalpha0
					I_0 = self.step_alpha()
					break
				limp /= 10
				uimp /= 10
				mimp /= 10
				a_0 = 0
				I_0 = 0
				reset_iter += 1
			else:
				newt_iter += 1
		self.improvement = I_0

	def search_alpha_CG(self,dalpha):
		prev_alpha = 0
		self.alpha = 0
		dalpha2 = dalpha*dalpha
		prev_o = self.prev_objective
		cur_o = self.prev_objective
		cur_alpha = 0
		dprint=False
		while True:
			obj_F = self.objective
			dOda = (obj_F - cur_o)/dalpha
			self.alpha = cur_alpha - dalpha
			self.step_alpha()
			d2Oda2 = (self.objective - 2*cur_o + obj_F)/dalpha2
			if d2Oda2 < self.min_delta_prec:
				self.cg_count = self.cg_max_count
				self.search_alpha_SD(dalpha)
				return
			prev_alpha = self.alpha
			self.alpha = prev_alpha - dOda/d2Oda2
			cur_alpha = self.alpha
			self.step_alpha()
			prev_o = cur_o
			cur_o = self.objective
			if abs(cur_o - prev_o)/prev_o < 1E-3:
				return
			self.alpha = cur_alpha + dalpha
			self.step_alpha()

	def get_dalpha(self):
		dalpha = self.dalpha
		self.alpha = dalpha
		rel_dObj = self.step_alpha()
		# if rel_dObj < 0:
		# 	return 0
		if rel_dObj < self.min_delta_prec:
			while rel_dObj < self.min_delta_prec:
				if dalpha > 1E5 and rel_dObj < self.min_delta_prec:
					return self.min_delta_prec
				prev_dalpha = dalpha
				dalpha *= 10
				self.alpha = dalpha
				rel_dObj = self.step_alpha()
			if rel_dObj > self.min_delta_prec and rel_dObj < self.max_delta_prec:
				return dalpha
			return self.bin_search_dalpha(prev_dalpha,dalpha,rel_dObj)
		elif rel_dObj > self.max_delta_prec:
			while rel_dObj > self.max_delta_prec:
				prev_dalpha = dalpha
				dalpha /= 10
				if dalpha < self.min_delta_prec:
					return self.min_delta_prec
				self.alpha = dalpha
				rel_dObj = self.step_alpha()
			if rel_dObj > self.min_delta_prec and rel_dObj < self.max_delta_prec:
				return dalpha
			return self.bin_search_dalpha(dalpha,prev_dalpha,rel_dObj)
		else:
			return dalpha

	def bin_search_dalpha(self,dalpha_low,dalpha_high,rel_dObj):
		dalpha = (dalpha_low + dalpha_high) / 2
		while rel_dObj < self.min_delta_prec or rel_dObj > self.max_delta_prec:
			self.alpha = dalpha
			rel_dObj = self.step_alpha()
			if rel_dObj > self.max_delta_prec:
				dalpha_high = dalpha
				if dalpha_high < self.min_delta_prec:
					return self.min_delta_prec
			elif rel_dObj < self.min_delta_prec:
				dalpha_low = dalpha
			else:
				if dalpha < self.min_delta_prec:
					return self.min_delta_prec
				return dalpha
			dalpha = (dalpha_low + dalpha_high) / 2

	# Used to update coefficient and objective data for output
	# Also prints some info to the terminal during optimization
	def process_data_output(self):
		if self.step_num % self.output_interval == 0:
			if self.step_num != 0:
				mkm_sys = self.objective_handler.mkm_systems[0]
				self.opt_parent.opt_dat[self.opt_parent.out_count,0] = self.step_num
				self.opt_parent.opt_dat[self.opt_parent.out_count,1:-1] = self.coeffs
				self.opt_parent.opt_dat[self.opt_parent.out_count,-1] = (
					self.objective)
				self.opt_parent.out_count += 1
			# self.data[j,:] = ([self.step_num,self.alpha,
			# 	self.objective,mkm_sys.ratio] + 
			# 	list(mkm_sys.ion_flows[0]) + 
			# 	list(mkm_sys.ion_flows[1]))
			# self.coeff_trajs[j,0] = self.step_num
			# self.coeff_trajs[j,1:] = np.copy(self.coeffs)
			print('{:<4d} obj: {:<6.1E} alpha: {:<6.1E} improvement: {:<6.1E}'.format(
					self.step_num, self.objective, 
					self.alpha, self.improvement
				))

	# Sets the prevous optimization variables to the current
	def finalize_step(self):
		np.copyto(self.prev_coeffs,self.coeffs)
		np.copyto(self.prev_gradient,self.gradient)
		np.copyto(self.prev_descent_vec,self.descent_vec)
		np.copyto(self.coeffs,self.prev_coeffs)
		# self.update_objective()
		self.prev_objective = self.objective
		self.opt_parent.prev_objective = self.objective
		self.step_num += 1

	# Performs an optimization step
	def step(self):
		self.update_descent_vec()
		self.search_alpha()
		self.finalize_step()
		return

	# Runs the Grad_CG optimization
	def run(self):
		j = 0
		self.process_data_output()
		while self.step_num < self.n_steps:
			self.step()
			self.process_data_output()
		# mkm_sys.export_coeffs('out_coeffs.csv')
		# outp.print_mat(self.data,'output.dat')
		# outp.print_mat(self.coeff_trajs,'coeff_trajs.dat')
		# outp.print_opt_summary('opt_summary.txt',self)