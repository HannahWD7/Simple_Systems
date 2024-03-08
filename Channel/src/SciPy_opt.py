import numpy as np
import scipy.optimize as sopt
from Simple_Systems.Channel.src.Config import read_config
import os.path
import sys

# Exception used to terminate optimizations
class FinishedOpt(Exception):
	pass

# Used as the parent optimizer's sub_opt class for SciPy
class SciPy_opt():
	def __init__(self,opt_parent):
		self.opt_parent = opt_parent
		self.opt_config = self.opt_parent.opt_config
		self.coeffs = opt_parent.coeffs
		self.init_coeffs = np.copy(self.coeffs)
		self.N_coeffs = opt_parent.N_coeffs
		self.bounds = sopt.Bounds(
			self.opt_parent.lbounds,
			self.opt_parent.ubounds,True)
		self.global_method = self.get_method('global_method')
		self.local_method = self.get_method('local_method')
		self.local_options = self.get_options(
			self.opt_config['local_options_file'])
		if 'global_options_file' in self.opt_config:
			self.global_options = self.get_options(
				self.opt_config['global_options_file'])
		self.n_steps = self.opt_config['n_steps']
		self.count = 1
		self.objective = 0
		self.min_obj = float('inf')
		self.min_coeffs = np.copy(self.coeffs)
		self.res = None
		self.yay = float('inf')

	# Returns the objective using self.coeffs
	def evaluate(self):
		self.objective = self.opt_parent.objective_handler.evaluate()
		if self.objective < self.min_obj:
			self.min_obj = self.objective
			np.copyto(self.min_coeffs, self.opt_parent.objective_handler.coeffs)
			print(self.min_obj)
		return self.objective

	# Returns the global or local optimization method string or None
	def get_method(self,method_str):
		if method_str in self.opt_config:
			return self.opt_config[method_str]
		return None

	# Returns a dict parsed from a SciPy options file if it exists
	def get_options(self,filename):
		if os.path.isfile(filename):
			return read_config(filename)[0]
		else:
			return {}

	# Returns the objective using coeffs
	def fun(self,coeffs):
		np.copyto(self.coeffs,coeffs)
		return self.evaluate()

	# Returns the gradient using coeffs
	def jac(self,coeffs):
		np.copyto(self.coeffs,coeffs)
		np.copyto(self.coeffs,self.opt_parent.prev_coeffs)
		self.opt_parent.update_objective()
		self.opt_parent.update_gradient()
		self.opt_parent.prev_objective = self.opt_parent.objective
		return self.opt_parent.gradient

	# Used to update coefficient and objective data for output
	def callback(self,xk,accept=None,context=None,f=None,fun_val=0,status=True,convergence=None):
		if not (self.global_method is None):
			if self.objective < self.min_obj:
				self.min_obj = self.objective
				np.copyto(self.min_coeffs,xk)
		if self.count % self.opt_parent.output_interval == 0:
			self.opt_parent.opt_dat[self.opt_parent.out_count,0] = self.count
			self.opt_parent.opt_dat[self.opt_parent.out_count,1:-1] = xk
			self.opt_parent.opt_dat[self.opt_parent.out_count,-1] = (
				self.objective)
			self.opt_parent.out_count += 1
		self.count += 1

		if self.count == self.n_steps + 1:
			raise FinishedOpt

	# Runs the SciPy optimization
	def run(self):
		if self.global_method is None:
			try:
				self.res = sopt.minimize(
					self.fun,self.init_coeffs,
					method=self.local_method,
					jac=self.jac,
					bounds=self.bounds,
					callback=self.callback,
					options=self.local_options)
			except (FinishedOpt):
				pass
		else:
			opt_fun = getattr(sopt,self.global_method)
			kwargs = dict()
			kwargs['func'] = self.fun
			kwargs['callback'] = self.callback
			kwargs.update(self.global_options)
			if self.global_method == 'basinhopping':
				basinbounds = BasinBounds(self.bounds)
				kwargs['accept_test'] = basinbounds
			elif (self.global_method == 'shgo' or
				self.global_method == 'dual_annealing'):
				kwargs['bounds'] = list(zip(self.opt_parent.lbounds,
					self.opt_parent.ubounds))
			else:
				kwargs['bounds'] = self.bounds
			if (self.global_method =='basinhopping' or
				self.global_method == 'differential_evolution' or
				self.global_method == 'dual_annealing'):
				kwargs['x0'] = self.init_coeffs
			if (self.global_method == 'basinhopping' or
				self.global_method == 'shgo' or
				self.global_method == 'dual_annealing'):
				if self.global_method == 'dual_annealing':
					local_str = 'local_search_options'
				else:
					local_str = 'minimizer_kwargs'
				kwargs[local_str] = dict()
				kwargs[local_str]['bounds'] = self.bounds
				kwargs[local_str]['method'] = self.local_method
				kwargs[local_str]['jac'] = self.jac
				kwargs[local_str]['options'] = self.local_options
			try:
				self.res = opt_fun(**kwargs)
			except (FinishedOpt):
				pass
			np.copyto(self.coeffs,self.min_coeffs)
		if not self.res is None:
			print(self.res)

# Used to define bounds for the 'basinhopping' global optimization
# method
class BasinBounds:
	def __init__(self,bounds):
		self.bounds = bounds
	def __call__(self,**kwargs):
		x = kwargs['x_new']
		tmin = np.all(x >= self.bounds.lb)
		tmax = np.all(x <= self.bounds.ub)
		print ('Bound check:', tmax and tmin)
		return tmax and tmin
