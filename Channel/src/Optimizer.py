import re
import numpy as np
import math
import copy
import pandas as pd
from matplotlib import pyplot as plt
from Simple_Systems.Channel.src.Objective_handler import Objective_handler
import Simple_Systems.Channel.src.Config as conf
import Simple_Systems.Channel.src.Output as outp
from Simple_Systems.Channel.src.Grad_CG import Grad_CG
from Simple_Systems.Channel.src.SciPy_opt import SciPy_opt
from Simple_Systems.Channel.src.SpotPy_opt import SpotPy_opt

# Used to perform optimization for the ClC-ec1 MKM system
class Optimizer():
	def __init__(self,filename):
		self.mkm_configs = conf.read_config_MKM(filename)
		self.opt_config = conf.read_config_OPT(
			self.mkm_configs[0]['opt_config_file'])
		self.objective_handler = Objective_handler(
			self.mkm_configs,self.opt_config)
		self.objective = 0
		self.step = 0
		self.min_delta_prec = 1E-11
		self.max_delta_prec = 1E-8
		self.update_objective()
		self.prev_objective = self.objective
		self.coeffs = self.objective_handler.coeffs
		self.prev_coeffs = np.copy(self.coeffs)
		self.gradient = np.copy(self.coeffs)
		self.N_coeffs = len(self.coeffs)
		self.coeff_deltas = np.copy(self.coeffs)
		self.rel_coeff_deltas = np.full((self.N_coeffs),
			(self.min_delta_prec + self.max_delta_prec)/2)
		self.lbounds = self.objective_handler.mkm_systems[0].lbounds
		self.ubounds = self.objective_handler.mkm_systems[0].ubounds
		self.sub_opt = self.init_sub_opt()
		self.output_interval = self.opt_config['output_interval']
		self.n_steps = self.opt_config['n_steps']
		self.opt_dat = self.init_opt_dat()
		self.out_count = 1
		if 'opt_dat_file' in self.opt_config:
			self.opt_dat_file = self.opt_config['opt_dat_file']
		else:
			self.opt_dat_file = 'opt.dat'
		self.obj_csv = self.objective_handler.opt_csv


	def save_obj(self):
		df = pd.DataFrame(self.obj_csv)
		df.to_csv('objective.csv')

	# Returns an initialized the output data matrix storing
	# coefficients and the objective
	def init_opt_dat(self):
		opt_dat = np.zeros((1+self.n_steps//self.output_interval,
			self.N_coeffs+2))
		opt_dat[0,1:-1] = self.coeffs
		opt_dat[0,-1] = self.objective
		return opt_dat

	# Returns the appropriate sub_opt class using self.opt_config
	def init_sub_opt(self):
		if 'opt_package' in self.opt_config:
			opt_package = self.opt_config['opt_package']
		else:
			opt_package = None
		if opt_package is None:
			sub_opt = Grad_CG(self)
		elif opt_package == 'scipy':
			sub_opt = SciPy_opt(self)
		elif opt_package == 'spotpy':
			sub_opt = SpotPy_opt(self)
		return sub_opt

	# Updates self.gradient, self.rel_coeff_deltas, and
	def update_gradient(self):
		for j in range(self.N_coeffs):
			self.calc_partial(j)
			self.coeffs[j] = self.prev_coeffs[j]
			self.validate_partial(j)

	def rel_to_delta(self,rel_delta,j):
		if self.prev_coeffs[j] == 0:
			return rel_delta
		return rel_delta*abs(self.prev_coeffs[j])

	# Updates self.gradient[j], self.rel_coeff_deltas[j],
	# and self.coeff_deltas[j]
	# Invalidates coeffs[j]
	def calc_partial(self,j):
		'''if j == 0:
			self.debug_plot_partial(j)
			quit()
		'''
		rel_delta = self.rel_coeff_deltas[j]
		delta = self.rel_to_delta(rel_delta,j)
		self.coeffs[j] = self.prev_coeffs[j] + delta
		self.update_objective()
		dObj = self.objective - self.prev_objective
		rel_dObj = abs(dObj)/self.prev_objective
		if rel_dObj < self.min_delta_prec:
			while rel_dObj < self.min_delta_prec:
				if rel_delta > 1E5 and rel_dObj < self.min_delta_prec:
					self.rel_coeff_deltas[j] = rel_delta
					self.coeff_deltas[j] = delta
					self.gradient[j] = dObj/delta
					return
				prev_rel_delta = rel_delta
				rel_delta *= 10
				delta = self.rel_to_delta(rel_delta,j)
				self.coeffs[j] = self.prev_coeffs[j] + delta
				self.update_objective()
				dObj = self.objective - self.prev_objective
				rel_dObj = abs(dObj)/self.prev_objective
			if rel_dObj > self.min_delta_prec and rel_dObj < self.max_delta_prec:
				self.rel_coeff_deltas[j] = rel_delta
				self.coeff_deltas[j] = delta
				self.gradient[j] = dObj/delta
				return
			self.bin_search_delta(prev_rel_delta,rel_delta,rel_dObj,j)
			return
		elif rel_dObj > self.max_delta_prec:
			while rel_dObj > self.max_delta_prec:
				prev_rel_delta = rel_delta
				rel_delta /= 10
				delta = self.rel_to_delta(rel_delta,j)
				if rel_delta < self.min_delta_prec:
					self.finalize_rel_delta(self.min_delta_prec,j)
					return
				self.coeffs[j] = self.prev_coeffs[j] + delta
				self.update_objective()
				dObj = self.objective - self.prev_objective
				rel_dObj = abs(dObj)/self.prev_objective
			if rel_dObj > self.min_delta_prec and rel_dObj < self.max_delta_prec:
				self.rel_coeff_deltas[j] = rel_delta
				self.coeff_deltas[j] = delta
				self.gradient[j] = dObj/delta
				return
			self.bin_search_delta(rel_delta,prev_rel_delta,rel_dObj,j)
			return
		else:
			self.rel_coeff_deltas[j] = rel_delta
			self.coeff_deltas[j] = delta
			self.gradient[j] = dObj/delta
			return

	def finalize_rel_delta(self,rel_delta,j):
		self.rel_coeff_deltas[j] = rel_delta
		delta = self.rel_to_delta(rel_delta,j)
		self.coeff_deltas[j] = delta
		self.coeffs[j] = self.prev_coeffs[j] + delta
		self.update_objective()
		dObj = self.objective - self.prev_objective
		self.gradient[j] = dObj/delta

	def bin_search_delta(self,rel_delta_low,rel_delta_high,rel_dObj,j):
		rel_delta = (rel_delta_low + rel_delta_high) / 2
		delta = self.rel_to_delta(rel_delta,j)
		while rel_dObj < self.min_delta_prec or rel_dObj > self.max_delta_prec:
			self.coeffs[j] = self.prev_coeffs[j] + delta
			self.update_objective()
			dObj = self.objective - self.prev_objective
			rel_dObj = abs(dObj)/self.prev_objective
			if rel_dObj > self.max_delta_prec:
				rel_delta_high = rel_delta
				if rel_delta_high < self.min_delta_prec:
					self.finalize_rel_delta(self.min_delta_prec,j)
					return
			elif rel_dObj < self.min_delta_prec:
				rel_delta_low = rel_delta
			else:
				if rel_delta < self.min_delta_prec:
					self.finalize_rel_delta(self.min_delta_prec,j)
					return
				self.rel_coeff_deltas[j] = rel_delta
				self.coeff_deltas[j] = delta
				self.gradient[j] = dObj/delta
				return
			rel_delta = (rel_delta_low + rel_delta_high) / 2
			delta = self.rel_to_delta(rel_delta,j)

	def debug_plot_partial(self,j):
		N = 30
		d0 = 1E-15
		x = np.zeros((N))
		y1 = np.copy(x)
		y2 = np.copy(x)
		for k in range(N):
			x[k] = d0 * 10 ** k
			delta = x[k]*abs(self.prev_coeffs[j])
			self.coeffs[j] = self.prev_coeffs[j] + delta
			self.update_objective()
			dObj = self.objective - self.prev_objective
			y1[k] = abs(dObj)/self.prev_objective
			y2[k] = abs(dObj)/delta
		plt.plot(x,y1)
		plt.yscale('log')
		plt.xscale('log')
		plt.ylabel('rel_d_obj')
		plt.xlabel('rel_d_coeff {:d}'.format(j))
		plt.plot()
		plt.figure()
		plt.plot(x,y2)
		plt.yscale('log')
		plt.xscale('log')
		plt.xlabel('rel_d_coeff {:d}'.format(j))
		plt.ylabel('|dO/dc|')
		plt.show()

	# Validates gradient entry j againsts bounds
	# NOTE: this invalidates flow matrices, ion flows, and ratios
	def validate_partial(self,j):
		#TODO: maybe remove conditional below for SciPy jac calculations
		if ((self.prev_coeffs[j] == self.lbounds[j] and self.gradient[j] > 0) or
			self.prev_coeffs[j] == self.ubounds[j] and self.gradient[j] < 0):
			self.gradient[j] = 0

	# Updates self.objective using self.coeffs
	def update_objective(self):
		self.objective = self.objective_handler.evaluate()


	# Saves output data
	def save_dat(self):
		header = ('Step\t' + '\t'.join( 
			self.objective_handler.mkm_systems[0].idents) +
			'\tObjective')
		np.savetxt(self.opt_dat_file,self.opt_dat,fmt='%.9e',
			header=header,comments='',delimiter='\t')
		self.objective_handler.mkm_systems[0].export_coeffs(
			'out_coeffs.csv', 'out_coeffs_A.csv')

	# Runs the optimization
	def run(self):
		self.sub_opt.run()
		self.save_dat()
		self.save_obj()
