from Simple_Systems.Channel.src.MKM_sys import MKM_sys
import Simple_Systems.Channel.src.Config as conf
import numpy as np
import math

# Used to handle objective calculation
class Objective_handler():
	def __init__(self,mkm_configs,opt_config):
		self.n_sys = len(mkm_configs)
		self.residual_params = self.gen_residual_params(
			opt_config['opt_residuals_file'])
		self.n_residual_params = len(self.residual_params)
		self.mkm_systems = []
		self.opt_csv = []
		self.mkm_systems.append(MKM_sys(mkm_configs[0]))
		self.coeffs = self.mkm_systems[0].coeffs
		self.ion_flows = [self.mkm_systems[0].ion_flows]
		self.flow_matrix = [self.mkm_systems[0].flow_mats]
		for j in range(1,self.n_sys):
			self.mkm_systems.append(MKM_sys(mkm_configs[j]))
			self.mkm_systems[-1].coeffs = self.coeffs
			self.ion_flows.append(self.mkm_systems[-1].ion_flows)
			self.flow_matrix.append(self.mkm_systems[-1].flow_mats)
		self.flow_alias_dict = self.gen_flow_alias_dict()
		self.constant1 = 1.99E-12
		self.constant2 = 1.6027E-19
		# self.sq_residuals = self.gen_sq_residuals()

	# Initializes flow indicies used for the 
	# method self.no_flow_sqresidual
	def no_flow_sqresidual(self):
		sq_res = 0
		value = 0
		sumation = 0
		number = 0
		for j in self.no_flow_sys:
			bio_flow_mat = self.mkm_systems[j].flow_mats[0]
			bio_flow_Cl = abs(self.ion_flows[j][0][0])
			for k in range(self.mkm_systems[j].N_states):
				for l in range(self.mkm_systems[j].N_states):
					if (bio_flow_mat[k][l] == 0) :
						pass
					else:
						value += (abs(bio_flow_mat[k][l]/self.constant1*self.constant2*1000) - abs(bio_flow_mat[l][k]/self.constant1*self.constant2*1000))**2
						sq_res += (value) / 2   # Divided by 2 to not account for more than 1 fwd/rev pair, summation to get avg difference
						number += 1 # because we are accounting for 1 pair per orientation
						sumation = 0
						value = 0
			sq_res /= number #To normalize it to the number of terms that we are looking at
		print('Reversibility')
		print(sq_res)
		return sq_res

	# Returns the summed square residual for the no flow systems,
	# which uses the minimum number of independent flow 
	# constraints to establish microscopic reversability.
	# Only using bio since opposite should be identical with 
	# zero gradient.


	# Returns the square residual for residual_parameter k, system j
	def get_sq_residual(self,j,k):
		sq_res = 0
		user_input = getattr(self, self.residual_params[k])[j]
		code_generated = self.flow_alias_dict[self.residual_params[k]](j)
		if getattr(self, self.residual_params[k])[j] > 1E99:
			sq_res = 0
		else:
			print('sq_res {}'.format(j))
			print(user_input)
			print(code_generated)
			sq_res = (code_generated - user_input) ** 2
			print(sq_res)
		return sq_res

	# Returns flow_alias_dict which contains lambda functions for
	# corresponding flow calculations
	def gen_flow_alias_dict(self):
		flow_alias_dict = dict()
		flow_alias_dict['Net_Cl_flow'] = lambda j:(
			self.ion_flows[j][0][0] + self.ion_flows[j][1][0])
		flow_alias_dict['Net_H_flow'] = lambda j:(
			self.ion_flows[j][0][1] + self.ion_flows[j][1][1])
		flow_alias_dict['Opp_Cl_flow'] = lambda j:(
			self.ion_flows[j][1][0])
		flow_alias_dict['Opp_H_flow'] = lambda j:(
			self.ion_flows[j][1][1])
		flow_alias_dict['Bio_Cl_flow'] = lambda j:(
			(self.ion_flows[j][0][0]/self.constant1)*self.constant2*1000)
		flow_alias_dict['Bio_H_flow'] = lambda j:(
			self.ion_flows[j][0][1])
		return flow_alias_dict

	# Returns the residual_params list of strings.
	# Also initializes the corresponding attribute's
	# systems vector
	def gen_residual_params(self,filename):
		residual_params = []
		self.res_configs = conf.read_config_RES(
			filename,self.n_sys)
		for key in self.res_configs[0]:
			if key != 'no_flow_sys':
				vec = np.zeros((self.n_sys))
				for j in range(self.n_sys):
					vec[j] = self.res_configs[j][key]
				setattr(self,key,vec)
				residual_params.append(key)
			else:
				vec = []
				for j in range(self.n_sys):
					if self.res_configs[j][key]:
						vec.append(j)
				if len(vec) > 0:
					setattr(self,key,vec)
		return residual_params

	# Returns the objective using the specified flow
	# residuals and self.coeffs
	def evaluate(self):
		objective = 0
		values = []
		no_flow_system = int
		for j in range(self.n_sys):
			if hasattr(self, 'no_flow_sys'):
				no_flow_system = self.no_flow_sys[0]
		for j in range(self.n_sys):
			self.mkm_systems[j].update_all()
		for j in range(self.n_sys):
			for k in range(self.n_residual_params):
				if math.isnan(self.flow_alias_dict[
								  self.residual_params[k]](j)):
					return float('NaN')
				sq_residual = self.get_sq_residual(j, k)
				if not math.isnan(sq_residual):
					objective += sq_residual
		if hasattr(self, 'no_flow_sys'):
			objective += self.no_flow_sqresidual()
		print(objective)
		return objective