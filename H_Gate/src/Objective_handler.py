from Simple_Systems.H_Gate.src.MKM_sys import MKM_sys
import Simple_Systems.H_Gate.src.Config as conf
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
		self.coeffs_A = self.mkm_systems[0].Accardi_coeff
		self.ion_flows_A = [self.mkm_systems[0].ion_flows_A]
		for j in range(1,self.n_sys):
			self.mkm_systems.append(MKM_sys(mkm_configs[j]))
			self.mkm_systems[-1].coeffs = self.coeffs
			self.mkm_systems[-1].Accardi_coeff = self.coeffs_A
			self.ion_flows.append(self.mkm_systems[-1].ion_flows)
			self.ion_flows_A.append(self.mkm_systems[-1].ion_flows_A)
		self.init_no_flow_residual()
		self.flow_alias_dict = self.gen_flow_alias_dict()
		# self.sq_residuals = self.gen_sq_residuals()

	# Initializes flow indicies used for the 
	# method self.no_flow_sqresidual
	def init_no_flow_residual(self):
		if not hasattr(self,'no_flow_sys'):
			return
		mkm_sys = self.mkm_systems[0]
		forward_adj = [[] for j in range(mkm_sys.N_coeffs)]
		for trans_set in mkm_sys.transitions:
			for transition in trans_set:
				if transition[1] < transition[0]:
					forward_adj[transition[1]].append(transition[0])
		forward_trans = []
		for j in range(mkm_sys.N_coeffs):
			N = len(forward_adj[j]) - 1
			for k in range(N):
				forward_trans.append((forward_adj[j][k],j))
		self.forward_inds = np.asarray(forward_trans)
		self.reverse_inds = np.copy(self.forward_inds)
		self.reverse_inds[:,0] = self.forward_inds[:,1]
		self.reverse_inds[:,1] = self.forward_inds[:,0]
		print('No flow constraints: {:d}'.format(
			self.forward_inds.shape[0]))

	# Returns the summed square residual for the no flow systems,
	# which uses the minimum number of independent flow 
	# constraints to establish microscopic reversability.
	# Only using bio since opposite should be identical with 
	# zero gradient.
	def no_flow_sqresidual(self):
		sq_res = 0
		for j in self.no_flow_sys:
			flow_mat = self.mkm_systems[j].flow_mats[0]
			flow_mat_A = self.mkm_systems[j].flow_mats_A[0]
			vals = np.take(flow_mat,self.forward_inds)
			vals -= np.take(flow_mat,self.reverse_inds)

			sq_res += np.sum(vals*vals)
		return sq_res


	# Returns the square residual for residual_parameter k, system j
	def get_sq_residual(self,j,k):
		sq_res = (self.flow_alias_dict[self.residual_params[k]](j) - getattr(self,self.residual_params[k])[j])**2
		return sq_res

	# Returns flow_alias_dict which contains lambda functions for
	# corresponding flow calculations
	def gen_flow_alias_dict(self):
		flow_alias_dict = dict()
		flow_alias_dict['M_net_Cl_flow'] = lambda j:(
			self.ion_flows[j][0][0] + self.ion_flows[j][1][0])
		flow_alias_dict['M_net_H_flow'] = lambda j:(
			self.ion_flows[j][0][1] + self.ion_flows[j][1][1])
		flow_alias_dict['M_opp_Cl_flow'] = lambda j:(
			self.ion_flows[j][1][0])
		flow_alias_dict['M_opp_H_flow'] = lambda j:(
			self.ion_flows[j][1][1])
		flow_alias_dict['M_bio_Cl_flow'] = lambda j:(
			self.ion_flows[j][0][0])
		flow_alias_dict['M_bio_H_flow'] = lambda j:(
			self.ion_flows[j][0][1])
		flow_alias_dict['A_net_Cl_flow'] = lambda j: (
			self.ion_flows_A[j][0][0] + self.ion_flows_A[j][1][0])
		flow_alias_dict['A_net_H_flow'] = lambda j: (
			self.ion_flows_A[j][0][1] + self.ion_flows_A[j][1][1])
		flow_alias_dict['A_opp_Cl_flow'] = lambda j: (
			self.ion_flows_A[j][1][0])
		flow_alias_dict['A_opp_H_flow'] = lambda j: (
			self.ion_flows_A[j][1][1])
		flow_alias_dict['A_bio_Cl_flow'] = lambda j: (
			self.ion_flows_A[j][0][0])
		flow_alias_dict['A_bio_H_flow'] = lambda j: (
			self.ion_flows_A[j][0][1])
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
		for j in range(self.n_sys):
			self.mkm_systems[j].update_all()
			for k in range(self.n_residual_params):
				if math.isnan(self.flow_alias_dict[
					self.residual_params[k]](j)):
					return float('NaN')
				sq_residual = self.get_sq_residual(j,k)
				if not math.isnan(sq_residual):
					objective += sq_residual
					self.opt_csv.append(objective)
		if hasattr(self,'no_flow_sys'):
			objective += self.no_flow_sqresidual()
		return objective