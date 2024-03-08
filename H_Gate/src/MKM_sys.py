import re
import numpy as np
import math
import copy
from matplotlib import pyplot as plt
import csv
import sys
from numpy.linalg import inv, det
import fractions
from decimal import Decimal

np.set_printoptions(threshold=sys.maxsize)
# Used to represent the ClC-ec1 MKM system
# enzyme surface concentration can be quantified and used to convert the
# calculated pseudo first-order rate constant
# to a second-order rate constant.
class MKM_sys():
	def __init__(self, config):
		self.surf_conc_mode = 'uniform'
		self.flow_transitions = self.gen_flow_transitions()
		self.map_states()
		self.concs = self.get_concs('Miller', config)
		self.concs_A = self.get_concs('Accardi', config)
		self.Voltage = float(config['Voltage'])
		self.lbounds = []
		self.ubounds = []
		self.coeffs_dict = self.get_coeffs_dict(config)
		self.init_coeffs = np.array(
			self.dict_to_list(self.coeffs_dict))
		self.idents = list(self.coeffs_dict)
		self.transitions = self.gen_transitions(config)
		self.n_uptakes = np.zeros((2,2),dtype=int)
		self.n_releases = np.zeros((2,2),dtype=int)
		self.uptake_transitions = self.gen_surf_transitions(True)
		self.release_transitions = self.gen_surf_transitions(False)
		self.coeffs = np.copy(self.init_coeffs)
		self.Accardi_coeff = np.copy(self.init_coeffs)
		self.N_coeffs = len(self.coeffs)
		self.N_sites = 4
		self.N_states = 2**self.N_sites
		self.N_orient = 2
		self.delta_F_mats = np.copy(self.coeffs)
		self.new_f_mats = np.copy(self.delta_F_mats)
		self.coeff_mats = np.zeros((self.N_orient,self.N_states,
			self.N_states))
		self.coeff_mats_A = np.zeros((self.N_orient, self.N_states,
			self.N_states))
		self.As = np.copy(self.coeff_mats)
		self.As[:,-1,:] = 1
		self.As_A = np.copy(self.coeff_mats_A)
		self.As_A[:, -1, :] = 1
		self.b = np.zeros([self.N_states,1])
		self.b[-1] = 1
		self.b_A = np.zeros([self.N_states, 1])
		self.b_A[-1] = 1
		self.ss_pops = np.zeros((self.N_states,self.N_orient))
		self.ss_pops_A = np.zeros((self.N_states, self.N_orient))
		self.flow_mats = np.copy(self.coeff_mats)
		self.flow_mats_A = np.copy(self.coeff_mats_A)
		self.ion_flows = np.zeros((2,2))
		self.ion_flows_A = np.zeros((2,2))
		self.ratio = 0
		self.expanded_coeff = np.zeros(2)
		self.expanded_coeff_A = np.zeros(2)
		if self.surf_conc_mode == 'uniform':
			self.surf_concs = np.copy(self.concs)
			self.surf_concs_A = np.copy(self.concs_A)
		else:
			self.surf_concs = np.zeros((2,2))
			self.surf_concs_A = np.zeros((2,2))
		self.update_all()
		self.population_analysis()
		self.coeffs_dict_V = self.calc_coeff_dict2()
		self.save_dat()

	def population_analysis(self):
		N = 16
		states = {}
		count = 0
		number = []
		state = []
		pop_bio = []
		pop_opp = []
		pop_bio_A = []
		pop_opp_A = []
		for j in range(N):  # rate matrix 1
			for k in range(N):  # rate matrix 2
				s1 = self.decode(j)
				if s1 not in states:
					states[s1] = count
					count += 1
				else:
					pass

		for key in states:
			state.append(key)
			number.append(states[key])
		for n in range(len(state)):
			pop_bio.append(self.ss_pops[n][0])
			pop_bio_A.append(self.ss_pops_A[n][0])
		with open('state_file_Miller.csv', 'w', newline='') as myfile:
			writer = csv.writer(myfile)
			writer.writerow(["State_No", "State_Binary", " State_Pop_Bio"])
			for num in number:
				writer.writerow([number[num], state[num], pop_bio[num]])
			myfile.close()
		with open('state_file_Accardi.csv', 'w', newline='') as myfile2:
			writer = csv.writer(myfile2)
			writer.writerow(["State_No", "State_Binary", " State_Pop_Bio"])
			for num in number:
				writer.writerow([number[num], state[num], pop_bio_A[num]])
			myfile2.close()

	def calc_coeff_dict2(self):
		coeffs_dict = dict()
		j = 0
		for identifier in self.coeffs_dict:
				coeffs_dict[identifier] = self.Accardi_coeff[j]
				j += 1
		return coeffs_dict

	# Writes self.coeffs to filename
	def export_coeffs(self,filename,filename2):
		bufflist = ['"'+'","'.join(self.idents)+'"\n']
		bufflist2 = ['"'+'","'.join(self.idents)+'"\n']
		for coeff in self.coeffs:
			bufflist.append('{:.15e}'.format(
				coeff
			))
			bufflist.append(',')
		bufflist[-1] = '\n'
		for coeff in self.Accardi_coeff:
			bufflist2.append('{:.15e}'.format(
				coeff
			))
			bufflist2.append(',')
		bufflist[-1] = '\n'
		for bound in self.lbounds:
			bufflist.append('{:.15e}'.format(
				bound
			))
			bufflist.append(',')
		bufflist[-1] = '\n'
		for bound in self.ubounds:
			bufflist.append('{:.15e}'.format(
				bound
			))
			bufflist.append(',')
		bufflist[-1] = '\n'
		with open(filename,'w') as myfile:
			myfile.write(''.join(bufflist))
		with open(filename2,'w') as myfile2:
			myfile2.write(''.join(bufflist2))

	# Updates all MKM system properties of self
	def update_all(self):
		self.update_coeffs()
		self.update_ss_pops()
		self.update_flow_mats()
		self.update_ion_flows()
		self.update_ratio()

	def update_coeffs(self):
		self.Accardi_coeff = self.get_new_coeff()

	# Updates self.flow_mats using self.ss_pops
	def update_flow_mats(self):
		np.copyto(self.flow_mats,self.coeff_mats)
		np.copyto(self.flow_mats_A,self.coeff_mats_A)
		for j in range(self.N_orient):
			np.fill_diagonal(self.flow_mats[j], 0)
			np.fill_diagonal(self.flow_mats_A[j], 0)
			for k in range(self.N_states):
				self.flow_mats[j,:,k] *= self.ss_pops[k,j] /2
				self.flow_mats_A[j, :, k] *= self.ss_pops_A[k, j] /2

	# Updates self.ss_pops using self.coeffs
	# TODO: allow repeated iteration calculations
	def update_ss_pops(self):
		niter = 1
		for j in range(niter):
			self.update_surf_concs()
			self.update_coeff_mats()
			self.iterate_ss_pops()

	# Updates self.ss_pops using self.coeffs via 1 iteration
	def iterate_ss_pops(self):
		for j in range(self.N_orient):
			try:
				self.ss_pops[:,j] = np.ravel(
					np.linalg.solve(self.As[j],self.b))
				self.ss_pops_A[:, j] = np.ravel(
					np.linalg.solve(self.As_A[j], self.b_A))
			except np.linalg.LinAlgError as e:
				if str(e) == 'Singular matrix':
					print('Warning: Singular matrix encountered. Invalidating populations.')
					self.ss_pops[:,j] = float('NaN')


	# Returns tuple containing surface transition indices
	# with format (location=0(int) or 1 (ext),ion=0(Cl) or 1(H))
	def get_surf_indices(self, transition, uptake):
		if uptake:
			j1 = '0'
			j2 = '1'
		else:
			j1 = '1'
			j2 = '0'
		keys = [
			(1,0,1), # (binary_index, adjacent_index, ion(0=Cl,1=H))  # E148
			(2,3,0), # Sout
			(3,2,0)  # Sint
		]
		s1 = self.decode(transition[0])
		s2 = self.decode(transition[1])
		for key in keys:
			if (s1[key[0]] == j1 and s2[key[0]] == j2):
				if int(s1[0]) > key[1] and key == keys[0]:
					return (1,key[2])
				elif (int(s1[0]) == key[1] and key == keys[0]):
					return (0,key[2])
				elif (key[1] > key[0] and key != keys[0] and s1[key[1]] == s2[key[1]]):
					return (1,key[2])
				elif (key[1] < key[0] and key != keys[0] and s1[key[1]] == s2[key[1]]):
					return (0,key[2])
		return tuple()

	# Generates surface transitions dict
	# with format [[Cl_int,H_int],[Cl_ext,H_ext]], this does not take into consideration orientation
	def gen_surf_transitions(self,uptake):
		surf_transitions = [[dict(),dict()],[dict(),dict()]]
		for j,transitions in enumerate(self.transitions):
			indices = self.get_surf_indices(transitions[0],uptake)
			if len(indices) != 0:
				surf_transitions[indices[0]][indices[1]][j] = transitions  #Mapping which surf_conc goes to which tuple
				# transition pair
				if uptake:
					self.n_uptakes[indices[0]][indices[1]] += len(transitions) #counting how many transitions per tuple pair
				else:
					self.n_releases[indices[0]][indices[1]] += len(transitions)
		return surf_transitions

	# Updates surface concentrations
	def update_surf_concs(self):
		if self.surf_conc_mode == 'uniform':
			return

	# Updates the intrinsic coefficients to uptake/release coefficients by multiplying by the appropriate concentration
	# produces a matrix that can be used to describe the int/ext_uptake Cl/H intrinsic rates by the intra and extra side
	# to produce a matrix that is effectively [bio, opposite]
	def update_expanded_coeff(self,j):
		self.expanded_coeff_A[:] = self.Accardi_coeff[j]
		self.expanded_coeff[:] = self.coeffs[j]
		for k in range(2):
				if j in self.uptake_transitions[0][k]:
					#This loop takes the int_uptake intrinsic rates and multiplies it 1st by the internal concentration
					#self.surf_concs[0,k] and then by the external concentrations: self.surf_concs[1,k] where k
					#represents either a Cl[0] or a H+[1] uptake transition.
					self.expanded_coeff[0] *= self.surf_concs[0, k]  # Int_Cl = 0,0 Int_H+ = 0,1 for Bio
					self.expanded_coeff[1] *= self.surf_concs[1, k]  # Ext_Cl = 1,0 Int_H+ = 1,1 for opp
					self.expanded_coeff_A[0] *= self.surf_concs_A[0, k]
					self.expanded_coeff_A[1] *= self.surf_concs_A[1, k]
					return
				elif j in self.uptake_transitions[1][k]:
					#This is the same as above but it takes the ext_uptake intrinsic rates
					self.expanded_coeff[0] *= self.surf_concs[1,k] #Ext_Cl and Ext_H+ for Bio
					self.expanded_coeff[1] *= self.surf_concs[0,k] #Int_Cl and Int_H+ for Opp
					self.expanded_coeff_A[0] *= self.surf_concs_A[1, k]  # Ext_Cl and Ext_H+ for Bio
					self.expanded_coeff_A[1] *= self.surf_concs_A[0, k]
					return

	# Produces rate matrices in a list with orientations
	# biological (self.coeff_mats[0]) and opposite (self.coeff_mats[1])
	# And associated solution matrices in self.As
	def update_coeff_mats(self):
		np.fill_diagonal(self.coeff_mats[0], 0)
		np.fill_diagonal(self.coeff_mats[1], 0)
		np.fill_diagonal(self.coeff_mats_A[0], 0)
		np.fill_diagonal(self.coeff_mats_A[1], 0)
		for j in range(self.N_coeffs): #for all of the rate_coefficients
			self.update_expanded_coeff(j)
			for t in self.transitions[j]: #for each tuple in transition matrix
				for k in range(self.N_orient):
					self.coeff_mats[k][t[1],t[0]]  = self.expanded_coeff[k]
					self.coeff_mats[k][t[0],t[0]] -= self.expanded_coeff[k]
					self.coeff_mats_A[k][t[1], t[0]] = self.expanded_coeff_A[k]
					self.coeff_mats_A[k][t[0], t[0]] -= self.expanded_coeff_A[k]
		for k in range(self.N_orient):
			np.copyto(self.As[k,:-1,:],self.coeff_mats[k,:-1,:])
			np.copyto(self.As_A[k, :-1, :], self.coeff_mats_A[k, :-1, :])


	# Produces orientational ion flows with format
	# [bio_flows,opp_flows] with subformat
	# [cl_flow,h_flow] with positive flows corresponding
	# to flow directed from extracellular to intracellular
	def update_ion_flows(self):
		for j in range(self.N_orient):
			for k in range(2):
				self.ion_flows[j,k] = 0
				self.ion_flows_A[j,k] = 0
		for t in self.flow_transitions:
			for k in range(2):
				self.ion_flows[0, t[2]] += (t[3] * self.flow_mats[0][t[1], t[0]]) #[0,1] is bio Cl-
				self.ion_flows[1, t[2]] -= (t[3] * self.flow_mats[1][t[1], t[0]])
				self.ion_flows_A[0, t[2]] += (t[3] * self.flow_mats_A[0][t[1], t[0]])
				self.ion_flows_A[1, t[2]] -= (t[3] * self.flow_mats_A[1][t[1], t[0]])

	# Updates the ion flow ratio self.ratio using self.ion_flows
	def update_ratio(self):
		denominator = self.ion_flows[0,1] + self.ion_flows[1,1]
		if denominator == 0:
			self.ratio = float('NaN')
		else:
			self.ratio = (
				(self.ion_flows[0,0] + self.ion_flows[1,0])/
				denominator
		)

	# Produces the integer corresponding to the
	# binary string ident.
	@staticmethod
	def encode(ident):
		N = len(ident)
		index = 0
		rev_indent = ident[::-1]
		for j in range(N):
			index += int(rev_indent[j]) * 2 ** j
		return index

	# Produces a binary string of length N corresponding to
	# an integer index.
	@staticmethod
	def decode(index, N = 4):
		result = ['0' for _ in range(N)]
		for j in range(N)[::-1]:
			result[j] = str(index // 2 ** j)
			index -= int(result[j]) * 2 ** j
		result.reverse()
		return ''.join(result)

	# Produces flow transitions with format
	# [(state_1,state_2,ion,sign),...]
	# where ion format 0=Cl,1=H, and sign format
	# 1 = positive flow 'going in' (bio. orient), -1 = negative flow 'going out'(bio. orient)
	def gen_flow_transitions(self):
		N = 16
		flow_transitions = []
		keys = [
			(1,1), # (binary_index, ion(0=Cl,1=H))
			(2,0)]
		for j in range(N): #rate matrix 1
			for k in range(N): #rate matrix 2
				for key in keys:
					s1 = self.decode(j)
					s2 = self.decode(k)
					s1_ext = s1[:key[0]] + s1[key[0]+1:]
					s2_ext = s2[:key[0]] + s2[key[0]+1:]
					if (s1[key[0]] == '0' and
							s2[key[0]] == '1' and
							s1_ext == s2_ext):
						if key == keys[0]:
							if int(s1[0]) == 1:
								flow_transitions.append(
								(j,k,key[1],1))
							else:
								flow_transitions.append((j,k,key[1],-1))
						elif  key != keys[0]:
							flow_transitions.append((j,k,key[1],1))
					elif (s1[key[0]] == '1' and
						s2[key[0]] == '0' and
						s1_ext == s2_ext):
						if 	key == keys[0]:
							if int(s1[0]) == 1:
								flow_transitions.append(
								(j,k,key[1],-1))
							else:
								flow_transitions.append((j,k,key[1],1))
						elif key != keys[0]:
							flow_transitions.append((j,k,key[1],-1))
		return flow_transitions

	# Returns the transition corresponding to to the state strings
	# m1 -> m2
	def get_transitions(self,m1,m2):
		transitions = []
		N_states = len(m1)
		N_fill = 0
		for j in range(N_states):
			if m1[j] == 'X':
				N_fill += 1
		N = 2**N_fill
		for j in range(N):
			filler = self.decode(j,N_fill)
			s1 = []
			s2 = []
			k = 0
			for l in range(N_states):
				if m1[l] == 'X':
					s1.append(filler[k])
					s2.append(filler[k])
					k += 1
				else:
					s1.append(m1[l])
					s2.append(m2[l])
			transitions.append((
				self.encode(''.join(s1)),
				self.encode(''.join(s2))
			))
		return tuple(transitions)

	# Produces a rate_map dictionary with format
	# rate_map[(state_1, state_2, identifier)] = rate_coefficient.
	# 'X' in a state is a wildcard.
	# Units are 1/ms for first order and 1/ms/mM for second order
	def gen_transition_dict(self,config):
		with open(config['rate_map_file'],'r') as myfile:
			buff = myfile.readlines()
		lines = []
		for line in buff:
			line = re.split('[,\n]+', ',' + line + ',')
			lines.append(line[1:-1])
		transition_dict = dict()
		N = len(lines[1][0][2:])
		for line in lines[1:]:
			s1 = line[0][2:]
			s2 = line[1][2:]
			mask = line[2][2:]
			m1 = []
			m2 = []
			for j in range(N):
				if mask[j] == '0':
					m1.append('X')
					m2.append('X')
				else:
					m1.append(s1[j])
					m2.append(s2[j])
			transitions = self.get_transitions(m1,m2)
			ident = line[3]
			transition_dict[ident] = transitions
		self.map_states()
		return transition_dict

	def map_states(self):
		N = 16
		states = {}
		count = 0
		number = []
		state = []
		pop_bio = []
		pop_opp = []
		for j in range(N):  # rate matrix 1
			for k in range(N):  # rate matrix 2
				s1 = self.decode(j)
				if s1 not in states:
						states[s1] = count
						count += 1
				else:
						pass


	#returns the list of transitions using config
	def gen_transitions(self,config):
		transition_dict = self.gen_transition_dict(config)
		return self.dict_to_list(transition_dict)

	# returns a list from a dict
	@staticmethod
	def dict_to_list(dict_in):
		list_out = []
		for ident in dict_in:
			list_out.append(dict_in[ident])
		return list_out

	# Produces dictionary rate_coeffs with format
	# rate_coeffs[identifier] = rate_coefficient.
	# Units are 1/ms for first order and 1/ms/mM for second order
	# Also sets self.lbounds and self.ubounds
	def get_coeffs_dict(self,config):
		with open(config['input_rate_file'],'r') as myfile:
			buff = myfile.readlines()
		lines = []
		for line in buff:
			line = re.split('[, \t"\n]+', ' ' + line + ' ')
			lines.append(line[1:-1])
		with open(config['voltage_rate_file'],'r') as myfile2:
			buff2 = myfile2.readlines()
		for line in buff2:
			line = re.split('[, \t"\n]+', ' ' + line + ' ')
			lines.append(line[1:-1])
		coeffs_dict = dict()
		N = len(lines[0])
		for j, identifier in enumerate(lines[0]):
			coeffs_dict[identifier] = float(lines[1][j])
		self.lbounds = np.zeros(N)
		self.ubounds = np.full(N,float('inf'))
		self.fz_Cl_R = np.zeros(N)
		self.fz_Cl_T = np.zeros(N)
		self.fz_H_R = np.zeros(N)
		self.fz_H_T = np.zeros(N)
		if(len(lines)>3):
			for j in range(N):
				self.lbounds[j] = (float(lines[2][j]))
				self.ubounds[j] = (float(lines[3][j]))
				self.fz_Cl_R[j] = (float(lines[4][j]))
				self.fz_Cl_T[j] = (float(lines[5][j]))
				self.fz_H_R[j] = (float(lines[6][j]))
				self.fz_H_T[j] = (float(lines[7][j]))

		return coeffs_dict

	def get_new_coeff(self):
			C = - 0.02568 #0.02568eV constants in the conversion for kT in a semiemperical capacitor-system with voltage in units eV, 0.01450 is roughly 2
			Accardi_coeff = np.copy(self.coeffs)
			Solution = np.copy(Accardi_coeff)
			for j in range(self.N_coeffs):
					value = (self.fz_Cl_T[j] - self.fz_Cl_R[j])* self.Voltage * -1 + (self.fz_H_T[j] - self.fz_H_R[j]) * self.Voltage
					if value == 0:
						Solution[j] = Accardi_coeff[j]
					else:
						Accardi_coeff[j] = Accardi_coeff[j] * math.exp(value / C)
						Solution[j] = Accardi_coeff[j]
			return Solution

	# Produces internal and external concentrations with format
	# concentrations[[Cl_int,H_int],[Cl_ext,H_ext]]
	# Units are M (mM/m^3)
	def get_concs(self,system,config):
		concs = np.zeros((2,2))
		if system == 'Miller':
			concs[0,0] = config['M_internal_Cl_conc']
			concs[1,0] = config['M_external_Cl_conc']
			concs[0,1] = 10**(-1*config['M_internal_pH']) * 1E3
			concs[1,1] = 10**(-1*config['M_external_pH']) * 1E3
		else:
			concs[0, 0] = config['A_internal_Cl_conc']
			concs[1, 0] = config['A_external_Cl_conc']
			concs[0, 1] = 10 ** (-1 * config['A_internal_pH']) * 1E3
			concs[1, 1] = 10 ** (-1 * config['A_external_pH']) * 1E3
		return concs

#Must manually call for this by modifying the above code to call for it.
	def save_dat(self):
		header = ('Step\t' + '\t'.join(
			self.idents) +
			'\tObjective')
		self.export_coeffs(
			'out_coeffs.csv', 'out_coeffs_A.csv')

'''
References
[1]: Iyer, R.; Iverson, M T.; Accardi, A.; Miller, C. Nature. 2002, 419. 715-718. 
[2]: Accardi, A.; Walden, M.; Nguitragool, W.; Jayaram, H.; Williams, C.; Miller, C. 
     J Gen Physiol. 2005,126(6).563-570. Doi: 10.1085/jgp.200509417
'''