#!/usr/bin/env python3

import argparse
import re
import numpy as np
import math
import copy
import os
import pickle
from matplotlib import pyplot as plt
from Simple_Systems.Channel.src.MKM_sys import MKM_sys
from CycFlowDec import CycFlowDec
import Simple_Systems.Channel.src.Config as conf

# Writes a numpy matrix mat to file filename.
def print_mat(mat,filename):
	bufflist = []
	for j in range(mat.shape[0]):
		for k in range(mat.shape[1]):
			if mat[j,k] == 0:
				exponent = 0
			else:
				exponent = int(math.log(abs(mat[j,k]),10))
			bufflist.append('{:.15f}E{:d}\t'.format(
				mat[j,k] / (10**exponent),
				exponent
			))
		bufflist.append('\n')
	with open(filename,'w') as myfile:
		myfile.write(''.join(bufflist))

# Generates a transiton->coefficient index map K_desc
def gen_K_desc(mkm_sys):
	K_desc = np.zeros((mkm_sys.N_coeffs,
		mkm_sys.N_coeffs),dtype=int)
	for j in range(mkm_sys.N_coeffs):
		N_transitions = len(mkm_sys.transitions[j])
		for k in range(N_transitions):
			K_desc[mkm_sys.transitions[j][k][1],
				mkm_sys.transitions[j][k][0]] = j
	return K_desc

# Runs cyclic flow decomposition for orientation orient.
# Generates corresponding subdirectories with pickled
# cycles dict
def oriented_decomp(j,mkm_sys, system):
	if j == 0:
		print('Running decomp (bio)...')
		subdir = 'biological/'
	else:
		print('Running decomp (opp)...')
		subdir = 'opposite/'
	if not os.path.isdir(subdir):
		os.mkdir(subdir)
	dat = dict()
	K_desc = gen_K_desc(mkm_sys)
	dat['K_desc'] = K_desc
	dat['idents'] = mkm_sys.idents
	if system == 'Miller':
		dat['coeffs_dict'] = mkm_sys.Miller_coeff_dict
		F = mkm_sys.flow_mats[j]
	else:
		dat['coeffs_dict'] = mkm_sys.Accardi_coeff_dict
		F = mkm_sys.flow_mats_A[j]
	dat['F'] = F
	nstep = 500
	run_post = 10
	burnin = nstep - run_post
	decomp = CycFlowDec(
		F,
		state=7,
		tol=1E-7
	)
	decomp.run(burnin,run_post)
	MRE = decomp.calc_MRE(1E-1)
	ntot = nstep
	MRE_tol = 5E-3
	while(ntot < 50000 and MRE > MRE_tol):
		decomp.run(burnin,run_post)
		MRE = decomp.calc_MRE(1E-1)
		ntot += nstep
		print('Step {:>6d}, MRE: {:.3e}'.format(ntot,MRE))
	dat['MRE'] = MRE
	dat['cycles'] = decomp.cycles
	with open(subdir + 'cycle_dat.pickle','wb') as myfile:
		pickle.dump(dat,myfile)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config',
		help='Configuration file (.txt)',
	)
	parser.add_argument('system', help='Miller or Accardi system')
	args = parser.parse_args()
	configs = conf.read_config_MKM(args.config)
	system = args.system
	mkm_sys = MKM_sys(configs[0])
	oriented_decomp(0,mkm_sys,system)

main()
