#!/usr/bin/env python3

import argparse
import numpy as np
import math
import copy
from matplotlib import pyplot as plt
from Simple_Systems.H_Gate.src.MKM_sys import MKM_sys
import Simple_Systems.H_Gate.src.Config as conf

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config',
		help='Configuration file (.txt)',
	)
	parser.add_argument('-f',
		help='Optimization output coefficients file (default="opt.dat")',
		default='opt.dat'
	)
	args = parser.parse_args()
	configs = conf.read_config_MKM(args.config)
	opt_dat = np.loadtxt(args.f, delimiter='\t',skiprows=1)
	out_dat = np.zeros((opt_dat.shape[0],5))
	header = 'Step\tBio_Cl_(ions/ms)\tBio_H_(ions/ms)\tOpp_Cl_(ions/ms)\tOpp_H_(ions/ms)'
	N = len(configs)
	for j in range(N):
		mkm_sys = MKM_sys(configs[j])
		for row in range(opt_dat.shape[0]):
			mkm_sys.coeffs = opt_dat[row,1:-1]
			mkm_sys.update_all()
			out_dat[row,0] = opt_dat[row,0]
			out_dat[row,1] = mkm_sys.ion_flows[0][0]
			out_dat[row,2] = mkm_sys.ion_flows[0][1]
			out_dat[row,3] = mkm_sys.ion_flows[1][0]
			out_dat[row,4] = mkm_sys.ion_flows[1][1]
		file_out = 'ion_flows_sys{:d}.dat'.format(j)
		np.savetxt(file_out,out_dat,fmt='%.9e',header=header,
			comments='',delimiter='\t')

main()
