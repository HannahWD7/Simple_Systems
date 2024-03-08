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
	parser.add_argument('system', help='Miller or Accardi system')
	args = parser.parse_args()
	configs = conf.read_config_MKM(args.config)
	system = args.system
	type = 'MKM'
	N = len(configs)
	for j in range(N):
		mkm_sys = MKM_sys(configs[j])
		if type == 'sensitivity':
			if system =='Miller':
				Cl_tot = mkm_sys.ion_flows[0][0] + mkm_sys.ion_flows[1][0]
				H_tot = mkm_sys.ion_flows[0][1] + mkm_sys.ion_flows[1][1]
				print('{}, {}'.format(Cl_tot, H_tot))
			else:
				Cl_tot = mkm_sys.ion_flows_A[0][0] + mkm_sys.ion_flows_A[1][0]
				H_tot = mkm_sys.ion_flows_A[0][1] + mkm_sys.ion_flows_A[1][1]
				print('{}, {}'.format(Cl_tot, H_tot))
		else:
			if system == 'Miller':
				print('System {:d} Ion flows (ions/ms):'.format(j))
				print('Bio Cl: {:>11.4e}'.format(mkm_sys.ion_flows[0][0]))
				print('Bio  H: {:>11.4e}'.format(mkm_sys.ion_flows[0][1]))

			else:
				print('System {:d} Ion flows (ions/ms):'.format(j))
				print('Bio Cl: {:>11.4e}'.format(mkm_sys.ion_flows_A[0][0]))
				print('Bio  H: {:>11.4e}'.format(mkm_sys.ion_flows_A[0][1]))

main()
