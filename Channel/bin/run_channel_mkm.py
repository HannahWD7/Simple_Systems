#!/usr/bin/env python3

import argparse
import numpy as np
import math
import copy
from matplotlib import pyplot as plt
from Simple_Systems.Channel.src.MKM_sys import MKM_sys
import Simple_Systems.Channel.src.Config as conf

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config',
		help='Configuration file (.txt)',
	)
	args = parser.parse_args()
	configs = conf.read_config_MKM(args.config)
	type = 'MKM'
	N = len(configs)
	for j in range(N):
		mkm_sys = MKM_sys(configs[j])
		if type == 'sensitivity':
			Cl_tot = mkm_sys.ion_flows[0][0] + mkm_sys.ion_flows[1][0]
			H_tot = mkm_sys.ion_flows[0][1] + mkm_sys.ion_flows[1][1]
			print('{}, {}'.format(Cl_tot, H_tot))
		else:
			ion_flow = mkm_sys.ion_flows[0][0] / 1.99E-12 * 1.602E-19 * 1000
			print('System {:d} Ion flows (ions/ms):'.format(j))
			print('Net K+: {:>11.4e}'.format(ion_flow))


main()
