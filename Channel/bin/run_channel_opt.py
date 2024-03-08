#!/usr/bin/env python3

import argparse
import numpy as np
import math
import copy
from matplotlib import pyplot as plt
from Simple_Systems.Channel.src.Optimizer import Optimizer

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config',
		help='Configuration file (.txt)',
	)
	args = parser.parse_args()
	mkm_opt = Optimizer(args.config)
	mkm_sys = mkm_opt.objective_handler
	mkm_opt.run()
	N = len(mkm_opt.mkm_configs)

main()
