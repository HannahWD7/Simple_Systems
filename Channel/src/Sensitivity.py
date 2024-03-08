import pandas as pd
import numpy as np
import csv
import os
from ClC_MKM.src.Objective_handler import Objective_handler
import ClC_MKM.src.Config as conf
import ClC_MKM.src.Output as outp
from ClC_MKM.src.MKM_sys import MKM_sys

class Sensitivity():
    def __init__(self,filename):
        self.mkm_configs = conf.read_config_MKM(filename)
        self.opt_config = conf.read_config_OPT(
            self.mkm_configs[0]['opt_config_file'])
        self.objective_handler = Objective_handler(
            self.mkm_configs, self.opt_config)
        self.n_sys = len(self.mkm_configs)
        self.objective = 0
        self.objective = self.objective_handler.evaluate()
        self.coeffs = self.objective_handler.coeffs
        self.N_coeffs = len(self.coeffs)
        self.prev_coeffs = np.copy(self.coeffs)
        self.ion_flows = self.objective_handler.ion_flows

    def sensitivity_analysis(self, coeffs, change_amount):
        sensitivity = {}
        new_flows = self.ion_flow
        grad = {}
        delta = 1E-8
        for j in range(self.N_coeffs):
            self.coeffs[j] += delta
            newRate = self.coeffs[j]
            MKM_sys.update_all()
            new_flows = MKM_sys.ion_flows


