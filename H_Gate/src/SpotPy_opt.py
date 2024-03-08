import numpy as np
import spotpy as sp


# Used to pass coeffs and objective information to SpotPy API
class spot_setup():
	def __init__(self,opt_parent):
		self.opt_parent = opt_parent
		self.coeffs = opt_parent.coeffs
		self.params = self.gen_params()
		self.objective = 0
		self.count = 1
		self.min_obj = float('inf')
		self.min_coeffs = np.copy(opt_parent.coeffs)


	# Transforms coeffs into SpotPy params
	def gen_params(self):
		params = []
		for j in range(self.coeffs.shape[0]):
			pname = 'x'+str(j)
			stepsize = (self.opt_parent.ubounds[j] -
				self.opt_parent.lbounds[j])/10
			params.append(sp.parameter.Uniform(pname,
				self.opt_parent.lbounds[j],self.opt_parent.ubounds[j],
				stepsize,self.coeffs[j],self.opt_parent.lbounds[j],
				self.opt_parent.ubounds[j]))
		return params

	#For internal SpotPy API
	def parameters(self):
		return sp.parameter.generate(self.params)

	# Performs the objective function calculation and returns
	# it as the simulation list that SpotPy expects
	def simulation(self,coeffs):
		np.copyto(self.coeffs,coeffs)
		self.objective = self.opt_parent.objective_handler.evaluate()
		return [self.objective]

	# Returns the objective as a list
	def evaluation(self):
		return [self.objective]

	# Returns the negated objective, since SpotPy is a maximizer
	def objectivefunction(self,simulation,evaluation):
		return -self.objective

	# Used to update coefficient and objective data for output
	def save(self,objfuncs,coeffs,sims,chains):
		if -objfuncs < self.min_obj:
			np.copyto(self.min_coeffs,coeffs)
			self.min_obj = -objfuncs
		if self.count % self.opt_parent.output_interval == 0:
			self.opt_parent.opt_dat[self.opt_parent.out_count,0] = self.count
			self.opt_parent.opt_dat[self.opt_parent.out_count,1:-1] = coeffs
			self.opt_parent.opt_dat[self.opt_parent.out_count,-1] = -objfuncs
			self.opt_parent.out_count += 1
		self.count += 1

# Used as the parent optimizer's sub_opt class for SpotPy
class SpotPy_opt():
	def __init__(self,opt_parent):
		self.opt_parent = opt_parent
		self.opt_config = self.opt_parent.opt_config

	# Runs the SpotPy optimization
	def run(self):
		my_spot_setup = spot_setup(self.opt_parent)
		sp_alg = getattr(sp.algorithms,self.opt_config['algorithm'])
		sampler = sp_alg(my_spot_setup,dbformat='custom',
			save_sim=False)
		sampler.sample(self.opt_config['n_steps'])
		np.copyto(self.opt_parent.coeffs,my_spot_setup.min_coeffs)
