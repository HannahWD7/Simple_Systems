import math
import numpy as np


def exponent(num):
	if num == 0:
		return int(0)
	else:
		return int(math.log(abs(num),10))

def base(num):
	if num == 0:
		return 0
	else:
		return num / (10**exponent(num))


# Writes a numpy matrix mat to file filename.
def print_mat(mat,filename):
	bufflist = []
	for j in range(mat.shape[0]):
		for k in range(mat.shape[1]):
			bufflist.append('{:.15f}E{:d}\t'.format(
				base(mat[j,k]),
				exponent(mat[j,k])))
		bufflist.append('\n')
	with open(filename,'w') as myfile:
		myfile.write(''.join(bufflist))

# Updates bufflist with numeric format format_spec
# With config identifier ident
def update_bufflist_row(bufflist,ident,configs,format_spec):
	n_sys = len(configs)
	sci_format = '{:>8.1e}'
	row_header_format = '{:>18s}'
	float_format = '{:>8.1f}'
	float_format_l = '{:>8.3f}'
	bufflist.append(row_header_format.format(ident))
	if format_spec =='e':
		for j in range(n_sys):
			bufflist.append(sci_format.format(
				configs[j][ident]))
	elif format_spec =='f':
		for j in range(n_sys):
			bufflist.append(float_format.format(
				configs[j][ident]))
	elif format_spec =='fl':
		for j in range(n_sys):
			bufflist.append(float_format_l.format(
				configs[j][ident]))
	bufflist.append('\n')

# Updates bufflist with achieved flows for the residuals specified
def update_bufflist_row_achflow(bufflist,ident,optimizer,n_sys,format_spec):
	flow_alias_dict = optimizer.objective_handler.flow_alias_dict
	sci_format = '{:>8.1e}'
	row_header_format = '{:>18s}'
	float_format = '{:>8.1f}'
	float_format_l = '{:>8.3f}'
	bufflist.append(row_header_format.format(ident))
	if format_spec =='e':
		for j in range(n_sys):
			bufflist.append(sci_format.format(
				flow_alias_dict[ident](j)))
	elif format_spec =='f':
		for j in range(n_sys):
			bufflist.append(float_format.format(
				flow_alias_dict[ident](j)))
	elif format_spec =='fl':
		for j in range(n_sys):
			bufflist.append(float_format_l.format(
				flow_alias_dict[ident](j)))
	bufflist.append('\n')

# Updates bufflist with coefficient statistics
def update_bufflist_coeffs(bufflist,optimizer):
	init_coeffs = np.copy(optimizer.objective_handler.
		mkm_systems[0].init_coeffs)
	final_coeffs = np.copy(optimizer.coeffs)
	idents = np.copy(optimizer.objective_handler.
		mkm_systems[0].idents)
	abs_devs = np.abs(init_coeffs-final_coeffs)
	zipped = zip(idents,init_coeffs,final_coeffs,abs_devs)
	zipped = sorted(zipped,key=lambda x:x[-1],reverse=True)
	n_coeffs = len(init_coeffs)
	median_ind = int(n_coeffs/2)
	quartile = int(median_ind/2)

	bufflist.append(
		'\nAbsolute coefficient deviation percentiles:\n')
	bufflist.append(
		 '0%: {:.1e}\n'.format(
			zipped[-1][-1]))
	bufflist.append(
		 '25%: {:.1e}\n'.format(
			zipped[-quartile][-1]))
	bufflist.append(
		 '50%: {:.1e}\n'.format(
			zipped[median_ind][-1]))
	bufflist.append(
		 '75%: {:.1e}\n'.format(
			zipped[quartile][-1]))
	bufflist.append(
		 '100%: {:.1e}\n'.format(
			zipped[0][-1]))
	N = 10
	zipped = zipped[:N]
	unzipped = list(zip(*zipped))
	#TODO: fix below to account for unzipped[-2] having values of zero
	rel_devs = np.divide(unzipped[-1],unzipped[-2])
	bufflist.append(
		'\nMost deviate coefficients:\n')
	bufflist.append('{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}\n'.format(
		'Identifier','Initial','Final','Abs_dev.','Rel_dev.'))
	for j in range(N):
		bufflist.append(
			'{:>10s}{:>10.1e}{:>10.1e}{:>10.1e}{:>10.1e}\n'.format(
			unzipped[0][j],unzipped[1][j],unzipped[2][j],unzipped[3][j],
			rel_devs[j]))

# Prints an optimization summary to filename
def print_opt_summary(filename,optimizer):
	sci_format = '{:>8.1e}'
	row_header_format = '{:>18s}'
	int_format = '{:>8d}'
	bufflist = []
	configs = optimizer.mkm_configs
	n_sys = len(configs)
	bufflist.append(
		'Number of optimized system configurations: {:d}\n'.format(
			n_sys))
	bufflist.append(
		'Number of optimization steps: {:d}\n'.format(
			optimizer.step_num))
	mystr = 'Final objective value: ' + sci_format + '\n'
	bufflist.append(mystr.format(optimizer.objective))
	bufflist.append(
		'\nSystem configurations:\n')
	bufflist.append(row_header_format.format('System'))
	for j in range(n_sys):
		bufflist.append(int_format.format(j+1))
	bufflist.append('\n')
	update_bufflist_row(bufflist,'internal_pH',configs,'f')
	update_bufflist_row(bufflist,'external_pH',configs,'f')
	update_bufflist_row(bufflist,'internal_Cl_conc',configs,'f')
	update_bufflist_row(bufflist,'external_Cl_conc',configs,'f')
	res_configs = optimizer.objective_handler.res_configs
	bufflist.append(
		'\nTarget flow values (ions/ms):\n')
	for key in res_configs[0]:
		update_bufflist_row(bufflist,key,res_configs,'fl')
	bufflist.append(
		'\nAchieved flow values (ions/ms):\n')
	for key in res_configs[0]:
		update_bufflist_row_achflow(bufflist,key,optimizer,n_sys,'fl')
	update_bufflist_coeffs(bufflist,optimizer)

	with open(filename,'w') as myfile:
		myfile.write(''.join(bufflist))