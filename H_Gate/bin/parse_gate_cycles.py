#!/usr/bin/env python3

import pickle
import numpy as np
import argparse
import re

class Cycle:
	def __init__(self,path,net_flow):
		self.path = path
		self.net_flow = net_flow

def cmp_to_key(mycmp):
	class K:
		def __init__(self, obj, *args):
			self.obj = obj
		def __lt__(self, other):
			return mycmp(self.obj, other.obj) < 0
		def __gt__(self, other):
			return mycmp(self.obj, other.obj) > 0
		def __eq__(self, other):
			return mycmp(self.obj, other.obj) == 0
		def __le__(self, other):
			return mycmp(self.obj, other.obj) <= 0
		def __ge__(self, other):
			return mycmp(self.obj, other.obj) >= 0
		def __ne__(self, other):
			return mycmp(self.obj, other.obj) != 0
	return K

def cmp(a,b):
	if abs(a[0]) != abs(b[0]):
		return abs(a[0]) - abs(b[0])
	elif a[0] == 0:
		if abs(a[1]) != abs(b[1]):
			return abs(a[1]) - abs(b[1])
		else:
			return a[1] - b[1]
	else:
		return a[0] - b[0]

def decode(n):
	sites = 4
	result = [0 for _ in range(sites)]
	for i in range(sites)[::-1]:
		result[i] = n // 2 ** i
		n -= result[i] * 2 ** i
	result.reverse()
	return tuple(result)

def decode_str(j):
	return ''.join(map(str,decode(j)))

def decode_list(mylist):
	list_out = []
	N = len(mylist)
	for j in range(N):
		list_out.append(decode_str(mylist[j]))
	return list_out

def update_stoich(t,stoich,args):
	if (t[0][2] == 0 and
		t[1][2] == 1 and
		t[0][1] == t[1][1]
	):
		if args.opp:
			stoich[0] -= 1
		else:
			stoich[0] += 1
	elif (t[0][2] == 1 and
		t[1][2] == 0 and
		t[0][1] == t[1][1]
	):
		if args.opp:
			stoich[0] += 1
		else:
			stoich[0] -= 1
	elif (t[0][3] == 0 and
		t[1][3] == 1 and
		t[0][1] == t[1][1]
	):
		if args.opp:
			stoich[0] += 1
		else:
			stoich[0] -= 1
	elif (t[0][3] == 1 and
		t[1][3] == 0 and
		t[0][1] == t[1][1]
	):
		if args.opp:
			stoich[0] -= 1
		else:
			stoich[0] += 1
	elif (t[0][1] == 0 and
		t[1][1] == 1 and
		t[0][0] == t[1][0] == 1
	):
		if args.opp:
			stoich[1] -= 1
		else:
			stoich[1] += 1
	elif (t[0][1] == 1 and
		t[1][1] == 0 and
		t[0][0] == t[1][0] == 1
	):
		if args.opp:
			stoich[1] += 1
		else:
			stoich[1] -= 1
	elif (t[0][1] == 0 and
		t[1][1] == 1 and
		t[0][0] == t[1][0] == 0
	):
		if args.opp:
			stoich[1] += 1
		else:
			stoich[1] -= 1
	elif (t[0][1] == 1 and
		t[1][1] == 0 and
		t[0][0] == t[1][0] == 0
	):
		if args.opp:
			stoich[1] -= 1
		else:
			stoich[1] += 1
	return stoich

def calc_stoich(cycle,args):
	stoich = [0,0]
	N = len(cycle)
	for j in range(N-1):
		s1 = decode(cycle[j])
		s2 = decode(cycle[j+1])
		t = (s1,s2)
		update_stoich(t,stoich,args)
	s1 = decode(cycle[N-1])
	s2 = decode(cycle[0])
	t = (s1,s2)
	update_stoich(t,stoich,args)
	return tuple(stoich)

def print_stoich(stoich):
	for cycle in stoich:
		print(decode_list(cycle.path),'{:.2e}'.format(cycle.net_flow))

def swap(x,y):
	x,y = y,x

def update_ion_flows(ion_flows,stoich,flow):
	for j in range(len(ion_flows)):
		ion_flows[j] += stoich[j]*flow


def get_rls(cycle,F):
	vec = range(len(cycle)-1)
	for j in vec:
		flow = F[cycle[j+1],cycle[j]] - F[cycle[j],cycle[j+1]]
		if j == 0:
			min_flow = flow
			min_idx = j
		else:
			if flow < min_flow:
				min_flow = flow
				min_idx = j
	flow = F[cycle[0],cycle[-1]] - F[cycle[-1],cycle[0]]
	if flow < min_flow:
		min_flow = flow
		min_idx = -1
	return ('({:d}, {:d})'.format(cycle[min_idx],cycle[min_idx+1]))


def print_summary(filename,stoichs,MRE):
	stoich_list = sorted(
		list(stoichs.keys()),
		key=cmp_to_key(cmp),
		reverse=True
	)
	stoich_sums = []
	for stoich in stoich_list:
		stoich_sum = 0
		for cycle in stoichs[stoich]:
			stoich_sum += cycle.net_flow
		stoich_sums.append(stoich_sum)
	ion_flows = [0,0]
	for j in range(len(stoich_list)):
		update_ion_flows(ion_flows,stoich_list[j],stoich_sums[j])

	buff_list = ['MRE:{:.3e} \nNet external to internal...\n'.
		format(MRE)
	]
	buff_list.append(
		'Cl flow (ions/ms): {:.3f}\n'.format(ion_flows[0])
	)
	buff_list.append(
		'H  flow (ions/ms): {:.3f}\n'.format(ion_flows[1])
	)
	
	buff_list.append(
		'\nstoichiometry (Cl,H)\tnet flow (ions/ms)\n'
	)
	for j in range(len(stoich_list)):
		buff_list.append('{:s}\t{:.3f}\n'.format(
			str(stoich_list[j]),
			stoich_sums[j]
		))

	buff_list.append('\n')
	tol = 1E-2
	for j in range(len(stoich_list)):
		buff_list.append('(Cl,H) stoichiometry: {:s}\n\n'.format(
			str(stoich_list[j]))
		)
		k = 0
		N = len(stoichs[stoich_list[j]])
		for k in range(N):
			cycle = stoichs[stoich_list[j]][k]
			if cycle.net_flow < tol:
				break
			buff_list.append('Path {:d}, Net flow: {:3f} ions/ms\n'.format(
				k,
				cycle.net_flow
				)
			)
			for state in cycle.path:
				buff_list.append('{:d} {:s}\n'.format(
						state,
						decode_str(state)
					)
				)
			buff_list.append('\n')

	with open(filename,'w') as myfile:
		myfile.write(''.join(buff_list))

def print_renview_species(filename,N):
	bufflist = ['Species_name Phase Surf_cov S1 S2 S3 S4 S5 S6 base\n']
	for j in range(N):
		sp_str=decode_str(j)
		line = '({:d})({:s}) Surface {:.2f} {:s} 1\n'.format(
			j,
			sp_str,
			0,
			' '.join(sp_str)
		)
		bufflist.append(line)
	with open(filename,'w') as myfile:
		myfile.write(''.join(bufflist))

def print_renview_paths(filename,stoichs,F):
	min_flow = 1E-2
	stoich_list = sorted(
		list(stoichs.keys()),
		key=cmp_to_key(cmp),
		reverse=True
	)
	buff_list = []
	for stoich in stoich_list:
		cycles = stoichs[stoich]
		if (stoich == (0,0) or cycles[0].net_flow < min_flow):
			continue
		buff_list.append('\n  Paths with ion flow  ')
		buff_list.append('  Cl:{:>3d}   H:{:>3d}\n'.format(
			stoich[0],stoich[1]
		))
		vec = range(len(cycles))
		for j in vec:
			cycle = cycles[j]
			if cycle.net_flow < min_flow:
				break
			buff_list.append('	Path{:>4d}: Flow = {:>10.3f}  RLS = {:s}\n'.format(
				j,
				cycle.net_flow,
				get_rls(cycle.path,F)
			))
			N = len(cycle.path)
			vec2 = range(N+1)
			for k in vec2:
				if k == N:
					s1 = cycle.path[-1]
					s2 = cycle.path[0]
				else:
					s1 = cycle.path[k-1]
					s2 = cycle.path[k]
				state = '{:>8d} {:s}'.format(s2,str(decode(s2)))
				buff_list.append(state)
				if k != 0:
					# buff_list.append('{:>11.3f}'.format(F[s2,s1]-F[s1,s2]))
					buff_list.append(' {:>10.3f} {:>10.3f}'.format(
						F[s2,s1],
						F[s1,s2]
					))
				buff_list.append('\n')
	with open(filename,'w') as myfile:
		myfile.write(''.join(buff_list))

def print_renview_flows(filename,F):
	bufflist = [' {:>13s} {:>13s} {:>13s} {:>4s} {:>46s}\n'.format(
		'Fwd_Rate',
		'Rev_Rate',
		'Net_Rate',
		'PEI',
		'Reaction_String'
		)
	]
	vec = range(F.shape[0])
	for j in vec:
		vec2 = range(j+1,F.shape[0])
		for k in vec2:
			if (F[j,k] != 0 or
				F[k,j] != 0):
				forward = F[k,j]
				reverse = F[j,k]
				net = forward - reverse
				# PEI = forward / (forward + reverse)
				PEI = 0.5
				s1 = '({:d})({:s})'.format(j,decode_str(j))
				s2 = '({:d})({:s})'.format(k,decode_str(k))
				rxn_str = '{:>20s}  <=> {:>20s}'.format(
					s1,
					s2
				)
				line = ' {:>13.7f} {:>13.7f} {:>13.7f} {:>4.2f} {:>46s}\n'.format(
					forward,
					reverse,
					net,
					PEI,
					rxn_str
				)
				bufflist.append(line)
	with open(filename,'w') as myfile:
		myfile.write(''.join(bufflist))

def print_paths(filename,stoichs,F,K_desc,idents,coeffs_dict,
	coeffs_dict_pre):
	min_flow = 1E-2
	stoich_list = sorted(
		list(stoichs.keys()),
		key=cmp_to_key(cmp),
		reverse=True
	)
	buff_list = []
	for stoich in stoich_list:
		cycles = stoichs[stoich]
		if (stoich == (0,0) or cycles[0].net_flow < min_flow):
			continue
		buff_list.append('\n  Paths with ion flow  ')
		buff_list.append('  Cl:{:>3d}   H:{:>3d}\n'.format(
			stoich[0],stoich[1]
		))
		vec = range(len(cycles))
		for j in vec:
			cycle = cycles[j]
			if cycle.net_flow < min_flow:
				break
			buff_list.append('	Path{:>4d}: Flow = {:>10.3f}  RLS = {:s}\n'.format(
				j,
				cycle.net_flow,
				get_rls(cycle.path,F)
			))
			N = len(cycle.path)
			vec2 = range(N+1)
			for k in vec2:
				if k == N:
					s1 = cycle.path[-1]
					s2 = cycle.path[0]
				else:
					s1 = cycle.path[k-1]
					s2 = cycle.path[k]
				state = '{:>8d} {:s}'.format(s2,str(decode(s2)))
				buff_list.append(state)
				if k == 0:
					buff_list.append(
					' {:>13s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s}'.format(
						'ID', 'k_opt (1/ms)', 'k_pre (1/ms)', 'MRE',
						'F_fwd (1/ms)', 'F_rev (1/ms)', 'F_net (1/ms)'
					))
				if k != 0:
					# buff_list.append('{:>11.3f}'.format(F[s2,s1]-F[s1,s2]))
					ident = idents[K_desc[s2,s1]]
					buff_list.append(
						' {:>13s} {:>13.1e} {:>13.1e} {:>13.1e} {:>13.3f} {:>13.3f} {:>13.3f}'.format(
						ident,
						coeffs_dict[ident],
						coeffs_dict_pre[ident],
						abs(coeffs_dict[ident]-coeffs_dict_pre[ident])/coeffs_dict[ident],
						F[s2,s1],
						F[s1,s2],
						F[s2,s1] - F[s1,s2]
					))
				buff_list.append('\n')
	with open(filename,'w') as myfile:
		myfile.write(''.join(buff_list))

def get_coeffs_dict(filename):
		with open(filename,'r') as myfile:
			buff = myfile.readlines()
		lines = []
		for line in buff:
			line = re.split('[, \t"\n]+', ' ' + line + ' ')
			lines.append(line[1:-1])
		coeffs_dict = dict()
		for j, identfier in enumerate(lines[0]):
			coeffs_dict[identfier] = float(lines[1][j])
		return coeffs_dict

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bn',
		help='Basename for output filenames (default "biological" or "opposite" depending on --opp)',
		default=None
	)
	parser.add_argument('-ci',
		help='Name of initial coeffs file (.csv)',
		default=None)
	parser.add_argument('-d',
		help='Directory containing cycle_dat.pickle (default "./")',
		default='./'
	)
	parser.add_argument('--opp', 
		help='Use to flip stoichiometry for opposite orientation',
		action='store_true'
	)
	args = parser.parse_args()
	if args.bn == None:
		if args.opp:
			args.bn = 'opposite'
		else:
			args.bn = 'biological'
	if args.d[-1] != '/':
		args.d += '/'
	with open(args.d + 'cycle_dat.pickle','rb') as myfile:
		cycle_dat = pickle.load(myfile)
	prev_cycles = cycle_dat['cycles']
	F = cycle_dat['F']
	MRE = cycle_dat['MRE']
	K_desc = cycle_dat['K_desc']
	idents = cycle_dat['idents']
	coeffs_dict = cycle_dat['coeffs_dict']
	coeffs_dict_pre = get_coeffs_dict(args.ci)
	# F = np.divide(F,2)
	# for cycle in prev_cycles:
	# 	prev_cycles[cycle] /= 2

	stoichs = dict()
	N = len(prev_cycles.keys())
	while N > 0:
		cycle = list(prev_cycles.keys())[0]
		for_flow = prev_cycles[cycle]
		if len(cycle) > 2:
			rev_cycle = tuple([cycle[0]]) + cycle[::-1][:-1]
			if rev_cycle in prev_cycles:
				rev_flow = prev_cycles[rev_cycle]
				prev_cycles.pop(cycle)
				prev_cycles.pop(rev_cycle)
				N -= 2
				if for_flow > rev_flow:
					net_flow = (for_flow - rev_flow)
				else:
					net_flow = (rev_flow - for_flow)
					cycle = rev_cycle
			else:
				net_flow = for_flow
				prev_cycles.pop(cycle)
				N -= 1
		else:
			net_flow = for_flow
			prev_cycles.pop(cycle)
			N -= 1
		stoich = calc_stoich(cycle,args)
		new_cycle = Cycle(cycle,net_flow)
		if stoich in stoichs:
			stoichs[stoich].append(new_cycle)
		else:
			stoichs[stoich] = [new_cycle]

	for stoich,cycles in stoichs.items():
		sorted_cycles = sorted(
			cycles,
			key=lambda mykey: mykey.net_flow,
			reverse=True
		)
		stoichs[stoich] = sorted_cycles

	print_summary(args.bn + '_summary.txt',stoichs,MRE)
	print_renview_species(args.bn + '_species_renview.txt',F.shape[0])
	print_renview_flows(args.bn + '_flows_renview.txt',F)
	print_renview_paths(args.bn + '_paths_renview.txt',stoichs,F)
	print_paths(args.bn + '_paths.txt',
		stoichs,F,K_desc,idents,coeffs_dict,coeffs_dict_pre)

main()