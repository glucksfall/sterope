#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
Project "Sensitivity Analysis of Rule-Based Models", Rodrigo Santibáñez, 2019
To be used with KaSim. Please refer to other subprojects for other stochastic simulators support
Citation:
'''

__author__  = 'Rodrigo Santibáñez'
__license__ = 'gpl-3.0'
__software__ = 'kasim-v4.0'

import argparse, glob, multiprocessing, os, random, re, shutil, subprocess, sys, time
import pandas, numpy

class custom:
	class random:
		def seed(number):
			if args.legacy:
				random.seed(args.seed)
			else:
				numpy.random.seed(args.seed)

		def random():
			if args.legacy:
				return random.random()
			else:
				return numpy.random.random()

		def uniform(lower, upper):
			if args.legacy:
				return random.uniform(lower, upper)
			else:
				return numpy.random.uniform(lower, upper, None)

		def lognormal(lower, upper):
			if args.legacy:
				return random.lognormvariate(lower, upper)
			else:
				return numpy.random.lognormal(lower, upper, None)

def safe_checks():
	error_msg = ''
	if shutil.which(opts['python']) is None:
		error_msg += 'python3 (at {:s}) can\'t be called to perform error calculation.\n' \
			'You could use --python {:s}\n'.format(opts['python'], shutil.which('python3'))

	# check for simulators
	#if shutil.which(opts['bng2']) is None:
		#error_msg += 'BNG2 (at {:s}) can\'t be called to perform simulations.\n' \
			#'Check the path to BNG2.'.format(opts['bng2'])
	if shutil.which(opts['kasim']) is None:
		error_msg += 'KaSim (at {:s}) can\'t be called to perform simulations.\n' \
			'Check the path to KaSim.'.format(opts['kasim'])
	#if shutil.which(opts['nfsim']) is None:
		#error_msg += 'NFsim (at {:s}) can\'t be called to perform simulations.\n' \
			#'Check the path to NFsim.'.format(opts['nfsim'])
	#if shutil.which(opts['piskas']) is None:
		#error_msg += 'PISKaS (at {:s}) can\'t be called to perform simulations.\n' \
			#'Check the path to PISKaS.'.format(opts['piskas'])

	# check for slurm
	if opts['slurm'] is not None or opts['slurm'] == '':
		if not sys.platform.startswith('linux'):
			error_msg += 'SLURM do not support WindowsOS and macOS (https://slurm.schedmd.com/platforms.html)\n'
		else:
			if shutil.which('sinfo') is None:
				error_msg += 'You specified a SLURM partition but SLURM isn\'t installed on your system.\n' \
					'Delete --slurm to use the python multiprocessing API or install SLURM (https://pleione.readthedocs.io/en/latest/SLURM.html)\n'
			else:
				cmd = 'sinfo -hp {:s}'.format(opts['slurm'])
				cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
				out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
				if out == b'':
					error_msg += 'You specified an invalid SLURM partition.\n' \
						'Please, use --slurm $SLURM_JOB_PARTITION or delete --slurm to use the python multiprocessing API.\n'

	# check if model file exists
	if not os.path.isfile(opts['model']):
		error_msg += 'The "{:s}" file cannot be opened.\n' \
			'Please, check the path to the model file.\n'.format(opts['model'])

	# print error
	if error_msg != '':
		print(error_msg)
		raise ValueError(error_msg)

	return 0

def parallelize(cmd):
	proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
	out, err = proc.communicate()
	proc.wait()
	return 0

def argsparser():
	parser = argparse.ArgumentParser(description = 'Perform a sensitivity analysis of RBM parameters employing the Morris sequence.')

	# required arguments
	parser.add_argument('--model'  , metavar = 'str'  , type = str  , required = True , nargs = 1  , help = 'RBM with tagged variables to analyze')
	parser.add_argument('--final'  , metavar = 'float', type = str  , required = True , nargs = 1  , help = 'limit time to simulate')
	parser.add_argument('--steps'  , metavar = 'float', type = str  , required = True , nargs = 1  , help = 'time steps to simulate')
	# choose one or more evaluation functions

	# useful paths
	#parser.add_argument('--bng2'   , metavar = 'path' , type = str  , required = False, default = '~/bin/bng2'    , help = 'BioNetGen path, default ~/bin/bng2')
	parser.add_argument('--kasim'  , metavar = 'path' , type = str  , required = False, default = '~/bin/kasim4'  , help = 'KaSim path, default ~/bin/kasim4')
	#parser.add_argument('--nfsim'  , metavar = 'path' , type = str  , required = False, default = '~/bin/nfsim'   , help = 'NFsim path, default ~/bin/nfsim')
	#parser.add_argument('--piskas' , metavar = 'path' , type = str  , required = False, default = '~/bin/piskas'  , help = 'PISKaS path, default ~/bin/piskas')
	parser.add_argument('--python' , metavar = 'path' , type = str  , required = False, default = '~/bin/python3' , help = 'python path, default ~/bin/python3')

	# distribute computation with SLURM, otherwise with python multiprocessing API
	parser.add_argument('--slurm'  , metavar = 'str'  , type = str  , required = False, default = None            , help = 'SLURM partition to use, default None')

	# general options
	parser.add_argument('--seed'   , metavar = 'int'  , type = int  , required = False, default = None            , help = 'random number generator seed, default None')
	parser.add_argument('--sims'   , metavar = 'int'  , type = int  , required = False, default = 10              , help = 'number of simulations per individual, default 100')
	parser.add_argument('--prec'   , metavar = 'str'  , type = str  , required = False, default = '7g'            , help = 'precision and format of parameter values, default 7g')
	parser.add_argument('--levels' , metavar = 'int'  , type = int  , required = False, default = 10              , help = 'number of levels, default 10')

	# other options
	parser.add_argument('--syntax' , metavar = 'str'  , type = str  , required = False, default = '4'             , help = 'KaSim syntax, default 4')
	parser.add_argument('--binary' , metavar = 'str'  , type = str  , required = False, default = 'model'         , help = 'KaSim binary prefix, default model')
	#parser.add_argument('--equil'  , metavar = 'float', type = float, required = False, default = 0               , help = 'equilibrate model before running the simulation, default 0')
	#parser.add_argument('--sync'   , metavar = 'float', type = str  , required = False, default = '1.0'           , help = 'time period to syncronize compartments, default 1.0')
	parser.add_argument('--output' , metavar = 'str'  , type = str  , required = False, default = 'outmodels'     , help = 'ranking files prefixes, default outmodels')
	parser.add_argument('--results', metavar = 'str'  , type = str  , required = False, default = 'results'       , help = 'output folder where to move the results, default results')
	parser.add_argument('--parsets', metavar = 'str'  , type = str  , required = False, default = 'individuals'   , help = 'folder to save the generated models, default individuals')
	parser.add_argument('--rawdata', metavar = 'str'  , type = str  , required = False, default = 'simulations'   , help = 'folder to save the simulations, default simulations')
	parser.add_argument('--fitness', metavar = 'str'  , type = str  , required = False, default = 'evaluation'    , help = 'folder to save the calculated sensitivity, default evaluation')
	parser.add_argument('--ranking', metavar = 'str'  , type = str  , required = False, default = 'ranking'       , help = 'folder to save the ranking summaries, default ranking')

	# TO BE DEPRECATED, only with publishing purposes.
	# the random standard library does not have a random.choice with an optional probability list, therefore, Pleione uses numpy.random.choice
	parser.add_argument('--legacy' , metavar = 'True' , type = str  , required = False, default = False           , help = 'use True to use random.random instead of False, the numpy.random package')
	# If the user wants to know the behavior of other functions, the option --dev should be maintained
	parser.add_argument('--dev'    , metavar = 'True' , type = str  , required = False, default = False           , help = 'calculate all evaluation functions, default False')

	args = parser.parse_args()

	if args.seed is None:
		if sys.platform.startswith('linux'):
			args.seed = int.from_bytes(os.urandom(4), byteorder = 'big')
		else:
			parser.error('pleione requires --seed integer')

	return args

def ga_opts():
	return {
		# user defined options
		'model'     : args.model[0],
		'final'     : args.final[0], # not bng2
		'steps'     : args.steps[0], # not bng2
		#'bng2'      : os.path.expanduser(args.bng2), # bng2, nfsim only
		'kasim'     : os.path.expanduser(args.kasim), # kasim4 only
		#'piskas'    : os.path.expanduser(args.piskas), # piskas only
		#'nfsim'     : os.path.expanduser(args.nfsim), # nfsim only
		'python'    : os.path.expanduser(args.python),
		'slurm'     : args.slurm,
		'rng_seed'  : args.seed,
		'num_sims'  : args.sims,
		'par_prec'  : args.prec,
		'p_levels'  : args.levels,
		'syntax'    : args.syntax, # kasim4 only
		#'binary'    : args.binary, # kasim4 beta only
		#'equil'     : args.equil, # nfsim only
		#'sync'      : args.sync, # piskas only
		'outfile'   : args.output,
		'results'   : args.results,
		'parsets'   : args.parsets,
		'rawdata'   : args.rawdata,
		'fitness'   : args.fitness,
		'ranking'   : args.ranking,
		# non-user defined options
		'home'      : os.getcwd(),
		'null'      : '/dev/null',
		'max_error' : numpy.nan,
		#'bin_file'  : args.model[0].split('.')[0] + '.bin', # kasim4 beta only
		'systime'   : str(time.time()).split('.')[0],
		# useful data
		#'num_pars'  : 0,
		'par_name'  : [],
		}

def configurate():
	# read the model
	data = []
	with open(opts['model'], 'r') as infile:
		for line in infile:
			data.append(line)

	# find variables to analyze
	regex = '%\w+: \'(\w+)\' ' \
		'([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)\s+(?:\/\/|#)\s+' \
		'(\w+)\[([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)\s+' \
		'([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)\]\n'

	#num_pars = 0
	parameters = {}

	for line in range(len(data)):
		matched = re.match(regex, data[line])
		if matched:
			parameters[line] = [
				'par',
				matched.group(1), # parameter name
				matched.group(2), # original value
				matched.group(3), # sensitivity keyword
				matched.group(4), # lower bound
				matched.group(5), # upper bound
				]
			opts['par_name'].append(matched.group(1))
		else:
			parameters[line] = data[line]

	if len(opts['par_name']) == 0:
		error_msg = 'No variables to analyze.\n' \
			'Check if selected variables follow the regex (See Manual).'
		print(error_msg)
		raise ValueError(error_msg)

	return parameters

def populate():
	# 'parameters' dictionary stores everything in the model, particularly the parameters to analyze
	par_keys = list(parameters.keys())

	population = {}
	model_string = 'model_level{:0' + str(len(str(opts['p_levels']))) + 'd}_{:s}'

	for par_name in opts['par_name']:
		for level in range(opts['p_levels']+1):
			var_key = par_name + 'level' + str(level)
			population[var_key, 'model'] = model_string.format(level, par_name)

			for line in range(len(par_keys)):
				if parameters[line][0] == 'par':
					lower = float(parameters[par_keys[line]][4])
					upper = float(parameters[par_keys[line]][5])
					morris = numpy.arange(lower, upper+1, upper // opts['p_levels'])

					if parameters[par_keys[line]][1] == par_name:
						population[var_key, parameters[par_keys[line]][1]] = morris[level]
					elif parameters[par_keys[line]][1] != par_name:
						population[var_key, parameters[par_keys[line]][1]] = float(parameters[par_keys[line]][2])
					else:
						raise ValueError('Use sensitivity[lower upper] for a valid range to define the levels.')
	#print(population)
	return population

def simulate():
	job_desc = {
		'nodes'     : 1,
		'ntasks'    : 1,
		'ncpus'     : 1,
		'null'      : opts['null'],
		'partition' : opts['slurm'],
		'job_name'  : 'child_{:s}'.format(opts['systime']),
		'stdout'    : 'stdout_{:s}.txt'.format(opts['systime']),
		'stderr'    : 'stderr_{:s}.txt'.format(opts['systime']),
		}

	# generate a kappa file per model
	par_keys = list(parameters.keys())
	par_string = '%var: \'{:s}\' {:.' + opts['par_prec'] + '}\n'

	for par_name in opts['par_name']:
		for level in range(opts['p_levels']):
			var_key = par_name + 'level' + str(level)
			model = population[var_key, 'model']

			if not os.path.exists(model + '.kappa'):
				with open(model + '.kappa', 'w') as file:
					for line in range(len(par_keys)):
						if parameters[par_keys[line]][0] == 'par':
							file.write(par_string.format(parameters[line][1], population[var_key, parameters[par_keys[line]][1]]))
						else:
							file.write(parameters[par_keys[line]])
					# add the DYN command at the end of the kappa file
					if opts['syntax'] == '4':
						file.write('%mod: [true] do $DIN \"' + var_key + '.dot\" [true];')
					else:
						file.write('%mod: [true] do $FLUX \"' + var_key + '.dot\" [true]')

	# submit simulations to the queue
	squeue = []
	model_string = '{:s}.{:0' + str(len(str(opts['p_levels']))) + 'd}.out.txt'

	for par_name in opts['par_name']:
		for level in range(opts['p_levels']):
			var_key = par_name + 'level' + str(level)
			for sim in range(opts['num_sims']):
				model = population[var_key, 'model']
				output = model_string.format(model, sim)

				if not os.path.exists(output):
					job_desc['exec_kasim'] = '{:s} -i {:s}.kappa -l {:s} -p {:s} -o {:s} -syntax {:s} --no-log'.format( \
						opts['kasim'], model, opts['final'], opts['steps'], output, opts['syntax'])

					# use SLURM Workload Manager
					if opts['slurm'] is not None:
						cmd = os.path.expanduser('sbatch --no-requeue -p {partition} -N {nodes} -c {ncpus} -n {ntasks} -o {null} -e {null} -J {job_name} \
							--wrap ""{exec_kasim}""'.format(**job_desc))
						cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
						out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
						while err == sbatch_error:
							out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
						squeue.append(out.decode('utf-8')[20:-1])

					# use multiprocessing.Pool
					else:
						cmd = os.path.expanduser(job_desc['exec_kasim'])
						print(cmd)
						cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
						squeue.append(cmd)

	# check if squeued jobs have finished
	if opts['slurm'] is not None:
		for job_id in range(len(squeue)):
			cmd = 'squeue --noheader -j{:s}'.format(squeue[job_id])
			cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
			out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
			while out.count(b'child') > 0 or err == squeue_error:
				time.sleep(1)
				out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

	#simulate with multiprocessing.Pool
	else:
		with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
			pool.map(parallelize, sorted(squeue), chunksize = 1)

	return population

def evaluate():
	job_desc = {
		'nodes'     : 1,
		'ntasks'    : 1,
		'ncpus'     : 1,
		'null'      : opts['null'],
		'partition' : opts['slurm'],
		'job_name'  : 'child_{:s}'.format(opts['systime']),
		'stdout'    : 'stdout_{:s}.txt'.format(opts['systime']),
		'stderr'    : 'stderr_{:s}.txt'.format(opts['systime']),
		'doerror'   : '{:s} -m pleione.kasim-doerror --crit {:s}'.format(opts['python'], opts['crit_vals']),
		'deverror'  : '{:s} -m pleione.kasim-allerror --crit {:s}'.format(opts['python'], opts['crit_vals']),
		}

	# submit error calculations to the queue
	squeue = []

	for ind in range(opts['pop_size']):
		model = population['model', ind]

		data = ' '.join(glob.glob(' '.join(opts['data'])))
		error = ' '.join(opts['error'])
		sims = ' '.join(glob.glob('{:s}.*.out.txt'.format(model)))
		output = '{:s}.txt'.format(model)

		job_desc['calc'] = job_desc['doerror'] + ' --data {:s} --sims {:s} --file {:s} --error {:s}'.format(data, sims, output, error)
		if args.dev:
			job_desc['calc'] = job_desc['deverror'] + ' --data {:s} --sims {:s} --file {:s}'.format(data, sims, output)

		# use SLURM Workload Manager
		if opts['slurm'] is not None:
			cmd = 'sbatch --no-requeue -p {partition} -N {nodes} -c {ncpus} -n {ntasks} -o {null} -e {null} -J {job_name} --wrap ""{calc}""'.format(**job_desc)
			cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
			out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
			while err == sbatch_error:
				out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
			squeue.append(out.decode('utf-8')[20:-1])

		# use multiprocessing.Pool
		else:
			cmd = os.path.expanduser(job_desc['calc'])
			cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
			squeue.append(cmd)

	# check if squeued jobs have finished
	if opts['slurm'] is not None:
		for job_id in range(len(squeue)):
			cmd = 'squeue --noheader -j{:s}'.format(squeue[job_id])
			cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
			out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
			while out.count(b'child') > 0 or err == squeue_error:
				time.sleep(1)
				out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

	# calc error with multiprocessing.Pool
	else:
		with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
			pool.map(parallelize, sorted(squeue), chunksize = 1)

	return population

def ranking():
	for ind in range(opts['pop_size']):
		with open('{:s}.txt'.format(population['model', ind]), 'r') as file:
			tmp = pandas.read_csv(file, delimiter = '\t', header = None)
			data = tmp.set_index(0, drop = False).rename_axis(None, axis = 0).drop(0, axis = 1).rename(columns = {1: 'value'})

		if args.dev:
			fitfunc = list(data.index)
		else:
			fitfunc = opts['error']

		# and store the error in the population dictionary
		for name in range(len(fitfunc)):
			population[fitfunc[name], ind] = data.loc[fitfunc[name], 'value']

	# now that everything is stored in the population dict, we proceed to rank models by the selected error function(s)
	jobs = {}
	rank = {}
	fitfunc = opts['error']

	for name in range(len(fitfunc)):
		for ind in range(opts['pop_size']):
			jobs[population['model', ind]] = population[fitfunc[name], ind]
		rank[fitfunc[name]] = sorted(jobs, key = jobs.get, reverse = False)

	for name in range(len(fitfunc)):
		for ind in range(opts['pop_size']):
			jobs[population['model', ind]] += { key : value for value, key in enumerate(rank[fitfunc[name]]) }[population['model', ind]]

	# create an 'ordered' list of individuals from the 'population' dictionary by increasing fitness
	rank = sorted(jobs, key = jobs.get, reverse = False)

	# find the index that match best individual with the 'population' dictionary keys and store the rank (a list) in the population dictionary
	ranked_population = []
	for best in range(opts['pop_size']):
		for ind in range(opts['pop_size']):
			if population['model', ind] == rank[best]:
				ranked_population.append(ind)
				break

	population['rank'] = ranked_population

	# save the population dictionary as a report file
	par_keys = list(set([x[0] for x in population.keys() if str(x[0]).isdigit()]))

	if args.dev:
		fitfunc = sorted(list(data.index))

	par_string = '{:.' + opts['par_fmt'] + '}\t'
	iter_string = '{:s}_{:0' + len(str(opts['num_iter'])) + 'd}.txt'
	with open(iter_string.format(opts['outfile'], iter), 'w') as file:
		file.write('# Output of {:s} {:s}\n'.format(opts['python'], subprocess.list2cmdline(sys.argv[0:])))
		file.write('Elapsed time: {:.0f} seconds\n'.format(time.time() - float(opts['systime'])))
		file.write('iteration: {:03d}\t'.format(iter))
		file.write('error: {:s}\t'.format(','.join(opts['error'])))
		file.write('seed: {:d}\n\n'.format(opts['rng_seed']))

		# header
		file.write('MODEL_ID\t')
		for i in range(len(fitfunc)):
			file.write('{:s}\t'.format(fitfunc[i]))
		for key in range(len(par_keys)):
			file.write('{:s}\t'.format(parameters[par_keys[key]][1].strip()))
		file.write('\n')

		for ind in ranked_population:
			file.write('{:s}\t'.format(population['model', ind]))
			for i in range(len(fitfunc)):
				if fitfunc[i] != 'MWUT':
					file.write(par_string.format(float(population[fitfunc[i], ind])))
				else:
					file.write('{:.0f}\t'.format(float(population[fitfunc[i], ind])))
			for key in range(len(par_keys)):
				file.write(par_string.format(float(population[par_keys[key], ind])))
			file.write('\n')

	return population

def mutate():
	# par_keys stores parameter values only
	par_keys = list(set([key[0] for key in population.keys() if str(key[0]).isdigit()]))

	# slice the population dictionary retrieving the best models, if needed
	ranked_population = population['rank']

	if opts['pop_best'] == 0:
		best_population = population
	else:
		best_population = {}
		for best in range(opts['pop_best']):
			for key in range(len(par_keys)):
				best_population[par_keys[key], best] = population[par_keys[key], ranked_population[best]]
			best_population['model', best] = population['model', ranked_population[best]]
			best_population['error', best] = population['error', ranked_population[best]]

	# fill the population dictionary with the elite, because population is a global variable
	for ind in range(opts['pop_best']):
		for par in range(len(par_keys)):
			population[par_keys[par], ind] = best_population[par_keys[par], ind]
		population['model', ind] = best_population['model', ind]
		population['error', ind] = best_population['error', ind]

	# User defined best population
	top = opts['pop_best']
	if opts['pop_best'] == 0:
		top = opts['pop_size']
	elif opts['pop_best'] == 1:
		opts['self_rec'] == True # allow self recombination to generate descendants from only one parent

	# probability distribution to select parents
	if opts['dist_type'] == 'uniform':
		# Define a uniform probability distribution according to the best population size
		dist = [1 for n in range(1, top + 1)]
	elif opts['dist_type'] == 'inverse':
		# Define an inverse probability distribution according to the best population size
		dist = [1/n for n in range(1, top + 1)]
	prob = numpy.divide(dist, float(numpy.sum(dist)))

	# fill the population dictionary with individuals from the best parents
	for ind in range(opts['pop_best'], opts['pop_size'], 2):
		if opts['pop_best'] == 0:
			# choose two random individuals from the ranked population (index start at zero)
			n1 = numpy.random.choice(ranked_population[0:top], p = prob[0:top])
			n2 = numpy.random.choice(ranked_population[0:top], p = prob[0:top])
			if opts['self_rec'] == False and not opts['pop_size'] == 1:
				while n2 == n1:
					n2 = numpy.random.choice(ranked_population[0:top], p = prob[0:top])

		elif opts['pop_best'] != 0:
			if not args.legacy:
				# choose two random individuals from the best population (reindexed from 0 to 'pop_best' size)
				n1 = numpy.random.choice(range(top), p = prob[0:top])
				n2 = numpy.random.choice(range(top), p = prob[0:top])
			else:
				n1 = random.choice(range(top))
				n2 = random.choice(range(top))

			if opts['self_rec'] == False and not opts['pop_size'] == 1:
				while n2 == n1:
					if args.legacy:
						n2 = numpy.random.choice(range(top), p = prob[0:top])
					else:
						n2 = random.choice(range(top))

		# perform multiple or single crossover
		if opts['xpoints'] == 'multiple':
			for par in range(len(par_keys)):
				# create children
				population[par_keys[par], ind] = best_population[par_keys[par], n1]
				population[par_keys[par], ind + 1] = best_population[par_keys[par], n2]

				# swap parameter values using a probability threshold
				if opts['mut_swap'] >= custom.random.random():
					population[par_keys[par], ind] = best_population[par_keys[par], n2]
					population[par_keys[par], ind + 1] = best_population[par_keys[par], n1]

		elif opts['xpoints'] == 'single':
			point = custom.random.uniform(0, len(par_keys))
			for par in range(len(par_keys)):
				# create children and do not swap parameters!
				if par <= point:
					population[par_keys[par], ind] = best_population[par_keys[par], n1]
					population[par_keys[par], ind + 1] = best_population[par_keys[par], n2]
				else:
					population[par_keys[par], ind] = best_population[par_keys[par], n2]
					population[par_keys[par], ind + 1] = best_population[par_keys[par], n1]

		# include the model id
		population['model', ind] = 'model_{:03d}_{:03d}'.format(iter + 1, ind)
		population['model', ind + 1] = 'model_{:03d}_{:03d}'.format(iter + 1, ind + 1)

		# include the error in the population dictionary
		population['error', ind] = opts['max_error']
		population['error', ind + 1] = opts['max_error']

	# mutate parameter values
	for ind in range(opts['pop_best'], opts['pop_size']):
		for par in range(len(par_keys)):
			if parameters[par_keys[par]][6] == 'factor':
				if float(parameters[par_keys[par]][7]) >= custom.random.random():
					population[par_keys[par], ind] = population[par_keys[par], ind] * \
						custom.random.uniform(1.0 - float(parameters[par_keys[par]][8]), 1.0 + float(parameters[par_keys[par]][8]))

			elif parameters[par_keys[par]][6] == 'uniform' or parameters[par_keys[par]][6] == 'loguniform':
				lower = float(parameters[par_keys[par]][7])
				upper = float(parameters[par_keys[par]][8])

				if parameters[par_keys[par]][9] is None and opts['mut_rate'] >= custom.random.random():
					if parameters[par_keys[par]][6] == 'uniform':
						population[par_keys[par], ind] = custom.random.uniform(lower, upper)
					if parameters[par_keys[par]][6] == 'loguniform':
						population[par_keys[par], ind] = numpy.exp(custom.random.uniform(numpy.log(lower), numpy.log(upper)))

				elif parameters[par_keys[par]][9] is not None and float(parameters[par_keys[par]][9]) >= custom.random.random():
					if parameters[par_keys[par]][6] == 'uniform':
						population[par_keys[par], ind] = custom.random.uniform(lower, upper)
					if parameters[par_keys[par]][6] == 'loguniform':
						population[par_keys[par], ind] = numpy.exp(custom.random.uniform(numpy.log(lower), numpy.log(upper)))

	return population

def clean():
	filelist = []
	fileregex = [
		'log*txt',    # log file
		'*.bngl',     # bng2 simulation files
		'*.xml',      # nfsim simulation files. Produced by bng2
		'*.rnf',      # nfsim configuration files
		'*.gdat',     # bng2 simulation outputs (SSA and others)
		'*.cdat',     # bng2 simulation outputs (CVODE)
		'*.species',  # bng2 network generation outputs
		'*.kappa',    # kasim original model and piskas simulations files.
		'model*.sh',  # kasim configuration files
		'model*.txt', # kasim, piskas simulation outputs. Also calculation error outputs
		opts['outfile'] + '*.txt', # summary per iteration
	]

	for regex in fileregex:
		filelist.append(glob.glob(regex))
	filelist = [ item for sublist in filelist for item in sublist ]

	for filename in filelist:
		if filename not in [ opts['model'] ]:
			os.remove(filename)

	return 0

def backup():
	results = opts['results'] + '_' + opts['systime']
	folders = {
		'ranking' : results + '/' + opts['ranking'],
		'parsets' : results + '/' + opts['parsets'],
		'rawdata' : results + '/' + opts['rawdata'],
		'fitness' : results + '/' + opts['fitness'],
	}

	# make backup folders
	os.mkdir(results)
	for folder in folders.values():
		os.mkdir(folder)

	# archive ranking files
	filelist = glob.glob('{:s}*.txt'.format(opts['outfile']))
	for filename in filelist:
		shutil.move(filename, folders['ranking'])

	# archive simulation outputs
	filelist = glob.glob('model_*.out.txt')
	for filename in filelist:
		shutil.move(filename, folders['rawdata'])

	# archive simulated models
	filelist = glob.glob('model_*.sh')
	for filename in filelist:
		shutil.move(filename, folders['parsets'])

	# archive goodness of fit outputs
	filelist = glob.glob('model_*.txt')
	for filename in filelist:
		shutil.move(filename, folders['fitness'])

	# archive a log file
	log_file = 'log_{:s}.txt'.format(opts['systime'])
	with open(log_file, 'w') as file:
		file.write('# Output of {:s} {:s}\n'.format(opts['python'], subprocess.list2cmdline(sys.argv[0:])))
	shutil.move(log_file, results)
	shutil.copy2(opts['model'], results)

	return 0

if __name__ == '__main__':
	sbatch_error = b'sbatch: error: slurm_receive_msg: Socket timed out on send/recv operation\n' \
		b'sbatch: error: Batch job submission failed: Socket timed out on send/recv operation'
	squeue_error = b'squeue: error: slurm_receive_msg: Socket timed out on send/recv operation\n' \
		b'slurm_load_jobs error: Socket timed out on send/recv operation'
	#sbatch_error = b'sbatch: error: Slurm temporarily unable to accept job, sleeping and retrying.'
	#sbatch_error = b'sbatch: error: Batch job submission failed: Resource temporarily unavailable'

	# general options
	args = argsparser()
	opts = ga_opts()
	seed = custom.random.seed(opts['rng_seed'])

	# perform safe checks prior to any calculation
	safe_checks()

	# clean the working directory
	clean()

	# read model configuration
	parameters = configurate()

	# Main Algorithm
	# generate k-dimensional grid of p-levels
	population = populate()
	population = simulate()
	#population = evaluate()
	#population = ranking()
	#population = mutate()

	# move and organize results
	#backup()
