 # -*- coding: utf-8 -*-

'''
Project "Sensitivity Analysis of Rule-Based Models", Rodrigo Santibáñez, 2019
Citation:
'''

__author__  = 'Rodrigo Santibáñez'
__license__ = 'gpl-3.0'
__software__ = 'kasim-v4.0'

import argparse, glob, json, multiprocessing, os, re, shutil, subprocess, sys, time, zipfile
import pandas, numpy

# import dask for distributed calculation
import dask, dask_jobqueue
from dask.distributed import Client, LocalCluster

# import astropy to perform bootstrapping
from astropy.stats import bootstrap

# import sensitivity samplers and methods
from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze
from SALib.sample.ff import sample as ff_sample
from SALib.analyze.ff import analyze as ff_analyze
from SALib.sample import fast_sampler, latin, saltelli
from SALib.analyze import delta, dgsm, fast, rbd_fast, sobol

def safe_checks():
	error_msg = ''
	if shutil.which(opts['kasim']) is None:
		error_msg += 'KaSim (at {:s}) can\'t be called to perform simulations.\n' \
			'Check the path to KaSim.'.format(opts['kasim'])

	# check if model file exists
	if not os.path.isfile(opts['model']):
		error_msg += 'The "{:s}" file cannot be opened.\n' \
		'Please, check the path to the model file.\n'.format(opts['model'])

	# print error
	if error_msg != '':
		print(error_msg)
		raise ValueError(error_msg)

	return 0

def _parallel_popen(cmd):
	proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
	out, err = proc.communicate()
	proc.wait()
	return out, err

def _parallel_analyze(index, method, problem, samples, data, seed):
	if method == 'sobol':
		result = sobol.analyze(problem, data, calc_second_order = True, print_to_console = False, seed = seed)
	elif method == 'fast':
		result = fast.analyze(problem, data, print_to_console = False, seed = seed)
	elif method == 'rbd-fast':
		result = rbd_fast.analyze(problem, samples, data, print_to_console = False, seed = seed)
	elif method == 'morris':
		result = morris_analyze(problem, samples, data, print_to_console = False, seed = seed)
	elif method == 'delta':
		result = delta.analyze(problem, samples, data, print_to_console = False, seed = seed)
	elif method == 'dgsm':
		result = dgsm.analyze(problem, samples, data, print_to_console = False, seed = seed)
	elif method == 'frac':
		result = ff_analyze(problem, samples, data, second_order = True, print_to_console = False, seed = seed)
	else:
		return 0

	for key in result.keys():
		result[key] = result[key].tolist()

	with open('{:s}.json'.format(index), 'w') as outfile:
		json.dump(result, outfile)

	return 0

def argsparser():
	parser = argparse.ArgumentParser(description = 'Perform a global sensitivity analysis of RBM parameters over the Dynamic Influence Network.',
			epilog = 'Method shortnames are FAST, RBD-FAST, Morris, Sobol, Delta, DGSM, and Frac (case-insensitive)\n' \
			'See https://salib.readthedocs.io/en/latest/api.html for more information',
		formatter_class = argparse.RawTextHelpFormatter)

	# required arguments to simulate models
	parser.add_argument('--model'  , metavar = 'str'  , type = str  , required = True , default = 'model.kappa', help = 'RBM with tagged variables to analyze. Default model.kappa.')
	parser.add_argument('--final'  , metavar = 'float', type = str  , required = True , default = '100'        , help = 'Limit time to simulate. Default 100.')
	parser.add_argument('--steps'  , metavar = 'float', type = str  , required = True , default = '1.0'        , help = 'Time step to simulate. Default 1.0.')

	# not required arguments to simulate models
	parser.add_argument('--tmin'   , metavar = 'float', type = str  , required = False, default = '0'          , help = 'Initial time to calculate the Dynamic Influence Network. Default 0.')
	parser.add_argument('--tmax'   , metavar = 'float', type = str  , required = False, default = None         , help = 'Final time to calculate the Dynamic Influence Network. Default --final.')
	parser.add_argument('--prec'   , metavar = 'str'  , type = str  , required = False, default = '7g'         , help = 'Precision and format of parameter values. Default 7g.')
	parser.add_argument('--syntax' , metavar = 'str'  , type = str  , required = False, default = '4'          , help = 'KaSim syntax. Default 4.')

	# useful paths
	parser.add_argument('--kasim'  , metavar = 'path' , type = str  , required = False, default = 'kasim4'     , help = 'KaSim path. Default kasim4')

	# general options for sensitivity analysis
	parser.add_argument('--method' , metavar = 'str'  , type = str  , required = False, default = 'Sobol'      , help = 'Methods supported by SALib. Default Sobol.')
	parser.add_argument('--seed'   , metavar = 'int'  , type = str  , required = False, default = None         , help = 'Seed for the sampler. Default None.')
	parser.add_argument('--grid'   , metavar = 'int'  , type = str  , required = False, default = '10'         , help = 'Number of parameter levels. Default 10.')
	parser.add_argument('--nsims'  , metavar = 'int'  , type = str  , required = False, default = '10'         , help = 'Number of simulations per parameter sample. Default 10.')
	parser.add_argument('--boots'  , metavar = 'int'  , type = str  , required = False, default = '100'        , help = 'Number of bootstraps to obtain a representative simulation. Default 100.')
	parser.add_argument('--min'    , metavar = 'int'  , type = str  , required = False, default = '1'          , help = 'Perform calculations in a cluster of min workers. Default 1.')
	parser.add_argument('--max'    , metavar = 'int'  , type = str  , required = False, default = '8'          , help = 'Perform calculations in a cluster of max workers. Default 8.')
	parser.add_argument('--memory' , metavar = 'str'  , type = str  , required = False, default = '1GB'        , help = 'Memory for each worker, including unit. Default 1GB.')

	# TODO slice the simulation and perform global sensitivity analysis for each slice (generalization of tmin-tmax)
	#parser.add_argument('--type'   , metavar = 'str'  , type = str  , required = False, default = 'total'		 , help = 'total or sliced sensitivity analysis')
	#parser.add_argument('--tick'   , metavar = 'float', type = str  , required = False, default = '0.0'		   , help = 'sliced SA: ...')
	#parser.add_argument('--size'   , metavar = 'float', type = str  , required = False, default = '1.0'		   , help = 'sliced SA: ...')
	#parser.add_argument('--beat'   , metavar = 'float', type = str  , required = False, default = '0.3'		   , help = 'sliced SA: time step to calculate DIN')

	#continue analysis without cleaning files
	parser.add_argument('--fstep'  , metavar = 'int'  , type = str  , required = False, default = '1'          , help = 'Continue from (Default 1):\n\t1) Generate models,\n\t2) Simulate models,\n\t3) Bootstrap simulations,\n\t4) Analyze bootstraps.')

	# other options
	parser.add_argument('--results', metavar = 'path' , type = str  , required = False, default = 'results'    , help = 'Output folder where to move the results. Default results.')
	parser.add_argument('--samples', metavar = 'path' , type = str  , required = False, default = 'samples'    , help = 'Subfolder to save the generated models. Default samples.')
	parser.add_argument('--rawdata', metavar = 'path' , type = str  , required = False, default = 'simulations', help = 'Subfolder to save the simulations. Default simulations.')
	parser.add_argument('--reports', metavar = 'path' , type = str  , required = False, default = 'reports'    , help = 'Subfolder to save the calculated sensitivity. Default reports.')

	args = parser.parse_args()

	if args.tmax is None:
		args.tmax = args.final

	if args.seed is None:
		if sys.platform.startswith('linux'):
			args.seed = int.from_bytes(os.urandom(4), byteorder = 'big')
		else:
			parser.error('sterope requires --seed int (to supply the samplers)')

	return args

def ga_opts():
	return {
		# user defined options
		# simulate models
		'model'     : args.model,
		'final'     : args.final,
		'steps'     : args.steps,
		# optional to simulate models
		'tmin'      : args.tmin,
		'tmax'      : args.tmax,
		'par_prec'  : args.prec,
		'syntax'    : args.syntax,
		# path to software
		'kasim'     : os.path.expanduser(args.kasim), # kasim4 only
		# global SA options
		'method'    : args.method.lower(),
		'seed'      : args.seed,
		'p_levels'  : args.grid,
		'min'       : int(args.min),
		'max'       : int(args.max),
		'nsims'     : int(args.nsims),
		'resamples' : int(args.boots),
		'memory'    : args.memory,
		# sliced global SA options
		#'type'  : args.type,
		#'size'  : args.size,
		#'tick'  : args.tick,
		#'beat'  : args.beat,
		# continue from
		'continue'  : args.fstep,
		# saving to
		'results'   : args.results,
		'samples'   : args.samples,
		'rawdata'   : args.rawdata,
		'reports'   : args.reports,
		# non-user defined options
		'home'      : os.getcwd(),
		'null'      : '/dev/null',
		'systime'   : str(time.time()).split('.')[0],
		# useful data
		'par_name'  : [],
		}

def configurate():
	# read the model
	data = []
	with open(opts['model'], 'r') as infile:
		for line in infile:
			data.append(line)

	# find parameters to analyze
	regex = '%\w+:\s+\'(\w+)\'\s+' \
			'([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)\s+(?:\/\/|#)\s+' \
			'(\w+)\[([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)\s+' \
			'([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)\]\n'

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
	# 'parameters' dictionary stores each line in the model
	par_keys = list(parameters.keys())

	# init problem definiton
	seed = int(opts['seed'])
	levels = int(opts['p_levels'])
	problem = {
		'names': opts['par_name'],
		'num_vars': len(opts['par_name']),
		'bounds': [],
		}

	# define bounds following the model configuration
	for line in range(len(par_keys)):
		if parameters[line][0] == 'par':
			if parameters[line][3] == 'range':
				lower = float(parameters[par_keys[line]][4])
				upper = float(parameters[par_keys[line]][5])
			if parameters[line][3] == 'factor':
				lower = float(parameters[line][2]) * (1 - float(parameters[par_keys[line]][4]))
				upper = float(parameters[line][2]) * (1 + float(parameters[par_keys[line]][5]))
			problem['bounds'].append([lower, upper])

	# create samples to simulate
	if opts['method'] == 'sobol':
		models = saltelli.sample(problem = problem, N = levels, calc_second_order = True, seed = seed)
	elif opts['method'] == 'fast':
		models = fast_sampler.sample(problem = problem, N = levels, seed = seed)
	elif opts['method'] == 'rbd-fast' or opts['method'] == 'delta' or opts['method'] == 'dgsm':
		models = latin.sample(problem = problem, N = levels, seed = seed)
	elif opts['method'] == 'morris':
		models = morris_sample(problem = problem, N = levels)
	elif opts['method'] == 'frac':
		models = ff_sample(problem, seed = seed)
	else:
		error_msg = 'Wrong method name.'
		print(error_msg)
		raise ValueError(error_msg)

	# add samples to population dict
	population = {}
	population['problem', 'samples'] = models

	# write models
	model_string = 'level{:0' + str(len(str(len(models)))) + 'd}'

	for model_index, model in enumerate(models):
		model_key = model_string.format(model_index+1)
		population[model_key, 'model'] = model_key
		for par_index, par_name in enumerate(opts['par_name']):
			population[model_key, par_name] = models[model_index][par_index]

	# generate a kappa file per model
	par_string = '%var: \'{:s}\' {:.' + opts['par_prec'] + '}\n'

	if opts['continue'] == '1':
		for model in sorted(population.keys()):
			if model[1] == 'model':
				for simulation in range(opts['nsims']):
					model_key = model[0]
					model_name = population[model_key, 'model']

					# define pertubation to the kappa model that indicates KaSim to calculates the Dinamic Influence Network
					#if opts['type'] == 'total':
					if opts['syntax'] == '4':
						flux = '%mod: [T] > {:s} do $DIN \"flux_{:s}_{:03d}.json\" [true];\n'.format(opts['tmin'], model_key, simulation)
						flux += '%mod: [T] > {:s} do $DIN \"flux_{:s}_{:03d}.json\" [false];'.format(opts['tmax'], model_key, simulation)
					else: # kappa3.5 uses $FLUX instead of $DIN
						flux = '%mod: [T] > {:s} do $FLUX \"flux_{:s}_{:03d}.json\" [true]\n'.format(opts['tmin'], model_key, simulation)
						flux += '%mod: [T] > {:s} do $FLUX \"flux_{:s}_{:03d}.json\" [false]'.format(opts['tmax'], model_key, simulation)

				#else: # sliced global sensitivity analysis
					#if opts['syntax'] == '4':
						#flux = '%mod: repeat (([T] > DIM_clock) && (DIM_tick > (DIM_length - 1))) do $DIN "flux_".(DIM_tick - DIM_length).".json" [false] until [false];'
					#else: # kappa3.5 uses $FLUX instead of $DIN
						#flux = '\n# Added to calculate a sliced global sensitivity analysis\n'
						#flux += '%var: \'DIN_beat\' {:s}\n'.format(opts['beat'])
						#flux += '%var: \'DIN_length\' {:s}\n'.format(opts['size'])
						#flux += '%var: \'DIN_tick\' {:s}\n'.format(opts['tick'])
						#flux += '%var: \'DIN_clock\' {:s}\n'.format(opts['tmin'])
						#flux += '%mod: repeat (([T] > DIN_clock) && (DIN_tick > (DIN_length - 1))) do '
						#flux += '$FLUX \"flux_{:s}\".(DIN_tick - DIN_length).\".json\" [false] until [false]\n'.format(model_key)
						#flux += '%mod: repeat ([T] > DIN_clock) do '
						#flux += '$FLUX "flux_{:s}".DIN_tick.".json" "probability" [true] until ((((DIN_tick + DIN_length) + 1) * DIN_beat) > [Tmax])\n'.format(model_key)
						#flux += '%mod: repeat ([T] > DIN_clock) do $UPDATE DIN_clock (DIN_clock + DIN_beat); $UPDATE DIN_tick (DIN_tick + 1) until [false]'

					model_path = './model_{:s}_{:03d}.kappa'.format(model_name, simulation)
					if not os.path.exists(model_path):
						with open(model_path, 'w') as file:
							for line in par_keys:
								if parameters[line][0] == 'par':
									file.write(par_string.format(parameters[line][1], population[model_key, parameters[line][1]]))
								else:
									file.write(parameters[line])
							# add the DIN perturbation at the end of the kappa file
							file.write(flux)

	# add problem definition to population dict
	population['problem', 'definition'] = problem

	return population

def simulate():
	# add simulations to the queue
	squeue = []

	for model in sorted(population.keys()):
		for simulation in range(opts['nsims']):
			if model[1] == 'model':
				model_name = population[model[0], 'model']
				output = 'model_{:s}_{:03d}.out.txt'.format(model_name, simulation)

				cmd = '{:s} -i model_{:s}_{:03d}.kappa -l {:s} -p {:s} -o {:s} -syntax {:s} --no-log'
				cmd = cmd.format(opts['kasim'], model_name, simulation, opts['final'], opts['steps'], output, opts['syntax'])
				cmd = os.path.expanduser(cmd)
				cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
				squeue.append(cmd)

	# simulate the queue with multiprocessing.Pool
	#with multiprocessing.Pool(opts['ntasks'] - 1) as pool:
		#pool.map(_parallel_popen, sorted(squeue), chunksize = opts['ntasks'] - 1)

	#results = []
	#for cmd in numpy.asarray(sorted(squeue)):
		#y = dask.delayed(_parallel_popen)(cmd)
		#results.append(y)

	#dask.compute(*results)

	futures = client.map(_parallel_popen, squeue)
	client.gather(futures)

	return 0

def bootstrapping():
	# read simulations of each model
	for model in sorted(population.keys()):
		if model[1] == 'model':
			din_hits = [] # list of column vectors, one vector per rule
			din_fluxes = [] # list of square numpy arrays, but not symmetrics

			model_key = model[0]

			files = sorted(glob.glob('./flux_{:s}_*json'.format(model_key)))
			for file in files:
				with open(file, 'r') as infile:
					data = pandas.read_json(infile)

				# vector column of lists
				din_hits.append(data['din_hits'].iloc[1:].values)
				# reshape matrix of fluxes into a vector column of lists
				tmp = [ x for x in data['din_fluxs'] ]
				din_fluxes.append(pandas.DataFrame(tmp).values) # easy conversion of a list of lists into a numpy array

			# DIN hits are easy to evaluate recursively (for-loop), parallelized (multiprocessing) or distributed (dask)
			din_hits = [ numpy.asarray(x) for x in numpy.transpose(din_hits) ]

			# DIN fluxes are not that easy to evaluate recursively; data needs to be reshaped
			a, b = numpy.shape(din_fluxes[0][1:, 1:])
			din_fluxes = [ x[0] for x in [ numpy.reshape(x[1:,1:], (1, a*b)) for x in din_fluxes ] ]
			din_fluxes = [ numpy.asarray(x) for x in numpy.transpose(din_fluxes) ]

			# bootstrap
			tmp = []
			for row in numpy.array(din_hits):
				tmp.append(numpy.mean(bootstrap(row, opts['resamples'], bootfunc = numpy.mean)))
			with open('./hits_bootstrapped_{:s}.txt'.format(model_key), 'w') as outfile:
				pandas.DataFrame(data = tmp).to_csv(outfile, sep = '\t', index = False, header = False)

			tmp = []
			for row in numpy.array(din_fluxes):
				tmp.append(numpy.mean(bootstrap(row, opts['resamples'], bootfunc = numpy.mean)))
			with open('./fluxes_bootstrapped_{:s}.txt'.format(model_key), 'w') as outfile:
				pandas.DataFrame(data = tmp).to_csv(outfile, sep = '\t', index = False, header = False)

	return 0

def evaluate():
	sensitivity = {
		'din_hits' : {},
		'din_fluxes' : {},
		}

	din_hits = [] # list of column vectors, one vector per rule
	din_fluxes = [] # list of square numpy arrays, but not sym(cmd)metric

	# read observations
	files = sorted(glob.glob('./hits_bootstrapped_*txt'))
	for file in files:
		with open(file, 'r') as infile:
			din_hits.append(pandas.read_csv(infile, header = None).values)
	din_hits = [ numpy.asarray(x) for x in numpy.transpose(din_hits) ]

	files = sorted(glob.glob('./fluxes_bootstrapped_*txt'))
	for file in files:
		with open(file, 'r') as infile:
			din_fluxes.append(pandas.read_csv(infile, header = None).values)
	din_fluxes = [ numpy.asarray(x) for x in numpy.transpose(din_fluxes) ]

	# estimate sensitivities
	#with multiprocessing.Pool(opts['ntasks'] - 1) as pool:
		#sensitivity['din_hits'] = pool.map(_parallel_analyze, din_hits, chunksize = opts['ntasks'] - 1)

	# queue to dask.delayed and compute
	results = []
	#for x in din_hits:
	#for index, x in enumerate(din_hits):
		#y = dask.delayed(_parallel_analyze)(index, x)
		#results.append(y)
	#dask.compute(*results)

	# submit to client and compute
	seed = int(opts['seed'])
	method = opts['method']
	samples = population['problem', 'samples']
	problem = population['problem', 'definition']

	for index, x in enumerate(din_hits[0]):
		print('hits: ' + str(index))
		results.append(client.submit(_parallel_analyze, 'sensitivity_indexes_hits_' + str(index), method, problem, samples, x, seed))
	client.gather(results)

	sensitivity['din_hits'] = []
	for index, x in enumerate(din_hits[0]):
		with open('sensitivity_indexes_hits_' + str(index) + '.json', 'r') as infile:
			sensitivity['din_hits'].append(json.load(infile))

	#with multiprocessing.Pool(opts['ntasks'] - 1) as pool:
		#sensitivity['din_fluxes'] = pool.map(_parallel_analyze, din_fluxes, chunksize = opts['ntasks'])

	# queue to dask.delayed and compute
	results = []
	#for x in din_fluxes:
	#for index, x in enumerate(din_fluxes):
		#y = dask.delayed(_parallel_analyze)(index, x)
		#results.append(y)
	#dask.compute(*results)

	# submit to client and compute
	for index, x in enumerate(din_fluxes[0]):
		print('fluxs: ' + str(index))
		results.append(client.submit(_parallel_analyze, 'sensitivity_indexes_fluxs_' + str(index), method, problem, samples, x, seed))
	client.gather(results)

	sensitivity['din_fluxes'] = []
	for index, x in enumerate(din_fluxes[0]):
		with open('sensitivity_indexes_fluxs_' + str(index) + '.json', 'r') as infile:
			sensitivity['din_fluxes'].append(json.load(infile))

	return sensitivity

def report():
	# get rule names from one DIN file
	files = sorted(glob.glob('./flux_level*_*.json'))
	with open(files[0], 'r') as file:
		lst = pandas.read_json(file)

	reports = {
		'DINhits' : {},
		'DINfluxes' : {},
		}

	# write reports for DIN hits
	x = sensitivity['din_hits']
	x = { k : v for k, v in zip(lst['din_rules'][1:], x) }

	# save corresponding indexes
	if opts['method'] == 'sobol':
		indexes = ['S1', 'S1_conf', 'ST', 'ST_conf']
	elif opts['method'] == 'fast':
		indexes = ['S1', 'ST']
	elif opts['method'] == 'rbd-fast':
		indexes = ['S1']
	elif opts['method'] == 'morris':
		indexes = ['mu', 'mu_star', 'sigma', 'mu_star_conf']
	elif opts['method'] == 'delta':
		indexes = ['delta', 'delta_conf', 'S1', 'S1_conf']
	elif opts['method'] == 'dgsm':
		indexes = ['vi', 'vi_std', 'dgsm', 'dgsm_conf']
	elif opts['method'] == 'frac':
		indexes = ['ME', 'IE']

	for key in indexes:
		reports['DINhits'][key] = pandas.DataFrame([ x[k][key] for k in x.keys() ],
			columns = opts['par_name'], index = lst['din_rules'][1:]).rename_axis('rules')

		with open('./report_DINhits_{:s}.txt'.format(key), 'w') as file:
			reports['DINhits'][key].to_csv(file, sep = '\t')

	if opts['method'] == 'sobol' or opts['method'] == 'frac':
		if opts['method'] == 'sobol':
			keys = ['S2', 'S2_conf']
		if opts['method'] == 'frac':
			keys = ['IE']
		for key in keys:
			tmp = [ pandas.DataFrame(x[k][key], columns = opts['par_name'], index = opts['par_name']).stack() for k in x.keys() ]
			reports['DINhits'][key] = pandas.DataFrame(tmp, index = lst['din_rules'][1:]).rename_axis('rules')

			with open('./report_DINhits_{:s}.txt'.format(key), 'w') as file:
				reports['DINhits'][key].to_csv(file, sep = '\t')

	# write reports for DIN fluxes; data need to be reshaped
	# name index: parameter sensitivities over the influence of a rule over a 2nd rule
	rules_names = list(lst['din_rules'][1:])
	first = [ y for x in [ [x]*len(rules_names) for x in rules_names ] for y in x ]
	second = rules_names * len(rules_names)

	x = sensitivity['din_fluxes']
	x = { (k1, k2) : v for k1, k2, v in zip(first, second, x) }
	for key in indexes:
		reports['DINfluxes'][key] = pandas.DataFrame([ x[k][key] for k in x.keys() ], columns = opts['par_name']).fillna(0)
		reports['DINfluxes'][key]['1st'] = first
		reports['DINfluxes'][key]['2nd'] = second
		reports['DINfluxes'][key].set_index(['1st', '2nd'], inplace = True)

		with open('./report_DINfluxes_{:s}.txt'.format(key), 'w') as file:
			reports['DINfluxes'][key].to_csv(file, sep = '\t')

	if opts['method'] == 'sobol' or opts['method'] == 'frac':
		if opts['method'] == 'sobol':
			keys = ['S2', 'S2_conf']
		if opts['method'] == 'frac':
			keys = ['IE']
		for key in keys:
			tmp = [pandas.DataFrame(x[k][key], columns = opts['par_name'], index = opts['par_name']).stack() for k in x.keys()]
			reports['DINfluxes'][key] = pandas.DataFrame(tmp).fillna(0)
			reports['DINfluxes'][key]['1st'] = first
			reports['DINfluxes'][key]['2nd'] = second
			reports['DINfluxes'][key].set_index(['1st', '2nd'], inplace = True)

			with open('./report_DINfluxes_{:s}.txt'.format(key), 'w') as file:
				reports['DINfluxes'][key].to_csv(file, sep = '\t')

	return sensitivity

def clean():
	filelist = []
	fileregex = [
		'slurm*',	  # slurm log files
		'*.json',	  # DIN files and sensitivity files
		'log*.txt',	# log file
		'*.kappa',	 # kasim model files.
		'model_*.txt', # kasim simulation outputs.
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
		'samples' : results + '/' + opts['samples'],
		'rawdata' : results + '/' + opts['rawdata'],
		'reports' : results + '/' + opts['reports'],
		'others'  : results + '/other_files'
	}

	# make backup folders
	os.mkdir(results)
	for folder in folders.values():
		os.mkdir(folder)

	# archive model files
	filelist = glob.glob('model_*.kappa')
	for filename in filelist:
		shutil.move(filename, folders['samples'])

	# archive fluxes outputs and simulations
	filelist = glob.glob('flux_*.json')
	for filename in filelist:
		shutil.move(filename, folders['rawdata'])

	filelist = glob.glob('model_*.out.txt')
	for filename in filelist:
		shutil.move(filename, folders['rawdata'])

	# archive reports
	filelist = glob.glob('report_*.txt')
	for filename in filelist:
		shutil.move(filename, folders['reports'])

	# archive others
	filelist = glob.glob('*bootstrapped*')
	for filename in filelist:
		shutil.move(filename, folders['others'])
	filelist = glob.glob('fluxs*.json')
	for filename in filelist:
		shutil.move(filename, folders['others'])

	# archive a log file
	log_file = 'log_{:s}.txt'.format(opts['systime'])
	with open(log_file, 'w') as file:
		file.write('# Output of python3 {:s}\n'.format(subprocess.list2cmdline(sys.argv[0:])))
		file.write('Elapsed time: {:.0f} seconds\n'.format(time.time() - float(opts['systime'])))
	shutil.move(log_file, results)
	shutil.copy2(opts['model'], results)

	# compress the results folder
	with zipfile.ZipFile(results + '.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
		for root, dirs, files in os.walk(results):
			for file in files:
				zipf.write(os.path.join(root, file))

	return 0

if __name__ == '__main__':
	# general options
	args = argsparser()
	opts = ga_opts()

	# perform safe checks prior to any calculation
	safe_checks()

	# clean the working directory
	if opts['continue'] == '1':
		clean()

	if opts['continue'] != '1':
		# create SLURM Cluster if available; if not, use multiprocessing
		slurm = os.getenv('SLURM_JOB_PARTITION', None)

		if slurm != None:
			cluster = dask_jobqueue.SLURMCluster(
				queue = os.environ['SLURM_JOB_PARTITION'],
				cores = 1, walltime = '0', memory = opts['memory'],
				local_directory = os.getenv('TMPDIR', '/tmp'))
			cluster.adapt(minimum_jobs = opts['min'], maximum_jobs = opts['max'])
			client = Client(cluster)

		else:
			cluster = LocalCluster(
				n_workers = opts['max'],
				processes = True,
				threads_per_worker = 1,
				local_directory = os.getenv('TMPDIR', '/tmp'))
			client = Client(cluster)

	# read model configuration
	parameters = configurate()

	# Sterope Main Algorithm
	population = populate()

	if opts['continue'] == '2':
		# simulate levels
		simulate()

	if opts['continue'] == '3':
		# bootstrapping
		bootstrapping()

	if opts['continue'] == '4':
		# evaluate sensitivity
		sensitivity = evaluate()

	# write reports
	report()

	# move and organize results
	backup()
	clean()
