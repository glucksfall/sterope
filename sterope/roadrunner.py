# -*- coding: utf-8 -*-

'''
Project "Sensitivity Analysis of Rule-Based Models", Rodrigo Santibáñez, 2019
Citation:
'''

__author__  = 'Rodrigo Santibáñez'
__license__ = 'gpl-3.0'
__software__ = 'kasim-v1.5.4'

import argparse, glob, multiprocessing, os, re, shutil, subprocess, sys, time, zipfile
import pandas, numpy, roadrunner

# import dask for distributed calculation
import dask, dask_jobqueue
from dask.distributed import Client

# import sensitivity samplers and methods
from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze
from SALib.sample.ff import sample as ff_sample
from SALib.analyze.ff import analyze as ff_analyze
from SALib.sample import fast_sampler, latin, saltelli
from SALib.analyze import delta, dgsm, fast, rbd_fast, sobol

def safe_checks():
	error_msg = ''
	if shutil.which(opts['kade']) is None:
		error_msg += 'KaDE (at {:s}) can\'t be called to perform simulations.\n' \
			'Check the path to KaDE.'.format(opts['kasim'])

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

def _parallel_analyze(data):
	seed = int(opts['seed'])
	samples = population['problem', 'samples']
	problem = population['problem', 'definition']

	if opts['method'] == 'sobol':
		return sobol.analyze(problem, data, calc_second_order = True, print_to_console = False)
	elif opts['method'] == 'fast':
		return fast.analyze(problem, data, print_to_console = False, seed = seed)
	elif opts['method'] == 'rbd-fast':
		return rbd_fast.analyze(problem, samples, data, print_to_console = False, seed = seed)
	elif opts['method'] == 'morris':
		return morris_analyze(problem, samples, data, print_to_console = False, seed = seed)
	elif opts['method'] == 'delta':
		return delta.analyze(problem, samples, data, print_to_console = False, seed = seed)
	elif opts['method'] == 'dgsm':
		return dgsm.analyze(problem, samples, data, print_to_console = False, seed = seed)
	elif opts['method'] == 'frac':
		return ff_analyze(problem, samples, data, second_order = True, print_to_console = False, seed = seed)
	else:
		return 0

def argsparser():
	parser = argparse.ArgumentParser(description = 'Perform a global sensitivity analysis of RBM parameters over the Dynamic Influence Network.', \
		epilog = 'Method shortnames are FAST, RBD-FAST, Morris, Sobol, Delta, DGSM, Frac\n' \
			'See https://salib.readthedocs.io/en/latest/api.html for more information',
		formatter_class = argparse.RawTextHelpFormatter)

	# required arguments to simulate models
	parser.add_argument('--model'  , metavar = 'str'  , type = str  , required = True , default = 'model.kappa'   , \
		help = 'RBM with tagged variables to analyze')
	parser.add_argument('--final'  , metavar = 'float', type = str  , required = True , default = '100'           , \
		help = 'limit time to simulate')
	parser.add_argument('--steps'  , metavar = 'float', type = str  , required = True , default = '1'             , \
		help = 'time step to simulate')

	# not required arguments to simulate models
	parser.add_argument('--tmin'   , metavar = 'float', type = str  , required = False, default = '0'             , \
		help = 'initial time to calculate the Dynamic Influence Network')
	parser.add_argument('--tmax'   , metavar = 'float', type = str  , required = False, default = None            , \
		help = 'final time to calculate the Dynamic Influence Network')
	parser.add_argument('--prec'   , metavar = 'str'  , type = str  , required = False, default = '7g'            , \
		help = 'precision and format of parameter values, default 7g')
	parser.add_argument('--syntax' , metavar = 'str'  , type = str  , required = False, default = '4'             , \
		help = 'KaSim syntax, default 4')

	# useful paths
	parser.add_argument('--kade'   , metavar = 'path' , type = str  , required = False, default = '~/bin/kade'    , \
		help = 'KaDE path, default ~/bin/kade')

	# general options for sensitivity analysis
	parser.add_argument('--method' , metavar = 'str'  , type = str  , required = False, default = 'Sobol'         , \
		help = 'methods supported by SALib')
	parser.add_argument('--seed'   , metavar = 'int'  , type = str  , required = False, default = None            , \
		help = 'seed for the sampler')
	parser.add_argument('--grid'   , metavar = 'int'  , type = str  , required = False, default = '10'            , \
		help = 'define the number of samples, default 10')
	parser.add_argument('--nprocs' , metavar = 'int'  , type = str  , required = False, default = '1'             , \
		help = 'perform calculations in parallel')

	# other options
	parser.add_argument('--results', metavar = 'path' , type = str  , required = False, default = 'results'       , \
		help = 'output folder where to move the results, default results (Sterope appends UNIX time string)')
	parser.add_argument('--samples', metavar = 'path' , type = str  , required = False, default = 'samples'       , \
		help = 'subfolder to save the generated models, default samples')
	parser.add_argument('--rawdata', metavar = 'path' , type = str  , required = False, default = 'simulations'   , \
		help = 'subfolder to save the simulations, default simulations')
	parser.add_argument('--reports', metavar = 'path' , type = str  , required = False, default = 'reports'       , \
		help = 'subfolder to save the calculated sensitivity, default reports')

	args = parser.parse_args()

	if args.tmax is None:
		args.tmax = args.final

	if args.seed is None:
		if sys.platform.startswith('linux'):
			args.seed = int.from_bytes(os.urandom(4), byteorder = 'big')
		else:
			parser.error('sterope requires --seed integer (to supply the samplers)')

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
		'kade'      : os.path.expanduser(args.kade), # kasim4 only
		# global SA options
		'method'    : args.method.lower(),
		'seed'      : args.seed,
		'p_levels'  : args.grid,
		'ntasks'    : int(args.nprocs),
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
	regex = '%\w+: \'(\w+)\' ' \
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

	population = {}
	# add samples to population dict
	population['problem', 'samples'] = models
	# add problem definition to population dict
	population['problem', 'definition'] = problem

	return population

def simulate():
	# write model in sbml format
	cmd = '{:s} -i {:s} --ode-backend sbml -syntax {:s} --output {:s}' \
		.format(opts['kade'], opts['model'], opts['syntax'], opts['model'].split('.')[0])
	cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
	_parallel_popen(cmd)

	# add simulations to the queue
	rr = roadrunner.RoadRunner(opts['model'].split('.')[0] + '.xml')

	population['problem', 'FluxControl'] = []
	for sample in population['problem', 'samples']:
		for index, par in enumerate(opts['par_name']):
			rr[par] = sample[index]
			population['problem', 'FluxControl'].append(
				numpy.asarray(rr.getUnscaledFluxControlCoefficientMatrix())
				)

	population['problem', 'reactions'] = []
	for key in rr.keys():
		if key.startswith('re'):
			population['problem', 'reactions'].append(key)

	return population

def evaluate():
	sensitivity = {
		'FluxControl' : {},
		}

	FluxControl = population['problem', 'FluxControl']

	# Control fluxes are not that easy to evaluate recursively; data needs to be reshaped
	a, b = numpy.shape(FluxControl[0])
	FluxControl = [ x[0] for x in [ numpy.reshape(x, (1, a*b)) for x in FluxControl ] ]
	FluxControl = [ numpy.asarray(x) for x in numpy.transpose(FluxControl) ]

	#with multiprocessing.Pool(opts['ntasks'] - 1) as pool:
		#sensitivity['din_fluxes'] = pool.map(_parallel_analyze, din_fluxes, chunksize = opts['ntasks'])

	# queue to dask.delayed
	results = []
	for x in FluxControl:
		y = dask.delayed(_parallel_analyze)(x)
		results.append(y)

	# compute results
	sensitivity['FluxControl'] = dask.compute(*results)

	return sensitivity

def report():
	reports = {
		'FluxControl' : {},
		}

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

	# write reports for FluxControl; data need to be reshaped
	# name index: parameter sensitivities over the influence of a rule over a 2nd rule
	rules_names = population['problem', 'reactions']
	first = [ y for x in [ [x]*len(rules_names) for x in rules_names ] for y in x ]
	second = rules_names * len(rules_names)

	x = sensitivity['FluxControl']
	x = { (k1, k2) : v for k1, k2, v in zip(first, second, x) }
	for key in indexes:
		reports['FluxControl'][key] = pandas.DataFrame([ x[k][key] for k in x.keys() ], columns = opts['par_name']).fillna(0)
		reports['FluxControl'][key]['1st'] = first
		reports['FluxControl'][key]['2nd'] = second
		reports['FluxControl'][key].set_index(['1st', '2nd'], inplace = True)

		with open('./report_FluxControl_{:s}.txt'.format(key), 'w') as file:
			reports['FluxControl'][key].to_csv(file, sep = '\t')

	if opts['method'] == 'sobol' or opts['method'] == 'frac':
		if opts['method'] == 'sobol':
			keys = ['S2', 'S2_conf']
		if opts['method'] == 'frac':
			keys = ['IE']
		for key in keys:
			tmp = [pandas.DataFrame(x[k][key], columns = opts['par_name'], index = opts['par_name']).stack() for k in x.keys()]
			reports['FluxControl'][key] = pandas.DataFrame(tmp).fillna(0)
			reports['FluxControl'][key]['1st'] = first
			reports['FluxControl'][key]['2nd'] = second
			reports['FluxControl'][key].set_index(['1st', '2nd'], inplace = True)

			with open('./report_FluxControl_{:s}.txt'.format(key), 'w') as file:
				reports['FluxControl'][key].to_csv(file, sep = '\t')

	return sensitivity

def clean():
	filelist = []
	fileregex = [
		'slurm*',     # slurm log files
		'*.xml'       # xml files
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
		'reports' : results + '/' + opts['reports'],
	}

	# make backup folders
	os.mkdir(results)
	for folder in folders.values():
		os.mkdir(folder)

	# archive reports
	filelist = glob.glob('report_*.txt')
	for filename in filelist:
		shutil.move(filename, folders['reports'])

	# archive a log file
	log_file = 'log_{:s}.txt'.format(opts['systime'])
	with open(log_file, 'w') as file:
		file.write('# Output of python3 {:s}\n'.format(subprocess.list2cmdline(sys.argv[0:])))
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
	clean()

	# read model configuration
	parameters = configurate()

	# Sterope Main Algorithm
	population = populate()
	# simulate levels
	population = simulate()
	# evaluate sensitivity
	sensitivity = evaluate()
	# write reports
	report()

	# move and organize results
	backup()
