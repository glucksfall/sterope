#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
Project "Genetic Algorithm for Rule-Based Models", Rodrigo Santibáñez, 2017 @ Dlab, FCV (rsantibanez@dlab.cl)
A Python implementation of Alberto Martin's Genetic Algorithm, 2016 @ Dlab, FCV (ajmm@dlab.cl)
To be used with KaSim. Please refer to other subprojects for other stochastic simulators support
Citation:
'''

__author__  = 'Rodrigo Santibáñez'
__license__ = 'gpl-3.0'
__software__ = 'kasim-v4.0'

import argparse, sys
import pandas, numpy

def argsparser():
	parser = argparse.ArgumentParser(description = 'Calculate goodness of fit between data and simulations.')
	parser.add_argument('--data' , metavar = 'path', type = str, required = True , nargs = '+', help = 'data files')
	parser.add_argument('--sims' , metavar = 'path', type = str, required = True , nargs = '+', help = 'KaSim output without further processing')
	parser.add_argument('--file' , metavar = 'path', type = str, required = True , nargs = 1  , help = 'output file name')
	parser.add_argument('--crit' , metavar = 'path', type = str, required = False, nargs = 1  , help = 'Mann-Whitney U-test critical values')
	parser.add_argument('--error', metavar = 'str' , type = str, required = True , nargs = '+', help = 'Goodness of Fit Function(s) to calculate')

	return parser.parse_args()

# read simulation files
def read_sims(files):
	sims = []
	for infile in files:
		with open(infile, 'r') as file:
			sims.append(pandas.read_csv(file, delimiter = ',', skiprows = 2, header = 0, engine = 'python').set_index('[T]', drop = False).rename_axis(None, axis = 0).drop('[T]', axis = 1))

	return pandas.concat(sims, keys = range(len(sims))), len(sims)

# read the data files
def read_data(files):
	data = []
	for infile in files:
		with open(infile, 'r') as file:
			data.append(pandas.read_csv(file, delimiter = ',', header = 0, engine = 'python').set_index('[T]', drop = False).rename_axis(None, axis = 0).drop('[T]', axis = 1))

	return pandas.concat(data, keys = range(len(data))), len(data)

def do(error):
	"""
	# Fitness Calculation Template:
	if set(args.error).issuperset(set(['the-acronysm'])):
		func = 0
		func = an algebraic expression combining the data average (data_avrg), data variance (data_stdv), simulation average (sims_stdv),
		single experimental files (data.loc[i]) and/or simulation files (sims.loc[i]).
		# Please consider this variables are DataFrames, meaning that division is a method (pandas.DataFrame.division)
		# Please consider use data.loc[i] and sims.loc[i] if average or standard deviation values are needed from them (as in MSE)
		# drop NaN values (from experimental data without simulation point or vice-versa), sum the two dimensions, and return a 6 float points scientific notation number
		error['acronysm'] = '{:.6e}'.format(func.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum())
	"""

	# mean square error
	if set(args.error).issuperset(set(['MSE'])):
		func = 0

		data_avrg = 0
		for i in range(len_data):
			data_avrg += data.loc[i].divide(len_data)

		sims_avrg = 0
		for j in range(len_sims):
			sims_avrg += sims.loc[j].divide(len_sims)

		func = (data_avrg - sims_avrg)**2
		func = func.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum()

		error['MSE'] = '{:.6e}'.format(func)

	# mean absolute error
	if set(args.error).issuperset(set(['MAE'])):
		func = 0

		data_avrg = 0
		for i in range(len_data):
			data_avrg += data.loc[i].divide(len_data)

		sims_avrg = 0
		for j in range(len_sims):
			sims_avrg += sims.loc[j].divide(len_sims)

		func = abs(data_avrg - sims_avrg)

		error['MAE'] = '{:.6e}'.format(func.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum())

	# sum of squares (from BioNetFit paper)
	if set(args.error).issuperset(set(['SSQ'])):
		func = 0

		for i in range(len_data):
			for j in range(len_sims):
				func += (data.loc[i] - sims.loc[j])**2

		error['SSQ'] = '{:.6e}'.format(func.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum())

	# chi-square (from BioNetFit paper)
	if set(args.error).issuperset(set(['CHISQ'])):
		func = 0

		data_avrg = 0
		for i in range(len_data):
			data_avrg += data.loc[i].divide(len_data)

		data_stdv = 0
		if len_data > 1:
			for i in range(len_data):
				data_stdv += ((data.loc[i] - data_avrg)**2).divide(len_data - 1)
		else:
			data_stdv = stdv**2

		for i in range(len_data):
			for j in range(len_sims):
				func += ((data.loc[i] - sims.loc[j]).divide(data_stdv**0.5))**2

		error['CHISQ'] = '{:.6e}'.format(func.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum())

	# mean normalized square error (from BioNetFit paper)
	if set(args.error).issuperset(set(['MNSE'])):
		func = 0

		data_avrg = 0
		for i in range(len_data):
			data_avrg += data.loc[i].divide(len_data)

		for i in range(len_data):
			for j in range(len_sims):
				func += ((data.loc[i] - sims.loc[j]).divide(data_avrg))**2

		error['MNSE'] = '{:.6e}'.format(func.replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum())

	# pair-wise square deviation
	if set(args.error).issuperset(set(['PWSD'])):
		func = 0

		for i in range(len_data):
			for j in range(len_sims):
				func += ((data.loc[i] - sims.loc[j])**2).divide(len_data * len_sims)

		error['PWSD'] = '{:.6e}'.format(func.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum())

	# pair-wise absolute deviation
	if set(args.error).issuperset(set(['APWSD'])):
		func = 0

		for i in range(len_data):
			for j in range(len_sims):
				func += (abs(data.loc[i] - sims.loc[j])).divide(len_data * len_sims)

		error['APWSD'] = '{:.6e}'.format(func.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum())

	# normalized pair-wise square deviation (also implemented in BioNetFit as equation 3, but not normalized by the number of data * sims)
	if set(args.error).issuperset(set(['NPWSD'])):
		func = 0

		for i in range(len_data):
			for j in range(len_sims):
				func += (((data.loc[i] - sims.loc[j]).divide(data.loc[i]))**2).divide(len_data * len_sims)

		error['NPWSD'] = '{:.6e}'.format(func.replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum())

	# normalized pair-wise absolute deviation
	if set(args.error).issuperset(set(['ANPWSD'])):
		func = 0

		for i in range(len_data):
			for j in range(len_sims):
				func += (abs((data.loc[i] - sims.loc[j]).divide(data.loc[i]))).divide(len_data * len_sims)

		error['ANPWSD'] = '{:.6e}'.format(func.replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').sum().sum())

	# Mann-Whitney U-test
	if set(args.error).issuperset(set(['MWUT'])):
		if ((len_data >= 3 and len_sims >= 3) or (len_data >= 2 and len_sims >= 5)):

			ucrit = pandas.read_table(args.crit[0], sep = None, engine = 'python', header = 0, index_col = 0)
			udata = pandas.DataFrame(index = sims.loc[0].index, columns = sims.loc[0].columns).fillna(0)
			usims = pandas.DataFrame(index = sims.loc[0].index, columns = sims.loc[0].columns).fillna(0)

			for i in range(len_data):
				for j in range(len_sims):
					diff = (data.loc[i] - sims.loc[j]).dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all')
					# if data < sims count -1.0
					diff[diff < 0] = -1.0
					# if data > sims count +1.0
					diff[diff > 0] = +1.0
					# if data = sims count +0.5
					diff[diff == 0] = +0.5
					# count how many times is data < sims (udata and usims are complementary)
					udata += diff[diff == -1.0].fillna(0).divide(-1) + diff[diff == +0.5].fillna(0)
					usims += diff[diff == +1.0].fillna(0).divide(+1) + diff[diff == +0.5].fillna(0)

			# U is significant if it is less than or equal to the table value
			U = len_data * len_sims - udata.where(udata >= usims).fillna(usims.where(usims >= udata))
			U[U <= ucrit.loc[len_sims, str(len_data)]] = +1.0
			U[U > ucrit.loc[len_sims, str(len_data)]] = +0.0

			error['MWUT'] = '{:.0f}'.format(U.sum().sum())
	else:
		error['MWUT'] = str(numpy.nan)

if __name__ == '__main__':
	args = argsparser()

	# read sims files
	sims, len_sims = read_sims(args.sims)
	# read data files
	data, len_data = read_data(args.data)

	# Filter out unavailable experimental data from simulation files and filter out unsimulated observables
	sims = sims.filter(items = list(data.columns))
	data = data.filter(items = list(sims.columns))

	# add here your favorite error function and call the genetic algorithm script with its acronysm
	error = {
		'MSE'   : str(numpy.nan),
		'MAE'   : str(numpy.nan),
		'SSQ'   : str(numpy.nan),
		'MWUT'  : str(numpy.nan),
		'PWSD'  : str(numpy.nan),
		'MNSE'  : str(numpy.nan),
		'CHISQ' : str(numpy.nan),
		'APWSD' : str(numpy.nan),
		'NPWSD' : str(numpy.nan),
		'ANPWSD': str(numpy.nan),
		}

	do(error)

	# write report file
	with open(args.file[0], 'w') as outfile:
		for fitfunc, value in sorted(error.items()):
			outfile.write('{:s}\t{:s}\n'.format(fitfunc, value))
