# -*- coding: utf-8 -*-

# Always prefer setuptools over distutils
#from ez_setup import use_setuptools
#use_setuptools()
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import versioneer

def main():

#	cmdclass = versioneer.get_cmdclass()

	here = path.abspath(path.dirname(__file__))

	# Get the long description from the README file
	with open(path.join(here, 'README.md'), encoding='utf-8') as f:
		long_description = f.read()

	setup(
		name='sterope',
		license='GPLv3+',
#		version=versioneer.get_version(),
		version='1.1',
		description='Sterope: Sensitivity analysis of kappa Rule-Based Models based on Dynamic Influence Networks',
		long_description=long_description,
		long_description_content_type='text/markdown',
		url='https://github.com/glucksfall/sterope',
		author='Rodrigo Santibáñez',
		author_email='glucksfall@users.noreply.github.com',
		classifiers=[
			#'Development Status :: 1 - Planning',
			#'Development Status :: 2 - Pre-Alpha',
			#'Development Status :: 3 - Alpha',
			#'Development Status :: 4 - Beta',
			'Development Status :: 5 - Production/Stable',
			#'Development Status :: 6 - Mature',
			#'Development Status :: 7 - Inactive',

			# Indicate who your project is intended for
			'Intended Audience :: Science/Research',
			'Topic :: Scientific/Engineering',

			# Pick your license as you wish
			'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

			# Specify the Python versions you support here. In particular, ensure
			# that you indicate whether you support Python 2, Python 3 or both.
			#'Programming Language :: Python :: 2',
			#'Programming Language :: Python :: 2.7',
			'Programming Language :: Python :: 3.6',
			'Programming Language :: Python :: 3.7',

			# other classifiers added by author
			'Environment :: Console',
			'Operating System :: Unix',
#			'Operating System :: Microsoft :: Windows',
#			'Operating System :: MacOS',

		],
#		cmdclass=cmdclass,
		keywords=['systems biology', 'stochastic modeling', 'parameter analysis'],
		python_requires='>=3.0',
		packages=find_packages(exclude=['contrib', 'docs', 'tests']),
		install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'salib', 'dask', 'dask_jobqueue'],
		package_data={
			'example': ['example'],
		},
		project_urls={
			'Manual': 'https://sterope.readthedocs.io',
			'Bug Reports': 'https://github.com/glucksfall/sterope/issues',
			'Source': 'https://github.com/glucksfall/sterope',
		},
	)

if __name__ == '__main__':
    main()
