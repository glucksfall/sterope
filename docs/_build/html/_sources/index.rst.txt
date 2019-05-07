Welcome to Sterope's documentation!
===================================

Sterope is a python3 package that implement a method based on the Dynamic
Influence Network (https://www.ncbi.nlm.nih.gov/pubmed/28866584) to analyze the
sensitivity of parameter values in the response of a Rule-Based Model written in
kappa (https://kappalanguage.org/)

Sterope creates models samples and analyze the Dynamic Influence Network employing
the Sobol method included in the SALib library (`SALibpaper`_). After samples are
created, Sterope simulates them in parallel employing SLURM (`SLURM`_) or the 
python multiprocessing API.

The plan to add methods into Pleiades (https://github.com/glucksfall/pleiades)
includes a parameterization employing a Particle Swarm Optimization protocol and
other analysis methods that are typical of frameworks like Ordinary Differential
Equations. You could write us if you wish to add methods into pleione or aid in
the development of them.


.. toctree::
   :maxdepth: 3

   Installation
..    ParameterEstimation
..    Python3
..    SLURM


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. refs
.. _KaSim: https://github.com/Kappa-Dev/KaSim
.. _NFsim: https://github.com/RuleWorld/nfsim
.. _BioNetGen2: https://github.com/RuleWorld/bionetgen
.. _PISKaS: https://github.com/DLab/PISKaS
.. _BioNetFit: https://github.com/RuleWorld/BioNetFit
.. _SLURM: https://slurm.schedmd.com/

.. _Kappa: https://www.kappalanguage.org/
.. _BioNetGen: http://www.csb.pitt.edu/Faculty/Faeder/?page_id=409
.. _pandas: https://pandas.pydata.org/

.. _27402907: https://www.ncbi.nlm.nih.gov/pubmed/27402907
.. _26556387: https://www.ncbi.nlm.nih.gov/pubmed/26556387
.. _29950016: https://www.ncbi.nlm.nih.gov/pubmed/29950016
.. _29175206: https://www.ncbi.nlm.nih.gov/pubmed/29175206
.. _26556387: https://www.ncbi.nlm.nih.gov/pubmed/26556387

.. _SALibpaper: https://joss.theoj.org/papers/431262803744581c1d4b6a95892d3343
