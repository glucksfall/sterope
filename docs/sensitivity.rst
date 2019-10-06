Global Sensitivity Analysis
===========================

1. **Prepare the model**

   Sterope finds which variables will be analyzed using
   the symbol ``//`` (doble slash, as C/C++) followed by:

	* : ``uniform``, ``range``, ``lognormal``
	* Bounds for each parameter: ``[min max]`` or ``[mean standard_deviation]``
	  in the case if ``lognormal`` was selected.

   For instace:

.. code-block:: bash

	%var: 'KD1__FREE__' 1.000000e+00 // range[0.01 100]
	%var: 'km1__FREE__' 1.000000e+00 // range[0.01 100]
	%var: 'K2RT__FREE__' 1.000000e+00 // range[0.01 100]
	%var: 'km2__FREE__' 1.000000e+00 // range[0.01 100]
	%var: 'kphos__FREE__' 1.000000e+00 // range[0.01 100]
	%var: 'kdephos__FREE__' 1.000000e+00 // range[0.01 100]

or the following configuration if the model is written in syntax 3 (KaSim v3):

.. code-block:: bash

	%var: 'KD1__FREE__' 1.000000e+00 # range[0.01 100]
	%var: 'km1__FREE__' 1.000000e+00 # range[0.01 100]
	%var: 'K2RT__FREE__' 1.000000e+00 # range[0.01 100]
	%var: 'km2__FREE__' 1.000000e+00 # range[0.01 100]
	%var: 'kphos__FREE__' 1.000000e+00 # range[0.01 100]
	%var: 'kdephos__FREE__' 1.000000e+00 # range[0.01 100]
