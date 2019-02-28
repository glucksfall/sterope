Installation
============

There are two different ways to install sterope:

1. **Install sterope natively (Recommended).**

   *OR*

2. **Clone the Github repository.** If you are familiar with git, sterope can
   be cloned and the respective folder added to the python path. Further details
   are below.

.. note::
	**Need Help?**
	If you run into any problems with installation, please visit our chat room:
	https://gitter.im/glucksfall/pleiades

Option 1: Install sterope natively on your computer
---------------------------------------------------

The recommended approach is to use system tools, or install them if
necessary. To install python packages, you could use pip, or download
the package from `python package index <https://test.pypi.org/project/sterope/>`_.

1. **Install with system tools**

   With pip, you simple need to execute and sterope will be installed on
   ``$HOME/.local/lib/python3.6/site-packages`` folder or similar.

   .. code-block:: bash

	pip3 install -i https://test.pypi.org/simple/ sterope --user

   If you have system rights, you could install sterope for all users with

   .. code-block:: bash

	sudo -H pip3 install -i https://test.pypi.org/simple/ sterope

2. **Download from python package index**

   Alternatively, you could download the package (useful when pip fails to
   download the package because of lack of SSL libraries) and then install with pip.
   For instance:

   .. code-block:: bash

	wget https://test-files.pythonhosted.org/packages/7a/d5/bd1f28031d9be331cbd5a7945a2934536c8ccf7c7171a80b4bde132ee245/sterope-1.0.1-py3-none-any.whl
	pip3 install sterope-1.0.1-py3-none-any.whl --user

   .. note::
	**Why Python3?**:
	Sterope is intended to be used with python3, because python2 won't receive
	further development past Jan 1st, 2020. Although, the code has specific python3
	functions over dictionaries and f-strings.

   .. note::
	**pip, Python and Anaconda**:
	Be aware which pip you invoque. You could install pip3 with
	``sudo apt-get install python3-pip`` if you have system rights, or
	install python3 from source, and adding ``<python3 path>/bin/pip3`` to the
	path, or linking it in a directory like ``$HOME/bin`` which is commonly
	added to the path at system login. Please be aware that, if you installed
	Anaconda, pip could be linked to the Anaconda specific version of pip, which
	will install sterope into Anaconda's installation folder.
	Type ``which pip3`` to find out the source of pip, and type ``python3 -m site``
	to find out where is more likely sterope would be installed.

Option 2: Clone the Github repository
-------------------------------------

1. **Clone with git**

   The source code is uploaded and maintained through Github at
   `<https://github.com/glucksfall/sterope>`_. Therefore, you could clone the
   repository locally, and then add the folder to the ``PYTHONPATH``. Beware
   that you should install the *pandas* (`pandas`_), *seaborn* (`seaborn`_), and
   *SALib* (`SALib`_) packages by any means.

   .. code-block:: bash

    git clone https://github.com/glucksfall/sterope /opt
    echo export PYTHONPATH="\$PYTHONPATH:/opt/sterope" >> $HOME/.profile

   .. note::
	Adding the path to ``$HOME/.profile`` allows python to find the package
	installation folder after each user login. Similarly, adding the path to
	``$HOME/.bashrc`` allows python to find the package after each terminal
	invocation.

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
.. _seaborn: https://seaborn.pydata.org/
.. _SALib: https://salib.readthedocs.io/en/latest/
