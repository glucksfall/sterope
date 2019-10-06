Compiling Python3 from source
=============================

If you don't have admin access to the cluster configuration, you could compile
and install python3 from source following these instructions:

.. code-block:: bash

	wget https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz \
	-O ~/opt/ubuntu-software/Python-3.6.5.tgz
	if [ -d ~/opt/Python-3.6.5 ]; then rm -rf ~/opt/Python-3.6.5; fi
	tar xvzf ~/opt/ubuntu-software/Python-3.6.5.tgz -C ~/opt
	cd ~/opt/Python-3.6.5
	if [ -f Makefile ]; then make clean; fi
	if [ -d $(HOME)/opt/python-3.6.5 ]; then rm -rf $(HOME)/opt/python-3.6.5; fi
	./configure --prefix=$(HOME)/opt/python-3.6.5
	make
	make install

.. note::
	Don't copy an installation folder from another machine since there may be
	libraries incompatibilities. Instead, the code will download, configure,
	compile, and install. To make accesible from anywhere, you could add an
	alias into ``~/.bashrc`` or a symbolic in your ``$HOME/bin`` folder for
	``$HOME/opt/python-3.6.5/bin/python3`` and ``pip3``.

To install numpy and pandas use the following instructions, in order since some
pandas dependencies has also dependencies:

.. code-block:: bash

	wget https://files.pythonhosted.org/packages/71/90/ca61e203e0080a8cef7ac21eca199829fa8d997f7c4da3e985b49d0a107d/numpy-1.14.3-cp36-cp36m-manylinux1_x86_64.whl
	wget https://files.pythonhosted.org/packages/dc/83/15f7833b70d3e067ca91467ca245bae0f6fe56ddc7451aa0dc5606b120f2/pytz-2018.4-py2.py3-none-any.whl
	wget https://files.pythonhosted.org/packages/67/4b/141a581104b1f6397bfa78ac9d43d8ad29a7ca43ea90a2d863fe3056e86a/six-1.11.0-py2.py3-none-any.whl
	wget https://files.pythonhosted.org/packages/cf/f5/af2b09c957ace60dcfac112b669c45c8c97e32f94aa8b56da4c6d1682825/python_dateutil-2.7.3-py2.py3-none-any.whl
	wget https://files.pythonhosted.org/packages/69/ec/8ff0800b8594691759b78a42ccd616f81e7099ee47b167eb9bbd502c02b9/pandas-0.23.0-cp36-cp36m-manylinux1_x86_64.whl

	pip3 install numpy-1.14.3-cp36-cp36m-manylinux1_x86_64.whl
	pip3 install pytz-2018.4-py2.py3-none-any.whl
	pip3 install six-1.11.0-py2.py3-none-any.whl
	pip3 install python_dateutil-2.7.3-py2.py3-none-any.whl
	pip3 install pandas-0.23.0-cp36-cp36m-manylinux1_x86_64.whl

If you have admin access (and willing to compile python3 from source) you could
install the following dependencies:

.. code-block:: bash

	apt-get install libssl-dev zlib1g-dev libncurses5-dev \
	libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev \
	libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev

Compiling python3 with all dependencies would make installation of packages
easier. Just follow the instructions:

.. code-block:: bash

	pip3 install pandas

.. note::
	Installing pandas with pip will install numpy as its dependency.

.. note::
	Be sure you are calling pip3 after creating an alias or a symbolic link.
	Without admin credentials, pip3 would fail to install pandas.
