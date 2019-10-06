.. _SLURM-instructions:

Installing SLURM in your machine
================================

To install SLURM you need admin access to the machine. Please follow this
instructions to start up running the workload manager, in the controller as well
in the controlled machines.

.. code-block:: bash

	sudo apt-get -y install slurm-wlm
	sudo nano /etc/slurm-llnl/slurm.conf

	sudo chown -R slurm:slurm /var/run/slurm-llnl/
	sudo chown -R slurm:slurm /var/lib/slurm-llnl/
	sudo chown -R slurm:slurm /var/log/slurm-llnl/
	sudo mkdir /var/spool/slurmd
	sudo chown -R slurm:slurm /var/spool/slurmd

	sudo systemctl start slurmd

Replace ``$HOST_NAME`` with your machine name that is going to act as the
controller. If you have multiple machines, this configuration file must be
identical and in all machines in the queue. Be aware that Dask requires
configuration of memory to be able to configure a SLURMCluster (More info in
https://jobqueue.dask.org/en/latest/examples.html)

.. code-block:: vim

	### slurm.conf - Slurm config file.

	#ClusterName=$HOST_NAME
	ControlMachine=$HOST_NAME
	SlurmUser=slurm
	AuthType=auth/munge

	SlurmctldPidFile=/var/run/slurm-llnl/slurmctld.pid
	SlurmdPidFile=/var/run/slurm-llnl/slurmd.pid
	SlurmdSpoolDir=/var/lib/slurm-llnl/slurmd
	StateSaveLocation=/var/lib/slurm-llnl/slurmctld

	SwitchType=switch/none
	ProctrackType=proctrack/pgid
	TaskPlugin=task/none

	MpiDefault=none
	MaxJobCount=100000
	MaxArraySize=64000

	# TIMERS
	SlurmdTimeout=300
	InactiveLimit=0
	MinJobAge=300
	KillWait=30
	Waittime=0

	# SCHEDULING
	DefMemPerCPU=18400
	SchedulerType=sched/backfill
	SelectType=select/cons_res
	SelectTypeParameters=CR_CPU_Memory
	FastSchedule=1

	# LOGGING
	SlurmctldDebug=3
	SlurmctldLogFile=/var/log/slurm-llnl/slurmctld.log
	SlurmdDebug=3
	SlurmdLogFile=/var/log/slurm-llnl/slurmd.log

	# COMPUTE NODES

	# Here you add the machine hardware configurations
	NodeName=$HOST_NAME Procs=56 TreadsPerCore=1 State=idle RealMemory=1031755

	# Here you add the machine(s) to a Partition
	PartitionName=MyCluster Nodes=$HOST_NAME Default=yes MaxTime=INFINITE State=up

.. note::
	Please refer to `SLURM`_ for advance configuration like limiting time, CPUs
	and RAM for users or groups, to balance load in your cluster.

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
