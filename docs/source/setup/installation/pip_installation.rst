.. _isaacsim-pip-installation:

Installation using Isaac Sim pip
================================


Installing Isaac Sim
--------------------

From Isaac Sim 4.0 release, it is possible to install Isaac Sim using pip. This approach is experimental and may have
compatibility issues with some Linux distributions. If you encounter any issues, please report them to the
`Isaac Sim Forums <https://docs.omniverse.nvidia.com/isaacsim/latest/common/feedback.html>`_.

.. attention::

   Installing Isaac Sim with pip requires GLIBC 2.34+ version compatibility.
   To check the GLIBC version on your system, use command ``ldd --version``.

   This may pose compatibility issues with some Linux distributions. For instance, Ubuntu 20.04 LTS has GLIBC 2.31
   by default. If you encounter compatibility issues, we recommend following the
   :ref:`Isaac Sim Binaries Installation <isaacsim-binaries-installation>` approach.

-  To use the pip installation approach for Isaac Sim, we recommend first creating a virtual environment.
   Ensure that the python version of the virtual environment is **Python 3.10**.

   .. tab-set::

      .. tab-item:: conda environment

         .. code-block:: bash

            conda create -n isaaclab python=3.10
            conda activate isaaclab

      .. tab-item:: venv environment

         .. tab-set::
            :sync-group: os

            .. tab-item:: :icon:`fa-brands fa-linux` Linux
               :sync: linux

               .. code-block:: bash

                  # create a conda environment named isaaclab with python3.10
                  python3.10 -m venv isaaclab
                  # activate the conda environment
                  source isaaclab/bin/activate

            .. tab-item:: :icon:`fa-brands fa-windows` Windows
               :sync: windows

               .. code-block:: batch

                  # create a virtual environment named isaaclab with python3.10
                  python3.10 -m venv isaaclab
                  # activate the virtual environment
                  isaaclab\Scripts\activate


-  Next, install a CUDA-enabled PyTorch 2.4.0 build based on the CUDA version available on your system. This step is optional for Linux, but required for Windows to ensure a CUDA-compatible version of PyTorch is installed.

   .. tab-set::

      .. tab-item:: CUDA 11

         .. code-block:: bash

            pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118

      .. tab-item:: CUDA 12

         .. code-block:: bash

            pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

-  Before installing Isaac Sim, ensure the latest pip version is installed. To update pip, run

   .. code-block:: bash

      pip install --upgrade pip


-  Then, install the Isaac Sim packages necessary for running Isaac Lab:

   .. code-block:: bash

      pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com


Installing Isaac Lab
--------------------

Cloning Isaac Lab
~~~~~~~~~~~~~~~~~

.. note::

   We recommend making a `fork <https://github.com/isaac-sim/IsaacLab/fork>`_ of the Isaac Lab repository to contribute
   to the project but this is not mandatory to use the framework. If you
   make a fork, please replace ``isaac-sim`` with your username
   in the following instructions.

Clone the Isaac Lab repository into your workspace:

.. tab-set::

   .. tab-item:: SSH

      .. code:: bash

         git clone git@github.com:isaac-sim/IsaacLab.git

   .. tab-item:: HTTPS

      .. code:: bash

         git clone https://github.com/isaac-sim/IsaacLab.git


.. note::
   We provide a helper executable `isaaclab.sh <https://github.com/isaac-sim/IsaacLab/blob/main/isaaclab.sh>`_ that provides
   utilities to manage extensions:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: text

            ./isaaclab.sh --help

            usage: isaaclab.sh [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-c] -- Utility to manage Isaac Lab.

            optional arguments:
               -h, --help           Display the help content.
               -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks (rl_games, rsl_rl, sb3, skrl) as extra dependencies. Default is 'all'.
               -f, --format         Run pre-commit to format the code and check lints.
               -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
               -s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.
               -t, --test           Run all python unittest tests.
               -o, --docker         Run the docker container helper script (docker/container.sh).
               -v, --vscode         Generate the VSCode settings file from template.
               -d, --docs           Build the documentation from source using sphinx.
               -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'isaaclab'.

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: text

            isaaclab.bat --help

            usage: isaaclab.bat [-h] [-i] [-f] [-p] [-s] [-v] [-d] [-c] -- Utility to manage Isaac Lab.

            optional arguments:
               -h, --help           Display the help content.
               -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks (rl_games, rsl_rl, sb3, skrl) as extra dependencies. Default is 'all'.
               -f, --format         Run pre-commit to format the code and check lints.
               -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
               -s, --sim            Run the simulator executable (isaac-sim.bat) provided by Isaac Sim.
               -t, --test           Run all python unittest tests.
               -v, --vscode         Generate the VSCode settings file from template.
               -d, --docs           Build the documentation from source using sphinx.
               -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'isaaclab'.

Installation
~~~~~~~~~~~~

-  Install dependencies using ``apt`` (on Ubuntu):

   .. code:: bash

      sudo apt install cmake build-essential

- Run the install command that iterates over all the extensions in ``source/extensions`` directory and installs them
  using pip (with ``--editable`` flag):

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh --install # or "./isaaclab.sh -i"

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: bash

         isaaclab.bat --install :: or "isaaclab.bat -i"

.. note::

   By default, this will install all the learning frameworks. If you want to install only a specific framework, you can
   pass the name of the framework as an argument. For example, to install only the ``rl_games`` framework, you can run

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh --install rl_games  # or "./isaaclab.sh -i rl_games"

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: bash

            isaaclab.bat --install rl_games :: or "isaaclab.bat -i rl_games"

   The valid options are ``rl_games``, ``rsl_rl``, ``sb3``, ``skrl``, ``robomimic``, ``none``.
