Developer's Guide
=================

For development, we suggest using `Microsoft Visual Studio Code
(VSCode) <https://code.visualstudio.com/>`__. This is also suggested by
NVIDIA Omniverse and there exists tutorials on how to `debug Omniverse
extensions <https://www.youtube.com/watch?v=Vr1bLtF1f4U&ab_channel=NVIDIAOmniverse>`__
using VSCode.


Setting up Visual Studio Code
-----------------------------

The ``orbit`` repository includes the VSCode settings to easily allow setting
up your development environment. These are included in the ``.vscode`` directory
and include the following files:

.. code-block:: bash

   .vscode
   ├── extensions.json
   ├── launch.json
   ├── settings.json
   └── tasks.json


To setup the IDE, please follow these instructions:

1. Open the ``orbit`` directory on Visual Studio Code IDE
2. Run VSCode
   `Tasks <https://code.visualstudio.com/docs/editor/tasks>`__, by
   pressing ``Ctrl+Shift+P``, selecting ``Tasks: Run Task`` and
   running the ``setup_python_env`` in the drop down menu.

   .. image:: ../_static/vscode_tasks.png
      :width: 600px
      :align: center
      :alt: VSCode Tasks

If everything executes correctly, it should create a file
``.python.env`` in the ``.vscode`` directory. The file contains the python
paths to all the extensions provided by Isaac Sim and Omniverse. This helps
in indexing all the python modules for intelligent suggestions while writing
code.

For more information on VSCode support for Omniverse, please refer to the
following links:

* `Isaac Sim VSCode support <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_standalone_python.html#isaac-sim-python-vscode>`__
* `Debugging with VSCode <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_advanced_python_debugging.html>`__


Configuring the python interpreter
----------------------------------

In the provided configuration, we set the default python interpreter to use the
python executable provided by Omniverse. This is specified in the
``.vscode/settings.json`` file:

.. code-block:: json

   {
      "python.defaultInterpreterPath": "${workspaceFolder}/_isaac_sim/kit/python/bin/python3",
      "python.envFile": "${workspaceFolder}/.vscode/.python.env",
   }

If you want to use a different python interpreter (for instance, from your conda environment),
you need to change the python interpreter used by selecting and activating the python interpreter
of your choice in the bottom left corner of VSCode, or opening the command palette (``Ctrl+Shift+P``)
and selecting ``Python: Select Interpreter``.

For more information on how to set python interpreter for VSCode, please
refer to the `VSCode documentation <https://code.visualstudio.com/docs/python/environments#_working-with-python-interpreters>`_.

Repository organization
-----------------------

The ``orbit`` repository is structured as follows:

.. code-block:: bash

   orbit
   ├── .vscode
   ├── LICENSE
   ├── orbit.sh
   ├── pyproject.toml
   ├── CHANGELOG.md
   ├── README.md
   ├── docs
   ├── source
   │   ├── extensions
   │   │   ├── omni.isaac.orbit
   │   │   └── omni.isaac.orbit_envs
   │   ├── standalone
   │   │   ├── demo
   │   │   ├── environments
   │   │   └── workflows
   │   └── tools
   └── VERSION

The ``source`` directory contains the source code for ``orbit`` *extensions*
and *standalone applications*. The two are the different development workflows
supported in `NVIDIA Isaac Sim <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_required_workflows.html>`__.

.. note::

   Instead of maintaining a `changelog <https://keepachangelog.com/en/1.0.0/>`__ for each
   extension, we maintain a common changelog file for the whole repository. This is
   located in the root directory of the repository and is named ``CHANGELOG.md``.


Extensions
~~~~~~~~~~

Extensions are the recommended way to develop applications in Isaac Sim. They are
modularized packages that formulate the Omniverse ecosystem. Each extension
provides a set of functionalities that can be used by other extensions or
standalone applications. A folder is recognized as an extension if it contains
an ``extension.toml`` file. More information on extensions can be found in the
`Omniverse documentation <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/extensions_basic.html>`__.

Orbit in itself provides extensions for robot learning. These are written into the
``source/extensions`` directory. Each extension is written as a python package and
follows the following structure:

.. code:: bash

   <extension-name>
   ├── config
   │   └── extension.toml
   ├── docs
   │   └── README.md
   ├── <extension-name>
   │   ├── __init__.py
   │   ├── ....
   │   └── scripts
   ├── setup.py
   └── tests

The ``config/extension.toml`` file contains the metadata of the extension. This
includes the name, version, description, dependencies, etc. This information is used
by Omniverse to load the extension. The ``docs`` directory contains the documentation
for the extension with more detailed information about the extension.

The ``<extension-name>`` directory contains the main python package for the extension.
It may also contains the ``scripts`` directory for keeping python-based applications
that are loaded into Omniverse when then extension is enabled using the
`Extension Manager <https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_extension-manager.html>`__.

More specifically, when an extension is enabled, the python module specified in the
``config/extension.toml`` file is loaded and scripts that contains children of the
:class:`omni.ext.IExt` class are executed.

.. code:: python

   import omni.ext

   class MyExt(omni.ext.IExt):
      """My extension application."""

      def on_startup(self, ext_id):
         """Called when the extension is loaded."""
         pass

      def on_shutdown(self):
         """Called when the extension is unloaded.

         It releases all references to the extension and cleans up any resources.
         """
         pass

While loading extensions into Omniverse happens automatically, using the python package
in standalone applications requires additional steps. To simplify the build process and
avoiding the need to understand the `premake <https://premake.github.io/>`__
build system used by Omniverse, we directly use the `setuptools <https://setuptools.readthedocs.io/en/latest/>`__
python package to build the python module provided by the extensions. This is done by the
``setup.py`` file in the extension directory.

.. note::

   The ``setup.py`` file is not required for extensions that are only loaded into Omniverse
   using the `Extension Manager <https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_extension-manager.html>`__.

Lastly, the ``tests`` directory contains the unit tests for the extension. These are written
using the `unittest <https://docs.python.org/3/library/unittest.html>`__ framework. It is
important to note that Omniverse also provides a similar
`testing framework <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/104.0/guide/testing_exts_python.html>`__.
However, it requires going through the build process and does not support testing of the python module in
standalone applications.

Standalone applications
~~~~~~~~~~~~~~~~~~~~~~~

In a typical Omniverse workflow, the simulator is launched first, after which the extensions are
enabled that load the python module and run the python application. While this is a recommended
workflow, it is not always possible to use this workflow. For example, for robot learning, it is
essential to have complete control over simulation stepping and all the other functionalities
instead of asynchronously waiting for the simulator to step. In such cases, it is necessary to
write a standalone application that launches the simulator using
`SimulationApp <https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html>`__
and allows complete control over the simulation through the
`SimulationContext <https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=simulation%20context#module-omni.isaac.core.simulation_context>`__
class.

.. code:: python

   """Launch Isaac Sim Simulator first."""

   from omni.isaac.kit import SimulationApp

   # launch omniverse app
   config = {"headless": False}
   simulation_app = SimulationApp(config)


   """Rest everything follows."""

   from omni.isaac.core.simulation_context import SimulationContext

   if __name__ == "__main__":
      # get simulation context
      simulation_context = SimulationContext()
      # rest and play simulation
      simulation_context.reset()
      # step simulation
      simulation_context.step()
      # stop simulation
      simulation_context.stop()


The ``source/standalone`` directory contains various standalone applications designed using the extensions
provided by ``orbit``. These applications are written in python and are structured as follows:

* **demo**: Contains various demo applications that showcase the core framework ``omni.isaac.orbit``.
* **environments**: Contains applications for running environments defined in ``omni.isaac.orbit_envs`` with different agents.
  These include a random policy, zero-action policy, teleoperation or scripted state machines.
* **workflows**: Contains applications for using environments with various learning-based frameworks. These include different
  reinforcement learning or imitation learning libraries.


Code style
----------

The code style used in ``orbit`` is based on the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`__.
For Python code, the PEP guidelines are followed. Most important ones are `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`__
for code comments and layout, `PEP-484 <http://www.python.org/dev/peps/pep-0484>`__ and
`PEP-585 <https://www.python.org/dev/peps/pep-0585/>`__ for type-hinting.

We use `pre-commit <https://pre-commit.com/>`__ checks, that runs
the `black <https://black.readthedocs.io/en/stable/>`__ formatter and
`flake8 <https://flake8.pycqa.org/en/latest/>`__ to check the code.
Please check `here <https://pre-commit.com/#install>`__ for instructions
to set this up.

To run over the entire repository, please execute the following command in the terminal:

.. code:: bash

   ./orbit.sh --format  # or `./orbit.sh -f`
