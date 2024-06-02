Registering an Environment
==========================

.. currentmodule:: omni.isaac.lab

In the previous tutorial, we learned how to create a custom cartpole environment. We manually
created an instance of the environment by importing the environment class and its configuration
class.

.. dropdown:: Environment creation in the previous tutorial
   :icon: code

   .. literalinclude:: ../../../../source/standalone/tutorials/03_envs/run_cartpole_rl_env.py
      :language: python
      :start-at: # create environment configuration
      :end-at: env = ManagerBasedRLEnv(cfg=env_cfg)

While straightforward, this approach is not scalable as we have a large suite of environments.
In this tutorial, we will show how to use the :meth:`gymnasium.register` method to register
environments with the ``gymnasium`` registry. This allows us to create the environment through
the :meth:`gymnasium.make` function.


.. dropdown:: Environment creation in this tutorial
   :icon: code

   .. literalinclude:: ../../../../source/standalone/environments/random_agent.py
      :language: python
      :lines: 36-47


The Code
~~~~~~~~

The tutorial corresponds to the ``random_agent.py`` script in the ``source/standalone/environments`` directory.

.. dropdown:: Code for random_agent.py
   :icon: code

   .. literalinclude:: ../../../../source/standalone/environments/random_agent.py
      :language: python
      :emphasize-lines: 36-37, 42-47
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

The :class:`envs.ManagerBasedRLEnv` class inherits from the :class:`gymnasium.Env` class to follow
a standard interface. However, unlike the traditional Gym environments, the :class:`envs.ManagerBasedRLEnv`
implements a *vectorized* environment. This means that multiple environment instances
are running simultaneously in the same process, and all the data is returned in a batched
fashion.

Similarly, the :class:`envs.DirectRLEnv` class also inherits from the :class:`gymnasium.Env` class
for the direct workflow.

Using the gym registry
----------------------

To register an environment, we use the :meth:`gymnasium.register` method. This method takes
in the environment name, the entry point to the environment class, and the entry point to the
environment configuration class.

.. note::
    The ``gymnasium`` registry is a global registry. Hence, it is important to ensure that the
    environment names are unique. Otherwise, the registry will throw an error when registering
    the environment.

Manager-Based Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^

For manager-based environments, the following shows the registration
call for the cartpole environment in the ``omni.isaac.lab_tasks.manager_based.classic.cartpole`` sub-package:

.. literalinclude:: ../../../../source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/cartpole/__init__.py
   :language: python
   :lines: 10-
   :emphasize-lines: 11, 12, 15

The ``id`` argument is the name of the environment. As a convention, we name all the environments
with the prefix ``Isaac-`` to make it easier to search for them in the registry. The name of the
environment is typically followed by the name of the task, and then the name of the robot.
For instance, for legged locomotion with ANYmal C on flat terrain, the environment is called
``Isaac-Velocity-Flat-Anymal-C-v0``. The version number ``v<N>`` is typically used to specify different
variations of the same environment. Otherwise, the names of the environments can become too long
and difficult to read.

The ``entry_point`` argument is the entry point to the environment class. The entry point is a string
of the form ``<module>:<class>``. In the case of the cartpole environment, the entry point is
``omni.isaac.lab.envs:ManagerBasedRLEnv``. The entry point is used to import the environment class
when creating the environment instance.

The ``env_cfg_entry_point`` argument specifies the default configuration for the environment. The default
configuration is loaded using the :meth:`omni.isaac.lab_tasks.utils.parse_env_cfg` function.
It is then passed to the :meth:`gymnasium.make` function to create the environment instance.
The configuration entry point can be both a YAML file or a python configuration class.

Direct Environemtns
^^^^^^^^^^^^^^^^^^^

For direct-based environments, the following shows the registration call for the cartpole environment
in the ``omni.isaac.lab_tasks.direct.cartpole`` sub-package:

.. code-block:: python

  import gymnasium as gym

  from . import agents
  from .cartpole_env import CartpoleEnv, CartpoleEnvCfg

  ##
  # Register Gym environments.
  ##

  gym.register(
      id="Isaac-Cartpole-Direct-v0",
      entry_point="omni.isaac.lab_tasks.direct.cartpole:CartpoleEnv",
      disable_env_checker=True,
      kwargs={
          "env_cfg_entry_point": CartpoleEnvCfg,
          "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
          "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
          "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
          "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
      },
  )

The ``id`` argument is the name of the environment. As a convention, we name all the environments
with the prefix ``Isaac-`` to make it easier to search for them in the registry.
For direct environments, we also add the suffix ``-Direct``. The name of the
environment is typically followed by the name of the task, and then the name of the robot.
For instance, for legged locomotion with ANYmal C on flat terrain, the environment is called
``Isaac-Velocity-Flat-Anymal-C-Direct-v0``. The version number ``v<N>`` is typically used to specify different
variations of the same environment. Otherwise, the names of the environments can become too long
and difficult to read.

The ``entry_point`` argument is the entry point to the environment class. The entry point is a string
of the form ``<module>:<class>``. In the case of the cartpole environment, the entry point is
``omni.isaac.lab_tasks.direct.cartpole:CartpoleEnv``. The entry point is used to import the environment class
when creating the environment instance.

The ``env_cfg_entry_point`` argument specifies the default configuration for the environment. The default
configuration is loaded using the :meth:`omni.isaac.lab_tasks.utils.parse_env_cfg` function.
It is then passed to the :meth:`gymnasium.make` function to create the environment instance.
The configuration entry point can be both a YAML file or a python configuration class.


Creating the environment
------------------------

To inform the ``gym`` registry with all the environments provided by the ``omni.isaac.lab_tasks``
extension, we must import the module at the start of the script. This will execute the ``__init__.py``
file which iterates over all the sub-packages and registers their respective environments.

.. literalinclude:: ../../../../source/standalone/environments/random_agent.py
   :language: python
   :start-at: import omni.isaac.lab_tasks  # noqa: F401
   :end-at: import omni.isaac.lab_tasks  # noqa: F401

In this tutorial, the task name is read from the command line. The task name is used to parse
the default configuration as well as to create the environment instance. In addition, other
parsed command line arguments such as the number of environments, the simulation device,
and whether to render, are used to override the default configuration.

.. literalinclude:: ../../../../source/standalone/environments/random_agent.py
   :language: python
   :start-at: # create environment configuration
   :end-at: env = gym.make(args_cli.task, cfg=env_cfg)

Once creating the environment, the rest of the execution follows the standard resetting and stepping.


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./isaaclab.sh -p source/standalone/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32


This should open a stage with everything similar to the previous :ref:`tutorial-create-rl-env` tutorial.
To stop the simulation, you can either close the window, or press ``Ctrl+C`` in the terminal.

In addition, you can also change the simulation device from GPU to CPU by adding the ``--cpu`` flag:

.. code-block:: bash

   ./isaaclab.sh -p source/standalone/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32 --cpu

With the ``--cpu`` flag, the simulation will run on the CPU. This is useful for debugging the simulation.
However, the simulation will run much slower than on the GPU.
