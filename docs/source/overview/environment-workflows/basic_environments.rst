Environment Workflows
=====================

Workflows
---------

With Isaac Lab, we also provide a suite of benchmark environments included
in the ``omni.isaac.lab_tasks`` extension. We use the OpenAI Gym registry
to register these environments. For each environment, we provide a default
configuration file that defines the scene, observations, rewards and action spaces.

The list of environments available registered with OpenAI Gym can be found by running:

.. code:: bash

   ./isaaclab.sh -p source/standalone/environments/list_envs.py


Basic agents
~~~~~~~~~~~~

These include basic agents that output zero or random agents. They are
useful to ensure that the environments are configured correctly.

-  Zero-action agent on the Cart-pole example

   .. code:: bash

      ./isaaclab.sh -p source/standalone/environments/zero_agent.py --task Isaac-Cartpole-v0 --num_envs 32

-  Random-action agent on the Cart-pole example:

   .. code:: bash

      ./isaaclab.sh -p source/standalone/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32


State machine
~~~~~~~~~~~~~

We include examples on hand-crafted state machines for the environments. These
help in understanding the environment and how to use the provided interfaces.
The state machines are written in `warp <https://github.com/NVIDIA/warp>`__ which
allows efficient execution for large number of environments using CUDA kernels.

.. code:: bash

   ./isaaclab.sh -p source/standalone/environments/state_machine/lift_cube_sm.py --num_envs 32
