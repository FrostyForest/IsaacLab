Overview
========

.. figure:: source/_static/isaaclab.jpg
   :width: 100%
   :alt: H1 Humanoid example using Isaac Lab

**Isaac Lab** is a unified and modular framework for robot learning that aims to simplify common workflows
in robotics research (such as RL, learning from demonstrations, and motion planning). It is built upon
`NVIDIA Isaac Sim`_ to leverage the latest simulation capabilities for photo-realistic scenes, and fast
and efficient simulation. The core objectives of the framework are:

- **Modularity**: Easily customize and add new environments, robots, and sensors.
- **Agility**: Adapt to the changing needs of the community.
- **Openness**: Remain open-sourced to allow the community to contribute and extend the framework.
- **Battery-included**: Include a number of environments, sensors, and tasks that are ready to use.

Key features available in Isaac Lab include fast and accurate physics simulation provided by PhysX,
tiled rendering APIs for vectorized rendering, domain randomization for improving robustness and adaptability,
and support for running in the cloud.

For more information about the framework, please refer to the `paper <https://arxiv.org/abs/2301.04195>`_
:cite:`mittal2023orbit`. For clarifications on NVIDIA Isaac ecosystem, please check out the
:doc:`/source/setup/faq` section.

.. figure:: source/_static/tasks.jpg
   :width: 100%
   :alt: Example tasks created using Isaac Lab


License
=======

The Isaac Lab framework is open-sourced under the BSD-3-Clause license.
Please refer to :ref:`license` for more details.

Acknowledgement
===============
Isaac Lab development initiated from the `Orbit <https://isaac-orbit.github.io/>`_ framework. We would appreciate if you would cite it in academic publications as well:

.. code:: bibtex

   @article{mittal2023orbit,
      author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
      journal={IEEE Robotics and Automation Letters},
      title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
      year={2023},
      volume={8},
      number={6},
      pages={3740-3747},
      doi={10.1109/LRA.2023.3270034}
   }


Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   source/setup/installation/index
   source/setup/developer
   source/setup/sample
   source/setup/template
   source/setup/faq

.. toctree::
   :maxdepth: 2
   :caption: Features

   source/features/task_workflows
   source/features/multi_gpu
   source/features/tiled_rendering
   source/features/environments
   source/features/actuators
   .. source/features/motion_generators

.. toctree::
   :maxdepth: 1
   :caption: Resources
   :titlesonly:

   source/migration/index
   source/tutorials/index
   source/how-to/index
   source/deployment/index

.. toctree::
   :maxdepth: 1
   :caption: Source API

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: References

   source/refs/migration
   source/refs/contributing
   source/refs/troubleshooting
   source/refs/issues
   source/refs/changelog
   source/refs/license
   source/refs/bibliography

.. toctree::
    :hidden:
    :caption: Project Links

    GitHub <https://github.com/isaac-sim/IsaacLab>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _NVIDIA Isaac Sim: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
