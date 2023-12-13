.. _how-to-create-articulation-config:

Creating an Articulation
========================

In this tutorial, we move beyond the use of pre-built Articulations such as
Anymal and Franka, focusing instead on the steps required to integrate
custom robots into Orbit. The tutorial provides a
step-by-step guide on importing a robot design in either USD format and
spawning it in Orbit as an :class:`Articulation`.

.. TODO: Talk about how to import via URDF


What is a Cartpole?
~~~~~~~~~~~~~~~~~~~

Cartpole, a variation of the inverted pendulum problem
(https://en.wikipedia.org/wiki/Inverted_pendulum), serves as a practical
example for learning traditional control and RL. The cartpole has a single
controllable degree of freedom (DOF) at the joint between the cart and
the rail. The attached pole has 1 DOF that allows it to rotate freely.

.. TODO: Add isaac sim screenshot and replace GIF with a webdb

In :ref:`tutorial-create-base-env` participants will learn to control the
pole to stabilize the cart, but this tutorial focuses on merely constructing
the :class:`ArticulationCfg` that defines the cartpole.

The Code
~~~~~~~~

In Orbit, we define an :class:`Articulation` by constructing its
configuration :class:`ArticulationCfg`. In the following sections we will break
down each part of the configuration.

.. dropdown:: Code for USD import configuration
   :icon: code

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/assets/config/cartpole.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Importing Cartpole's USD
^^^^^^^^^^^^^^^^^^^^^^^^

The next chunk of code handles the USD import of the Cartpole:

* Defining the USD file path from which to spawn the Cartpole
* Defining the rigid body properties of the Cartpole
* Defining properties of the root of the Cartpole

.. dropdown:: Code for USD import configuration
   :icon: code

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/assets/config/cartpole.py
      :language: python
      :start-after: # USD file configuration
      :end-before: # Initial state definition

.. note::
   To import articulation from a URDF file instead of a USD file, use ``UrdfFileCfg`` found in
   ``source/extensions/omni.isaac.orbit/omni/isaac/orbit/sim/spawners/from_files/from_file_cfg``
   and replace ``usd_path`` argument with ``urdf_path``. For more details, see the API documentation.

.. TODO: Either add an example of this here or make a separate tutorial

Defining Cartpole's USD File Path
"""""""""""""""""""""""""""""""""

First we define the path the Cartpole USD file will be loaded from. In this
case ``cartpole.usd`` is included in the Nucleus server.

.. TODO: Document Nucleus server somewhere or link it if docs exist

.. literalinclude:: ../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/assets/config/cartpole.py
   :language: python
   :start-after: # Location of USD file
   :end-before: # Rigid body properties

Defining Cartpole's Rigid Body Properties
"""""""""""""""""""""""""""""""""""""""""

The rigid body properties define how the cartpole will interact with its
environment. The settings we want to modify in this example are:

* The rigid body to be enabled
* | the maximum values for linear and angular velocity and depenetration
  | velocity which defines the speed at which objects in collision with one
  | another move away from one another
* The Gyroscopic forces on our cartpole to be enabled

.. TODO: Either go into more detail here, or add tutorial on rigid body properties

.. literalinclude:: ../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/assets/config/cartpole.py
   :language: python
   :start-after: # Rigid body properties
   :end-before: # Articulation root properties


Defining the Initial State of Cartpole
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`InitialStateCfg` object defines the initial state of the root of
an articulation in addition to the initial state of any joints. In this
example, we will spawn the Cartpole at the origin of the XY plane at a Z height
of 2.0 meters. The cart's joints will default to 0.0 as defined by the ``joint_pos``
parameter.

.. literalinclude:: ../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/assets/config/cartpole.py
   :language: python
   :start-after: # Initial state definition
   :end-before: # Actuators definition

Defining the Cartpole's Actuators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cartpole articulation has two actuators, one corresponding to each joint
``cart_to_pole`` and ``slider_to_cart``. for more details on actuators, see
:ref:`feature-actuators`.

.. literalinclude:: ../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/assets/config/cartpole.py
   :language: python
   :start-after: # Actuators definition
   :end-before: # End cartpole articulation configuration
