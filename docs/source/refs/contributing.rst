Contribution Guidelines
=======================

We wholeheartedly welcome contributions to the project to make the framework more mature
and useful for everyone. These may happen in forms of:

* Bug reports: Please report any bugs you find in the `issue tracker <https://github.com/NVIDIA-Omniverse/orbit/issues>`__.
* Feature requests: Please suggest new features you would like to see in the `discussions <https://github.com/NVIDIA-Omniverse/Orbit/discussions>`__.
* Code contributions: Please submit a `pull request <https://github.com/NVIDIA-Omniverse/orbit/pulls>`__.

  * Bug fixes
  * New features
  * Documentation improvements
  * Tutorials and tutorial improvements


.. note::

   We prefer GitHub `discussions <https://github.com/NVIDIA-Omniverse/Orbit/discussions>`_ for discussing ideas,
   asking questions, conversations and requests for new features.

   Please use the
   `issue tracker <https://github.com/NVIDIA-Omniverse/orbit/issues>`_ only to track executable pieces of work
   with a definite scope and a clear deliverable. These can be fixing bugs, new features, or general updates.


Contributing Code
-----------------

We use `GitHub <https://github.com/NVIDIA-Omniverse/orbit>`__ for code hosting. Please
follow the following steps to contribute code:

1. Create an issue in the `issue tracker <https://github.com/NVIDIA-Omniverse/orbit/issues>`__ to discuss
   the changes or additions you would like to make. This helps us to avoid duplicate work and to make
   sure that the changes are aligned with the roadmap of the project.
2. Fork the repository.
3. Create a new branch for your changes.
4. Make your changes and commit them.
5. Push your changes to your fork.
6. Submit a pull request to the `main branch <https://github.com/NVIDIA-Omniverse/orbit/compare>`__.
7. Ensure all the checks on the pull request template are performed.

After sending a pull request, the maintainers will review your code and provide feedback.

Please ensure that your code is well-formatted, documented and passes all the tests.

.. tip::

   It is important to keep the pull request as small as possible. This makes it easier for the
   maintainers to review your code. If you are making multiple changes, please send multiple pull requests.
   Large pull requests are difficult to review and may take a long time to merge.


Coding Style
------------

We follow the `Google Style
Guides <https://google.github.io/styleguide/pyguide.html>`__ for the
codebase. For Python code, the PEP guidelines are followed. Most
important ones are `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`__
for code comments and layout,
`PEP-484 <http://www.python.org/dev/peps/pep-0484>`__ and
`PEP-585 <https://www.python.org/dev/peps/pep-0585/>`__ for
type-hinting.

We use the following tools for maintaining code quality:

* `pre-commit <https://pre-commit.com/>`__: Runs a list of formatters and linters over the codebase.
* `black <https://black.readthedocs.io/en/stable/>`__: The uncompromising code formatter.
* `flake8 <https://flake8.pycqa.org/en/latest/>`__: A wrapper around PyFlakes, pycodestyle and
  Ned Batchelder's McCabe script.

Please check `here <https://pre-commit.com/#install>`__ for instructions
to set these up. To run over the entire repository, please execute the
following command in the terminal:

.. code:: bash

   ./orbit.sh --format  # or `./orbit.sh -f`

Maintaining a changelog
-----------------------

Each extension maintains a changelog in the ``CHANGELOG.rst`` file in the ``docs`` directory. The
file is written in `reStructuredText <https://docutils.sourceforge.io/rst.html>`__ format. It
contains a curated, chronologically ordered list of notable changes for each version of the extension.

The goal of this changelog is to help users and contributors see precisely what notable changes have
been made between each release (or version) of the extension. This is a *MUST* for every extension.

For updating the changelog, please follow the following guidelines:

* Each version should have a section with the version number and the release date.
* The version number is updated according to `Semantic Versioning <https://semver.org/>`__. The
  release date is the date on which the version is released.
* Each version is divided into subsections based on the type of changes made.

  * ``Added``: For new features.
  * ``Changed``: For changes in existing functionality.
  * ``Deprecated``: For soon-to-be removed features.
  * ``Removed``: For now removed features.
  * ``Fixed``: For any bug fixes.

* Each change is described in its corresponding sub-section with a bullet point.
* The bullet points are written in the past tense and in imperative mode.

For example, the following is a sample changelog:

.. code:: rst

    Changelog
    ---------

    0.1.0 (2021-02-01)
    ~~~~~~~~~~~~~~~~~~

    Added
    ^^^^^

    * Added a new feature.

    Changed
    ^^^^^^^

    * Changed an existing feature.

    Deprecated
    ^^^^^^^^^^

    * Deprecated an existing feature.

    Removed
    ^^^^^^^

    * Removed an existing feature.

    Fixed
    ^^^^^

    * Fixed a bug.

    0.0.1 (2021-01-01)
    ~~~~~~~~~~~~~~~~~~

    Added
    ^^^^^

    * Added a new feature.


Contributing Documentation
--------------------------

Contributing to the documentation is as easy as contributing to the codebase. All the source files
for the documentation are located in the ``orbit/docs`` directory. The documentation is written in
`reStructuredText <https://docutils.sourceforge.io/rst.html>`__ format.

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`__ with the
`Book Theme <https://sphinx-book-theme.readthedocs.io/en/stable/>`__
for maintaining the documentation.

Sending a pull request for the documentation is the same as sending a pull request for the codebase.
Please follow the steps mentioned in the `Contributing Code`_ section.

To build the documentation, we recommend creating a `virtual environment <https://docs.python.org/3/library/venv.html>`__
to install the dependencies. This can also be a `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.

Execute the following commands in the terminal:

1. Enter the ``orbit/docs`` directory.

   .. code:: bash

     # enter the location of the docs directory (relative to the root of the repository)
     cd docs

2. Install the dependencies (preferably in a virtual/conda environment).

   .. code:: bash

     # install the dependencies
     pip install -r requirements.txt

3. Build the documentation.

   .. code:: bash

     # build the documentation
     make html

4. Open the documentation in a browser.

   .. code:: bash

     # open the documentation in a browser
     xdg-open _build/html/index.html


Contributing assets
-------------------

Currently, we host the assets for the extensions on `NVIDIA Nucleus Server <https://docs.omniverse.nvidia.com/prod_nucleus/prod_nucleus/overview.html>`__.
Nucleus is a cloud-based storage service that allows users to store and share large files. It is
integrated with the `NVIDIA Omniverse Platform <https://developer.nvidia.com/omniverse>`__.

Since all assets are hosted on Nucleus, we do not need to include them in the repository. However,
we need to include the links to the assets in the documentation.

The included assets are part of the `Isaac Sim Content <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/reference_assets.html>`__.
To use this content, you need to download the files to a Nucleus server or create an **Isaac** Mount on
a Nucleus server.

Please check the `Isaac Sim documentation <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_faq.html#assets-and-nucleus>`__
for more information on how to download the assets.

.. note::
  We are currently working on a better way to contribute assets. We will update this section once we
  have a solution. In the meantime, please follow the steps mentioned below.

To host your own assets, the current solution is:

1. Create a separate repository for the assets and add it over there
2. Make sure the assets are licensed for use and distribution
3. Include images of the assets in the README file of the repository
4. Send a pull request with a link to the repository

We will then verify the assets, its licensing, and include the assets into the Nucleus server for hosting.
In case you have any questions, please feel free to reach out to us through e-mail or by opening an issue
in the repository.
