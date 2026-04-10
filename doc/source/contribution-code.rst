.. _contribute_code:

*******************************
Contributing to the source code
*******************************

Build a development environment
===============================

This page describes how to get the development environment set up so that
you can begin contributing code, documentation or examples to QuTiP-qip.  Please
stop by to talk to us either in the `QuTiP Google group`_ or in the issues page
on the `qutip-qip repository`_ if you have suggestions for new features, so we
can discuss the design and suitability with you.

To contribute to QuTiP-qip development, you will need to have a working knowledge of
``git``.  If you're not familiar with it, you can read `GitHub's simple introduction`_
or look at the official `Git book <https://git-scm.com/book/en>`_ which also has the basics,
but then goes into much more detail if you're interested.

.. _QuTiP Google group: https://groups.google.com/forum/#!forum/qutip
.. _qutip-qip repository: https://github.com/qutip/qutip-qip/
.. _GitHub's simple introduction: https://guides.github.com/introduction/git-handbook


Requirements
============

To build ``qutip-qip`` from source, you will need recent versions of

- ``python`` (at least version 3.10)
- ``setuptools``

You should set up a separate virtual environment to house your development
version of qutip-qip so it does not interfere with any other installation you might
have. If you use conda, this can be done with the command

.. code-block::

   conda create -n qutip-dev python>=3.10

This will create the virtual environment ``qutip-dev``, which you can then
switch to by using the command ``conda activate qutip-dev``.

.. note::
   You do not need to use ``conda``---any suitable virtual environment manager
   like ``venv``, ``uv`` should work just fine.


Creating a Local Copy
=====================

At some point you will (hopefully) want to share your changes with us, so you
should fork the main repository on GitHub into your account, and then clone
that forked copy.  If you do not create a fork on GitHub, you will be able to
install and modify QuTiP-qip, but you will not be able to push any changes you make
back to GitHub so you can share them with us.

To create a fork, go to the relevant repository's page on GitHub (for example,
the `qutip-qip repository`_), and click the fork
button in the top right.  This will create a linked version of the repository in
your own GitHub account. For additional details you may refer to GitHub's documentation
on `forking <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_.

You can now "clone" your fork onto your local computer.  The command will look
something like

.. code-block::

   git clone https://github.com/<user>/qutip-qip

where ``<user>`` is your GitHub username (i.e. *not* ``qutip``).  This will
create a folder in your current directory called ``qutip-qip`` which contains the
repository.  This is your *local copy*.

.. note::
   You can put your local copy wherever you like, and call the top-level
   directory whatever you like, including moving and renaming it after the
   ``clone`` operation.


Building From Source
====================

Make sure you have activated your QuTiP development virtual environment that you
set up earlier. If you are in the root of the ``qutip-qip`` repository
(you should see the file ``pyproject.toml``), then the command to install is

.. code-block::

   pip install -e .[full]

After you have done this, you should
be able to ``import qutip_qip`` from anywhere as long as you have this development
environment active. Also you should install the dev dependencies for testing, linting
by running the command:

.. code-block::

   pip install --upgrade pip
   pip install --group dev

.. note::
   You do not need to re-run above commands even when you make changes
   to the project's Python files. The ``-e`` flag automatically ensures that you are
   using the latest local changes when you do ``import qutip_qip``

You should now be able to run the tests.  From the root of the repository, or in
``qutip-qip/tests`` folder, you can simply run ``pytest`` to run
all the tests.  The full test suite will take around 1-2 minutes, depending on
your computer.  You can test specific files by passing them as arguments to
``pytest``.


Contributing Code
=================

QuTiP development follows the "GitHub Flow" pattern of using Git.  This is a
simple triangular workflow, and you have already done the first step by creating
a fork.  In general, the process for contributing code follows a short list of
steps:

#. Fetch changes from the QuTiP organisation's copy
#. Create a new branch for your changes
#. Add commits with your changes to your new branch
#. Push your branch to your fork on GitHub
#. Make a pull request (PR) to the qutip/qutip-qip repository

You can read more documentation about this pattern in the
`GitHub guide to Flow`_, and see
`the GitHub blog post
<https://github.blog/2015-07-29-git-2-5-including-multiple-worktrees-and-triangular-workflows/#improved-support-for-triangular-workflows>`_
about when the Git tool added greater support for this type of triangular work.

While using this pattern, you should keep your ``master`` branch looking the
same as ours, or at least you should not add any commits to it that we do not
have.  Always use topic branches, and do not merge them directly into
``master``.  Wait until your PR has been accepted and merged into our version,
then pull down the changes into your ``master`` branch.

To fetch changes from our copy, you will need to add our version (the repository
that you clicked "Fork" on) as a Git remote.  The base command is

.. code-block::

   git remote add upstream https://github.com/qutip/qutip-qip

This will add a remote called ``upstream`` to your local copy.  You will not
have write access to this, so you will not be able to push to it.  You will,
however, be able to fetch from it.  While on ``master``, do

.. code-block::

   git pull upstream master

Unless you have made changes to your own version of ``master``, this will bring
you up-to-speed with ours.  To create and swap to a new branch to work on, use

.. code-block::

   git branch <branchname>
   git switch <branchname>

``git branch <branchname>`` will create a new branch with the specified branch-name, while
the ``switch`` command switches to the specified branch. To add commits to a branch, make the
changes you want to make, then call

.. code-block::

   git add <file1> [<file2> ...]

on all the files you changed, and do

.. code-block::

   git commit -m "<your message>"

to commit them.  Once you've made all the commits you want to make, push them to
your GitHub fork with

.. code-block::

   git push -u origin

and make the Pull Request (PR) using the GitHub web interface in the qutip-qip repository.


.. _GitHub guide to Flow: https://guides.github.com/introduction/flow


Pull Requests
=============

Please give the pull request a short, clear title that gives us an idea of what
your proposed change does.  It's good if this is not more than ten words, and starts
with an imperative verb.  Good examples of titles are:

- `Add QROT and MS gate <https://github.com/qutip/qutip-qip/pull/193>`_
- `Simplify the structure of CircuitSimulator <https://github.com/qutip/qutip-qip/pull/225>`_
- `Define max_step separately for CavityQED <https://github.com/qutip/qutip-qip/pull/199>`_

In the body of the PR, please use the template provided.  In particular,
describe in words what you've done and why you've done it and link to any issues
that are related.  Please also write a short comment (about a sentence) for the
changelog.  If you have made quite a small PR, the changelog can just be a
copy-paste of the title.

All PRs will be reviewed by the admin team, and merged subject to their
comments.  We're happy to answer questions and help you if we ask for changes.
If you have lots of questions before you start, please consider raising an issue
on GitHub (on our copy of the repository) first, so we can discuss it with you
before you start coding.  If you've noticed a bug and you have submitted a PR to
fix it, you may also want to check it hasn't been reported before as an issue,
and comment on it if it has to let us know you're working on it.

For any major new features, we strongly recommend creating an issue first, so we
can tell you if we think it's appropriate for the library, or point you to the
repository of a more suitable plugin, and organise the design with you before
you start coding.

When you make the PR, our CI/CD will run the full test suite on a variety of machines,
and check the code style.  You should run the tests locally before you make the PR to check
to save you time, because waiting for the CI to complete can take a while if the repository
is active.  We will not accept any PR with failing tests, unless the failure was not caused by you.


Code Style
==========

All new Python code should follow the standard `PEP 8 style guide`_.  Our CI
pipelines will test this when you make a PR. You can run ``black --check .`` to check
whether the code is formatted as per ``PEP 8`` standards and stays within the
88-character line-length requirement for readability. You can also auto format the
code in accordance with the ``PEP 8 standards`` using the command ``black .``.

All functions, classes and methods should have up-to-date docstrings.  We use
`Sphinx's autodoc extension`_ to generate API documentation, so please ensure
that your docstrings follow the `NumPy docstring format`_.

New or changed functionality should have comprehensive unit tests.  Add these
into the ``tests`` folder, following the conventions in there.  Try to add
them to the relevant file if it already exists.  We use ``pytest`` to run the
tests, so write your tests in this style.

New features should be supported by an example `notebook`_ in the
separate ``qutip-tutorial`` repository (`qutip/qutip-tutorials on GitHub`_).
This will require making a separate PR to that repository, and it's helpful if
you add links between the two in the descriptions.

Please use the same parameter and attribute names that are already in use within
the library when referring to similar objects. Please prefix private attributes and
methods in classes with an underscore, and use new classes sparingly; QuTiP is designed
for scientific notebook use, and a "pure" object-orientated style does not fit this
use case.  Please do not change existing public parameter and attribute names
without consulting with us first, even if the names are ugly, because many users
may depend on the names staying the same.


.. _PEP 8 style guide: https://www.python.org/dev/peps/pep-0008
.. _Sphinx's autodoc extension: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _NumPy docstring format: https://numpydoc.readthedocs.io/en/latest/format.html
.. _parametrisation routines: https://docs.pytest.org/en/stable/parametrize.html
.. _notebook: https://jupyter.org
.. _qutip/qutip-tutorials on GitHub: https://github.com/qutip/qutip-tutorials


Docstrings for the code
=======================

Each class and function should be accompanied with a docstring
explaining the functionality, including input parameters and returned values.
The docstring should follow
`NumPy Style Python Docstrings <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html>`_.


To ensure the codebase follows the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_
style guidelines, we use the ``pre-commit`` framework and the ``black`` formatter.

Automated Checking with pre-commit
----------------------------------

The easiest way to maintain compliance is to automate the checks. you only need to install
the git hooks once:

.. code-block:: bash

   pre-commit install

This ensures that linting checks run automatically every time you run git commit.
To run all checks manually across the entire repository without committing, use:

.. code-block:: bash

  pre-commit run --all-files

Alternatively with Black
------------------------

In the directory that contains ``some_file.py``, use

.. code-block::

  black some_file.py --check
  black some_file.py --diff --color
  black some_file.py

Using ``--check`` will show if any of the file will be reformatted or not.

  * `Code 0 <https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#the-basics>`_ means nothing will be reformatted.
  * Code 1 means one or more files could be reformatted. More than one files could
    be reformatted if ``black some_directory --check`` is used.

Using ``--diff --color`` will show a `difference <https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#diffs>`_ of
the changes that will be made by ``Black``. If you would prefer these changes to be made, use the last line of above code block.

.. note::
  We are currently in the process of checking format of existing code in ``qutip-qip``.
  Running ``black existing_file.py`` will attempt to format existing code. We
  advise you to create a separate issue for ``existing_file.py`` or skip re-formatting
  ``existing_file.py`` in the same PR as your new contribution.

  It is advised to keep your new contribution ``PEP8`` compliant.

Checking tests locally
=======================

Optionally you can generate code coverage report locally. First make sure
required packages have been installed.

.. code-block::

  pip install pytest-cov

A code coverage report in ``html`` format  can be generated locally for
``qutip-qip`` using the code line given below. By default the coverage report
is generated in a temporary directory ``htmlcov``. The report can be output
in `other formats <https://pytest-cov.readthedocs.io/en/latest/reporting.html>`_
besides ``html``.

.. code-block::

  pytest --cov-report html --cov=qutip_qip tests/

If you would prefer to check the code coverage of one specific file, specify
the location of this file. Same as above the report can be accessed in ``htmlcov``.

.. code-block::

  pytest --cov-report html --cov=qutip_qip tests/test_something.py
