.. _contribute_code:

*******************************
Contributing to the source code
*******************************

Build up an development environment
===================================

This page describes how to get a QuTiP development environment set up so that
you can begin contributing code, documentation or examples to QuTiP.  Please
stop by to talk to us either in the `QuTiP Google group`_ or in the issues page
on the `main QuTiP repository`_ if you have suggestions for new features, so we
can discuss the design and suitability with you.

To contribute to QuTiP development, you will need to have a working knowledge of
``git``.  If you're not familiar with it, you can read `GitHub's simple
introduction`_ or look at the official
`Git book <https://git-scm.com/book/en>`_ which also has the basics, but
then goes into much more detail if you're interested.

.. _QuTiP Google group: https://groups.google.com/forum/#!forum/qutip
.. _main QuTiP repository: https://github.com/qutip/qutip
.. _GitHub's simple introduction: https://guides.github.com/introduction/git-handbook


Requirements
============

To build ``qutip`` from source and to run the tests, you will need recent
versions of

- ``python`` (at least version 3.6)
- ``setuptools``
- ``numpy``
- ``scipy``
- ``pytest``
- ``Cython``

You will also need a working C++ compiler.  On Linux or Mac, there should
already be a suitable version of ``gcc`` or ``clang`` available, but on Windows
you will likely need to use a recent version of the Visual Studio compiler.

You should set up a separate virtual environment to house your development
version of QuTiP so it does not interfere with any other installation you might
have.  This can be done with ::

   conda create -n qutip-dev python>=3.6 setuptools numpy scipy pytest Cython

This will create the virtual environment ``qutip-dev``, which you can then
switch to by using ``conda activate qutip-dev``.  Note that this does *not*
install any version of ``qutip``, because we will be building that from source.

.. note::
   You do not need to use ``conda``---any suitable virtual environment manager
   should work just fine.


Creating a Local Copy
=====================

At some point you will (hopefully) want to share your changes with us, so you
should fork the main repository on GitHub into your account, and then clone
that forked copy.  If you do not create a fork on GitHub, you will be able to
read and install QuTiP, but you will not be able to push any changes you make
back to GitHub so you can share them with us.

To create a fork, go to the relevant repository's page on GitHub (for example,
the main QuTiP repository is
`qutip/qutip on GitHub <https://github.com/qutip/qutip>`_), and click the fork
button in the top right.  This will create a linked version of the repository in
your own GitHub account.  GitHub also has `its own documentation on forking
<https://guides.github.com/activies/forking>`_.

You can now "clone" your fork onto your local computer.  The command will look
something like ::

   git clone https://github.com/<user>/qutip

where ``<user>`` is your GitHub username (i.e. *not* ``qutip``).  This will
create a folder in your current directory called ``qutip`` which contains the
repository.  This is your *local copy*.

.. note::
   You can put your local copy wherever you like, and call the top-level
   directory whatever you like, including moving and renaming it after the
   ``clone`` operation.  As there is more than one QuTiP organisation
   repository, you may find it convenient to have ``qutip``, ``docs`` and
   ``notebooks`` all in a containing folder called ``qutip``.


Building From Source
====================

Make sure you have activated your QuTiP development virtual environment that you
set up earlier.  You should not have any version of QuTiP installed here.  If
you are in the root of the ``qutip`` repository (you should see the file
``setup.py``), then the command to build is ::

   python setup.py develop

If you need to test OpenMP support, add the flag ``--with-openmp`` to the end of
the command.

The ``develop`` target for
`setuptools <https://setuptools.readthedocs.io/en/latest/>`_ will compile and
link all of the Cython extensions, package the resulting files into an egg, and
add the package to the Python search path.  After you have done this, you should
be able to ``import qutip`` from anywhere as long as you have this development
environment active, and you will get your development environment.

.. note::
   In general, you do not need to re-run ``setup.py`` if you only make changes
   to the Python files.  These changes should appear immediately when you
   re-import ``qutip``, albeit with the standard Python proviso that you may
   need to re-open the Python interpreter or use :func:`importlib.reload` to
   clear the package cache.

   If you make changes to any Cython files, you *must*
   re-run ``setup.py develop`` in the same manner, or your extensions will not
   be built.

You should now be able to run the tests.  From the root of the repository, or
inside ``qutip`` or ``qutip/tests``, you can simply run ``pytest`` to run
everything.  The full test suite will take about 20--30 minutes, depending on
your computer.  You can test specific files by passing them as arguments to
``pytest``, or you can use the ``-m "not slow"`` argument to disable some of the
slowest tests.


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
#. Make a pull request (PR) to the QuTiP repository

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
that you clicked "Fork" on) as a Git remote.  The base command is ::

   git remote add upstream https://github.com/qutip/qutip

This will add a remote called ``upstream`` to your local copy.  You will not
have write access to this, so you will not be able to push to it.  You will,
however, be able to fetch from it.  While on ``master``, do ::

   git pull upstream master

Unless you have made changes to your own version of ``master``, this will bring
you up-to-speed with ours.  To create and swap to a new branch to work on, use
::

   git checkout -b <branchname>

You can then swap branches by using ``git checkout <branchname>`` without the
``-b`` option.  To add commits to a branch, make the changes you want to make,
then call ::

   git add <file1> [<file2> ...]

on all the files you changed, and do ::

   git commit -m "<your message>"

to commit them.  Once you've made all the commits you want to make, push them to
your GitHub fork with ::

   git push -u origin

and make the PR using the GitHub web interface in the main QuTiP repository.


.. _GitHub guide to Flow: https::guides.github.com/introduction/flow


Pull Requests
=============

Please give the pull request a short, clear title that gives us an idea of what
your proposed change does.  It's good if this not more ten words, and starts
with an imperative verb.  Good examples of titles are:

- `Create PR and issue templates <https://github.com/qutip/qutip/pull/1198>`_
- `Fix spin Husimi/Wigner functions <https://github.com/qutip/qutip/pull/1195>`_
- `Fix function QubitCircuit.add_circuit <https://github.com/qutip/qutip/pull/1269>`_
- `Remove eigh usage on mac <https://github.com/qutip/qutip/pull/1288>`_

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

When you make the PR, our continuous integration server will run the full test
suite on a variety of machines, and check the code style.  You should run the
tests locally before you make the PR to check to save you time, because waiting
for the CI to complete can take a while if the repository is active.  We will
not accept any PR with failing tests, unless the failure was not caused by you.


Code Style
==========

All new Python code should follow the standard `PEP 8 style guide`_.  Our CI
pipelines will test this when you make a PR, although in limited circumstances
we may accept a PR which breaks the 79-character line-length requirement by a
small amount if doing so improves readability.

You can locally check that you are following PEP 8 by using the ``pycodestyle``
(`link <https://pycodestyle.pycqa.org>`__) or ``flake8`` (`link
<https://flake8.pycqa.org>`__) tools, which can be installed by either ``conda``
or ``pip``.  We *strongly* recommend that you do this before commiting.

You do not need to fix all existing PEP 8 issues in a file that you are editing.
If you want to do so, please ensure that these changes are added as a separate
commit, and ideally a completely separate PR.  It is difficult to review the
meaningful code changes you have made when they are crowded out by minor
formatting ones.

All functions, classes and methods should have up-to-date docstrings.  We use
`Sphinx's autodoc extension`_ to generate API documentation, so please ensure
that your docstrings follow the `NumPy docstring format`_.

New or changed functionality should have comprehensive unit tests.  Add these
into the ``qutip/tests`` folder, following the conventions in there.  Try to add
them to the relevant file if it already exists.  We use ``pytest`` to run the
tests, so write your tests in this style; in particular, there is no need for
``unittest``-type ``unittest.TestCase`` classes with ``setUp`` and ``tearDown``
methods, and multiple tests on similar objects should make use of the
`parametrisation routines`_.

New features should be supported by an example `Jupyter notebook`_ in the
separate ``qutip-notebooks`` repository (`qutip/qutip-notebooks on GitHub`_).
This will require making a separate PR to that repository, and it's helpful if
you add links between the two in the descriptions.

Please use the same parameter and attribute names that are already in use within
the library when referring to similar objects, even if to do so would break the
PEP 8 rules for new names.  Please prefix private attributes and methods in
classes with an underscore, and use new classes sparingly; QuTiP is designed for
scientific notebook use, and a "pure" object-orientated style does not fit this
use case.  Please do not change existing public parameter and attribute names
without consulting with us first, even if the names are ugly, because many users
may depend on the names staying the same.


.. _PEP 8 style guide: https://www.python.org/dev/peps/pep-0008
.. _Sphinx's autodoc extension: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _NumPy docstring format: https://numpydoc.readthedocs.io/en/latest/format.html
.. _parametrisation routines: https://docs.pytest.org/en/stable/parametrize.html
.. _Jupyter notebook: https://jupyter.org
.. _qutip/qutip-notebooks on GitHub: https://github.com/qutip/qutip-notebooks


Docstrings for the code
=======================

Each class and function should be accompanied with a docstring
explaining the functionality, including input parameters and returned values.
The docstring should follow
`NumPy Style Python Docstrings <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html>`_.

Checking Code Style and Format
==============================

In order to check if your code in ``some_file.py`` follows `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_
style guidelines, `Black <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_
has to be installed.

.. code-block::

  pip install black

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

You can run tests and generate code coverage report locally. First make sure
required packages have been installed.

.. code-block::

  pip install pytest pytest-cov

``pytest`` is used to test files containing tests. If you would like to test all the
files contained in a directory then specify the path to this directory. In order to run
tests in ``test_something.py`` then specify the exact path to this file for ``pytest``
or navigate to the file before running the tests.

.. code-block::

  pytest path_to_some_directory
  pytest /path_to_test_something/test_something.py
  ~/path_to_test_something$ pytest test_something.py

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
