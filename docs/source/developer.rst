Developer Documentation
=======================

This section describes the setup steps for development on :mod:`conformer_rl`.

Installing Developer Dependencies
---------------------------------
To install the dependencies needed for development, simply run::

    pip install conformer-rl[dev]

This command installs tools such as Sphinx (for documentation) and pytest (for testing).

Building Documentation
----------------------
Documentation for :mod:`conformer_rl` is located in the :code:`docs` directory. The documentation is written in restructured text (.rst) and built into html using Sphinx. The following command will build the documentation into the :code:`build` subdirectory of :code:`docs`::

    cd docs
    make html

Testing
-------
Unit tests are written in the :code:`tests` directory and are run using :code:`pytest`. To run the tests and generate a coverage report run the following command::

    coverage run --source src -m pytest tests
    coverage html

This would generate a html file in the directory :code:`htmlcov`. Current coverage is around 96%.
