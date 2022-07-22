Developer Documentation
=======================

Thank you for your interest in contributing to :mod:`conformer_rl`. This section walks through how to submit
bug reports and feature requests, as well as setup instructions and best practices for developing on :mod:`conformer_rl`.

The best way to contribute code to :mod:`conformer_rl` is by `forking the project on GitHub <https://github.com/ZimmermanGroup/conformer-rl/fork>`_ and making a pull request. We encourage developers to document new features and write unit tests (if applicable).

Bug Reports and Feature Requests
--------------------------------
We are actively adding new features to this project and are open to all suggestions. If you believe you have encountered a bug, or if you have a feature that you would like to see implemented, please feel free to file an
`issue on GitHub <https://github.com/ZimmermanGroup/conformer-rl/issues>`_

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
