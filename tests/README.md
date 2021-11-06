# Unit tests

The unit tests make use of pytest and pytest-mocker.

To download a coverage report for all the unit tests run the following command in the root directory:
```
$ coverage run --source src -m pytest tests
$ coverage html
```

Then open the `index.html` file in the generated `htmlcov` directory.

Current code coverage is ~96%.