import pathlib
from setuptools import setup
import setuptools

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setuptools.setup(
    name="torsionnet",
    version="1.0.0",
    package_dir={"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires=">=3.7",

    description="",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ZimmermanGroup/conformer-ml",
)