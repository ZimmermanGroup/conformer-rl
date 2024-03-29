import pathlib
from setuptools import setup
import setuptools

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setuptools.setup(
    name="conformer-rl",
    version="1.1.0",
    description="Deep Reinforcement Library for Conformer Generation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ZimmermanGroup/conformer-rl",
    author="Runxuan Jiang",
    author_email="runxuanj@umich.edu",
    license="MIT",
    package_dir={"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=(
        'numpy',
        'tensorflow',
        'gym',
        'stable-baselines3>=1.0'
    ),
    extras_require={
        "dev": ["sphinx", "sphinx_rtd_theme", "pytest", "coverage", "pytest-mock"],
        "generate_molecules": ['common_wrangler == 0.3.5', 'ligninkmc', 'stk',],
        "visualization": ['py3Dmol', 'seaborn', 'jupyterlab']
    },
)