import pathlib
from setuptools import setup
import setuptools

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setuptools.setup(
    name="conformer_rl",
    version="1.0.0",
    package_dir={"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=(
        'numpy',
        'torch',
        'torchvision',
        'tensorboard',
        'gym',
        'stable-baselines3',
        'ligninkmc',
        'seaborn',
        'stk'
    ),
    description="Deep Reinforcement Library for Conformer Generation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ZimmermanGroup/conformer-ml",
)