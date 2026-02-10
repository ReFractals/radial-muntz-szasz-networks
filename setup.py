from setuptools import setup, find_packages

setup(
    name="radial-muntz-szasz-networks",
    version="0.1.0",
    description="Radial Müntz-Szász Networks (RMN): learnable radial power bases for multidimensional singularities",
    author="Gnankan Landry Regis N'guessan, Bum Jun Kim",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
    ],
    python_requires=">=3.9",
)
