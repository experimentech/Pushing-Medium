from setuptools import setup, find_packages

setup(
    name="pmflow_bnn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "tqdm",
        "matplotlib",
        "numpy",
        "torchvision"
    ],
    author="Copilot, Tristan Mumford",
    description="PMFlow dynamics and PMFlow-BNN models for PyTorch",
    python_requires=">=3.8",
)

