# PMFlow BNN v0.2.0 Setup Configuration
# This setup.py provides fallback compatibility for pip installations
# when pyproject.toml might not be fully supported

from setuptools import setup, find_packages
import os

# Read version from version.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'pmflow_bnn', 'version.py')
    with open(version_file, 'r') as f:
        exec(f.read())
    return locals()['__version__']

# Read README for long description
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "PMFlow Biological Neural Networks with Temporal Parallelism"

setup(
    name="pmflow-bnn",
    version=get_version(),
    author="PMFlow Development Team",
    author_email="contact@pmflow.dev",
    description="PMFlow Biological Neural Networks with Temporal Parallelism - Enhanced implementation of gravitational flow dynamics for neural computation",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/experimentech/pushing-medium",
    project_urls={
        "Homepage": "https://github.com/experimentech/pushing-medium",
        "Repository": "https://github.com/experimentech/pushing-medium",
        "Bug Tracker": "https://github.com/experimentech/pushing-medium/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="neural-networks biological-networks gravitational-dynamics temporal-parallelism pushing-medium",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.4.0",
            "jupyter>=1.0.0",
            "tqdm>=4.60.0",
        ],
        "cuda": [
            "torch[cuda]>=1.12.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)