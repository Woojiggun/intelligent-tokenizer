"""
Setup script for Intelligent Tokenizer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="intelligent-tokenizer",
    version="0.1.0",
    author="Woo Jinhyun",
    author_email="ggunio5782@gmail.com",
    description="Pure learning-based universal tokenizer with hierarchical boundary detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/intelligent-tokenizer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "training": [
            "wandb",
            "tensorboard",
            "matplotlib",
        ],
    },
    entry_points={
        "console_scripts": [
            "intelligent-tokenizer=demo.interactive_demo:main",
        ],
    },
    keywords="tokenizer, nlp, deep-learning, transformer, compression, multilingual",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/intelligent-tokenizer/issues",
        "Source": "https://github.com/yourusername/intelligent-tokenizer",
    },
)