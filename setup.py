"""
Setup script for TrustDiff.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trustdiff",
    version="1.0.0",
    author="TrustDiff Team",
    author_email="your.email@example.com",
    description="H-CAF Framework for AI Platform Cognitive Assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/trustdiff",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "trustdiff=trustdiff.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 