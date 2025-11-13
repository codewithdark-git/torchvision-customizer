"""Setup configuration for torchvision-customizer package."""

from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    include_package_data=True,
)
