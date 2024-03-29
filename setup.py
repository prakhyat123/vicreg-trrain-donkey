import os

from setuptools import find_packages, setup

with open("requirements.txt", "r") as file_handler:
    requirements = file_handler.read().strip().split(os.linesep)

setup(
    name="vicregdonkey",
    packages=[package for package in find_packages() if package.startswith("vicregdonkey")],
    install_requires=requirements,
    extras_require={},
    description="vicreg for donkey car",
    author="Prakhyat Shankesi",
    license="MIT",
    version="0.1.0",
    python_requires=">=3.7",
    package_data={'': ['*.pth']},
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
