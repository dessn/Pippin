# Package imports
from setuptools import find_packages, setup


with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Setup function declaration
setup(
    name="Pippin",
    version="0.1.1",
    author="Samuel Hinton",
    author_email="samuelreay@gmail.com",
    description="Pipeline for Supernova Cosmology",
    long_description="Pipeline for Supernova Cosmology",
    url="https://github.com/dessn/Pippin",
    license="MIT",
    platforms=["Linux", "Unix"],
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Utilities",
    ],
    python_requires=">=3.7.*",
    packages=find_packages(),
    package_dir={"pippin": "pippin"},
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
)
