# pip install -e .
import os
import sys
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="equiformer_v2",
    packages=["equiformer_v2"],  # same as name
    # https://stackoverflow.com/questions/17155804/confused-about-the-package-dir-and-packages-settings-in-setup-py
    # the paths in package_dir should stop at the parent directory of the directories which are Python packages.
    package_dir={"": "../"},  # empty string means the root package
    version="0.1",
)
