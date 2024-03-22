# pip install -e .
import os
import sys
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="deq2ff",
    # https://setuptools.pypa.io/en/latest/userguide/quickstart.html#package-discovery
    packages=["deq2ff"],  # same as name
    package_dir={'': 'src'}, # empty string means the root package
    # package_dir={'deq2ff': 'src/deq2ff'},
    # package_dir={'deq2ff':'src'},
    # package_dir={'deq2ff': 'deq2ff'},
    version="0.1",
    description="Force Fields with Deep Equilibrium Equivariant Transformers",
    author="Andreas Burger, Luca Thiede",
    # extra_requires=[
    #     "wandb",
    # ],
)

# setuptools.setup(
#    name='proj',
#    packages=['proj'],
#    package_dir={'':'src'},
# )