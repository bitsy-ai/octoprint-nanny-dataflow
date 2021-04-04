# coding: utf-8

"""
    Contact: leigh@bitsy.ai
"""


from setuptools import setup, find_packages  # noqa: H301

NAME = "print-nanny-dataflow"
VERSION = "0.1.0"
# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = [
    "apache-beam[gcp]",
    "nptyping",
    "numpy",
    "pillow",
    "print_nanny_client>=0.5.0-dev60",
    "gcsfs",
    "pyarrow",
    "pandas",
]

setup(
    name=NAME,
    version=VERSION,
    description="",
    author="Leigh Johnson",
    author_email="leigh@bitsy.ai",
    url="",
    install_requires=REQUIRES,
    packages=["print_nanny_dataflow"],
    package_dir={"print_nanny_dataflow": "print_nanny_dataflow/"},
    include_package_data=True,
)
