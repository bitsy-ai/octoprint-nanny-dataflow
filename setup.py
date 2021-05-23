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
    "print_nanny_client==0.6.2",
    "gcsfs",
    "pyarrow",
    "pandas",
    "tensorflow",
    "tensorflow-transform",
    "backoff",
]

setup(
    name=NAME,
    version=VERSION,
    description="",
    author="Leigh Johnson",
    author_email="leigh@bitsy.ai",
    url="",
    install_requires=REQUIRES,
    scripts=["print_nanny_dataflow/scripts/render_video.sh"],
    packages=find_packages(),
    include_package_data=True,
)
