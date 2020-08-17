import os
from setuptools import setup, find_packages

data_files = ['convert.sh']

setup(
    name="NeoMlmodelConverter",
    version="1.0",

    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},

    # include data files
    data_files=data_files,

    scripts=['convert.sh'],

    # Note: You almost certainly don't want to do that.
    root_script_source_version=True,

    # Use the pytest brazilpython runner. Provided by BrazilPython-Pytest.
    test_command='pytest',
)
