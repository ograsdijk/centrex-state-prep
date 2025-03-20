from setuptools import find_packages, setup

VERSION = "0.1"
DESCRIPTION = (
    "Package for simulating coherent state preparation and evolution for centrex"
)

setup(
    name="state_prep",
    version=VERSION,
    author="Oskari Timgren",
    description=DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2",
        "scipy",
        "centrex_tlf",
        "matplotlib",
        "pandas",
        "joblib",
        "dill",
    ],
    include_package_data=True,
)
