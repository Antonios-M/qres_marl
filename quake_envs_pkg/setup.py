from setuptools import setup, find_packages

setup(
    name="quake_envs",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geopandas",
        "numpy",
        "networkx",
        "overpy",
        "osmnx",
        "jupyter",
        "geopy",
        "openmatrix"
    ],
)
