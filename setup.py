from setuptools import setup, find_packages
setup(
    name = 'VizCalibrateEDMF',
    version = '0.1',
    packages = find_packages(include = ["src", "src.*", "entr_viz", "entr_viz.*"]),
)