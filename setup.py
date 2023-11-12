from setuptools import find_packages, setup


setup(
    name="ml-from-scratch",
    version="0.0.1",
    author="Mathias Gout",
    packages=find_packages(exclude=["tests"]),
    install_requires=["numpy==1.26.1"],
    python_requires="==3.9.*",
)
