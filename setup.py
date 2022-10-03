from setuptools import setup, find_packages

PACKAGES = find_packages(".", include=["pygment*"])

with open("requirements.txt") as f:
    REQUIRES = f.readlines()

setup(
    name="pygment",
    version="0.0.1",
    description="A collection of Variational Autoencoder models in PyTorch.",
    author="Dimitris Poulopoulos",
    author_email="dimitris.a.poulopoulos@gmail.com",
    license="Apache License Version 2.0",
    packages=PACKAGES,
    install_requires=REQUIRES,
    python_requires=">=3.8.0",
    include_package_data=True,
    zip_safe=False
)
