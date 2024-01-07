from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="FakeNewsPredictor",
    version="0.0.1",
    description="A package to predict fake news",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vasilis Panagaris",
    author_email="va.panagaris@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=load_requirements(),
    python_requires=">=3.10",
)

