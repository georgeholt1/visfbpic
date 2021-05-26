import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="visfbpic",
    version="0.0.1",
    author="George K. Holt",
    description="A package for visualising FBPIC results",
    long_description=long_description,
    license="MIT",
    packages=["visfbpic"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8"
    ]
)