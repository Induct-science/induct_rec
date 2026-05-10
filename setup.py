from setuptools import setup, find_packages

setup(
    name="induct_rec",
    version="0.1.0",
    description="Induct's recommendation algorithm",
    author="Induct",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.0.0"
    ],
    python_requires=">=3.9",
)
