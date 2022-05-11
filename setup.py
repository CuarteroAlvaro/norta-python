from setuptools import setup, find_packages

setup(
    name="norta",
    version="0.1.0",
    author="CuarteroAlvaro",
    url="hhttps://github.com/CuarteroAlvaro/norta-python.git",
    packages=find_packages(),
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "numpy",
        "scipy",
        "statistics", 
        "tqdm",
        "pandas"
    ],
)