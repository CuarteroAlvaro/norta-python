from setuptools import setup, find_packages

setup(
    name="norta-python",
    version="0.1.0",
    author="CuarteroAlvaro",
    url="hhttps://github.com/CuarteroAlvaro/norta-python.git",
    packages=find_packages(),
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    keywords=["statistics", "norta"],
    download_url="https://github.com/CuarteroAlvaro/norta-python/archive/refs/tags/0.1.0.tar.gz",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "statistics", 
        "tqdm",
        "pandas",
        "matplotlib"
    ],
)
