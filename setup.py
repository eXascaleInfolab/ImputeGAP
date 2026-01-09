import pathlib
import setuptools

setuptools.setup(
    name="imputegap",
    version="1.1.1",
    description="A Library of Imputation Techniques for Time Series Data",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eXascaleInfolab/ImputeGAP",
    author="Quentin Nater",
    author_email="quentin.nater@unifr.ch",
    license="MIT License",
    project_urls = {
        "Documentation": "https://imputegap.readthedocs.io/",
        "Source" : "https://github.com/eXascaleInfolab/ImputeGAP"
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=(
            open('requirements.txt').read().splitlines()
    ),    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        'imputegap': [
            'assets/*.png',  # Include logo
            'env/*.toml',  # Include TOML files from env
            'params/*.toml',  # Include TOML files from params
            'datasets/*.txt',  # Include TXT files from dataset
            'algorithms/lib/*.dll',  # Include DLL files from algorithms/lib (for Windows)
            'algorithms/lib/*.so'  # Include SO files from algorithms/lib (for Linux/Unix)
            'algorithms/lib/*.dylib'  # Include dylib files from algorithms/lib (for MACOS)
        ],
    },
    entry_points={"console_scripts": ["imputegap = imputegap.runner_display:display_title"]}
)