from os.path import join

from setuptools import find_packages, setup


def get_version_and_cmdclass(package_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


install_requires = [
    "menpo<0.12.0",
    "menpo3d",
    "sanic",
    "sanic-cors",
    "loguru",
    "joblib",
    "PyYaml",
]
version, cmdclass = get_version_and_cmdclass("landmarkerio")

setup(
    name="landmarkerio",
    version=version,
    cmdclass=cmdclass,
    description="Menpo-based server for www.landmarker.io",
    author="James Booth",
    author_email="james.booth08@imperial.ac.uk",
    url="https://github.com/menpo/landmarkerio-server/",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    package_data={"landmarkerio": ["default_templates/*"]},
    packages=find_packages(),
    install_requires=install_requires,
    scripts=[
        join("landmarkerio", "lmio"),
        join("landmarkerio", "lmioserve"),
        join("landmarkerio", "lmiocache"),
        join("landmarkerio", "lmioconvert"),
    ],
)
