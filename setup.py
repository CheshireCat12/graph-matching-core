# -*- coding: utf-8 -*-
from glob import glob
import os
from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop

install_requires = [
    'cython',
    'numpy',
    'scipy',
    'powerline-status',
    'pytest',
    'xmltodict',
    'pytest-profiling',
    'networkx',
    'pandas',
    'progress',
    'pyyaml',
    'notebook',
    'psutil',
    'gitpython',
    'easydict',
    'bunch',
    'sklearn',
    'pygad'
]


def extension_modules():
    """
    Find the cython extension modules to install
    :return:
    """
    import numpy
    ext = []
    files = glob('**/*.pyx', recursive=True)
    packages = ['graph_pkg_core']
    for file in files:
        if any(file.startswith(pkg) for pkg in packages):
            ext_name = file[:-4].replace('/', '.')
            source_name = './' + file
            new_extension = Extension(name=ext_name,
                                      sources=[source_name],
                                      include_dirs=[numpy.get_include()],
                                      extra_compile_args=['-ffast-math', '-march=native'],
                                      define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
            ext.append(new_extension)

            print(f'Create new Extension for: {ext_name.split(".")[-1]}')

    return ext



# Create automatically the extenstions
extensions = extension_modules()

for e in extensions:
    e.cython_directives = {'language_level': "3",  # all are Python-3
                           'embedsignature': True}

setup(name='graph-matching-core',
      version='0.1.2',
      description='A graph module using cython',
      author='Anthony Gillioz',
      author_email='anthony.gillioz@outlook.com',
      install_requires=install_requires,
      setup_requires=[
          'setuptools>=18.0',  # automatically handles Cython extensions
          'cython>=0.28.4',
      ],
      ext_modules=extensions,
      )
