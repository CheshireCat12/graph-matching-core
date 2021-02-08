# -*- coding: utf-8 -*-
from glob import glob

from setuptools import setup, Extension

install_requires = [
    'cython',
    'numpy',
    'scipy',
    'powerline-status',
    'pytest',
    'xmltodict',
]


def extension_modules():
    ext = []
    files = glob('**/*.pyx', recursive=True)
    for file in files:
        if file.startswith('graph_pkg'):
            ext_name = file[:-4].replace('/', '.')
            source_name = './' + file
            print(ext_name)
            print(source_name)
            new_extension = Extension(name=ext_name, sources=[source_name])
            ext.append(new_extension)
            print(f'Create new Extension for: {ext_name.split(".")[-1]}')

    return ext

# Create automatically the extenstions
extensions = extension_modules()

for e in extensions:
    e.cython_directives = {'language_level': "3",  # all are Python-3
                           'embedsignature': True}

setup(name='graph_pkg',
      version='0.0.1',
      description='A graph module',
      author='Anthony Gillioz',
      author_email='anthony.gillioz@outlook.com',
      install_requires=install_requires,
      setup_requires=[
          'setuptools>=18.0',  # automatically handles Cython extensions
          'cython>=0.28.4',
      ],
      ext_modules=extensions
      )
