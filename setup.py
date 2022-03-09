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


# Install the external libraries (e.g. sigma.js used for the visualization of the graphs).
def friendly(command_subclass):
    """
    A decorator to customized setuptools install command
    - Download the external libraries.
    """
    origin_run = command_subclass.run

    def modified_run(self):
        origin_run(self)

        dir_ = './external/sigma.js/'
        github_repo = 'https://github.com/jacomyal/sigma.js.git'
        if not os.path.isdir(dir_):
            import git
            git.Repo.clone_from(github_repo, dir_, branch='main')

    command_subclass.run = modified_run
    return command_subclass


@friendly
class CustomDevelopCommand(develop):
    pass


@friendly
class CustomInstallCommand(install):
    pass


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
                                      extra_compile_args=['-ffast-math', '-march=native'])
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
      description='A graph module',
      author='Anthony Gillioz',
      author_email='anthony.gillioz@outlook.com',
      install_requires=install_requires,
      setup_requires=[
          'setuptools>=18.0',  # automatically handles Cython extensions
          'cython>=0.28.4',
      ],
      ext_modules=extensions,
      cmdclass={
          'develop': CustomDevelopCommand,
          'install': CustomInstallCommand,
      },
      )
