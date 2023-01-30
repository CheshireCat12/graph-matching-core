# -*- coding: utf-8 -*-
from glob import glob

from setuptools import setup, Extension
#
# install_requires = [
#     'cython',
#     'numpy',
#     'scipy',
#     'powerline-status',
#     'pytest',
#     'xmltodict',
#     'pytest',
#     'networkx',
#     'progress',
#     'pyyaml',
#     'psutil',
#     'gitpython',
#     'easydict',
#     'bunch',
#     'pandas',
#     'tqdm'
# ]


def extension_modules():
    """
    Find the cython extension modules to install
    :return:
    """
    import numpy
    ext = []
    files = glob('**/*.pyx', recursive=True)
    packages = ['cyged']
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

setup(name='cyged',
      version='0.2.3',
      description='A python module to compute GED using cython',
      author='Anthony Gillioz',
      author_email='anthony.gillioz@outlook.com',
      # install_requires=install_requires,
      ext_modules=extensions,
      )
