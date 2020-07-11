from setuptools import setup, find_packages

version = '0.1.0'

with open('README.md', 'r') as fh:
    long_description_text = fh.read()

setup(
    name='torch-SM3',
    version=version,

    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=['torch'],

    author='Eleanor Holland',
    author_email='holland.dwight@gmail.com',

    description='Adds the memory efficient SM3 optimizer to PyTorch.',
    long_description=long_description_text,
    long_description_content_type='text/markdown',
    url='https://github.com/Enealor/PyTorch-SM3',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache-2',
)
