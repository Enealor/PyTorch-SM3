from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description_text = fh.read()

with open('LICENSE', 'r') as fh:
    license_text = fh.read()
setup(
    name='torch-SM3',
    version='0.0.1',

    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=['torch'],

    author='Eleanor Holland',
    author_email='holland.dwight@gmail.com',

    description='Adds the memory efficient SM3 optimizer to PyTorch.',
    long_description=long_description_text,
    long_description_content_type='text/markdown',
    url='https://github.com/Enealor/PyTorch-SM3',
    license=license_text,

)
