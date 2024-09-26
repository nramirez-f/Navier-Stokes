from setuptools import setup, find_packages

# Función para leer el archivo requirements.txt
def read_requirements(file):
    with open(file, 'r') as f:
        return f.read().splitlines()

setup(
    name='Navier-Stokes',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=read_requirements('requirements.txt'),
    author='nramirez-f',
    author_email='nramirez@uma.es',
    description='Implementation of Navier-Stokes Equation by MVF',
    url='https://github.com/nramirez-f/Navier-Stokes',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
