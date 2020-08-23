import sys
from setuptools import setup


def description_text():
    description = 'A python package for the computation and visualisation of orbital mechanics.'

    return description


def readme():
    with open('README.md') as f:
        return f.read()


if sys.version_info[0] < 3 or sys.version_info[0] == 3 and sys.version_info[1] < 6:
    sys.exit('Sorry, Python < 3.6 is not supported')

install_requires = ['numpy', 'scipy', 'matplotlib', 'plotly']

setup(
    name='orbitalmechanics',
    version='1.0.0',
    description=description_text(),
    long_description=readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
    ],
    url='https://github.com/robbievanleeuwen/orbitalmechanics',
    author='Robbie van Leeuwen',
    author_email='robbie.vanleeuwen@gmail.com',
    license='MIT',
    packages=[
        'orbitalmechanics'
    ],
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False
)
