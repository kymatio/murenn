from importlib.machinery import SourceFileLoader
from setuptools import setup


version = SourceFileLoader('murenn.version', 'murenn/version.py').load_module()

with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='murenn',
    version=version.version,
    description='Multiresolution neural networks',
    author='The Kymatio consortium',
    url='https://gitlab.univ-nantes.fr/kymatio/murenn',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: ISC License (ISCL)',
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='wavelets, deep learning, signal processing, audio, speech, music',
    license='3-clause BSD',
    install_requires=['dtcwt>=0.13.0', 'torch>2.0.0']
)
