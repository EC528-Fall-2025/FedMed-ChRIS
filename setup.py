from setuptools import setup
import re

_version_re = re.compile(r"(?<=^__version__ = (\"|'))(.+)(?=\"|')")

def get_version(rel_path: str) -> str:
    """
    Searches for the ``__version__ = `` line in a source code file.
    """
    with open(rel_path, 'r') as f:
        matches = map(_version_re.search, f)
        filtered = filter(lambda m: m is not None, matches)
        version = next(filtered, None)
        if version is None:
            raise RuntimeError(f'Could not find __version__ in {rel_path}')
        return version.group(0)

setup(
    name='pl-matt',
    version=get_version('app.py'),
    description='A ChRIS plugin for federated learning with OpenFL on MNIST',
    author='FNNDSC',
    author_email='dev@babyMRI.org',
    url='https://github.com/FNNDSC/python-chrisapp-template',
    py_modules=['app'],
    install_requires=[
        'chris_plugin==0.4.0',
        'openfl>=1.5',
        'torch>=1.12',
        'torchvision>=0.13',
        'numpy',
    ],
    license='MIT',
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'pl-matt = app:main'
        ]
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    extras_require={
        'none': [],
        'dev': [
            'pytest~=7.1'
        ]
    }
)