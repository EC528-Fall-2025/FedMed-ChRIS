from setuptools import setup
import re

_version_re = re.compile(r"(?<=^__version__ = (\"|'))(.+)(?=\"|')")

def get_version(rel_path: str) -> str:
    """
    Searches for the ``__version__ = `` line in a source code file.

    https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
    """
    with open(rel_path, 'r') as f:
        matches = map(_version_re.search, f)
        filtered = filter(lambda m: m is not None, matches)
        version = next(filtered, None)
        if version is None:
            raise RuntimeError(f'Could not find __version__ in {rel_path}')
        return version.group(0)


setup(
    name='pl-flwrClient',
    version=get_version('app.py'),
    description='A Flower client ChRIS plugin for distributed federated learning',
    author='BU FedMed Cloud Group',
    author_email='jedelist@bu.edu',
    url='https://github.com/EC528-Fall-2025/FedMed-ChRIS/tree/flwr-david/flwrDemo',
    py_modules=['app'],
    packages=['MNIST_root'],
    install_requires=[
        "chris_plugin==0.4.0",
        "flwr==1.8.0",
        "torch",
        "torchvision",
    ],
    license='MIT',
    entry_points={
        'console_scripts': [
            'flwrClient = app:main'
        ]
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Federated Learning',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    python_requires=">=3.10",
    extras_require={'none': [],'dev': ['pytest~=7.1']}
)
