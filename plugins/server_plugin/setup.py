from pathlib import Path

from setuptools import setup


def read_version() -> str:
    for line in Path("app.py").read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to determine version from app.py")


setup(
    name="fedmed-fl-server",
    version=read_version(),
    description="Flower-based coordinator ChRIS plugin for the FedMed demo",
    author="FedMed",
    author_email="dev@fedmed.org",
    url="https://github.com/FedMed-ChRIS",
    py_modules=["app"],
    install_requires=[
        "chris_plugin==0.4.0",
        "flwr==1.8.0",
    ],
    license="MIT",
    entry_points={
        "console_scripts": [
            "fedmed-fl-server = app:main",
        ]
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    extras_require={"none": [], "dev": []},
)
