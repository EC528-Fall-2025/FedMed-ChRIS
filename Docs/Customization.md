# Customizing the Plugin

## Changing the name

To change the name of the plugin (AKA the command used in the command line):
- Navigate to `ChRIS/python-chrisapp-template/setup.py`
- Change `name='pl-matt'` to `name='[your name]'`
- Change `'pl-matt = app:main'` to `'[your name] = app:main'`

- Navigate to `ChRIS/python-chrisapp-template/Dockerfile`
- Change `CMD["pl-matt"]` to `CMD["[your name]"]`

- Navigate to `ChRIS/miniChRIS-docker/chrisomatic.yml`
- Change `- localhost/pl-matt:1.0.9` to '- localhost/[your name]:1.0.9`

## Changing the version

To change the version number of the plugin:
- Navigate to `ChRIS/python-chrisapp-template/app.py`
- Change `__version__ = '1.0.9'` to `'__version__ = '[your version]'`

## Changing the python version (not recommended)

If you want to change the version of python the Docker image uses (I don't recommend doing this, as it may break certain dependencies):
- Navigate to `ChRIS/python-chrisapp-template/Dockerfile`
- Replace `FROM docker.io/python:3.12.1-slim-bookworm` to `FROM docker.io/[your python version]`
