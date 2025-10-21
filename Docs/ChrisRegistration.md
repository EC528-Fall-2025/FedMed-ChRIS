# Registering plugin to ChRIS

## Create Docker Image

- Make sure that Docker Desktop is open on your device.
- In ChRIS/python-chrisapp-template, run `docker build -t localhost/pl-matt:1.0.9 .'
- If you modify app.py, make sure that :1.0.9 is changed to match `__version__`.
- Building the docker image can take ~10 minutes, as there are many dependencies to install.

## Register with Chrisomatic

- Navigate into ChRIS/miniChRIS-docker
- Make sure that an entry for `- localhost/pl-matt:1.0.9 .` in chrisomatic.yml exists, replacing :1.0.9 with the proper version number, if necessary.
- Run ./minichris.sh
- Open [this link](http://localhost:8020/catalog) to view your plugin.
