# FedMed Flower SuperLink

This folder contains the complete _ChRIS_ plugin that launches the Flower SuperLink for the FedMed demo. Run the commands below from inside `plugins/superlink_plugin/`. The SuperLink coordinates training, waits for a configurable number of Flower SuperNodes to connect over gRPC, and writes a JSON summary of the run to `/outgoing`.

## Build
```bash
docker build -t fedmed/fl-superlink .
```

## Run (example)
```bash
docker run --rm --name fedmed-superlink --network fedmed-net \
  -v $(pwd)/demo/server-in:/incoming:ro \
  -v $(pwd)/demo/server-out:/outgoing:rw \
  fedmed/fl-superlink \
    fedmed-fl-superlink --host 0.0.0.0 --port 9091 --rounds 1 --expected-clients 1 \
    /incoming /outgoing
```

Use `docker inspect fedmed-superlink` to obtain the IPv4 address and pass it to SuperNode containers (Flower prefers literal addresses on Docker networks).
