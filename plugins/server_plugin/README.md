# FedMed Flower Server

This folder contains the complete _ChRIS_ plugin that launches the Flower coordinator for the FedMed demo. Run the commands below from inside `plugins/server_plugin/`. The server waits for a configurable number of Flower NumPyClients to connect over gRPC and writes a JSON summary of the run to `/outgoing`.

## Build
```bash
docker build -t fedmed/fl-server .
```

## Run (example)
```bash
docker run --rm --name fedmed-server --network fedmed-net \
  -v $(pwd)/demo/server-in:/incoming:ro \
  -v $(pwd)/demo/server-out:/outgoing:rw \
  fedmed/fl-server \
    fedmed-fl-server --host 0.0.0.0 --port 9091 --rounds 1 --expected-clients 1 \
    /incoming /outgoing
```

Use `docker inspect fedmed-server` to obtain the IPv4 address and pass it to client containers (Flower prefers literal addresses on Docker networks).
