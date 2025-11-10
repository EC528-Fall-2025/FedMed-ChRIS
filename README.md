## FedMed Flower Demo (Docker + ChRIS Plugins)

This repo packages a minimal Flower-based federated learning demo as two standalone _ChRIS_ plugins so you can simulate “different networks” using two Docker containers:

- `plugins/server_plugin`: launches the Flower coordinator and waits for a configurable number of trainers.
- `plugins/client_plugin`: runs a local logistic-regression trainer that only knows how to fit its own data shard; it simply reports updates back to the server.

The client’s training loop is completely unaware of federated learning. Flower only wraps the trainer at the plugin boundary so we can show the two isolated containers talking to each other.

## Build the Plugin Images

Run all commands from the repo root.

```bash
docker build -t fedmed/fl-server plugins/server_plugin
docker build -t fedmed/fl-client plugins/client_plugin
```

## Local Demo (two terminals)

1. Create a network and output folders:
   ```bash
   docker network create fedmed-net || true
   mkdir -p demo/server-in demo/server-out demo/client-in demo/client-out
   ```
2. **Terminal 1 – server plugin**
   ```bash
   docker run --rm --name fedmed-server \
     --network fedmed-net \
     -v $(pwd)/demo/server-in:/incoming:ro \
     -v $(pwd)/demo/server-out:/outgoing:rw \
     fedmed/fl-server \
       fedmed-fl-server --host 0.0.0.0 --port 9091 \
       --rounds 1 --expected-clients 1 \
       /incoming /outgoing
   ```
   Grab the container’s IP (Flower prefers a literal IPv4 address on Docker networks):
   ```bash
   export FEDMED_SERVER_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' fedmed-server)
   echo "Server reachable at ${FEDMED_SERVER_IP}:9091"
   ```
   The server logs stream to the terminal and `demo/server-out/server_summary.json` captures the aggregated metrics.

3. **Terminal 2 – client plugin**
   ```bash
   docker run --rm --name fedmed-client \
     --network fedmed-net \
     -v $(pwd)/demo/client-in:/incoming:ro \
     -v $(pwd)/demo/client-out:/outgoing:rw \
     fedmed/fl-client \
       fedmed-fl-client --cid 0 --total-clients 1 \
       --server-host ${FEDMED_SERVER_IP} --server-port 9091 \
       /incoming /outgoing
   ```
   The client trains locally, reports metrics to stdout, and stores `client_metrics.json` inside `demo/client-out/`.

Once the Flower round finishes, both containers exit. Destroy the network with `docker network rm fedmed-net` if desired.

## File Overview

| Path                          | Purpose |
|-------------------------------|---------|
| `plugins/server_plugin/`     | Complete ChRIS plugin for the Flower server (Dockerfile, setup.py, etc.) |
| `plugins/client_plugin/`     | Client-side ChRIS plugin with its own Dockerfile and training loop |
| `requirements.txt`           | Optional helper for local Python development (mirrors the plugin deps) |

Feel free to duplicate `client_plugin` for additional sites—only CLI flags like `--cid`/`--total-clients` need tweaking to make them unique.
