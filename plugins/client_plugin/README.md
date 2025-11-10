# FedMed Flower Client

This plugin packages a Flower `NumPyClient` inside a ChRIS-friendly container. Run the commands below from inside `plugins/client_plugin/`. The actual training loop only knows how to fit a local logistic regression model; Flower merely orchestrates the parameter exchange.

## Build
```bash
docker build -t fedmed/fl-client .
```

## Run (example)
```bash
docker run --rm --name fedmed-client --network fedmed-net \
  -v $(pwd)/demo/client-in:/incoming:ro \
  -v $(pwd)/demo/client-out:/outgoing:rw \
  fedmed/fl-client \
    fedmed-fl-client --cid 0 --total-clients 1 \
    --server-host ${FEDMED_SERVER_IP} --server-port 9091 \
    /incoming /outgoing
```

Make sure `${FEDMED_SERVER_IP}` is populated via `docker inspect fedmed-server` (see the server README). After the Flower round completes, the plugin writes `client_metrics.json` into `/outgoing` so downstream ChRIS components (or you) can inspect the results.
