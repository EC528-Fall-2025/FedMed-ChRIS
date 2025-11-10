## FedMed Flower Apps + miniChRIS Walkthrough

This repository now ships a complete Flower Deployment Engine demo that runs entirely inside miniChRIS. Two ChRIS plugins wrap the Flower binaries:

- `fedmed/fl-superlink:0.2.0` – spins up a Flower **SuperLink** and launches the bundled FedMed **ServerApp**.
- `fedmed/fl-supernode:0.2.0` – starts a Flower **SuperNode** that executes the FedMed **ClientApp**.

The Flower App (stored in `plugins/superlink_plugin/fedmed_flower_app`) uses SuperLink/SuperNode APIs exclusively and never touches deprecated `start_server`/`NumPyClient` helpers.

Below is the full workflow for building the images, booting miniChRIS, importing the pipelines, and running an end-to-end round.

---

### 1. Build (or rebuild) the OCI images

```bash
docker build -t docker.io/fedmed/fl-superlink:0.2.0 plugins/superlink_plugin
docker build -t docker.io/fedmed/fl-supernode:0.2.0 plugins/supernode_plugin
```

> Rebuild whenever you change plugin or Flower App code. The SuperLink image bundles the Flower App sources, so every rebuild keeps SuperNodes in sync.

---

### 2. Start miniChRIS

From the repo root:

```bash
./minichris.sh
```

The wrapper script boots the stack, runs `chrisomatic`, and registers both FedMed plugins. Watch for lines such as:

```
✔ docker.io/fedmed/fl-superlink:0.2.0 http://chris:8000/api/v1/plugins/6/
✔ docker.io/fedmed/fl-supernode:0.2.0 http://chris:8000/api/v1/plugins/7/
```

Those checkmarks mean the plugins are available to pipelines.

---

### 3. Import the pipelines via the UI

1. Open http://localhost:8020/pipelines and log in (`chris/chris1234`).
2. Upload both YAML definitions from `pipelines/`:
   - `pipelines/fedmed_fl_superlink.yml`
   - `pipelines/fedmed_fl_supernode.yml`

---

### 4. Launch demo analyses (SuperLink first, then SuperNodes)

Create two throwaway analyses named `superlink` and `supernode` (any seed file works).

#### SuperLink analysis

1. Open the `superlink` analysis, right-click its lone node, and select **Add Pipeline → FedMed Flower SuperLink v0.2.0**.
2. Leave defaults unless you need extra rounds/clients. The plugin log prints something like:
   ```
   [fedmed-fl-superlink] SuperNodes should target Fleet API at 172.22.0.3:9092
   [fedmed-fl-superlink] reachable IPv4 addresses: 172.22.0.3
   ```
3. Copy the Fleet API IP/port—clients must point `superlink_host`/`superlink_port` at that address.
4. When training finishes the log emits:
   ```
   [fedmed-superlink-app] SUMMARY {...}
   ```
   and `/outgoing/server_summary.json` receives the parsed JSON (round metrics, run id, etc.).

#### SuperNode analysis

1. Open the `supernode` analysis, right-click the node, and select **Add Pipeline → FedMed Flower SuperNode v0.2.0**.
2. Set `superlink_host` to the IP reported by the SuperLink pipeline (e.g. `172.22.0.3`). Keep `superlink_port=9092` unless you changed it.
3. Launch one analysis per participant (`cid` ranges from `0` to `total_clients-1`). Each SuperNode connects to the running SuperLink, receives the bundled ClientApp, and logs
   ```
   [fedmed-supernode-app] SUMMARY {"kind":"train", ... }
   ```
4. `/outgoing/client_metrics.json` captures the last `SUMMARY` payload for auditing.

> **Tip:** The SuperNode container shuts down automatically once the SuperLink stops, so you always run the SuperLink pipeline first and tear it down last.

---

### 5. Cleanup

1. Tear the stack down from the repo root:
   ```bash
   ./minichris.sh down
   ```
   (This runs `minichris/unmake.sh`, stopping containers and removing associated networks/volumes.)
2. Remove the OCI images if you no longer need them:
   ```bash
   docker image rm docker.io/fedmed/fl-superlink:0.2.0 docker.io/fedmed/fl-supernode:0.2.0
   ```
3. Optional: prune Flower state caches created during the run:
   ```bash
   rm -rf /tmp/fedmed-flwr /tmp/fedmed-flwr-node*
   ```

Re-run the workflow anytime by rebuilding the images (Step 1) and executing `./minichris.sh` again. The Flower App version stays synchronized with the containers, so any change in `fedmed_flower_app/` takes effect as soon as you rebuild `fedmed/fl-superlink`.
