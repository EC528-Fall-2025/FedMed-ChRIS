## FedMed Flower Plugins + miniChRIS Walkthrough

This repo contains two Flower-based _ChRIS_ plugins:

- `fedmed/fl-server:0.1.0` – runs the Flower coordinator.
- `fedmed/fl-client:0.1.0` – runs a single Flower trainer that connects to the coordinator.

Below is the end-to-end workflow for building the images, spinning up miniChRIS, importing the pipelines, and running a demo round entirely from the repo root.

---

### 1. Build (or rebuild) the OCI images

```bash
docker build -t docker.io/fedmed/fl-server:0.1.0 plugins/server_plugin
docker build -t docker.io/fedmed/fl-client:0.1.0 plugins/client_plugin
```

> Rerun these commands whenever you change the plugin code.

---

### 2. Start miniChRIS

From the repo root:

```bash
./minichris
```

This wraps `minichris/minichris.sh`, brings up the full miniChRIS stack, and automatically runs `chrisomatic`. Watch the console for lines like:

```
✔ docker.io/fedmed/fl-server:0.1.0 http://chris:8000/api/v1/plugins/6/
✔ docker.io/fedmed/fl-client:0.1.0 http://chris:8000/api/v1/plugins/7/
```

Those checkmarks confirm the plugins were registered successfully.

---

### 3. Import the pipelines via the UI

1. Open the UI at http://localhost:8020/pipelines (default credentials: `chris/chris1234`).
2. Click **Upload Pipeline**, then import both yml files from `pipelines/`:
   - `pipelines/fedmed_fl_server.yml`
   - `pipelines/fedmed_fl_client.yml`

Each upload should now appear in the “Pipelines” list.

---

### 4. Launch demo analyses (server first, then client)

1. Go to **Analysis → New and Existing Analyses**.
2. Create two throwaway analyses with any dummy file you want and name them anything. For this example we will call them `server` and `client`.

#### Server analysis
1. Open the `server` analysis.
2. Right-click the lone root node and choose **Add Pipeline**.
3. Select **FedMed Flower Server v0.1.0**.
4. Once the job starts, open the node details and check the **Logs** tab. Near the top you’ll see something like:
   ```
   [fedmed-fl-server] reachable IPv4 addresses: 172.19.0.2
   ```
   if that is not the exact address you see, make sure to change the **server_host** in `pipelines/fedmed_fl_client.yml`

#### Client analysis
1. Open the `client` analysis.
2. Right-click the root node → **Add Pipeline** → select **FedMed Flower Client v0.1.0**.
3. Submit the dialog. The client log should show `connecting to <IP>:9091` followed by the usual Flower messages.

---

### 5. Cleanup

1. Tear the stack down from the repo root:
   ```bash
   ./minichris down
   ```
   (This runs `minichris/unmake.sh`, stopping containers and removing associated networks/volumes.)
2. Optional: remove the OCI images if you don’t need them anymore:
   ```bash
   docker image rm docker.io/fedmed/fl-server:0.1.0 docker.io/fedmed/fl-client:0.1.0
   ```
3. Optional: prune residual volumes if you ran multiple demos:
   ```bash
   docker volume rm minichris-files minichris_db_data minichris_pfdcm
   ```

You can now reboot the workflow at any time by rebuilding the images (Step 1) and running `./minichris` again.
