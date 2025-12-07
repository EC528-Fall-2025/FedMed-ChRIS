## FedMed Flower Apps + miniChRIS Walkthrough

This repo ships a Flower Deployment Engine demo that runs inside miniChRIS. Two ChRIS plugins wrap the Flower binaries:

- `plugins/superlink_plugin` – spins up a Flower **SuperLink** and launches the bundled FedMed **ServerApp**.
- `plugins/supernode_plugin` – starts a Flower **SuperNode** that executes the FedMed **ClientApp**.

### Architecture at a glance

- **Long-lived components**: the Flower **SuperLink** acts as the control plane; each Flower **SuperNode** reconnects and waits for tasks.
- **Short-lived components**: every `flwr run` triggers a **ServerApp** plus one **ClientApp** per SuperNode; they pull the Flower App bundle from SuperLink, run, and exit.
- **ChRIS orchestration**: SuperLink stages the Flower App bundle, starts the control plane, and kicks off the run. SuperNode starts `flower-supernode`, streams metrics, and writes a JSON summary to `/outgoing`.

Below is the workflow for building images, booting miniChRIS, importing pipelines, and running end-to-end. It also covers the reverse SSH setup (bastion) and the single-machine fallback.

---

### 0) Pre-run checklist
- Decide which mode you’ll use:
  - Reverse SSH (cross-machine): bastion running (e.g., AWS), SSH keys available (`id_ed25519` + matching `known_hosts`).
  - Single-machine (demo): all components on one host; Docker network only.
- If using reverse SSH on AWS:
  - Launch a small EC2 instance as a bastion (Linux AMI; allow inbound SSH/22).
  - Generate or use an SSH keypair (`id_ed25519`); add the bastion host to `known_hosts`.
  - Verify you can `ssh -i id_ed25519 <user>@<bastion-ip>` from your machine(s) before wiring it into pipelines.
- Ensure Docker is running.
- If you run into other issues, see the **Common Issues** section at the bottom. 

### 1) Build (or rebuild) the OCI images

```bash
docker build -t docker.io/fedmed/pl-superlink:x.x.x plugins/superlink_plugin
docker build -t docker.io/fedmed/pl-supernode:x.x.x plugins/supernode_plugin
```

Rebuild whenever you change plugin or Flower App code (SuperLink bundles the Flower App).

---

### 2) Start miniChRIS

From the repo root:

```bash
./minichris.sh
```

This boots the stack, runs `chrisomatic`, and registers both FedMed plugins.

---

### 3) Prepare feeds for reverse SSH (bastion) or single-machine fallback

**Create feeds**
- Go to `http://localhost:8020/library/home/chris` to create feeds. (Sign in with *usr: chris* and *pass: chris1234* when prompted).

**Reverse SSH (recommended for cross-machine runs):**
- Have a bastion running (we use AWS). Save its SSH keys (`id_ed25519`, matching `known_hosts` entry).
- Create a feed that contains a folder with `known_hosts` and `id_ed25519`. Name the feed *Superlink*. 

**Single-machine fallback (recommended for demo runs):**
- If you don’t want to set up a bastion, create a feed from any file for SuperLink. It will default to the Docker network, and communication will be local-only (all SuperNodes must run on the same machine).

---

### 4) Import the pipelines via the UI

1. Open `http://localhost:8020/pipelines`
2. If you decide to use the reverse SSH method, go to the plugins folder and change bastion host and bastion ports based on how you set it up. If you are running the demo version, go to the supernode yml files and change the superlink host to the ip the superlink prints out when you run it. (To see the output, click the "**>_**" symbol at the top of the screen). 
3. Upload all YAMLs from `pipelines/`:
   - `pipelines/fedmed_pl_superlink.yml`
   - `pipelines/fedmed_pl_supernode_x.yml` do this for 0, 1, and 2. 
   > If you are running this on a single machine and not using the bastion, you may want to see step 5 on how to adjust the yml files in this case.

---

### 5) Launch analyses (SuperLink first, then SuperNodes)

#### SuperLink analysis
1. Open the `superlink` analysis, right-click its lone node, and select **Add Pipeline → FedMed Flower SuperLink v x.x.x**.
2. When training finishes, `/outgoing/server_summary.json` contains the parsed summary.

#### SuperNode analysis
1. Make sure `superlink_host` is set to the SuperLink address in the pipeline yml file (bastion-resolvable if using reverse SSH; Docker network IP if local-only). 
2. Open each `supernode` analysis, right-click the node, and select **Add Pipeline → FedMed Flower SuperNode v x.x.x**.
3. Launch one analysis per participant. Each SuperNode logs `SUMMARY` messages; `/outgoing/client_metrics.json` captures the last one.

**Order:** Run SuperLink first; SuperNodes connect while it’s up.

---

### 6) Cleanup

1. Tear down the stack:
   ```bash
   ./minichris.sh down
   ```
2. Remove images if desired:
   ```bash
   docker image rm docker.io/fedmed/pl-superlink:x.x.x docker.io/fedmed/pl-supernode:x.x.x
   ```
3. Optional: prune state caches:
   ```bash
   rm -rf /tmp/fedmed-flwr /tmp/fedmed-flwr-node*
   ```

Re-run by rebuilding (Step 1) and running `./minichris.sh`. The Flower App stays in sync once you rebuild SuperLink.

## Common Issues (With Fixes)

**It is highly recommended that you run this on a Unix machine if available. If you do not have one available, a VM could also work. If you run into any issues, this should be the first thing you should try.**

- Can't get Supernode to connect to the Superlink:
   - Make sure you have set the correct host address in all of the pipelines. The IP address will be printed when you run the superlink (or you can get it from your AWS instance). 
   - Sometimes the Supernode will not connect to the Superlink through reverse SSH when you run them both plugins on the same device. Either run them on seperate devices, or connect them through the docker network instead. 
- The Superlink/Supernode randomly stops it's run before finishing: 
   - This is seemingly from a UI bug within ChRIS that causes a running plugin to end when you close the tab. Run the Superlink and Supernodes in different tabs and make sure you don't close them. 
- General Fixes:
   - Run `./minichris.sh down` from the repo root then pull it back up. 
   - Clear cache in browser. 
   - Restart Docker. 
   - Restart Computer.

**With any other issues, reach out to the FedMed team and we can try to help you through it.**