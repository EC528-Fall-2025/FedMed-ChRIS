# ChrRIS Server (SuperLink) and Client (SuperNode) Plugin for Distributed Federated Learning Using Flower Framework
## Local Testing Steps: 
### 1. Create Venv for both plugins:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Flower Server (SuperLink):
From `/pl-flwrServer`:
```bash
mkdir -p /tmp/fl_server_in /tmp/fl_server_out

python app.py \
  --host 0.0.0.0 \
  --port 9091 \
  --rounds 3 \
  --expected-clients 2 \
  /tmp/fl_server_in \       # inputdir
  /tmp/fl_server_out        # outputdir
```

#### Explanation:
--host 0.0.0.0 : binds on all interfaces (so 127.0.0.1:9091 works for clients).
--expected-clients 2 : server will wait for two clients each round.
rounds 3 : three global FL rounds.
inputdir = /tmp/fl_server_in (unused, but required by the plugin interface).
outputdir = /tmp/fl_server_out (server will write server_summary.json there).

#### Expected Logs:
```
waiting for 2 client(s) at 0.0.0.0:9091 for 3 FL round(s)
(and some IP discovery printing)
The process will sit there waiting for clients.
```

### 3. Start SuperNode #0 and #1 (Flower Clients):
In a second terminal from `/pl-flwrClient` run `source venv/bin/activate`, then: 

```bash
mkdir -p /tmp/fl_client0_in /tmp/fl_client0_out

# launch Clinet #0
python app.py \
  --cid 0 \
  --total-clients 2 \
  --server-host 127.0.0.1 \
  --server-port 9091 \
  --learning-rate 0.01 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 2 \
  --seed 13 \
  --device auto \
  --data-root MNIST_root/data/ \
  /tmp/fl_client0_in \
  /tmp/fl_client0_out
```

#### Explanation:
* Client 0 takes its partition of MNIST (1/2 of train set).
* Connects to 127.0.0.1:9091.
* Participates in all 3 FL rounds.
* Writes its final metrics to /tmp/fl_client0_out/client_metrics.json.

#### Expected Logs: 
```
[mnist-fl-client:0] connecting to 127.0.0.1:9091 with XXXX train samples
```
Start Supernode #1: In a third terminal from the same client source directory run the following.

```bash
mkdir -p /tmp/fl_client1_in /tmp/fl_client1_out

python app.py \
  --cid 1 \
  --total-clients 2 \
  --server-host 127.0.0.1 \
  --server-port 9091 \
  --learning-rate 0.01 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 2 \
  --seed 13 \
  --device auto \
  --data-root MNIST_root/data/ \
  /tmp/fl_client1_in \
  /tmp/fl_client1_out
```


## Containerize Locally

### 1. Build the Server and Client Docker Images:

From `/pl-flwrClient`: 
```bash 
docker build -t jedelist/pl-flwr-client:latest .

# OR

docker build -t pl-flwr-client:latest .
```

Then, from `/pl-flwrServer`:
```bash
docker build -t jedelist/pl-flwr-server:latest .

# OR 

docker build -t jedelist/pl-flwr-server:latest .
```

### 2. Create Local Docker Network:

Run the following line so containers can see each other by name:
```bash
docker network create flwr-net
```

### 3. Run pl-flwrServer in Container:

```bash
mkdir -p /tmp/fl_server_in /tmp/fl_server_out

docker run --rm \
  --name flwr-server \
  --network flwr-net \
  -v /tmp/fl_server_in:/in \
  -v /tmp/fl_server_out:/out \
  -p 9091:9091 \
  jedelist/pl-flwr-server:latest \
  flwrServer \
  --host 0.0.0.0 \
  --port 9091 \
  --rounds 3 \
  --expected-clients 2 \
  /in /out
  ```

### 4. Run Clinets

Ensure you are in the directory `/pl-flwrClient`

#### a. Client #0
```bash
# Client 0
mkdir -p /tmp/fl_client0_in /tmp/fl_client0_out

docker run --rm \
  --name flwr-client-0 \
  --network flwr-net \
  -v /tmp/fl_client0_in:/in \
  -v /tmp/fl_client0_out:/out \
  -v "$(pwd)/MNIST_root/data:/mnist_data:ro" \
  jedelist/pl-flwr-client:latest \
  flwrClient \
  --cid 0 \
  --total-clients 2 \
  --server-host flwr-server \
  --server-port 9091 \
  --learning-rate 0.01 \
  --epochs 1 \
  --batch-size 128 \
  --num-workers 2 \
  --seed 13 \
  --device auto \
  --data-root /mnist_data \
  /in /out
```
- Ensure that the 'cid' and 'total-clients' arguments are correct (the cid cannot be the same between 2 clients)


#### b. Client #1:
```bash
# Client 1
mkdir -p /tmp/fl_client1_in /tmp/fl_client1_out

# Note that the cid is different from client 0, and that total-clinets match
docker run --rm \
  --name flwr-client-1 \
  --network flwr-net \
  -v /tmp/fl_client1_in:/in \
  -v /tmp/fl_client1_out:/out \
  -v "$(pwd)/MNIST_root/data:/mnist_data:ro" \
  jedelist/pl-flwr-client:latest \
  flwrClient \
  --cid 1 \
  --total-clients 2 \
  --server-host flwr-server \
  --server-port 9091 \
  --learning-rate 0.01 \
  --epochs 1 \
  --batch-size 128 \
  --num-workers 2 \
  --seed 13 \
  --device auto \
  --data-root /mnist_data \
  /in /out
```

#### c. Clean Up:

```bash 
docker network prune    

#OR

docker network rm flwr-net
``` 

### Running in miniChRIS:

Once the local images have been registered in `chrisomatic.yaml` file, open the minichris UI in a browser and log in as the admin user. 

#### 1. Server CLI:

Use the following CL arguments to launch the server pipeline (from an empty feed):

```bash
--host 0.0.0.0 --port 9091 --rounds 3 --expected-clients 2
```

#### 2. Client CLI:

Only upload /data to the feed, not MNIST_root, then use the following command line arguments to launch the client pipelines:
```bash
--cid 0 --total-clients 2 --server-host 172.17.0.2 --server-port 9091 --learning-rate 0.01 --epochs 1 --batch-size 64 --num-workers 2 --seed 13 --device auto --data-root data
```