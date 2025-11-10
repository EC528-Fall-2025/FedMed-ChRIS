# ChrRIS Server (SuperLink) Plugin for Federated Learning using Flower Framework
## Local Testing Steps: 
### 1. Create Venv for both plugins:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Flower Server (SuperLink):
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
In a second terminal from the client source directory, run `source venv/bin/activate`
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
Start Supernode #1: In a third terminal from the client source directory run the following.
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


## 
