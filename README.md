# Federated Learning in Medical Imaging Using the ChRIS Platform

** **
### Collaborators

| Name          | Email           |
|---------------|-----------------|
| Amado Diallo | [amadod@bu.edu](mailto:amadod@bu.edu) |
| David Edelist  | [jedelist@bu.edu](mailto:jedelist@bu.edu) |
| Julie Green | [jugreen@bu.edu](mailto:jugreen@bu.edu) |
| Matthew Hendsch    | [mhendsch@bu.edu](mailto:mhendsch@bu.edu) |
| Anisa Qureshi    | [anisaqu@bu.edu](mailto:anisaqu@bu.edu) |
| Ryan Smith    | [rpsmith@bu.edu](mailto:rpsmith@bu.edu) |

** **

## Sprint presentations
[Sprint 1](https://drive.google.com/file/d/1YTHomhBlerSoSIfBh1ot-iieM9HAaYIk/view?usp=sharing)

[Sprint 2](https://drive.google.com/file/d/11G4dycd98x9-rH8zcnjA942ix2_vBDy_/view?usp=sharing)

## 1.   Vision and Goals Of The Project:
Our vision is to produce a Federated Learning workflow for medical imaging so that data never leaves its collection site. The flow will have multiple ChRIS instances train a shared model locally and exchange only model updates with a central aggregator. We will then have created a reproducible, auditable, and secure reference implementation that a hospital can run across isolated datasets.

Our Primary Goals:
1. Deploy ≥3 ChRIS nodes + 1 aggregator (local/cloud VMs OK), each with its own dataset.

2. Integrate a federated framework (OpenFL or equivalent) with ChRIS pipelines (CUBE) to run FedAvg rounds.

3. Containerize training/inference plugins (Python) with clear inputs/outputs and versioned provenance.

4. Automate orchestration (script or Makefile) for bring-up, tear-down, and N training rounds.

5. Security & privacy: TLS between sites; verify that no raw images or PHI traverse the network (logs/artifacts).

6. Metrics: track AUC/accuracy, round time, bandwidth per round, and resource usage; export to a simple dashboard.

7. Baseline comparison: show federated model within ≤5% of a centrally trained baseline on a held-out test set.

8. Docs: architecture diagram, runbook, and “one-command demo” instructions.

## 2. Users/Personas Of The Project:

* **Lab/Researcher (academic use)**: Researchers use the platform to train machine learning models on their own datasets within ChRIS. They contribute model updates (not raw data) to a central aggregator, enabling collaborative learning across institutions while preserving privacy.
* **Clinical user**: Clinical users aim to integrate their existing medical data systems with the federated learning workflow. They want to securely and independently contribute data so that clinical researchers can leverage it for developing and validating medical imaging models.
* **ChRIS operator (demo/admin)**: ChRIS operators set up and manage multiple local ChRIS instances—each with its own dataset—plus a central aggregator. They are focused on ensuring reliable deployment and demonstration of federated learning across nodes.


** **

## 3.   Scope and Features Of The Project:

### In scope
- Multi-node ChRIS deployment: Each node (or container image) hosts an isolated set of medical images to be trained in parallel

- Central aggregation server: Receives trained models from each node and outputs combined model to each node until desired accuracy is achieved

- Secure & private flow of data: Ensure that only models are being transferred between nodes and aggregators. Outputs of each node will be automatically assessed for patient data. 

- OpenFL: This federated learning framework will be integrated with the ChRIS pipeline for generalizability of training models and privacy regulations

- Convenient orchestration: Simple process for setup, training round execution, shut-down, and clean up. There should be minimal commands used to orchestrate a complete machine learning pipeline.

### Out of scope
- Custom UI for data analysis

- Advanced federated learning algorithms
  
- Development of individual machine learning models for medical imaging


## 4. Solution Concept

The project will follow a multi-step approach. Our goal is to build a machine learning model that performs well while respecting data privacy. The steps are:
- **Stand up ChRIS instances**
 Install Docker on two VMs to simulate two different hospitals, then clone and run miniChRIS. This gives us two reachable CUBE endpoints (each hospital has its own CUBE, the ChRIS backend API).


- **Set up a secure central aggregator**
 Deploy a central “director/aggregator” service that coordinates rounds of federated training, using OpenFL. It will use TLS (transport layer security), which is a protocol that gives encrypted connections and authentication (certificates), this ensures the traffic is encrypted and the server knows which client is connecting.


- **Write and containerize Python training apps**
 Develop Python training code for the local nodes (to run training loops/epochs), and containerize it so each site can run it reproducibly. The output will be a versioned docker container image that runs on both ChRIS sites and participates in OpenFL rounds. 


- **Orchestrate federated training rounds**
 Run multiple rounds of FedAvg: the director schedules tasks, each site trains locally, sends weight deltas, and the director aggregates and redistributes the global model. This entails defining an FL plan, starting envoys at sites, and tuning the loop. The output will be a repeatable script or ChRIS pipeline that runs x rounds end-to-end and leaves an artifacted (version of model produced at specified round) global model per round.

- **Collect metrics**
Gather accuracy, performance, and stability metrics. These metrics demonstrate that the system works without ever centralizing the data. The output will be a report with per round global metrics, per site local metrics, total runtime and network traffic. 


<p align="center">
<img width="100%" alt="Federated Learning Flow Chart" src="https://github.com/user-attachments/assets/0c120a77-8fad-438b-bcdf-996a4375243b" />
</p>


## 5. Acceptance criteria

Overall, our minimum acceptance criteria is to build an intelligent model that can utilize a large set of data from an aggregated set of weights to solve a medically relevant problem using the ChRIS platform. Our MVP is a dummy app that can demonstrate these capabilities. This app can be run on three machines. Two machines can act as workers during the learning phase, each learning from one half of the training data set. A third machine can then act as an aggregator, combining the weights from the workers. Ideally, this aggregated model will have a higher accuracy than either of the worker models. If this is the case, then our project will be considered a success.

We also have some stretch goals. Ideally, we'd make a pipeline with our app that would be able to solve a problem with some important medical significance. The ultimate challenge for our project would be to create a pipeline that would be used by real medical professionals. This requires the model to be very accurate and easy to use. Medical professionals will be resistant to any changes that will affect their workflow, so we have to ensure the impact is minimal and the pipeline does not take a long time to learn.

## 6.  Release Planning:

This project will be delivered across a series of 2-week sprints (iterations) from September to December. Each release will deliver functionality allowing us to validate progress with our mentor (and potentially professors).

### Sprint 1 (Sept 22 – Oct 1): Environment & Baseline Setup

**User stories:**

As a ChRIS operator, I want to open a single ChRIS instance locally.

As a researcher, I want to be able to train and run inference on the MNIST handwritten digits dataset locally in a standard Python environment. 

**Deliverables:**

* Local ChRIS or miniChRIS deployment.
* Functional training, testing and inference Python setup locally (not a plugin yet). 
* Documentation of setup instructions and baseline results.

### Sprint 2 (Oct 1 – Oct 15): Multi-Node Deployment

**User stories:**

As a ChRIS operator or medical researcher, I want to be able to run my MNIST Python classifier ChRIS plugin on my local instance.

**Deliverables:**


* Containerized baseline training plugin (Python) with clear inputs/outputs.
* Verification that ChRIS instances can ingest its own dataset and run the baseline training plugin/app.
* Run the MNIST classifier plugin successfully on a ChRIS instance.

### Sprint 3 (Oct 15 – Oct 29): Federated Learning Integration

**User stories:**

As a researcher, I want local model updates sent to an aggregator server securely and reliably.

As a ChRIS operator or medical researcher, I want to deploy at least three ChRIS nodes and 1 aggregator to demo and test the reliability of weights passing to the aggregator server.

**Deliverables:**

* Integration of OpenFL with our existing ChRIS plugins/apps.
* Networked aggregator stub in place to test weight passing (without any actual federated logic yet).
* End-to-end demo showing model weights exchanged reliably, no raw data transfer.

### Sprint 4 (Oct 29 – Nov 12): Security & Privacy Layer

**User stories:**

As a hospital IT or security stakeholder, I need peace of mind that no raw images or Protected Health Information (PHI) leaves my site.

As a clinical researcher or developer, I need at least 2 nodes to be reliably sharing weights along with proper central aggregation computations at the aggregator server.
 
**Deliverables:**

* TLS-secured (encrypted) communication between nodes and the aggregator (might be free with OpenFL).
* Federated aggregation working across at least two nodes with central aggregation.
* Deployment scripts to help for multiple ChRIS instances.
* Early draft of documentation that FL setup and ML plugin design.

### Sprint 5 (Nov 12 – Nov 26): Metrics & Monitoring

**User stories:**

As a medical practitioner or researcher, I want to be able to train a model on MNIST dataset using FL across 3 nodes and one aggregator server.

As a user, I want to be able to track accuracy, runtime, and bandwidth to evaluate FL effectiveness.

As an operator, I want a lightweight dashboard that displays progress across epochs (FL training rounds).

**Deliverables:**

* Federated learning and aggregation reliable and functional with MNIST dataset across at least 3 nodes and one aggregation server; fully integrated with ChRIS.
* Lightweight dashboard for training metrics monitoring (accuracy, time, projections, bandwidth, and resource usage).
* Audit of logs to confirm no raw data leaves the network.
* Draft of documentation that outlines security and pipeline configuration.

### Sprint 6 (Nov 26 – Dec 10): Documentation & Demo Packaging

**User stories:**

As a clinical user, I want to easily run the demo inference through ChRIS without any computer skills required.

As a demo operator or researcher, I want a reproducible setup and the necessary documentation to train and deploy my own instances of this pipeline.

**Deliverables:**

* Finalized launch interface for end-to-end inference.
* Architecture diagram, polished runbook, and thorough documentation.
* Modular final pipeline and plugins/apps that enable future scalability.
* Stretch goals if time permits (framework substitution, extended monitoring, example datasets, additional features).
* Maybe: Comparison or benchmark of federated model to central baseline (single site).

** **
