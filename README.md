# Federated Learning in Medical Imaging Using the ChRIS Platform

** **
### Collaborators

| Name          | Email           |
|---------------|-----------------|
| Amado Diallo | [amadod@bu.edu](mailto:amadod@bu.edu) |
| David Edelisr  | [jedelist@bu.edu](mailto:jedelist@bu.edu) |
| Julie Green | [jugreen@bu.edu](mailto:jugreen@bu.edu) |
| Matthew Hendsch    | [mhendsch@bu.edu](mailto:mhendsch@bu.edu) |
| Anisa Qureshi    | [anisaqu@bu.edu](mailto:anisaqu@bu.edu) |
| Ryan Smith    | [rpsmith@bu.edu](mailto:rpsmith@bu.edu) |

** **

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

* **Lab/Researcher (academic use)**: Runs training on their own data inside ChRIS. Contributes model updates to a central aggregator without moving raw data. They care about simple steps to start runs and seeing results.
* **Clinical user**: Just runs the plugin on local studies and looks at results. They only care about the output quality, convenience, and speed.
* **ChRIS operator (demo/admin)**: Spins up multiple local ChRIS instances with separate data plus one aggregator. They care about having a local demo that shows it reliably working. 


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


 <img width="1214" height="788" alt="image" src="https://github.com/user-attachments/assets/65df7f4d-8fb6-406a-9289-16185780cede" />


Design Implications and Discussion:

This section discusses the implications and reasons of the design decisions made during the global architecture design.

## 5. Acceptance criteria

Overall, our minimum acceptance criteria is to build an intelligent model that can utilize a large set of data from an aggregated set of weights to solve a medically relevant problem using the ChRIS platform. Our MVP is a dummy app that can demonstrate these capabilities. This app can be run on three machines. Two machines can act as workers during the learning phase, each learning from one half of the training data set. A third machine can then act as an aggregator, combining the weights from the workers. Ideally, this aggregated model will have a higher accuracy than either of the worker models. If this is the case, then our project will be considered a success.

We also have some stretch goals. Ideally, we'd make a pipeline with our app that would be able to solve a problem with some important medical significance. The ultimate challenge for our project would be to create a pipeline that would be used by real medical professionals. This requires the model to be very accurate and easy to use. Medical professionals will be resistant to any changes that will affect their workflow, so we have to ensure the impact is minimal and the pipeline does not take a long time to learn.

## 6.  Release Planning:

This project will be delivered across a series of 2-week sprints (iterations) from September to December. Each release will deliver functionality allowing us to validate progress with our mentor (and potentially professors).

### Sprint 1 (Sept 22 – Oct 6): Environment & Baseline Setup

**User stories:**

As a ChRIS operator, I want to open a single ChRIS instance locally so that I can test data ingestion and pipeline execution.

**Deliverables:**

* Local ChRIS deployment.
* Containerized baseline training plugin (Python) with clear inputs/outputs.
* Documentation of setup instructions and baseline results.

### Sprint 2 (Oct 7 – Oct 20): Multi-Node Deployment

**User stories:**

As a ChRIS operator or medical researcher, I want to deploy at least three ChRIS nodes and 1 aggregator to demo and test distributed data environments.

**Deliverables:**

* Deployment scripts to help for multiple ChRIS instances (VMs or containers).
* Verification that each node can ingest its own dataset and run the baseline training plugin/app.
* Networked aggregator stub in place to test weight passing (without any actual federated logic yet).

### Sprint 3 (Oct 21 – Nov 3): Federated Learning Integration

**User stories:**

As a researcher, I want local model updates sent to an aggregator so that federated averaging can begin.

**Deliverables:**

* Integration of OpenFL with our existing ChRIS plugins/apps.
* Federated aggregation working across multiple nodes with central aggregation.

### Sprint 4 (Nov 4 – Nov 17): Security & Privacy Layer

**User stories:**

As a hospital IT or security stakeholder, I need peace of mind that no raw images or Protected Health Information (PHI) leaves my site.
 
**Deliverables:**

* TLS-secured (encrypted) communication between nodes and the aggregator.
* Audit of logs to confirm no raw data leaves the network.
* Early draft of documentation that outlines security configuration.
* End-to-end proof showing model parameters exchanged, no raw data transfer.

### Sprint 5 (Nov 18 – Dec 1): Metrics & Monitoring

**User stories:**

As a user, I want to be able to track accuracy, runtime, and bandwidth to evaluate FL effectiveness.

As an operator, I want a dashboard that displays progress across epochs (FL training rounds) and other throughput metrics.

**Deliverables:**

* Collection of accuracy, time, projections, bandwidth, and resource usage.
* Lightweight dashboard.
* Maybe: Comparison or benchmark of federated model to central baseline (single site).

### Sprint 6 (Dec 2 – Dec 15): Documentation & Demo Packaging

**User stories:**

As a clinical user, I want to easily run the demo inference through ChRIS without any computer skills required.

As a demo operator or researcher, I want a reproducible setup and the necessary documentation to train and deploy my own instances of this pipeline.

**Deliverables:**

* Finalized single-command launch interface for end-to-end inference.
* Architecture diagram, polished runbook, and thorough documentation.
* Modular final pipeline and plugins/apps that enable future scalability.
* Stretch goals if time permits (framework substitution, extended monitoring, example datasets, additional features).

** **

## General comments

Remember that you can always add features at the end of the semester, but you can't go back in time and gain back time you spent on features that you couldn't complete.

** **

For more help on markdown, see
https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

In particular, you can add images like this (clone the repository to see details):

![alt text](https://github.com/BU-NU-CLOUD-SP18/sample-project/raw/master/cloud.png "Hover text")

