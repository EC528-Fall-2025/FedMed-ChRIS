# EC528-Fall-2025-template-repo

** **

## Project Description Template

The purpose of this Project Description is to present the ideas proposed and decisions made during the preliminary envisioning and inception phase of the project. The goal is to analyze an initial concept proposal at a strategic level of detail and attain/compose an agreement between the project team members and the project customer (mentors and instructors) on the desired solution and overall project direction.

This template proposal contains a number of sections, which you can edit/modify/add/delete/organize as you like.  Some key sections we’d like to have in the proposal are:

- Vision: An executive summary of the vision, goals, users, and general scope of the intended project.

- Solution Concept: the approach the project team will take to meet the business needs. This section also provides an overview of the architectural and technical designs made for implementing the project.

- Scope: the boundary of the solution defined by itemizing the intended features and functions in detail, determining what is out of scope, a release strategy and possibly the criteria by which the solution will be accepted by users and operations.

Project Proposal can be used during the follow-up analysis and design meetings to give context to efforts of more detailed technical specifications and plans. It provides a clear direction for the project team; outlines project goals, priorities, and constraints; and sets expectations.

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

The Scope places a boundary around the solution by detailing the range of features and functions of the project. This section helps to clarify the solution scope and can explicitly state what will not be delivered as well.

It should be specific enough that you can determine that e.g. feature A is in-scope, while feature B is out-of-scope.

** **

## 4. Solution Concept

This section provides a high-level outline of the solution.

Global Architectural Structure Of the Project:

This section provides a high-level architecture or a conceptual diagram showing the scope of the solution. If wireframes or visuals have already been done, this section could also be used to show how the intended solution will look. This section also provides a walkthrough explanation of the architectural structure.

 

Design Implications and Discussion:

This section discusses the implications and reasons of the design decisions made during the global architecture design.

## 5. Acceptance criteria

This section discusses the minimum acceptance criteria at the end of the project and stretch goals.

## 6.  Release Planning:

This project will be delivered across a series of 2-week sprints (iterations) from September to December. Each release will deliver functionality allowing us to validate progress with our mentor (and potentially professors).

### Sprint 1 (Sept 22 – Oct 1): Environment & Baseline Setup

**User stories:**

As a ChRIS operator, I want to open a single ChRIS instance locally.

As a researcher, I want to be able to train and run inference on the MNIST handwitten digits dataset locally in a standard Python environment. 

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

As clinical researcher or developer, I need at least 2 nodes to be reliably sharing weights along with proper central aggregation computations at the aggregator server.
 
**Deliverables:**

* TLS-secured (encrypted) communication between nodes and the aggregator (migth be free with OpenFL).
* Federated aggregation working across at least two nodes with central aggregation.
* Deployment scripts to help for multiple ChRIS instances.
* Early draft of documentation that FL setup and ML plugin design.

### Sprint 5 (Nov 12 – Dec 26): Metrics & Monitoring

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

## General comments

Remember that you can always add features at the end of the semester, but you can't go back in time and gain back time you spent on features that you couldn't complete.

** **

For more help on markdown, see
https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

In particular, you can add images like this (clone the repository to see details):

![alt text](https://github.com/BU-NU-CLOUD-SP18/sample-project/raw/master/cloud.png "Hover text")

