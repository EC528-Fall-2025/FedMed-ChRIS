# FedMed Reverse SSH Bastion Setup Guide

### **How to Deploy and Configure the Bastion on AWS for Cross-Machine Federated Learning in ChRIS**

This guides you through exactly how to set up the AWS bastion host used for enabling reverse SSH tunnels between:

* **SuperLink** running inside **miniChRIS**, and
* **Remote SuperNodes** running on entirely different networks (hospitals, universities, laptops, etc.).

This is the *only* networking component required to support **multi-institution federated learning** using the FedMed ChRIS plugins.

**Note**: This document is meant to complement `run_tutorial.md`, which describes how to run the our plugins *inside* a single machine or miniChRIS instance. You must complete this bastion setup before running distributed FL in ChRIS.

# 1. Overview & Motivation:

ChRIS imposes networking limitations:

* SuperLink runs *inside* Docker and cannot expose ports publicly
* SuperNodes in hospitals cannot accept inbound connections
* Only outbound TCP is allowed from remote sites
* NAT, firewalls, and VPNs block incoming federation requests

### **Our solution involves reverse SSH tunneling**

SuperLink creates a **single outbound SSH connection** to a cloud bastion.

Via this SSH session, it exposes:

| Bastion Public Port | Forwards To (inside SuperLink container) | Purpose            |
| ------------------- | ---------------------------------------- | ------------------ |
| **19092**           | `127.0.0.1:9092`                         | Fleet API          |
| **19093**           | `127.0.0.1:9093`                         | Control API        |
| **19091**           | `127.0.0.1:9091`                         | ServerAppIO Stream |

Remote SuperNodes then connect to:

```
tcp://<Elastic-IP>:19092
tcp://<Elastic-IP>:19093
tcp://<Elastic-IP>:19091
```

This allows global FL orchestration without exposing ChRIS, Docker, or hospital firewalls.


# 2. Launch the AWS Bastion EC2 Instance

Create a new EC2 instance with the following configuration:

| Setting               | Value                  |
| --------------------- | ---------------------- |
| AMI                   | Ubuntu 22.04 LTS       |
| Instance type         | t2.micro or larger     |
| Network               | Public subnet          |
| Auto-assign Public IP | Enabled                |
| Storage               | Default                |
| Security Group        | NEW (see next section) |

SSH into the instance using AWS’ default keypair:

```bash
ssh -i my_aws_key.pem ubuntu@<public-ip>
```

# 3. Security Group Configuration

Create a Security Group with these inbound rules:

| Port      | Protocol | Source    | Purpose                                  |
| --------- | -------- | --------- | ---------------------------------------- |
| **22**    | TCP      | 0.0.0.0/0 | Allows SuperLink to build the SSH tunnel |
| **19092** | TCP      | 0.0.0.0/0 | SuperNodes → Fleet API                   |
| **19093** | TCP      | 0.0.0.0/0 | SuperNodes → Control API                 |
| **19091** | TCP      | 0.0.0.0/0 | SuperNodes → ServerAppIO                 |

Note: These port forwards reach only the SuperLink container, not ChRIS itself.

# 4. Create the Dedicated “fedmed” User on the Bastion

On the EC2 instance, run:

```bash
sudo adduser fedmed
sudo usermod -aG sudo fedmed
```

Create the `.ssh` directory:

```bash
sudo mkdir -p /home/fedmed/.ssh
sudo chown -R fedmed:fedmed /home/fedmed/.ssh
sudo chmod 700 /home/fedmed/.ssh
```

# 5. Assign Static Elastic IP to the Bastion

### Steps:
1. AWS Console navigate to EC2, then **Elastic IPs**
2. Click "Allocate Elastic IP"
3. Select your region
4. Click ""Allocate""
5. Select the IP, then "Actions", then Associate"
6. Choose your EC2 instance and associate the only private IP shown.

Bastion will now have a static public IP of the form:
```
A.B.C.D
```

This value is used in:

* The SuperLink feed (`known_hosts`)
* The pipeline YAMLs (`superlink-host`)
* Every SuperNode’s configuration

# 6. Generate SSH Credentials (`id_ed25519` and `known_hosts`)

These two files are what CHRIS users will upload into the SuperLink feed.

They facilitate:
* Secure authentication from SuperLink to the bastion
* Protected host verification
* Password free reverse SSH tunnels

Note: This only needs to be done once, after just distribute the files.

## 6.1 Generate the SSH Keypair (on local computer)

Run:

```bash
ssh-keygen -t ed25519 -C "fedmed-bastion" -f id_ed25519
```

This creates:
```bash
id_ed25519        # PRIVATE key (SuperLink uses this)
id_ed25519.pub    # PUBLIC key (installed on the bastion)
```

## 6.2 Install the public key on the bastion

```bash
ssh -i my_aws_key.pem ubuntu@<elastic-ip>
sudo su - fedmed
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

Then, append the generated public key:

```bash
echo "<contents of id_ed25519.pub>" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Now the SuperLink can authenticate over SSH using the private key.

## 6.3 Generate the correct `known_hosts` entry

On local machine:

```bash
ssh-keyscan -t ed25519 <elastic-ip> > known_hosts
```

This records the bastion’s server fingerprint, for example:

```
203.0.113.10 ssh-ed25519 AAAAC3Nza...
```

This ensures secure host authentication/identification.

## 6.4 Test the credential pair

Try logging into the bastion using the new key and known_hosts file:

```bash
ssh -i id_ed25519 -o UserKnownHostsFile=known_hosts fedmed@<elastic-ip>
```

If able to reach a shell without entering a password the setup is correct.

# 7. Configure SSH Daemon for Reverse Tunneling

Open the SSH daemon config:

```bash
sudo vim /etc/ssh/sshd_config
```

Add or confirm the following lines:

```
PasswordAuthentication no
PermitRootLogin no

AllowTcpForwarding yes
GatewayPorts clientspecified
```

These two are essential and required:

* `AllowTcpForwarding yes` — allows reverse SSH tunnels
* `GatewayPorts clientspecified` — allows public binding of forwarded ports

Finally, restart:

```bash
sudo systemctl restart ssh
```

# 8. Verify SSH and Port Reachability

Check that SSH is listening on the bastion:

```bash
ss -lntp | grep ssh
```

Test externally on local machine:

```bash
nc -vz <elastic-ip> 22
nc -vz <elastic-ip> 19092
nc -vz <elastic-ip> 19093
nc -vz <elastic-ip> 19091
```

All ports must be reachable.

# 9. What CHRIS Users Upload to the SuperLink Feed

Users must include these two files in the SuperLink feed:

```
id_ed25519
known_hosts
```

# 10. Full System Verification

When SuperLink is launched in ChRIS, logs must show:

```
remote forward success for listen 0.0.0.0:19092
remote forward success for listen 0.0.0.0:19093
remote forward success for listen 0.0.0.0:19091
```

Then start a SuperNode on any machine in miniChRIS by uploading the `id_ed25519` and `known_hosts` files (refer to `run_tutorial`):

```
[fedmed-pl-supernode] Connected to <elastic-ip>:19092
```

If instead you see the following in the logs:

```
Connection attempt failed, retrying...
```

The bastion is misconfigured.

---

# 11. Security Hardening for Hospitals

Recommended for hospital deployments:

* Restrict inbound ports to known IP ranges in AWS
* Store keys in a dedicated secret manager
* Decommission and rotate SSH keys every week/month
* Create a dedicated IAM role/VPC for the bastion
* Use fail2ban to throttle brute-force login attempts
* Add security to the TCP connections between SuperNodes and the SuperLink (OpenSSL recommended)

# 12. Bastion Deployment Complete

Now have:

* A working AWS bastion with a static Elastic IP
* Correct SSH daemon configuration
* Reverse tunnel compatible credentials
* Verified port reachability
