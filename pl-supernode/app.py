#!/usr/bin/env python
  2 from pathlib import Path
  3 from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
  4 from chris_plugin import chris_plugin
  5 import subprocess
  6 import os
  7
  8 __version__ = '1.0.0'
  9
 10 DISPLAY_TITLE = r"""
 11 ChRIS Flower SuperNode Plugin
 12 """
 13
 14 parser = ArgumentParser(
 15     description="Run a Flower SuperNode connected to a SuperLink",
 16     formatter_class=ArgumentDefaultsHelpFormatter,
 17 )
 18
 19 # SuperNode connection/config
 20 parser.add_argument('--superlink-host', default='127.0.0.1',
 21                     help='SuperLink host (service name/hostname/IP)')
 22 parser.add_argument('--fleet-port', default='9092',
 23                     help='SuperLink Fleet gRPC port')
 24 parser.add_argument('--clientapp-addr', default='127.0.0.1:9094',
 25                     help='ClientAppIo API bind address for this SuperNode (unique per instance)')
 26 parser.add_argument('--partition-id', type=int, required=True,
 27                     help='Partition ID for this site/node')
 28 parser.add_argument('--num-partitions', type=int, required=True,
 29                     help='Total number of participating partitions/sites')
 30 parser.add_argument('--insecure', action='store_true',
 31                     help='Use insecure mode (DEV ONLY). For prod, configure TLS instead.')
 32
 33 @chris_plugin(
 34     parser=parser,
 35     title='Flower SuperNode for ChRIS',
 36     category='',                 # ref. https://chrisstore.co/plugins
 37     min_memory_limit='100Mi',
 38     min_cpu_limit='1000m',
 39     min_gpu_limit=0
 40 )
 41 def main(options: Namespace, inputdir: Path, outputdir: Path):
 42     """
 43     Launch a Flower SuperNode and stream its logs to stdout so ChRIS captures them.
 44     """
 45     print(DISPLAY_TITLE, flush=True)
 46
 47     cmd = [
 48         "flower-supernode",
 49         "--superlink", f"{options.superlink_host}:{options.fleet_port}",
 50         "--clientappio-api-address", options.clientapp_addr,
 51         "--node-config", f"partition-id={options.partition_id} num-partitions={options.num_partitions}",
 52     ]
 53     if options.insecure:
 54         cmd.insert(1, "--insecure")
 55
 56     # Environment (optional, but handy if your ClientApp reads these)
 57     env = os.environ.copy()
 58     env.update({
 59         "SUPERLINK_HOST": str(options.superlink_host),
 60         "FLEET_PORT": str(options.fleet_port),
 61         "CLIENTAPP_ADDR": str(options.clientapp_addr),
 62         "PARTITION_ID": str(options.partition_id),
 63         "NUM_PARTITIONS": str(options.num_partitions),
 64         "INSECURE": "true" if options.insecure else "false",
 65     })
 66
 67     print(f"[SuperNode] starting: {' '.join(cmd)}", flush=True)
 68     # Run and stream logs; returns when the SuperNode exits or the container is stopped
 69     subprocess.run(cmd, env=env, check=True)
 70
 71
 72 if __name__ == '__main__':
 73     main()
 74
