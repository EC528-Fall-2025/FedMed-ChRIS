#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

from chris_plugin import chris_plugin, PathMapper
import subprocess

__version__ = '1.0.0'

DISPLAY_TITLE = r"""
PLugin for Flower superlink
"""


parser = ArgumentParser(description='!!!CHANGE ME!!! An example ChRIS plugin which '
                                    'counts the number of occurrences of a given '
                                    'word in text files.',
                        formatter_class=ArgumentDefaultsHelpFormatter)
#parser.add_argument('-w', '--word', required=True, type=str,
#                    help='word to count')
#parser.add_argument('-p', '--pattern', default='**/*.txt', type=str,
#                    help='input file filter glob')
#parser.add_argument('-V', '--version', action='version',
#                    version=f'%(prog)s {__version__}')


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='FedMed Testing',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='100Mi',    # supported units: Mi, Gi
    min_cpu_limit='1000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """
    subprocess.run('ls chris-flwr-test; flower-superlink --insecure', shell=True, capture_output=True, text=True)


if __name__ == '__main__':
    main()
