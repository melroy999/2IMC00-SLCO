# Race Condition Explorer tool, v0.1, 2017
#
# Update History:
#

# Start of program

import argparse
import logging
import logging.config
import re
import sys
from os.path import dirname, join
from lts import *
from VarDependencyGraph import VarDependencyGraph
from textx.metamodel import metamodel_from_file

from help_on_error_argument_parser import HelpOnErrorArgumentParser

this_folder = dirname(__file__)
slco_mm = metamodel_from_file(join(this_folder,'slco2.tx'))

def main():
	# setup logging
	file_handler    = logging.FileHandler(filename='rc_explorer.log', mode='w')
	console_handler = logging.StreamHandler(stream=sys.stdout)
	file_handler.setLevel(logging.INFO)
	console_handler.setLevel(logging.DEBUG)
	logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
						datefmt='%Y-%m-%d %H:%M:%S',
						level=logging.DEBUG,
						handlers=[file_handler, console_handler])
	
	# parse arguments
	parser = HelpOnErrorArgumentParser(
		description='Core-SLCO to Atomicity-Level-SLCO transformation',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument("INPUT_MODEL", type=str, help="SLCO input model")
	parser.add_argument("INPUT_SUGGESTIONS", type=str, help="Suggestions file generated by mCRL2 analysis")
	parser.add_argument("-o", "--out", type=str, default="INPUT_MODEL", help="Output slco model")
	parser.add_argument("-q", "--quiet", action='store_true', help="Quiet mode, print no messages to screen")
	parser.add_argument("-m", "--mute", action='store_true', help="Mute mode, no messages are printed or logged")
	
	args = vars(parser.parse_args())

	# set options
	if args['quiet']:
		console_handler.setLevel(logging.CRITICAL + 1)

	if args['mute']:
		console_handler.setLevel(logging.CRITICAL + 1)
		file_handler.setLevel(logging.CRITICAL + 1)
	
	slco_path = args['INPUT_MODEL']
	sugg_path = args['INPUT_SUGGESTIONS']
	
	out_path = args['out']
	if out_path == 'INPUT_MODEL':
		out_path = slco_path[:-5] + '_AL.slco'
	elif not out_path.endswith('.slco'):
		out_path = out_path + '.slco'
	
	logging.info('Starting SLCO Atomicity-free code generator')
	logging.info('Input model       : %s', slco_path)
	logging.info('Input suggestions : %s', sugg_path)
	logging.info('Output File       : %s', out_path)

	# read model
	model = slco_mm.model_from_file(slco_path)
	# translate
	out_model = translate(model, )

	outFile = open(out_path, 'w')
	outFile.write(out_model)
	outFile.close()
	
	logging.info('Finished, output written to %s', out_path + '_[quick/minimal].rc')
	logging.shutdown()


if __name__ == '__main__':
	main()
