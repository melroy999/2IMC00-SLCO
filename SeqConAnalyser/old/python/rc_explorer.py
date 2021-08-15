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
from lts import *
from VarDependencyGraph import VarDependencyGraph
from lts.MHSMurakami import HyperGraph
import time

from help_on_error_argument_parser import HelpOnErrorArgumentParser

# matching of actions
GROUP_LABEL             = 'label'
GROUP_SRC_OBJECT        = 'src_obj'
GROUP_TGT_OBJECT        = 'tgt_obj'
GROUP_SRC_READ_VARS     = 'src_reads'
GROUP_TGT_READ_VARS     = 'tgt_reads'
GROUP_SRC_WRITE_VARS    = 'src_writes'
GROUP_TGT_WRITE_VARS    = 'tgt_writes'
GROUP_SRC_STATE_MACHINE = 'src_sm'
GROUP_TGT_STATE_MACHINE = 'tgt_sm'
GROUP_SRC_PORT          = 'src_port'
GROUP_TGT_PORT          = 'tgt_port'
GROUP_MSG               = 'msg'

test_regex = None#"rw_globalObject(Frog,{var_a(10)},{var_a(9),var_a(10)})"
#(?P<GROUP_LABEL>rw|send|receive|peek|comm)_(?P<GROUP_SRC_OBJECT>[a-zA-Z0-9]+)(_(?P<GROUP_SRC_PORT>[a-zA-Z0-9]+))?(_(?P<GROUP_TGT_OBJECT>[a-zA-Z0-9]+)_(?P<GROUP_TGT_PORT>[a-zA-Z0-9]+))?[(](?P<GROUP_SRC_STATE_MACHINE>\w+),[{](?P<GROUP_SRC_READ_VARS>(\w+([(]\d[)])?)?([,]\w+([(]\d[)])?)*)[}],[{](?P<GROUP_SRC_WRITE_VARS>(\w+([(]\d[)])?)?([,]\w+([(]\d[)])?)*)[}](,(?P<GROUP_TGT_STATE_MACHINE>\w+),[{](?P<GROUP_TGT_READ_VARS>(\w+([(]\d[)])?)?([,]\w+([(]\d[)])?)*)[}],[{](?P<GROUP_TGT_WRITE_VARS>(\w+([(]\d[)])?)?([,]\w+([(]\d[)])?)*)[}],(?P<GROUP_MSG>\w+).*[)])?
action_matcher = re.compile('(?P<'+GROUP_LABEL+'>rw|send|receive|peek|comm)'
							'_(?P<'+GROUP_SRC_OBJECT+'>[a-zA-Z0-9]+)'

							'(_(?P<'+GROUP_SRC_PORT+'>[a-zA-Z0-9]+))?'

							'(_(?P<'+GROUP_TGT_OBJECT+'>[a-zA-Z0-9]+)_(?P<'+GROUP_TGT_PORT+'>[a-zA-Z0-9]+))?'

							'[(](?P<'+GROUP_SRC_STATE_MACHINE+'>\w+),'
							'[\{\[](?P<'+GROUP_SRC_READ_VARS+'>(\w+([(]\d+[)])?)?([,]\w+([(]\d+[)])?)*)[\]\}],'
							'[\{\[](?P<'+GROUP_SRC_WRITE_VARS+'>(\w+([(]\d+[)])?)?([,]\w+([(]\d+[)])?)*)[\]\}]'

							'(,(?P<'+GROUP_TGT_STATE_MACHINE+'>\w+),'
							'[\{\[](?P<'+GROUP_TGT_READ_VARS+'>(\w+([(]\d+[)])?)?([,]\w+([(]\d+[)])?)*)[\]\}],'
							'[\{\[](?P<'+GROUP_TGT_WRITE_VARS+'>(\w+([(]\d+[)])?)?([,]\w+([(]\d+[)])?)*)[\]\}],'
							'(?P<'+GROUP_MSG+'>\w+).*[)])?')

class ActionSyntaxException(Exception):
	def __init__(self, value):
		self.parameter = value
	def __str__(self):
		return repr(self.parameter)

def RCE_get_race_conditions_from_file(path):
	dep_ltss = initial_locks = None
	if (path.endswith('.aut')):
		lts = LTS.create(path)
		contains_tau, lts = LTS_remove_peek(lts)
		if contains_tau:
			lts = lts.minimise(LTS.Equivalence.BRANCHING_BISIM)

		# calculate dependency LTSs
		start = time.perf_counter()
		dep_ltss, initial_locks = get_dependency_ltss(lts)
		time_dep_ltss = time.perf_counter() - start
		logging.info("initial locks: %s" % initial_locks)
		logging.info("#locks: %s" % len(initial_locks))
		logging.info("Calculating dependency LTSs took " + str(time_dep_ltss))
	else:
		start = time.perf_counter()
		in_type, locks, competing_statements = read_competing_statements(path)
		time_read_data = time.perf_counter() - start
		print("Parsing input took " + str(time_read_data))
		
		if (in_type == InputType.LOCK_ALL):
			return {"var_all_objects'lock_all"}
		
		start = time.perf_counter()
		dep_ltss, initial_locks = get_dependency_ltss_new(in_type, locks, competing_statements)
		time_dep_ltss = time.perf_counter() - start
		logging.info("initial locks: %s" % initial_locks)
		logging.info("Calculating dependency LTSs took " + str(time_dep_ltss))

	cnt = 0
	for lts in dep_ltss.values():
		cnt += 1
		dep_lts = lts.dependency_graph
		logging.info("Dependency Graph : %i", cnt)
		logging.info("# Vertices : %i", len(dep_lts.keys()))
		logging.info("# Edges    : %i", sum(map(lambda edges: len(edges), dep_lts.values())))

	# Quickly find un-optimal locks
	dep_ltss_copy = {frozenset(key): dep_lts.get_copy() for key, dep_lts in dep_ltss.items()}
	locked = {x for x in initial_locks}
	start = time.perf_counter()
	quick_locks = get_race_conditions(dep_ltss_copy, locked)
	time_quick = time.perf_counter() - start
	logging.info("Quick locks: %s" % quick_locks)
	logging.info("#locks: %s" % len(quick_locks))
	logging.info("Quick analysis took " + str(time_quick))

	# Find optimal locking
	sets = {frozenset({x}) for x in initial_locks}
	start = time.perf_counter()
	cycle_sets = get_cycle_sets(dep_ltss)
	minimal_locks = get_minimal_hitting_set(sets | cycle_sets)
	time_opt = time.perf_counter() - start
	logging.info("cycle_sets: %s" % cycle_sets)
	logging.info("MHS locks: %s" % minimal_locks)
	logging.info("#locks: %s" % len(minimal_locks))
	logging.info("Optimal analysis took " + str(time_opt))
	
	return quick_locks, minimal_locks
	

def get_dependency_ltss_new(in_type, locks, competing_statements):
	dep_ltss = dict()
	for statements in competing_statements.values():
		signature = set(statements)  # a set of all outgoing edges that are relevant for race condition detection
		frozen_signature = frozenset(signature)
		if frozen_signature in dep_ltss:
			continue
			
		rw_list = []
		for statement in statements:
			value = statement.split(', ')
			#object = value[0]
			state_machine = value[1]
			reads = string_to_list(value[2])
			writes = string_to_list(value[3])
			rw_list.append((statement, state_machine, reads, writes))
	
		locked = locks
		do_checks = in_type == InputType.RW
		dep_ltss[frozen_signature] = VarDependencyGraph(rw_list, locked, do_checks)
		if do_checks:
			# get locked sets (variable sets that require being locking together)
			for dep_lts in dep_ltss.values():
				locked |= (dep_lts.get_locked_sets())
			
			# remove (to be) locked variables from the graphs
			for dep_lts in dep_ltss.values():
				dep_lts.remove_locked_labels_from_transitions(locked)

	return (dep_ltss, locked)

# precondition: peeks are removed from the LTS
def get_dependency_ltss(lts):
	### Begin switch "append_rw" ###
	# absence of ports are ignored as they do not contribute to the dependency ltss
	# a postfix is added to the action label to ensure two nodes for the comm action
	# and consistent naming for the nodes
	def append_standard():
		rw_list.append((a + src_obj + src_sm, src_sm, src_read_vars, src_write_vars))
	
	def append_receive():
		if (tgt_pre is None) or (tgt_sm is None) or (tgt_read is None) or (tgt_write is None):
			raise ActionSyntaxException(
				'action label \"%s\" is missing value for target state_machine, read, and/or write.' % a)
		rw_list.append((a + tgt_pre + tgt_sm, tgt_sm, tgt_read_vars, tgt_write_vars))
	
	def append_comm():
		append_standard()
		append_receive()
	
	append_rw = {'rw': append_standard, 'send': append_standard, 'receive': append_receive,
		'peek': append_standard, 'comm': append_comm}
	### End switch ###
	
	transitions = lts.transition_dict
	dep_ltss = dict()
	for src, trans in transitions.items():
		rw_list = []
		signature = set()  # a set of all outgoing edges that are relevant for race condition detection
		# get read/write actions
		for a, tgts in trans.items():
			if a.startswith('tau'):
				continue
			
			match_result = action_matcher.match(a)
			if not match_result:
				raise ActionSyntaxException('action label \"%s\" does not match the required format.' % a)
				
			label = match_result.group(GROUP_LABEL)
			
			src_obj  = match_result.group(GROUP_SRC_OBJECT)
			src_sm   = match_result.group(GROUP_SRC_STATE_MACHINE)
			src_port = match_result.group(GROUP_SRC_PORT)
			tgt_obj  = match_result.group(GROUP_TGT_OBJECT)
			tgt_sm   = match_result.group(GROUP_TGT_STATE_MACHINE)
			tgt_port = match_result.group(GROUP_TGT_PORT)
			
			src_read  = match_result.group(GROUP_SRC_READ_VARS)
			src_write = match_result.group(GROUP_SRC_WRITE_VARS)
			tgt_read  = match_result.group(GROUP_TGT_READ_VARS)
			tgt_write = match_result.group(GROUP_TGT_WRITE_VARS)
			
			#msg = match_result.group(GROUP_MSG)
			
			src_read_vars  = set(src_read.split(',')) - {''}
			src_write_vars = set(src_write.split(',')) - {''}
			tgt_read_vars = set(tgt_read.split(',')) - {''} if tgt_read else set()
			tgt_write_vars = set(tgt_write.split(',')) - {''} if tgt_write else set()
			
			# add object owning the variable
			tgt_pre = src_obj if label == 'receive' else tgt_obj
			src_read_vars  = {src_obj + '.' + var for var in src_read_vars}
			src_write_vars = {src_obj + '.' + var for var in src_write_vars}
			tgt_read_vars  = {tgt_pre + '.' + var for var in tgt_read_vars}
			tgt_write_vars = {tgt_pre + '.' + var for var in tgt_write_vars}
			
			append_rw[label]() # switch on label to push to rw_list
			signature.add(a)
		
		# analyse read/write performed by src state
		frozen_signature = frozenset(signature)
		locked = set()
		if frozen_signature not in dep_ltss:
			dep_ltss[frozen_signature] = VarDependencyGraph(rw_list, locked, True)
		
		# get locked sets (variable sets that require being locking together)
		for dep_lts in dep_ltss.values():
			locked |= (dep_lts.get_locked_sets())
		
		# remove (to be) locked variables from the graphs
		for dep_lts in dep_ltss.values():
			dep_lts.remove_locked_labels_from_transitions(locked)
	return (dep_ltss, locked)


def get_race_conditions(dep_ltss, pre_locked):
	locked = pre_locked
	for dep_lts in dep_ltss.values():
		# locked is both used as set of existing locks and as accumulator of locks
		dep_lts.get_locked(locked)
	return locked


def get_cycle_sets(dep_ltss):
	cycles = set()
	for dep_lts in dep_ltss.values():
		# Find the cycles in the dependency LTS.
		found_cycles = dep_lts.find_cycles()
		logging.info("cycles: %s" % found_cycles)

		# Construct the transition label sets from the cycles.
		for cycle in found_cycles:
			transition_cycle = list()

			for i in range(1, len(cycle)):
				transition_labels = dep_lts.get_labels_of_transition(cycle[i-1], cycle[i])

				if len(transition_labels) == 0:
					logging.warning("No label found on transition from " + cycle[i-1] + "' to '" + cycle[i] + "'")
				elif len(transition_labels) > 1:
					logging.warning(
						"Multiple labels found on transition from " + cycle[i - 1] + "' to '" + cycle[i] + "'")

				transition_cycle += transition_labels

			cycles.add(frozenset(transition_cycle))
	return cycles


def get_minimal_hitting_set(sets):
	if not sets:
		return set()

	# Run the minimal hitting set algorithm.
	graph = HyperGraph(sets)
	mhss = graph.mmcs()

	# No minimal hitting set has been found.
	if not mhss:
		return None

	logging.info("MHSs: %s" % mhss)

	# Select the minimal hitting set of the smallest length.
	smallest = mhss[0]
	for mhs in mhss:
		if len(mhs) < len(smallest):
			smallest = mhs

	return smallest
	
	
def LTS_remove_peek(my_lts):
	temp_act = 'temp_tau'
	contains_tau = my_lts.rename_action_labels({'tau': temp_act})
	hidden = my_lts.hide_action_labels({'peek_[a-zA-Z0-9]+_[a-zA-Z0-9]+[(].*'})
	if hidden:
		my_lts = my_lts.minimise(LTS.Equivalence.WEAK_BISIM)
	my_lts.rename_action_labels({temp_act: 'tau'})
	return (contains_tau, my_lts)

GROUP_ACTION            = 'act'
GROUP_STATE             = 'state'

TYPE_RW                 = 'rw'
TYPE_LOCK               = 'lock'
TYPE_REPORT             = 'report'
TYPE_LOCK_ALL           = 'lock_all'

detected_action_matcher = \
	re.compile(
		"Detected action '(?P<"+GROUP_LABEL+">"+ TYPE_RW + "|" + TYPE_LOCK_ALL + "|" + TYPE_REPORT + "|" + TYPE_LOCK + ")" +
		"(?P<"+GROUP_ACTION+">.*)' \(state index (?P<"+GROUP_STATE+">[0-9]+)\)"
	)

class InputType:
	NONE = 0
	RW = 1
	LOCK_REPORT = 2
	LOCK_ALL = 3
	
class StatementType:
	REPORT = 0
	RW = 1
	LOCK = 2
	LOCK_ALL = 3


report_remove_strings = {"AP'(" : "", "]))" : "]"}
report_filter = re.compile('|'.join(re.escape(key) for key in report_remove_strings.keys()))
def read_competing_statements(path):
	f = open(path, 'r')
	lines = f.readlines()
	f.close()
	
	locks = set()
	competing_statements = {}
	input_type = InputType.NONE
	for line in lines:
		m = detected_action_matcher.match(line)
		if not m:
			raise ActionSyntaxException('Line \"%s\" does not match the required format.' % line)
		
		label = m.group(GROUP_LABEL)
		new_input_type = InputType.NONE
		if label == TYPE_RW:
			statement_type = StatementType.RW
			new_input_type = InputType.RW
		elif label == TYPE_REPORT:
			statement_type = StatementType.REPORT
			new_input_type = InputType.LOCK_REPORT
		elif label == TYPE_LOCK:
			statement_type = StatementType.LOCK
			new_input_type = InputType.LOCK_REPORT
		else:
			return (InputType.LOCK_ALL, set(), {})
		
		if input_type != 0 and input_type != new_input_type:
			raise ActionSyntaxException('Combination of types %s and %s is used' % (input_type, new_input_type))
		
		input_type = new_input_type
		
		action = m.group(GROUP_ACTION)
		state  = m.group(GROUP_STATE)
		
		action = action[1:-1]
		
		if statement_type == StatementType.LOCK:
			locks |= {x for x in string_to_list(action)}
		else:
			statements = competing_statements.get(state, [])
			if statement_type == StatementType.REPORT:
				stripped_action = report_filter.sub(lambda x: report_remove_strings[x.group()], action)
				statement_list = stripped_action.split(", OA'(")
				if len(statement_list) > 0:
					statement_list[0] = statement_list[0][5:]
					statement_list[-1] = statement_list[-1][:-1]
				statements += statement_list
			else:
				statements.append(action)
			competing_statements[state] = statements
	return (input_type, locks, competing_statements)
	
splitter = re.compile('[^,]*\([^)]*\)|[^,]+')
def string_to_list(txt):
	return splitter.findall(txt[1:-1])

def main():
	if test_regex:
		match_result = action_matcher.match(test_regex)
		if match_result:
			print("match successful!")
		else:
			print("match failed!")
		exit()

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
		description='Race Condition Explorer: finds race conditions in state spaces in aut format, where labels are formatted as rw_Object(Read set, Write set)',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument("INPUT_LTS", type=str, help="LTS file in aut format")
	parser.add_argument("-o", "--out", type=str, default="INPUT_LTS", help="Output file name")
	parser.add_argument("-q", "--quiet", action='store_true', help="Quiet mode, print no messages to screen")
	parser.add_argument("-m", "--mute", action='store_true', help="Mute mode, no messages are printed or logged")
	
	args = vars(parser.parse_args())

	# set options
	if args['quiet']:
		console_handler.setLevel(logging.CRITICAL + 1)

	if args['mute']:
		console_handler.setLevel(logging.CRITICAL + 1)
		file_handler.setLevel(logging.CRITICAL + 1)
	
	lts_path = args['INPUT_LTS']
	#if not lts_path.endswith('.aut'):
	#	lts_path = lts_path + ".aut"
	
	out_path = args['out']
	if out_path == 'INPUT_LTS':
		out_path = lts_path
	
	logging.info('Starting Race Condition Explorer')
	logging.info('Input LTS  : %s', lts_path)
	logging.info('Output File: %s', out_path + '_[quick/minimal].rc')
	
	quick_locks, minial_locks = RCE_get_race_conditions_from_file(lts_path)
	write_locks(quick_locks, out_path + '_quick.rc')
	write_locks(minial_locks, out_path+ '_minimal.rc')
	
	logging.info('Finished, output written to %s', out_path + '_[quick/minimal].rc')
	logging.shutdown()


def write_locks(locks, out_path):
	object_locks = {}
	for lock in locks:
		obj, _, var = lock.partition("'")
		obj = obj[4:]
		lock_list = object_locks.get(obj, [])
		lock_list.append(var)
		object_locks[obj] = lock_list
	f = open(out_path, 'w')
	for obj, locks in object_locks.items():
		f.write(obj + ":\n")
		for lock in locks:
			f.write("\t%s\n" % lock)
	f.close()

if __name__ == '__main__':
	main()
