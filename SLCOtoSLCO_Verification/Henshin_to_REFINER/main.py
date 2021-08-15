import sys
import re
import pyecore.ecore as Ecore
from pyecore.resources import ResourceSet

import refiner
import textualSlco as SLCO
import henshin


EXT = ".henshin"
HENSHIN_URI = "http://www.eclipse.org/emf/2011/Henshin"


SLCO_SM = 'StateMachine'
SLCO_VERTEX = 'Vertex'
SLCO_TRANSITION = 'Transition'
SLCO_DOACTION = 'DoAction'

SLCO_TRANSITION_TARGET = 'http://www.xtext.org/slco/textualslco/TextualSlco#//Transition/target'
SLCO_TRANSITION_SOURCE = 'http://www.xtext.org/slco/textualslco/TextualSlco#//Transition/source'
SLCO_TRANSITION_STATEMENTS = 'http://www.xtext.org/slco/textualslco/TextualSlco#//Transition/statements'

class NotSupportedException(Exception):
	def __init__(self, message):
		# Call the base class constructor with the parameters it needs
		super().__init__(message)


node_to_state_mapping = {}
node_counter = 0


def transition_node_to_LTS_transition(trans_node, node_map):
	global node_to_state_mapping, node_counter
	# A transition node has outgoing edges to source and target Vertex, and statements
	# We assume that each transition has a single action associated with it
	s_node = None
	t_node = None
	a_node = None
	for e in trans_node.outgoing:
		edge_name = e.type._proxy_path
		if edge_name == SLCO_TRANSITION_SOURCE and not s_node:
			s_node = e.target
		elif edge_name == SLCO_TRANSITION_TARGET and not t_node:
			t_node = e.target
		elif edge_name == SLCO_TRANSITION_STATEMENTS and not a_node:
			a_node = e.target
		elif s_node or t_node or a_node:
			raise NotSupportedException(
				"A transition may only have one source state, target state, and statement (a DoAction statement)")

	if not (s_node and t_node and a_node) or (a_node and a_node.type.name != SLCO_DOACTION):
		raise NotSupportedException(
			"A transition must have one source state, target state, and statement (a DoAction statement)")

	if len(a_node.attributes.items) != 1:
		raise NotSupportedException("The DoAction statement must have (only) the name attribute set")

	s_node = node_map.get(s_node, s_node)
	t_node = node_map.get(t_node, t_node)

	s = node_to_state_mapping.get(s_node, node_counter)
	if s == node_counter:
		node_to_state_mapping[s_node] = s
		node_counter += 1

	t = node_to_state_mapping.get(t_node, node_counter)
	if t == node_counter:
		node_to_state_mapping[t_node] = t
		node_counter += 1

	a = a_node.attributes.items[0].value.strip('"')

	return s, a, t


def hensin_graph_to_lts(g, node_map):
	# Resolve the types of the nodes
	for n in g.nodes:
		n.type.force_resolve()

	sm_nodes = [n for n in g.nodes if n.type.name == SLCO_SM]
	if len(sm_nodes) > 1:
		raise NotSupportedException("A rule may define a transformation over only one State Machine")
	elif len(sm_nodes) == 0:
		return None
	sm_node = sm_nodes[0]
	# Select transitions from transition nodes associated with the state machine
	transitions = {transition_node_to_LTS_transition(e.target, node_map) for e in sm_node.outgoing.items
					if e.target.type.name == SLCO_TRANSITION}

	lts = refiner.LTS(transitions)
	return lts


def henshin_rule_to_refiner_rule(rule):
	ref_laws_L = []
	ref_laws_R = []

	# create a mapping of the vertices in LHS to RHS
	vertex_mapping = {x.origin: x.image for x in rule.mappings.items if x.image.type.name == SLCO_VERTEX}

	R = hensin_graph_to_lts(rule.rhs, {}) # no mapping required
	L = hensin_graph_to_lts(rule.lhs, vertex_mapping) # mapping from lhs nodes to rhs nodes

	if not (L and R):
		return None, None, None

	if not L:
		L = refiner.LTS(set())
	elif not R:
		R = refiner.LTS(set())

	ref_rule = refiner.Rule(L, R)
	return ref_rule, ref_laws_L, ref_laws_R


def henshin_to_rulesystem(hroot):
	rules = []
	for unit in hroot.units:
		# NOTE: it seems the commented conditions are not read properly by pyecore
		# if not unit.activated:
		# 	continue
		if not isinstance(unit, henshin.Rule):
			raise NotSupportedException("Only 'Rule' elements are allowed in the transformation specification")
		# if not unit.checkDangling:
		# 	raise ElementNotSupportedException("Only 'Rule' elements with 'checkDangling=true' are allowed")
		# if not unit.injectiveMatching:
		# 	raise ElementNotSupportedException("Only 'Rule' elements with 'injectiveMatching=true' are allowed")
		rules.append(unit)

	rule_system = refiner.RuleSystem()
	for rule in rules:
		ref_rule, ref_laws_L, ref_laws_R = henshin_rule_to_refiner_rule(rule)
		if ref_rule:
			rule_system.rules.append(ref_rule)
			rule_system.sync_left.extend(ref_laws_L)
			rule_system.sync_right.extend(ref_laws_R)

	return rule_system


RE_PRE_PROC = '".*TextualSlco.ecore'


def preprocess_input_file(input_file):
	f = open(input_file, 'r')
	contents = f.read()
	f.close()

	new_contents = re.sub(RE_PRE_PROC, '"%s' % SLCO.nsURI, contents)
	new_input_file = "temp.henshin"
	f = open(new_input_file, 'w')
	f.write(new_contents)
	f.close

	return new_input_file

def main(args):
	"""The main function"""

	output_filename = ""
	if len(args) > 0:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 main <henshin_file>")
			print("")
			print("Extract SLCO/REFINER semantics from henchin file.")
			sys.exit(0)

	input_file = args[0]
	if not input_file.endswith(EXT):
		input_file = input_file + EXT

	output_file = input_file[:-len(EXT)] + ".rs"
	print("Input: %s" % input_file)
	print("Output: %s" % output_file)

	new_input_file = preprocess_input_file(input_file)

	rset = ResourceSet()
	rset.metamodel_registry[Ecore.nsURI] = Ecore
	rset.metamodel_registry[henshin.nsURI] = henshin
	rset.metamodel_registry[SLCO.nsURI] = SLCO

	input_model = rset.get_resource(new_input_file)
	root_module = input_model.contents[0]
	rs = henshin_to_rulesystem(root_module)
	f = open(output_file, 'w')
	f.write(rs.to_string())
	f.close()


if __name__ == '__main__':
	args = []
	for i in range(1, len(sys.argv)):
		args.append(sys.argv[i])
	main(args)
