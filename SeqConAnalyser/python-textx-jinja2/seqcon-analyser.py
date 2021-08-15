import sys
from os import mkdir
from os.path import exists, dirname, basename, join, split
from copy import deepcopy
import itertools
import glob
from enum import Enum
import re, traceback
import time

this_folder = dirname(__file__)

# import libraries
sys.path.append(join(this_folder,'../../libraries'))
from slcolib import *
from SCCTarjan import identifySCCs_lower_bound, identifySCCs

class MemoryModel(Enum):
	SC = 0
	TSO = 1
	PSO = 2
	ARM = 3

# Characteristics of memory model
RW_order = True
WR_order = False
RR_order = True
WW_order = True
StoreBuffering = True
WriteAtomicity = True

# purely apply static analysis
pure_static_analysis = False

memory_model = MemoryModel.TSO
model_statespace = {}
modelname = ""
model = []
DG = {}
DG_P = {}
DG_C = {}

# Set of global variables per object
globalvars = {}
# dictionary to look up state machine owning a given statement
statemachine = {}
# dictionary to look up class owning a given state machine
smclass = {}
# per class / statemachine, give a set of local variables
smlocalvars = {}
# per class / statemachine, give a dictionary, making the scope of variables explicit
scopedvars = {}
# dictionary providing statement object, given the ID of its transition
STID_to_object = {}

# Access patterns of statements
accesspattern = {}
# Transitive closure of access order of statements
access_smaller_than = {}
access_bigger_than = {}

# dictionary providing info on conflicting accesses between statements due to speculative reading
#DG_C_spec_conflicts = {}

# structures providing info on the structure of the SLCO model
statements_objects = {}
statements_IDs = {}
statements_accesses = {}
SMowner = {}
# dictionary providing a set of variables accessed to compute the index of a given array access
index_accesses = {}

# structures for Johnson's algorithm
blocked = {}
B = {}

#structure containing the cycles
critical_cycles = {}
# structure to count the number of unsafe P traces in a critical cycle per statement
unsafe_cyclic_P_traces = {}

# list of traces (action, action, instruction-level trace triples) that need to receive a fence somewhere between their statements
statement_fence_requirements = []
# statement level and access level synch suggestions
statement_fence_suggestions = {}
access_fence_suggestions = {}

# count number of transactions and fences needed
transaction_counter = 0
fence_counter_intra = 0
fence_counter_inter = 0

autscanner=re.Scanner([
  (r"des",															lambda scanner,token:("HEADER", token)),
  (r"[0-9]+",														lambda scanner,token:("NAT", token)),
  (r"[\s\(\),]+", None), # None == skip token.
  (r"[A-Za-z0-9!\_\-?.\*\']+\([A-Za-z0-9!\_\-?,.\*\'\(\)\[\]]+\)|[A-Za-z0-9!\_\-?.\*\'\[\]]+|[\"\'][A-Za-z0-9!?\_\-\#\s.,\*\'\(\)\[\]]+[\"\']",	lambda scanner,token:("ACTION", token)),
])

rwscanner=re.Scanner([
  (r"rw\(", None),
  (r"[\"\s\(\),]+", None), # None == skip token.
  (r"[A-Za-z0-9!\_\-?.\*\'\(\)]+", lambda scanner,token:("NAME", token)),
  (r"ST\'[0-9]+", lambda scanner,token:("STID", token)),
  (r"\[", lambda scanner,token:("BEGINLIST", token)),
  (r"\]", lambda scanner,token:("ENDLIST", token)),
])

# SLCO model functions - BEGIN

def printstatement(s):
	"""print the given statement"""
	result = ''
	if s.__class__.__name__ == "Assignment":
		result += s.left.var.name
		if s.left.index != None:
			result += "(" + printstatement(s.left.index) + ")"
		result += " := " + printstatement(s.right)
	elif s.__class__.__name__ == "Composite":
		result += "("
		if s.guard != None:
			result += printstatement(s.guard)
			if len(s.assignments) > 1:
				result += ";"
		for i in range(0,len(s.assignments)):
			result += " " + printstatement(s.assignments[i])
			if i < len(s.assignments)-1:
				result += ";"
		result += ")"
	elif s.__class__.__name__ == "ReceiveSignal":
		result += "receive " + s.signal + '('
		first = True
		for p in s.params:
			if not first:
				result += ', '
			else:
				first = False
			result += printstatement(p)
		if s.guard != None:
			result += " | " + printstatement(s.guard)
		result += ") from " + s.target.name
	elif s.__class__.__name__ == "SendSignal":
		result += "send " + s.signal + '('
		first = True
		for p in s.params:
			if not first:
				result += ', '
			else:
				first = False
			result += printstatement(p)
		result += ") to " + s.target.name
	elif s.__class__.__name__ == "Delay":
		result += "after " + str(s.length) + " ms"
	elif s.__class__.__name__ == "Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2" or s.__class__.__name__ == "ExprPrec1":
		if s.op != '':
			result += printstatement(s.left) + " " + s.op + " " + printstatement(s.right)
		else:
			result += printstatement(s.left)
	elif s.__class__.__name__ == "Primary":
		result += s.sign
		if s.sign == "not":
			result += " "
		if s.value != None:
			newvalue = s.value
			result += str(newvalue).lower()
		elif s.ref != None:
			result += s.ref.ref
			if s.ref.index != None:
				result += "(" + printstatement(s.ref.index) + ")"
		else:
			result += '(' + printstatement(s.body) + ')'
	elif s.__class__.__name__ == "VariableRef":
		result += s.var.name
		if s.index != None:
			result += "(" + printstatement(s.index) + ")"
	return result

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def statement_has_guard(i):
	"""Returns whether i has a guard"""
	global statements_objects, statements_accesses
	# obtain corresponding object
	s = statements_objects[statements_accesses[i][1]]
	if s.__class__.__name__ == "Expression":
		return True
	if s.__class__.__name__ == "Composite":
		if s.guard != None:
			return True
	return False

def operator(s):
	"""Maps SLCO expression operators to mCRL2 operators"""
	if s == '=':
		return '=='
	elif s == '<>':
		return '!='
	elif s == '<=':
		return '<='
	elif s == '>=':
		return '>='
	elif s == '<':
		return '<'
	elif s == '>':
		return '>'
	elif s == '+':
		return '+'
	elif s == '-':
		return '-'
	elif s == 'or':
		return '||'
	elif s == 'xor':
		return 'xor'
	elif s == 'and':
		return '&&'
	elif s == '*':
		return '*'
	elif s == '/':
		return '/'
	elif s == '%':
		return 'mod'
	elif s == '**':
		return 'exp'
	elif s == 'not':
		return '!'
	return ''

def expression(s,stm,c,primmap, owner):
	"""Maps SLCO expression to mCRL2 expression. Statemachine stm and Class c owning the statement are also given. Primmap is a dictionary for rewriting primaries.
	Owner is the class/object owning the statement."""
	global actions, scopedvars
	output = ''
	# special case: s is a variableref. In this case, it is the left-hand side of an assignment, and s refers to an array
	if s.__class__.__name__ == "Variable":
		output = primmap.get(s.name, owner.name + "'" + scopedvars[c.name + "'" + stm.name][s.name])
	if s.__class__.__name__ == "Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2":
		#if s.op != '':
		#	output = '('
		if s.op != '' and s.op != 'xor':
			output += expression(s.left,stm,c,primmap,owner) + " " + operator(s.op) + " " + expression(s.right,stm,c,primmap,owner)
		elif s.op == 'xor':
			output += operator(s.op) + '(' + expression(s.left,stm,c,primmap,owner) + ', ' + expression(s.right,stm,c,primmap,owner) + ')'
		else:
			output += expression(s.left,stm,c,primmap,owner)
	elif s.__class__.__name__ == "ExprPrec1":
		if s.op != '':
			output += operator(s.op) + '(' + expression(s.left,stm,c,primmap,owner) + ', ' + expression(s.right,stm,c,primmap,owner) + ')'
		else:
			output += expression(s.left,stm,c,primmap,owner)
	elif s.__class__.__name__ == "Primary":
		output = operator(s.sign)
		if s.value != None:
			newvalue = s.value
			output += str(newvalue).lower()
		elif s.ref != None:
			if s.ref.ref in actions:
				output += s.ref.ref
			else:
				# is an index to an array given?
				if s.ref.index != None:
					varname = scopedvars[c.name + "'" + stm.name][s.ref.ref]
					varname_with_index = varname + "(" + expression(s.ref.index,stm,c,primmap,owner) + ")"
					if primmap.get(varname_with_index) != None:
						output += primmap.get(varname_with_index)
						return output
				if s.ref.index != None:
					output += "get'("
				output += primmap.get(scopedvars[c.name + "'" + stm.name][s.ref.ref], owner.name + "'" + scopedvars[c.name + "'" + stm.name][s.ref.ref])
				if s.ref.index != None:
					output += ", " + expression(s.ref.index,stm,c,primmap,owner) + ")"
		else:
			output += '(' + expression(s.body,stm,c,primmap,owner) + ')'
	return output

def statement_structure_accesspattern(s, o):
	"""Provide the access pattern of the given statement. o is Object owning statement s."""
	global statemachine, smclass

	if s.__class__.__name__ == "Assignment":
		#print(smclass)
		readset = expression_full_varset(s.right,statemachine[s],smclass[statemachine[s]],{},o,False)
		if s.left.index != None:
			readset |= expression_full_varset(s.left.index,statemachine[s],smclass[statemachine[s]],{},o,False)
		writeset = expression_full_varset(s.left,statemachine[s],smclass[statemachine[s]],{},o,True)
		return [tuple([readset, writeset])]
	elif s.__class__.__name__ == "Composite":
		alist = []
		readset = set([])
		if s.guard != None:
			readset |= expression_full_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False)
			alist.append(tuple([expression_full_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False),set([])]))
		vardict = {}
		writeset = set([])
		for st in s.assignments:
			readset2 = expression_full_varset(st.right,statemachine[s],smclass[statemachine[s]],vardict,o,False)
			if st.left.index != None:
				readset2 |= expression_full_varset(st.left.index,statemachine[s],smclass[statemachine[s]],vardict,o,False) - writeset
			writeset2 = expression_full_varset(st.left,statemachine[s],smclass[statemachine[s]],vardict,o,True)
			# update vardict (to correctly handle possible array index use in subsequent assignments)
			newright = expression(st.right,statemachine[s],smclass[statemachine[s]],vardict,o)
			varname = scopedvars[smclass[statemachine[s]].name + "'" + statemachine[s].name][st.left.var.name]
			if st.left.index != None:
				vardict[varname] = "set'(" + expression(st.left.var,statemachine[s],smclass[statemachine[s]],vardict,o) + ", " + expression(st.left.index,statemachine[s],smclass[statemachine[s]],vardict,o) + ", " + newright + ")"
			else:
				vardict[varname] = "(" + newright + ")"
			alist.append(tuple([readset2, writeset2]))
			readset |= readset2
			writeset |= writeset2
		return alist
	elif s.__class__.__name__ == "Delay":
		return []
	elif s.__class__.__name__ == "SendSignal":
		readset = set([])
		for st in s.params:
			readset |= expression_full_varset(st,statemachine[s],smclass[statemachine[s]],{},o,False)
		return tuple([readset, set([])])
	elif s.__class__.__name__ == "ReceiveSignal":
		for st in s.params:
			writeset |= expression_full_varset(st,statemachine[s],smclass[statemachine[s]],{},o,True)
		readset = set([])
		# in SLCO ReceiveSignal, it is not possible to refer to the old value of a variable to which you are reading. Hence, reading AND writing to the same variable cannot occur
		readset = readset - writeset
		return [tuple([readset, writeset])]
	elif s.__class__.__name__ == "Expression":
		if expression_is_actionref(s):
			return tuple([set([]), set([])])
		else:
			readset = expression_full_varset(s,statemachine[s],smclass[statemachine[s]],{},o,False)
			return [tuple([readset, set([])])]

def expression_full_varset(s,stm,c,primmap,owner,ignore_indices):
	"""Produce set of variables including the local ones"""
	return expression_variables(s,stm,c,primmap,owner,ignore_indices,True)

def expression_variables(s,stm,c,primmap,owner,ignore_indices,add_local):
	"""Produces set of variables referenced in given SLCO expression or variableref. Statemachine stm and Class c owning the statement are also given. Primmap is a dictionary for rewriting primaries, which is relevant when array indices are used.
	The owner of the statement is also given. ignore_indices is a Boolean flag indicating whether array indices should be ignored or not.
	add_local is a Boolean flag indicating whether local variables should be included or nor."""
	global actions, scopedvars, smlocalvars
	output = set([])
	if s.__class__.__name__ == "VariableRef":
		# is the variable not local to state machine?

		if add_local or (s.var.name not in smlocalvars.get(c.name + "'" + stm.name,set([]))):
			if s.var.name in smlocalvars.get(c.name + "'" + stm.name,set([])):
				varname = "var_" + owner.name + "'" + stm.name + "'" + s.var.name
			else:
				varname = "var_" + owner.name + "'" + s.var.name
			if s.index != None:
				e = expression(s.index,stm,c,primmap,owner)
				if RepresentsInt(e):
					varname += "(" + e + ")"
				else:
					varname += "(*)"
				if not ignore_indices:
					output |= expression_variables(s.index,stm,c,primmap,owner,ignore_indices,add_local)
			output.add(varname)
	elif s.__class__.__name__ != "Primary":
		output |= expression_variables(s.left,stm,c,primmap,owner,ignore_indices,add_local)
		if s.op != '':
			output |= expression_variables(s.right,stm,c,primmap,owner,ignore_indices,add_local)
	else:
		if s.ref != None:
			if s.ref.ref not in actions:
				# IGNORE STATE MACHINE LOCAL VARS?
				if add_local or (s.ref.ref not in smlocalvars.get(c.name + "'" + stm.name,set([]))):
					if s.ref.ref in smlocalvars.get(c.name + "'" + stm.name,set([])):
						varname = "var_" + owner.name + "'" + stm.name + "'" + s.ref.ref
					else:
						varname = "var_" + owner.name + "'" + s.ref.ref
					if s.ref.index != None:
						e = expression(s.ref.index,stm,c,primmap,owner)
						if RepresentsInt(e):
							varname += "(" + e + ")"
						else:
							varname += "(*)"
						if not ignore_indices:
							output |= expression_variables(s.ref.index,stm,c,primmap,owner,ignore_indices,add_local)
					output.add(varname)
		if s.body != None:
			output |= expression_variables(s.body,stm,c,primmap,owner,ignore_indices,add_local)
	return output

# SLCO model functions - END

def var_is_local(a, i):
	"""Return whether variable accessed by a is local to SM of i or not"""
	global statements_accesses, statements_objects, globalvars
	o = statements_accesses[i][0]
	V = globalvars.get(o, set([]))
	a_splitted = a.split("(")
	if len(a_splitted) > 1:
		a2 = a_splitted[0]
	else:
		a2 = a
	return (a2 not in V)

def peek(stack):
	"""Return but do not pop top element of stack"""
	return stack[len(stack) - 1]

def unblock(u):
	"""Procedure for Johnson's algorithm"""
	global blocked, B

	blocked[u] = False
	Bu = B.get(u, set([]))
	for w in Bu:
		Bup = deepcopy(Bu)
		Bup.remove(w)
		B[u] = Bup
		if blocked[w]:
			unblock(w)

# global variables to iterate over the circuits
circuit_callstack = []
circuit_stackset = set([])
circuit_T = set([])

def init_circuit_search(L, s):
	"""Initialise a new circuit search starting from node s"""
	global circuit_callstack, circuit_stackset, circuit_T, SMowner, statements_accesses, blocked

	# Keep track of visited threads in cycle detection, to enforce condition 1 of Shasha & Snir for critical cycles
	initial_thread = SMowner[statements_accesses[s][1]]
	circuit_T = set([])
	circuit_callstack = []
	outgoing = L.get(s, set([]))
	circuit_callstack.append((s, list(outgoing), False, initial_thread))
	blocked[s] = True
	# Note: stackset should not be required, but blocked does not prevent element duplicates on the stack (check bakery.3.slco)
	circuit_stackset = set([s])
	#print("initial: " + str(initial_thread.name))

def next_circuit(L, s, o):
	"""Procedure for Johnson's algorithm"""
	global blocked, B, statements_accesses, SMowner, circuit_callstack, circuit_stackset

	initial_thread = SMowner[statements_accesses[s][1]]

	while circuit_callstack != []:
		#print(callstack)
		#print(blocked[2])
		v, targets, f, current_thread = peek(circuit_callstack)
		# print(v)
		# print(targets)
		# print(f)
		# print(current_thread)
		move_to_next = False
		#if v == 2:
		#	if len(callstack) > 2 and targets != []:
		#		if targets[len(targets)-1] == 2 and blocked[v] == False:
		#			print(callstack)
		#			print(blocked[v])
		while len(targets) > 0:
			w = targets.pop()
			# update callstack element
			circuit_callstack[len(circuit_callstack)-1] = v, targets, f, current_thread
			#if v == 2 and w == 2 and blocked[w] == False:
			#	print("next2: " + str(callstack))
			#print(w)
			#print(s)
			#print(T)
			# print(blocked)
			if w == s and circuit_T != set([]):
				# unfold trace on stack, and add to list of critical cycles (at least two threads are involved)
				trace = []
				for n, tgt, f2, nt in circuit_callstack:
					trace.append(n)
				# print(callstack)
				#print("adding " + str(trace) + " to list of critical cycles")
				# critlist = critical_cycles.get(o, [])
				# critlist.append(trace)
				# critical_cycles[o] = critlist

				# update callstack element
				circuit_callstack[len(circuit_callstack)-1] = (v, targets, True, current_thread)
				return (o, trace)
			elif not blocked[w] and w not in circuit_stackset:
				# condition for criticality (thread visitation)
				w_thread = SMowner[statements_accesses[w][1]]
				#print("wthread: " + str(w_thread.name))
				#print("current thread: " + str(current_thread.name))
				#print(T)
				if (w_thread not in circuit_T or w_thread == current_thread) and not (circuit_T != set([]) and current_thread == initial_thread and w_thread != initial_thread):
					#print("appending " + str(w))
					# store f value on callstack
					# circuit_callstack[len(circuit_callstack)-1] = (v, targets, f, current_thread)
					if w_thread != current_thread and w_thread != initial_thread:
						circuit_T.add(w_thread)
						current_thread = w_thread
					# put w on the callstack
					woutgoing = L.get(w, set([]))
					circuit_callstack.append((w, list(woutgoing), False, w_thread))
					circuit_stackset.add(w)
					#print("pushing " + str(w))
					#print(callstack)
					blocked[w] = True
					move_to_next = True
					break
		if not move_to_next:
			if f:
				unblock(v)
				# if v == 2:
				# 	if len(callstack) > 1:
				# 		if callstack[1][0] == 2:
		 	# 				print("unblock2: " + str(callstack))
		 	# 				exit(1)
			else:
				# put v in B sets of successors
				for w in L.get(v, set([])):
					Bw = B.get(w, set([]))
					Bw.add(v)
					B[w] = Bw
			circuit_callstack.pop()
			circuit_stackset.remove(v)
			# update predecessor
			if len(circuit_callstack) > 0:
				n, tgt, f2, nt = peek(circuit_callstack)
				if f:
					circuit_callstack[len(circuit_callstack)-1] = (n, tgt, f, nt)
				if nt != current_thread:
					if current_thread in circuit_T:
						circuit_T.remove(current_thread)
	# callstack is empty
	return ''
	#print(critical_cycles)

# variables to detect critical cycles via an iterator
dc_object = ''
dc_DG = {}
dc_s = -1
dc_s_circuits_done = True
dc_max_state = -1
dc_scc_index = 0
dc_minimal_index = 0
dc_SCCs = []

def init_detect_critical_cycles():
	global dc_object, dc_DG, dc_s, dc_max_states, dc_s_circuits_done, dc_scc_index, dc_minimal_index, dc_SCCs
	dc_object = ''
	dc_DG = {}
	dc_s = -1
	dc_s_circuits_done = True
	dc_max_state = -1
	dc_scc_index = 0
	dc_minimal_index = 0
	dc_SCCs = []

def get_next_critical_cycle():
	"""Detect the next conflict cycle in given graph. Based on Johnson's algorithm for the detection of elementary circuits"""
	global blocked, B, DG_P, DG_C, T, model, statements_accesses, SMowner, dc_object, dc_DG, dc_s, dc_max_state, dc_s_circuits_done, dc_scc_index, dc_minimal_index, dc_SCCs

	while True:
		if dc_s_circuits_done:
			dc_s += 1
		# move on to next object?
		if dc_s == dc_max_state+1:
			if dc_object == '':
				dc_object = model.objects[0]
			else:
				for i in range(0, len(model.objects)):
					if dc_object == model.objects[i]:
						if i == len(model.objects)-1:
							return ''
						else:
							dc_object = model.objects[i+1]
							break
			# Combine the C and P relations
			dc_DG = deepcopy(DG_P[dc_object])
			for i, tgts in DG_C[dc_object].items():
				p_tgts = dc_DG.get(i, set([]))
				p_tgts |= DG_C[dc_object][i]
				dc_DG[i] = p_tgts

			states = set(dc_DG.keys())
			for s in dc_DG.keys():
				states |= dc_DG[s]
			dc_max_state = max(states)

			for s in states:
				blocked[s] = False
			dc_s = 0
			dc_s_circuits_done = True

		if dc_s_circuits_done:
			dc_s_circuits_done = False
			# compute subgraph of DG induced by {s, ..., n}
			dc_SCCs = list()
			#print(s)
			#print("call")
			identifySCCs_lower_bound(dc_DG, {}, dc_SCCs, dc_s)
			#print(SCCs)
			# Detect SCC with minimal vertex number
			dc_scc_index = 0
			dc_minimal_vertex = dc_max_state+1
			for i in range(0, len(dc_SCCs)):
				nodes = list((dc_SCCs[i][1].keys()))
				nodes.sort()
				if nodes[0] < dc_minimal_vertex and dc_SCCs[i][1][nodes[0]] != set([]):
					dc_scc_index = i
					dc_minimal_vertex = nodes[0]
			if dc_minimal_vertex < dc_max_state+1:
				dc_s = dc_minimal_vertex
				for t in dc_SCCs[dc_scc_index][1].keys():
					blocked[t] = False
					B[t] = set([])
				init_circuit_search(dc_SCCs[dc_scc_index][1], dc_s)
			else:
				dc_s_circuits_done = True
				dc_s = dc_max_state
		if not dc_s_circuits_done:
			c = next_circuit(dc_SCCs[dc_scc_index][1], dc_s, dc_object)
			if c == '':
				dc_s_circuits_done = True
			else:
				return c

def access_follows(i, a1, a2):
	"""Could a1 follow a2 in i? It is assumed that a2 is in i, but whether a1 is as well needs to be checked"""
	global statements_accesses
	ap = statements_accesses[i][2]
	# 0: no, 1: maybe, 2: yes
	condreads = set([])
	reads = set([])
	writes = set([])
	i = 0
	condreads - set([])
	if statement_has_guard(i):
		condreads = ap[0][0]
		if len(ap) > 1:
			i = 1
	while i < len(ap):
		reads |= ap[0] - writes - condreads
		writes |= ap[1]
		i += 1
	if a1[0] == 'C' and a1[1] in condreads:
		if a2[0] == 'C':
			return 1
		else:
			return 2
	if a1[0] == 'R' and a1[1] in reads:
		if a2[0] == 'C':
			return 0
		elif a2[0] == 'R':
			return 1
		else:
			return 2
	if a1[0] == 'W' and a1[1] in writes:
		if a2[0] != 'W':
			return 0
		else:
			return 1

def P_trace_has_condition_with_access(x, i1, a1, i2, a2, AP):
	"""Returns whether the P-trace from i1 to i2 with access pattern AP has a condition accessing variable x"""
	if x in AP[0]:
		return True
		return True
	if access_has_smaller_than(a2, ('C',x), i2) == 2:
		return True
	return False

def access_is_on_trace(a, a1, i1, a2, i2, tAP):
	"""Return whether access a is somewhere on the trace starting with access a1 (in i1), end with access a2 (in i2), with t access pattern tAP"""
	if a[0] == 'C':
		if a[1] in tAP[0]:
			return True
	if a[0] == 'R':
		if a[1] in tAP[1]:
			return True
	if a[0] == 'W':
		if a[1] in tAP[2]:
			return True
	if access_has_bigger_than(a1, a, i1):
		return True
	if access_has_smaller_than(a2, a, i2):
		return True
	return False

def get_access_classes(s, a, b):
	"""Get the accesses of statement s, in a list of ordered accesses (based on their order inside s), from access a up to access b"""
	global access_smaller_than, accesspattern
	L = []
	done = set([])
	todo1 = []
	todo2 = []
	#print("to obtain: " + str(s))
	#print(a)
	#print(access_smaller_than[s])
	ap = accesspattern[s]
	# print(ap)
	if a != None:
		S = access_smaller_than[s].get(a, set([]))
		done = deepcopy(S)
		done.add(a)
	for a in ap[0]:
		if ('C',a) not in done:
			todo1.append(('C',a))
	for a in ap[1]:
		if ('R',a) not in done:
			todo1.append(('R',a))
	for a in ap[2]:
		if ('W',a) not in done:
			todo1.append(('W',a))
	# print(todo1)
	while todo1 != []:
		#print(todo1)
		#print("done: " + str(done))
		C = (set([]), set([]), set([]))
		done_tmp = set([])
		for a1 in todo1:
			S = access_smaller_than[s].get(a1, set([]))
			#print("smaller than " + str(a1) + ": " + str(S))
			#print(S - done)
			if S - done == set([]):
				if a1[0] == 'C':
					C[0].add(a1[1])
				if a1[0] == 'R':
					C[1].add(a1[1])
				if a1[0] == 'W':
					C[2].add(a1[1])
				done_tmp.add(a1)
			else:
				#print("add " + str(a1) + " to todo2")
				todo2.append(a1)
		todo1 = todo2
		todo2 = []
		done |= done_tmp
		#print(todo1)
		#print(todo2)
		#print(done)
		if C != (set([]), set([]), set([])):
			if b != None:
				if b[0] == 'C':
					if b[1] in C[0]:
						break
				if b[0] == 'R':
					if b[1] in C[1]:
						break
				if b[0] == 'W':
					if b[1] in C[2]:
						break
			L.append((s,C))
	# print(L)
	return L

def get_access_classes_trace(t, a, b):
	"""Get a list of access classes for a trace t"""
	if len(t) == 1:
		L = get_access_classes(t[0], a, b)
	else:
		L = get_access_classes(t[0], a, None)
		for i in range(1, len(t)-1):
			L += get_access_classes(t[i], None, None)
		L += get_access_classes(t[len(t)-1], None, b)
	start = (set([]), set([]), set([]))
	end = (set([]), set([]), set([]))
	if a[0] == 'C':
		start[0].add(a[1])
	if a[0] == 'R':
		start[1].add(a[1])
	if a[0] == 'W':
		start[2].add(a[1])
	if b[0] == 'C':
		end[0].add(b[1])
	if b[0] == 'R':
		end[1].add(b[1])
	if b[0] == 'W':
		end[2].add(b[1])
	L = [(t[0], start)] + L + [(t[len(t)-1], end)]
	return L

def P_trace_is_safe(a1, a2, t):
	"""Returns whether the trace t from statement executing a1 to statement executing a2 via accesses in t is safe or not."""
	# In case a1 and a2 stem from the same statement, the P-trace is by default considered unsafe (to detect atomicity violations)
	if len(t) == 1:
		return False
	# First construct list of access patterns for the involved statements, together with a dictionary to indicate their ordering in t
	C = get_access_classes_trace(t, a1, a2)
	Cnext = {}
	Cprev = {}
	Ccounter = len(C)
	for i in range(0,len(C)-1):
		Cnext[i] = i+1
	for i in range(1,len(C)):
		Cprev[i] = i-1
	# indices for a1, a2, and frontier of reordering
	a1_index = 0
	a2_index = len(C)-1
	frontier_index = 0
	
	# print("to reorder: " + str(C))

	# reordering algorithm
	while frontier_index != None and a1_index < a2_index:
		#print("outer loop: " + str(C))
		# reorder right neighbour of frontier to the left over a1_index
		current = Cnext.get(frontier_index)
		if current == None:
			break
		pred = frontier_index
		stop_inner = False
		while True:
			#print("inner loop: " + str(C))
			# print("reordering " + str(C[pred]) + " " + str(C[current]))
			L = reorder_accesses(C[pred], C[current])
			node_pos = 0
			node_counter = 0
			for i in range(0,4):
				if L[i][1] != (set([]), set([]), set([])):
					node_counter += 1
			newpos = []
			counter = node_counter
			# assign indices to the new nodes
			for i in range(0,4):
				if L[i][1] != (set([]), set([]), set([])):
					if counter == node_counter:
						newpos.append(pred)
					elif counter > 1:
						newpos.append(Ccounter)
						Ccounter += 1
					else:
						newpos.append(current)
					counter -= 1
				else:
					newpos.append(-1)
			# print(newpos)
			previous_index = -1
			for i in range(0,4):
				if L[i][1] != (set([]), set([]), set([])):
					if newpos[i] >= len(C):
						C.append(L[i])
					else:
						C[newpos[i]] = L[i]
					if previous_index != -1:
						Cnext[previous_index] = newpos[i]
						Cprev[newpos[i]] = previous_index
					previous_index = newpos[i]
			# update ai indices
			if a1_index == pred and (newpos[0] != -1 or newpos[3] != -1):
				if newpos[3] != -1:
					a1_index = newpos[3]
				else:
					a1_index = newpos[1]
				stop_inner = True
			if a2_index == current and (newpos[0] != -1 or newpos[3] != -1):
				if newpos[0] != -1:
					a2_index = newpos[0]
				else:
					a2_index = newpos[2]
			# print("result of reorder: ")
			j = 0
			while j != None:
				# print(C[j])
				j = Cnext.get(j)
			if L[0][1] != (set([]), set([]), set([])) and not stop_inner:
				current = pred
				#print(Cprev)
				pred = Cprev[current]
				# print("current: " + str(current))
				# print("pred: " + str(pred))
			else:
				break
		frontier_index = Cnext.get(frontier_index)
		# print(Cnext)
	return (a1_index < a2_index)

# def P_trace_is_safe(a1, a2, t):
# 	"""Returns whether the trace t from statement executing a1 to statement executing a2 via accesses in t is safe or not."""
# 	global RR_order, RW_order, WR_order, WW_order, statements_accesses, accesspattern, access_bigger_than

# 	i1 = t[0]
# 	i2 = t[len(t)-1]
# 	if i1 == i2:
# 		# single statement case
# 		return a2 in access_bigger_than[i1].get(a1, set([]))
# 	else:
# 		if a1[1] == a2[1] and (a1[0] == 'W' or a2[0] == 'W'):
# 			return True
# 		if a1[0] != 'W' and a2[0] != 'W' and RR_order:
# 			return True
# 		if a1[0] != 'W' and a2[0] == 'W' and RW_order:
# 			return True
# 		if a1[0] == 'W' and a2[0] != 'W' and WR_order:
# 			return True
# 		if a1[0] == 'W' and a2[0] == 'W' and WW_order:
# 			return True
# 		#print("UNSAFE?")
# 		# obtain summarised access pattern for t between first and last instruction
# 		t_accesses = (set([]), set([]), set([]))
# 		if len(t) > 2:
# 			for i in range(1, len(t)-1):
# 				ap = accesspattern[t[i]]
# 				t_accesses = (t_accesses[0]|ap[0], t_accesses[1]|ap[1], t_accesses[2]|ap[2])
# 		if a1[0] != 'W' and a2[0] == 'W':
# 			# simple, single access occurrence checks
# 			if not RW_order and not WR_order and not RR_order and not WW_order:
# 				if a1[1] in t_accesses[0]:
# 					return True
# 			if RR_order:
# 				if t_accesses[0] != set([]):
# 					return True
# 				if a2[1] in t_accesses[0] or a2[1] in t_accesses[1]:
# 					return True
# 			if WW_order:
# 				# is a write to a1[1] somewhere on the trace?
# 				if access_is_on_trace(('W',a1[1]), a1, i1, a2, i2, t_accesses):
# 					return True
# 			# simple, single access occurrence within start and end instructions checks
# 			if not RW_order and not WR_order and not RR_order and not WW_order:
# 				# read a1[1] occurs before write a2[1]
# 				if access_has_smaller_than(a2, ('R',a1[1]), i2):
# 					return True
# 				if access_has_bigger_than(a1, ('W',a2[1]), i1):
# 					return True
# 			if RR_order:
# 				if access_has_read_smaller_than(a2, i2):
# 					return True
# 				if access_has_bigger_than(a1, ('W',a2[1]), i1):
# 					return True
# 				if access_has_condition_bigger_than(a1, i1):
# 					return True
# 				if access_has_read_bigger_than(a1, ('R',a2[1]), i1):
# 					return True
# 			if WR_order and not RR_order and not WW_order:
# 				# check whether a write to a1[1] is followed by a read from a2[1] on a safe subtrace
# 				# if RR_order or WW_order, then those checks have already established that this condition cannot be satisfied.

# 				# find statement with first write to a1[1]
# 				start = -1
# 				end = -1
# 				for i in range(0,len(t)):
# 					ap = accesspattern[i]
# 					if a1[1] in ap[2]:
# 						start = i
# 						break
# 				if start != -1:
# 					# find statement with last read from a2[1]
# 					for i in reversed(range(0,len(t))):
# 						if i < start:
# 							break
# 						if a2[1] in ap[0] or a2[1] in ap[1]:
# 							end = i
# 							break
# 				if end != -1:
# 					subtrace = []
# 					for i in range(start,end):
# 						subtrace.append(t[i])
# 					return P_trace_is_safe(('W',a1[1]), ('R', a2[1]), subtrace)
# 			return False
# 		if a1[0] != 'W' and a2[0] != 'W':
# 			if not RW_order and not WR_order and not RR_order and not WW_order:
# 				if access_has_bigger_than(a1, ('W',a2[1]), i1):
# 					return True
# 				if access_has_smaller_than(a2, ('W',a1[1]), i2):
# 					return True
# 			if RW_order:
# 				if access_is_on_trace(('W',a2[1]), a1, i1, a2, i2, t_accesses):
# 					return True
# 			if WR_order:
# 				if access_is_on_trace(('W',a1[1]), a1, i1, a2, i2, t_accesses):
# 					return True
# 			if WW_order and not RW_order and not WR_order:
# 				# check whether a write to a1[1] is followed by a write to a2[1] on a safe subtrace
# 				# if RW_order or WR_order, then those checks have already established that this condition cannot be satisfied.

# 				# find statement with first write to a1[1]
# 				start = -1
# 				end = -1
# 				for i in range(0,len(t)):
# 					ap = accesspattern[i]
# 					if a1[1] in ap[2]:
# 						start = i
# 						break
# 				if start != -1:
# 					# find statement with last write to a2[1]
# 					for i in reversed(range(0,len(t))):
# 						if i < start:
# 							break
# 						if a2[1] in ap[2]:
# 							end = i
# 							break
# 				if end != -1:
# 					subtrace = []
# 					for i in range(start,end):
# 						subtrace.append(t[i])
# 					return P_trace_is_safe(('W',a1[1]), ('W', a2[1]), subtrace)
# 			return False
# 		if a1[0] == 'W' and a2[0] == 'W':
# 			if not RW_order and not WR_order and not RR_order and not WW_order:
# 				if access_has_smaller_than(a2, ('R',a1[1]), i2):
# 					return True
# 				if access_has_bigger_than(a1, ('R',a2[1]), i1):
# 					return True
# 				if access_is_on_trace(('C',a1[1]), a1, i1, a2, i2, t_accesses):
# 					return True
# 			if RW_order:
# 				if access_is_on_trace(('R',a1[1]), a1, i1, a2, i2, t_accesses):
# 					return True
# 			if WR_order:
# 				if access_is_on_trace(('R',a2[1]), a1, i1, a2, i2, t_accesses):
# 					return True
# 			if RR_order:
# 				if access_has_smaller_than(a2, ('R',a1[1]), i2):
# 					return True
# 				if access_is_on_trace(('C',a1[1]), a1, i1, a2, i2, t_accesses):
# 					return True
# 			if RR_order and not RW_order and not WR_order:
# 				# check whether a read from a1[1] is followed by a read from a2[1] on a safe subtrace
# 				# if RW_order or WR_order, then those checks have already established that this condition cannot be satisfied.

# 				# find statement with first read from a1[1]
# 				start = -1
# 				end = -1
# 				for i in range(0,len(t)):
# 					ap = accesspattern[i]
# 					if a1[1] in ap[0] or a1[1] in ap[1]:
# 						start = i
# 						break
# 				if start != -1:
# 					# find statement with last read from a2[1]
# 					for i in reversed(range(0,len(t))):
# 						if i < start:
# 							break
# 						if a2[1] in ap[0] or a2[1] in ap[1]:
# 							end = i
# 							break
# 				if end != -1:
# 					subtrace = []
# 					for i in range(start,end):
# 						subtrace.append(t[i])
# 					return P_trace_is_safe(('R',a1[1]), ('R', a2[1]), subtrace)
# 			return False
# 		if a1[0] == 'W' and a2[0] != 'W':
# 			if not RW_order and not WR_order and not RR_order and not WW_order:
# 				return False
# 			if RR_order:
# 				if access_is_on_trace(('R',a1[1]), a1, i1, a2, i2, t_accesses):
# 					return True
# 			if WW_order:
# 				if access_is_on_trace(('W',a2[1]), a1, i1, a2, i2, t_accesses):
# 					return True
# 			if RW_order and not RR_order and not WW_order:
# 				# check whether a read from a1[1] is followed by a write to a2[1] on a safe subtrace
# 				# if RW_order or WR_order, then those checks have already established that this condition cannot be satisfied.

# 				# find statement with first read from a1[1]
# 				start = -1
# 				end = -1
# 				for i in range(0,len(t)):
# 					ap = accesspattern[i]
# 					if a1[1] in ap[0] or a1[1] in ap[1]:
# 						start = i
# 						break
# 				if start != -1:
# 					# find statement with last write to a2[1]
# 					for i in reversed(range(0,len(t))):
# 						if i < start:
# 							break
# 						if a2[1] in ap[2]:
# 							end = i
# 							break
# 				if end != -1:
# 					subtrace = []
# 					for i in range(start,end):
# 						subtrace.append(t[i])
# 					return P_trace_is_safe(('R',a1[1]), ('W', a2[1]), subtrace)
# 			return False

def process_critical_cycles():
	"""Postprocess detected cycles to identify fence insertion suggestions"""
	global DG_P, DG_C, statements_accesses, unsafe_cyclic_P_traces, SMowner, statement_fence_suggestions, access_fence_suggestions, statement_fence_requirements, accesspattern, WriteAtomicity

	# for efficiency: set of (instruction level) P-traces for which a critical cycle has already been found
	critical_P_traces = set([])
	# for efficiency: set of (instruction level) P-traces for which it has been determined that they cannot be unsafe
	safe_P_traces = set([])

	# Perform the following loop twice: first for single instruction P-traces, then for multiple instruction P-traces
	# for check_P_length_one in [True, False]:
	# 	if check_P_length_one:
	# 		print("processing cycles on P-paths of length one")
	# 		sys.stdout.flush()
	# 	else:
	# 		print("processing cycles on P-paths of length greater than one")
	# 		sys.stdout.flush()
	print("processing P-paths in cycles")
	init_detect_critical_cycles()
	result = get_next_critical_cycle()
	while result != '':
		o_name = result[0]
		CY = result[1]
		#crit_count = 0
		#print(crit_count)
		#crit_count += 1
		#print("here: " + str(CY))
		# Compress the P-traces
		CY_summary = []
		CY_summary_cycle_starts = []
		P_traces = {}
		scanning_start = 0
		for i in range(0, len(CY)):
			s = CY[i]
			t = CY[(i+1)%len(CY)]
			if SMowner[statements_accesses[s][1]] != SMowner[statements_accesses[t][1]]:
				# here we should start scanning for P-traces
				scanning_start = (i+1)%len(CY)
				break
		i = scanning_start
		trace = []
		first = True
		while i != scanning_start or first:
			first = False
			s = CY[i]
			t = CY[(i+1)%len(CY)]
			if trace == []:
				P_trace_start = s
			trace.append(s)
			if t in DG_P[o_name].get(s, set([])) and SMowner[statements_accesses[s][1]] == SMowner[statements_accesses[t][1]]:
				if s == t:
					CY_summary.append((P_trace_start,s))
					P_traces[tuple((P_trace_start,s))] = trace
					# if we have not yet found a critical cycle for this P trace, mark it as a starting point
					if tuple(trace) not in critical_P_traces:
						CY_summary_cycle_starts.append(len(CY_summary)-1)
					trace = []
			else:
				if s == P_trace_start:
					t_summary = tuple([(s)])
				else:
					t_summary = (P_trace_start,s)
				CY_summary.append(t_summary)
				P_traces[t_summary] = trace
				# if we have not yet found a critical cycle for this P trace, and if we do not know that it is definitely safe, mark it as a starting point
				if tuple(trace) not in critical_P_traces and tuple(trace) not in safe_P_traces:
					CY_summary_cycle_starts.append(len(CY_summary)-1)
				trace = []
			i = (i+1)%len(CY)
		# next, we try to identify for each P-trace in CY whether we can construct an access-level cycle in which that P-trace is unsafe
		# (and no C-chords are present)

		# first limit the accesses per statement to take C-edges into account
		cycle_accesses = {}
		#print(CY_summary)
		if len(CY_summary_cycle_starts) > 0:
			for i in range(0,len(CY_summary)):
				index = CY_summary_cycle_starts[i]
				t_start = CY_summary[index][0]
				t_end = CY_summary[index][len(CY_summary[index])-1]
				previous = (index-1)%len(CY_summary)
				next_index = (index+1)%len(CY_summary)
				predecessor = CY_summary[previous][len(CY_summary[previous])-1]
				successor = CY_summary[next_index][0]
				ap = accesspattern[t_start]
				#print(t_start)
				L = [('C',a) for a in ap[0]] + [('R',a) for a in ap[1]] + [('W',a) for a in ap[2]]
				cycle_accesses[t_start] = tuple([L])
				#print(L)
				if t_start != t_end:
					ap = accesspattern[t_end]
					L = [('C',a) for a in ap[0]] + [('R',a) for a in ap[1]] + [('W',a) for a in ap[2]]
					cycle_accesses[t_end] = tuple([L])
				A = cycle_accesses[t_start][0]
				A_filtered = []
				for a,x in A:
					if conflicting_accesses(x, (a == 'W'), predecessor) != (set([]),set([])):
						A_filtered.append((a,x))
				#print("filtered: " + str(A_filtered))
				cycle_accesses[t_start] = tuple([A_filtered])
				if t_start != t_end:
					A = cycle_accesses[t_end][0]
				A_filtered = []
				for a,x in A:
					if conflicting_accesses(x, (a == 'W'), successor) != (set([]),set([])):
						A_filtered.append((a,x))
				if t_start == t_end:
					# we write a tuple of two lists, to distinguish how to enter, and how to leave the statement
					cycle_accesses[t_end] = (cycle_accesses[t_end][0], A_filtered)
				else:
					cycle_accesses[t_end] = tuple([A_filtered])

		# depth-first like access cycle construction
		number_of_accesses = len(CY_summary)*2
		#print("*** " + str(CY) + " " + str(CY_summary))

		callstack = []
		for ti in range(0,len(CY_summary_cycle_starts)):
			t = CY_summary[ti]
			# if tuple(P_traces[tuple(t)]) not in critical_P_traces:
			# 	# consider t?
			# 	if (check_P_length_one and len(t) == 1) or (not check_P_length_one and len(t) > 1):
			# keep track of whether the starting trace can be made unsafe
			t_can_be_unsafe = False
			callstack = []
			accesses = deepcopy(cycle_accesses[t[0]][0])
			# currently selected access in current statement
			current_access = ()
			# previously selected access in current statement (have we moved inside statement?)
			previous_access = ()
			callstack.append((t[0], accesses, current_access, previous_access))
			i = ti
			Ppos = 0
			cycle_found = False
			# the two accesses at the start and end of the considered P-trace t
			first_access = ()
			second_access = ()
			# variable counter to avoid C-chords (condition 2 of Shasha & Snir's critical cycle definition)
			var_counter = {}
			while callstack != []:
				#print(CY_summary)
				v, A, current_access, previous_access = peek(callstack)
				move_to_next = False
				while len(A) > 0:
					a = A.pop()
					# increment counter if we are not considering the same access as the one in the previous step (in which case
					# we stay at the same access in the same statement)
					if a != previous_access:
						#print("increment " + str(a[1]))
						var_counter[a[1]] = var_counter.get(a[1],0)+1
					if len(callstack) == 1:
						first_access = a
					current_access = a
					move_to_next = False
					# if we have selected a[1] already three times, reject in this case
					if var_counter[a[1]] > 3:
						var_counter[a[1]] -= 1
						continue
					# if we are considering trace t, make sure it will be unsafe
					if len(callstack) == 2:
						#print("checking safety")
						t_start = callstack[0][0]
						t_end = v
						# print(v)
						if t_start == t_end:
							if  P_trace_is_safe(first_access, a, P_traces[tuple([(t_start)])]):
								#print("safe!")
								continue
							else:
								#print("unsafe!")
								second_access = a
								t_can_be_unsafe = True
						else:
							# print(P_traces)
							# print(CY_summary)
							# print(callstack)
							if P_trace_is_safe(first_access, a, P_traces[(t_start,t_end)]):
								continue
							else:
								second_access = a
								t_can_be_unsafe = True
					# are we about to close a cycle?
					if len(callstack) == number_of_accesses:
						if accesses_are_conflicting(first_access, a):
							if len(t) == 1:
								AS = access_fence_suggestions.get(t[0], set([]))
								AS.add((first_access, second_access))
								access_fence_suggestions[t[0]] = AS
								# add fence suggestion to statement access order
								#(first_access, second_access, t[0])
								#print(first_access)
								#print(second_access)
							else:
								# add trace to the list of fence requirements
								trace = P_traces[tuple(t)]
								statement_fence_requirements.append(first_access, second_access, trace)
								# print(trace)
							critical_P_traces.add(tuple(P_traces[tuple(t)]))
							cycle_found = True
							# update current access on callstack
							callstack[len(callstack)-1] = (v, A, a, previous_access)
							break
						else:
							continue
					# put next statement on call stack
					else:
						new_previous = ()
						if len(CY_summary[i]) == 2 or previous_access != ():
							if Ppos == 0 and len(CY_summary[i]) == 2:
								Ppos = 1
								find_conflicts = False
							else:
								Ppos = 0
								i = (i+1)%len(CY_summary)
								find_conflicts = True
							accesses = deepcopy(cycle_accesses[CY_summary[i][Ppos]][0])
						elif previous_access == ():
							#print(i)
							#print(Ppos)
							#print(CY_summary)
							#print(cycle_accesses)
							accesses = deepcopy(cycle_accesses[CY_summary[i][Ppos]][1])
							new_previous = current_access
							find_conflicts = False
							# take a out of accesses in case we are processing t
							if len(callstack) == 1:
								accesses = set(accesses)
								if a in accesses:
									accesses.remove(a)
								accesses = list(accesses)
						# filter accesses
						if find_conflicts:
							accesses_filtered = []
							for a1 in accesses:
								if accesses_are_conflicting(a, a1):
									accesses_filtered.append(a1)
							if accesses_filtered != []:
								accesses = accesses_filtered
							else:
								continue
						# update current access on callstack
						callstack[len(callstack)-1] = (v, A, a, previous_access)
						# put new access on callstack
						callstack.append((CY_summary[i][Ppos], accesses, (), new_previous))
						move_to_next = True
						break
				if cycle_found:
					break
				if not move_to_next:
					callstack.pop()
					# decrement variable counter, if needed
					if previous_access != current_access and current_access != ():
						count = var_counter.get(current_access[1])
						if count != None:
							count -= 1
							var_counter[current_access[1]] = count
					if Ppos == 1:
						Ppos = 0
					elif previous_access == ():
						i = (i-1)%len(CY_summary)
						Ppos = len(CY_summary[i])-1
			if not t_can_be_unsafe:
				safe_P_traces.add(tuple(P_traces[tuple(t)]))
		# process write atomicity issues, if applicable
		if len(callstack) > 0 and not WriteAtomicity:
			i = 0
			first = True
			while i != 0 or first:
				first = False
				j = (i+1)%len(callstack)
				v1 = callstack[i][0]
				v2 = callstack[j][0]
				if SMowner[statements_accesses[v1][1]] != SMowner[statements_accesses[v2][1]]:
					# from Write to Read?
					# print(callstack[i][2])
					if callstack[i][2][0] == 'W' and callstack[j][2][0] != 'W':
						# check for fences
						found = False
						for t in P_traces:
							St = set(t)
							if v1 in St or v2 in St:
								found = True
								break
						if not found:
							# we need to add a fence. We will do so on the succeeding trace
							for t in CY_summary:
								if v2 in set(t):
									# print("adding a fence to handle write non-atomicity")
									critical_P_traces.add(tuple(P_traces[tuple(t)]))
									break
				i = (i+1)%len(callstack)
		result = get_next_critical_cycle()
	# Postprocess P-traces
	print("optimising transaction and intra-instruction fence placement")
	sys.stdout.flush()
	optimise_transactions()
	print("optimising inter-instruction fence placement")
	sys.stdout.flush()
	optimise_fences()

def optimise_transactions():
	"""Optimise the placement of transactions and intra-instruction fences"""
	global access_fence_suggestions, access_smaller_than, access_bigger_than, transaction_counter, fence_counter_intra
	# for each statement mentioned in the suggestions, add suggestions to access order and detect SCCs to identify transactions
	transaction_counter = 0
	for i, S in access_fence_suggestions.items():
		O = deepcopy(access_bigger_than[i])
		for a1, a2 in S:
			B2 = O.get(a2, set([]))
			B1 = O.get(a1, set([]))
			B1.add(a2)
			B1 |= B2
			O[a1] = B1
		# detect SCCs
		SCCs = list()
		SCCdict = {}
		identifySCCs(O, SCCdict, SCCs)
		# count SCCs of size > 1. Those require transactions.
		for scc in SCCs:
			if scc[0] > 1:
				transaction_counter += 1
				print("transaction in " + str(i) + ": " + str(scc))
				sys.stdout.flush()
		# filter suggestions, removing edges within an SCC
		S_filtered = []
		for a1, a2 in S:
			if SCCdict[a1] != SCCdict[a2]:
				S_filtered.append((a1,a2))
		# add remaining suggestions to access relations to strengthen them
		for a1, a2 in S_filtered:
			update_smaller_than(a2, a1, i)
			update_bigger_than(a1, a2, i)
		# group accesses in i into equivalence classes based on their predecessors and successors
		EC_counter = 0
		EC_numbers = {}
		ECs = {}
		accesses = set(access_smaller_than[i].keys()) | set(access_bigger_than[i].keys())
		for a in accesses:
			B = list(access_smaller_than[i].get(a,set([])))
			B.sort()
			B = tuple(B)
			G = list(access_bigger_than[i].get(a,set([])))
			G.sort
			G = tuple(G)
			ecid = EC_numbers.get((B,G))
			if ecid == None:
				EC_numbers[(B,G)] = EC_counter
				ecid = EC_counter
				EC_counter += 1
			ECs[a] = ecid
		# filter suggestions, keeping at most one for every equivalence class pair
		S_EQ_filtered = []
		sug_added = set([])
		for a1, a2 in S_filtered:
			if (ECs[a1],ECs[a2]) not in sug_added:
				S_EQ_filtered.append((a1,a2))
				sug_added.add((ECs[a1],ECs[a2]))
		# count number of intra-instruction fences
		fence_counter_intra += len(S_EQ_filtered)

def filter_statements_fence_requirements():
	"""Function to check which fence requirements are still relevant (still unsafe after intra-instruction fences have been placed),
	and preparing the relevant ones for optimised fence placement"""
	global statement_fence_requirements, statement_fence_suggestions

	statement_fence_counter = 0
	for (first, second, trace) in statement_fence_requirements:
		if not P_trace_is_safe(first, second, trace):
			for tj in range(0,len(trace)-1):
				tjid = (statements_accesses[trace[tj]][0],statements_accesses[trace[tj]][1])
				fencing = statement_fence_suggestions.get(tjid, set([]))
				fencing.add(statement_fence_counter)
				statement_fence_suggestions[tjid] = fencing
				statement_fence_counter += 1

def statement_fence_sorting_value(e):
	"""Function for sorting statement fence suggestions"""
	global statement_fence_suggestions
	size = len(statement_fence_suggestions[e])
	if size == None:
		return 0
	else:
		return size

def optimise_fences():
	"""Optimise the placement of inter-instruction fences"""
	global statement_fence_requirements, statement_fence_suggestions, fence_counter_inter, statements_accesses
	# filter the statements fence requirements
	filter_statements_fence_requirements()

	L = list(statement_fence_suggestions.keys())
	fence_counter_inter = 0
	# print(statement_fence_suggestions)
	while L != []:
		L.sort(reverse=True, key=statement_fence_sorting_value)
		# pick first item to put a fence in front of it (todo: store this list somewhere)
		# and remove all associated trace entries
		select = L[0]
		print("fence before: " + str(select))
		sys.stdout.flush()
		fence_counter_inter += 1
		S = statement_fence_suggestions[select]
		sug_tmp = {}
		for i, S2 in statement_fence_suggestions.items():
			newS2 = S2 - S
			if newS2 != set([]):
				sug_tmp[i] = newS2
		statement_fence_suggestions = sug_tmp
		L = list(statement_fence_suggestions.keys())

def readAut(autfile):
	"""Read a .aut file and place the data in a dictionary"""
	trans = {}

	try:
		inFile = open(autfile,'r')
	except IOError:
		print_error(".aut file of SLCO model does not exist!")
		sys.stdout.flush()
		return {}
	# scan the first line
	line = inFile.readline()
	global token, next
	results, remainder = autscanner.scan(line)
	inFile.close()

	# read the header
	if results[0][0] != 'HEADER':
		print_error("unexpected start of .aut description!")
		sys.stdout.flush()
	transheader = []
	# an aut file has three values in the header
	for index in range(1,4):
		transheader.append(results[index][1])

	# count to remove double entries
	ntrans = 0
	actset = set([])

	#store transitions in memory
	currentsrc = ''
	newentry = {}
	first = True
	for transline in open(autfile):
		tl, remainder = autscanner.scan(transline)
		src = tl[0][1]
		act = tl[1][1]
		tgt = tl[2][1]
		if not first:
			# add quotes if not present
			act.replace('\'', '"')
			if act[0] != '"':
				act = '"' + act + '"'
			# put transition label in actset
			actset.add(act)
			# are we building a new dictionary entry?
			if src != currentsrc:
				# possibly add old entry if not empty
				if currentsrc != '':
					trans[currentsrc] = newentry
				currentsrc = src
				newentry = {}
			outgoingwithlabel = newentry.get(act)
			if outgoingwithlabel == None:
				newentry[act] = set([tgt])
				ntrans += 1
			else:
				if tgt not in outgoingwithlabel:
					# insert new element
					outgoingwithlabel.add(tgt)
					newentry[act] = outgoingwithlabel
					ntrans += 1
		else:
			first = False
	# add final entry
	trans[currentsrc] = newentry
	transheader[1] = str(ntrans)
	return [transheader, trans, actset]

def read_statespace(m):
	"""Read the state space of the model"""
	global model_statespace, WR_order

	statespace_name = m[:-4] + "aut"
	model_statespace = readAut(statespace_name)

def preprocess():
	global statements_objects, SMowner, model, globalvars, statemachine, smclass, smlocalvars, pure_static_analysis, STID_to_object
	# build dictionaries providing for a given statement the state machine owning it, and for a given state machine, the class owning it
	for c in model.classes:
		for stm in c.statemachines:
			smclass[stm] = c
			for trn in stm.transitions:
				for stat in trn.statements:
					statemachine[stat] = stm
	# build a dictionary providing sets of variables local for given state machines
	for c in model.classes:
		for stm in c.statemachines:
			varset = set([])
			for var in stm.variables:
				varset.add(var.name)
			smlocalvars[c.name + "'" + stm.name] = varset
	# build dictionary making variable scopes explicit
	for c in model.classes:
		for stm in c.statemachines:
			vdict = {}
			for var in c.variables:
				vdict[var.name] = var.name
			for var in stm.variables:
				vdict[var.name] = stm.name + "'" + var.name
			scopedvars[c.name + "'" + stm.name] = vdict
	# construct list of statement names with pointers to the objects
	statements_objects = {}
	for c in model.classes:
		for sm in c.statemachines:
			for tr in sm.transitions:
				statements_objects["ST'" + str(tr._tx_position)] = tr.statements[0]
	# construct function providing for given statement the owning SM
	SMowner = {}
	for c in model.classes:
		for sm in c.statemachines:
			for tr in sm.transitions:
				SMowner["ST'" + str(tr._tx_position)] = sm
	# build set of global variables per object
	for o in model.objects:
		V = set([])
		c = o.type
		for v in c.variables:
			V.add("var_" + o.name + "'" + v.name)
		globalvars[o] = V
	# build dictionary providing statement object, given the ID of its transition
	STID_to_object = {}
	for c in model.classes:
		for sm in c.statemachines:
			for tr in sm.transitions:
				STID_to_object["ST'" + str(tr._tx_position)] = tr.statements[0]

def get_smallest_accesses(id):
	global statements_accesses
	"""Return the set of smallest accesses of the given statement (indicated by number) id"""
	instr = statements_accesses[id]

	if len(instr[2]) > 0:
		if instr[2][0][0] != set([]) and instr[2][0][1] == set([]):
			return instr[2][0]
	rset = set([])
	wset = set([])
	writes = set([])
	for i in range(0,len(instr[2])):
		ap = instr[2][i]
		rset |= (ap[0] - writes)
		if ap[0] == set([]):
			wset |= ap[1]
		writes |= ap[1]
	return (rset, wset)

def get_largest_accesses(id):
	global statements_accesses
	"""Return the set of largest accesses of the given statement (indicated by number) id"""
	instr = statements_accesses[id]

	if len(instr[2]) == 1 and instr[2][0][1] == set([]):
		return instr[2][0]
	else:
		wset = set([])
		for i in range(0, len(instr[2])):
			wset |= instr[2][i][1]
		return (set([]),wset)

def access_can_be_moved_over_set(s, a, t, A2, to_right):
	"""Return whether access a can be swapped with set A2. to_right indicates the direction of moving.
	s and t are owners of a and A2, respectively."""
	global RR_order, RW_order, WR_order, WW_order, StoreBuffering, access_smaller_than, access_bigger_than
	# print(s)
	# print(a)
	# print(t)
	# print(A2)
	# print(to_right)
	# print(RW_order)
	# print(WR_order)
	if a[0] != 'W':
		if RR_order:
			if A2[0] != set([]) or A2[1] != set([]):
				# print("here1")
				return False
		if (to_right and RW_order) or ((not to_right) and WR_order):
			if A2[2] != set([]):
				# print("here2")
				return False
		# no speculative writing
		if to_right and a[0] == 'C' and A2[2] != set([]):
			# print("here3")
			return False
		# no dependent accesses reordering inside an instruction
		if to_right and s == t:
			S = access_bigger_than[s].get(a, set([]))
			for a2 in S:
				if a2[1] in A2[2]:
					# print("here4")
					return False
		# conflict with read (unless moving to the left and StoreBuffering is supported)
		# print("here5")
		# print((a[1] not in A2[2]) or ((not to_right) and StoreBuffering))
		return (a[1] not in A2[2]) or ((not to_right) and StoreBuffering)
	else:
		if (to_right and WR_order) or ((not to_right) and RW_order):
			if A2[0] != set([]) or A2[1] != set([]):
				# print("here6")
				return False
		if WW_order:
			if A2[2] != set([]):
				# print("here7")
				return False
		# no speculative writing
		if (not to_right) and A2[0] != set([]):
			# print("here8")
			return False
		# conflict with identical write
		if a[1] in A2[2]:
			# print("here9")
			return False
		# no dependent accesses reordering inside an instruction
		if (not to_right) and s == t:
			S = access_smaller_than[s].get(a, set([]))
			for a2 in S:
				if a2[1] in A2[1]:
					# print("here10")
					return False
		# conflict with read (unless moving to the right and StoreBuffering is supported)
		# print("here11")
		return (a[1] not in A2[0] and a[1] not in A2[1]) or (to_right and StoreBuffering)

def reorder_accesses(A1, A2):
	"""Reorder access sets A1 and A2. Produces a tuple of four access patterns: (Part of A2 moved over A1), (left over of A1), (left over of A2),
	(Part of A1 moved over A2)"""
	L = (set([]), set([]), set([]))
	R = (set([]), set([]), set([]))
	newA1 = (set([]), set([]), set([]))
	newA2 = (set([]), set([]), set([]))
	s = A1[0]
	t = A2[0]
	ap1 = A1[1]
	ap2 = A2[1]
	for a in ap1[0]:
		if access_can_be_moved_over_set(s, ('C',a), t, ap2, True):
			R[0].add(a)
		else:
			newA1[0].add(a)
	for a in ap1[1]:
		if access_can_be_moved_over_set(s, ('R',a), t, ap2, True):
			R[1].add(a)
		else:
			newA1[1].add(a)
	for a in ap1[2]:
		if access_can_be_moved_over_set(s, ('W',a), t, ap2, True):
			R[2].add(a)
		else:
			newA1[2].add(a)
	for a in ap2[0]:
		if access_can_be_moved_over_set(s, ('C',a), t, newA1, False):
			L[0].add(a)
		else:
			newA2[0].add(a)
	for a in ap2[1]:
		if access_can_be_moved_over_set(s, ('R',a), t, newA1, False):
			L[1].add(a)
		else:
			newA2[1].add(a)
	for a in ap2[2]:
		if access_can_be_moved_over_set(s, ('W',a), t, newA1, False):
			L[2].add(a)
		else:
			newA2[2].add(a)
	# print("((" + str(t) + ", " + str(L) + "), (" + str(s) + ", " + str(newA1) + "), (" + str(t) + ", " + str(newA2) + "), (" + str(s) + ", " + str(R) + "))")
	return ((t, L), (s, newA1), (t, newA2), (s, R))

def statements_are_conflicting(i1, i2):
	"""Return whether the first statement conflicts with the second"""
	global accesspattern
	ap = accesspattern[i1]
	reads = ap[0] | ap[1]
	writes = ap[2]
	return accesspattern_conflicts_with_statement((reads, writes), i2)

def accesspattern_conflicts_with_statement(ap, i1):
	"""Return whether the given access pattern ap conflicts with the statement i1"""
	global statements_accesses

	reads = ap[0]
	writes = ap[1]
	conflicting = False
	for a in reads:
		c1 = conflicting_accesses(a, False, i1)
		if c1[0] != set([]) or c1[1] != set([]):
			return True
	for a in writes:
		c1 = conflicting_accesses(a, True, i1)
		if c1[0] != set([]) or c1[1] != set([]):
			return True
	return False

def statements_are_parallel(a1, a2):
	"""Return whether the two statements can possibly be executed in parallel (are from different statemachines of the same object)"""
	global statements_accesses, SMowner

	if a1 == '\"tau\"' or a2 == '\"tau\"':
		return True
	ac1 = statements_accesses[a1]
	ac2 = statements_accesses[a2]
	if ac1[0] == ac2[0]:
		return SMowner[ac1[1]] != SMowner[ac2[1]]
	else:
		return False

def accesses_are_conflicting(a1, a2):
	"""Return whether accesses are conflicting"""
	global pure_static_analysis
	if a1[0] != 'W' and a2[0] != 'W':
		return False
	if not pure_static_analysis:
		return (a1[1] == a2[1])
	else:
		if "(*)" in a1[1]:
			if "(*)" in a2[1]:
				return (a1[1] == a2[1])
			else:
				a1_splitted = a1[1].split("(*)")
				a2_splitted = a2[1].split("(")
				return a1_splitted[0] == a2_splitted[0]
		else:
			if "(*)" in a2[1]:
				a1_splitted = a1[1].split("(")
				a2_splitted = a2[1].split("(*)")
				return a1_splitted[0] == a2_splitted[0]
			else:
				return (a1[1] == a2[1])
		# a1_splitted = a1[1].split("(")
		# a2_splitted = a2[1].split("(")
		# if len(a1_splitted) == 1 and len(a2_splitted) == 1:
		# 	return (a1[1] == a2[1])
		# if (len(a1_splitted) == 1 and len(a2_splitted) > 1) or (len(a1_splitted) > 1 and len(a2_splitted) == 1):
		# 	return False
		# a1_splitted = a1[1].split("(*)")
		# if a1_splitted[0] != a1[1]:
		# 	return (a1_splitted[0] == a2_splitted[0])
		# a1_splitted = a1[1].split("(")
		# a2_splitted = a2[1].split("(*)")
		# if a2_splitted[0] != a2[1]:
		# 	return (a1_splitted[0] == a2_splitted[0])
		# return False

def conflicting_accesses(a, iswrite, ins):
	global accesspattern, pure_static_analysis
	"""For the given access a, provide the accesses of statement ins conflicting with a. Boolean iswrite indicates whether a is a write or not"""
	#print("a: " + str(a))
	#print(iswrite)
	ap = accesspattern[ins]
	#print(ap)
	# merge conditional reads and reads
	readsap = ap[0] | ap[1]
	writesap = ap[2]
	conflicting = (set([]), set([]))
	array_static_access = False
	array_dynamic_access = False
	if pure_static_analysis:
		a_splitted = a.split("(*)")
		if a_splitted[0] != a:
			array_dynamic_access = True
			a2 = a_splitted[0]
		else:
			a_splitted = a.split("(")
			if len(a_splitted) > 1:
				array_static_access = True
				a2 = a_splitted[0]
		#print("a_splitted: " + str(a_splitted))
		#print("a2: " + str(a2))
	if iswrite:
		if array_static_access:
			if a in readsap:
				conflicting[0].add(a)
			elif a2 + "(*)" in readsap:
				conflicting[0].add(a2 + "(*)")
		elif array_dynamic_access:
			if a in readsap:
				conflicting[0].add(a)
			else:
				for a3 in readsap:
					a3_splitted = a3.split("(")
					if len(a3_splitted) > 1:
						if a2 == a3_splitted[0]:
							conflicting[0].add(a3)
							break
		elif a in readsap:
			conflicting[0].add(a)
	if array_static_access:
		if a in writesap:
			conflicting[1].add(a)
		elif a2 + "(*)" in writesap:
			conflicting[1].add(a2 + "(*)")
	elif array_dynamic_access:
		if a in writesap:
			conflicting[1].add(a)
		else:
			for a3 in writesap:
				a3_splitted = a3.split("(")
				if len(a3_splitted) > 1:
					if a2 == a3_splitted[0]:
						conflicting[1].add(a3)
						break
	elif a in writesap:
		conflicting[1].add(a)
	#print(conflicting)
	return conflicting

def print_STIDs():
	global model
	for c in model.classes:
		for sm in c.statemachines:
			for tr in sm.transitions:
				print("ST'" + str(tr._tx_position) + ": " + printstatement(tr.statements[0]))
				sys.stdout.flush()

def obtain_statements_accesses():
	"""Analyse the statements occurring in the state space, and construct lists of IDs and accesses"""
	global model, model_statespace, statements_IDs, statements_accesses, accesspattern, access_smaller_than, access_bigger_than, globalvars, index_accesses, STID_to_object

	# construct dictionary of statement IDs
	L = list(model_statespace[2])
	statements_IDs = {}
	j = 0
	for i in range(0,len(L)):
		if L[i] != '\"tau\"':
			statements_IDs[L[i]] = j
			j += 1

	# construct dictionary of accesses for the statements
	statements_accesses = {}
	# construct dictionary of accesses needed to compute the index of an array element access
	index_accesses = {}
	for a in model_statespace[2]:
		#print(a)
		if a != '\"tau\"':
			rw, remainder = rwscanner.scan(a)
			#print(rw)
			i = 4
			reads = set([])
			writes = set([])
			read = False
			accessmode = False
			alist = []
			while i < len(rw):
				if rw[i][0] == 'BEGINLIST':
					accessmode = True
					if not read:
						reads = set([])
					else:
						writes = set([])
					i += 1
				elif rw[i][0] == 'ENDLIST':
					if accessmode:
						if not read:
							read = True
						else:
							alist.append((reads, writes))
							read = False
							accessmode = False
					else:
						break
					i += 1
				elif rw[i][1] == 'A\'(':
					i += 1
				else:
					if not read:
						reads.add(rw[i][1])
					else:
						writes.add(rw[i][1])
					i += 1
			# get object
			owner = ""
			for o in model.objects:
				if o.name == rw[0][1]:
					owner = o
					break
			# record index_accesses
			# first obtain corresponding static access pattern
			static_alist = statement_structure_accesspattern(STID_to_object[rw[2][1]], o)
			#print(alist)
			#print(static_alist)
			#for (reads,writes) in alist:
			#	for r in reads:
			#		print("here")
			# filter out-of-bound array accesses
			alist_filtered = []
			for (reads,writes) in alist:
				newreads = set([])
				newwrites = set([])
				for r in reads:
					if r.find("(") != -1:
						access = r.split("(")
						access = access[0].split("'")
						V = owner.type.variables
						# possibly look up variables of state machine (if access is to state machine local variable)
						if len(access) > 2:
							for sm in owner.type.statemachines:
								if sm.name == access[1]:
									V = sm.variables
									break
						# look up variable
						for v in V:
							if v.name == access[len(access)-1]:
								break
						# check for dimension of array v
						access = r.split("(")
						index = int(access[1][:-1])
						if index >= 0 and index < v.type.size:
							newreads.add(r)
					else:
						newreads.add(r)
				for w in writes:
					if w.find("(") != -1:
						access = w.split("(")
						access = access[0].split("'")
						V = owner.type.variables
						# possibly look up variables of state machine (if access is to state machine local variable)
						if len(access) > 2:
							for sm in owner.type.statemachines:
								if sm.name == access[1]:
									V = sm.variables
									break
						# look up variable
						for v in V:
							if v.name == access[len(access)-1]:
								break
						# check for dimension of array v
						access = w.split("(")
						index = int(access[1][:-1])
						if index >= 0 and index < v.type.size:
							newwrites.add(w)
					else:
						newwrites.add(w)
				if newreads != set([]) or newwrites != set([]):
					alist_filtered.append((newreads, newwrites))
			statements_accesses[statements_IDs[a]] = (owner, rw[2][1], alist_filtered)
			#print(statements_accesses[statements_IDs[a]])

def static_obtain_statements_accesses():
	"""Analyse the statements occurring in the model statically, and construct list of IDs and accesses"""
	global model, model_statespace, statements_IDs, statements_accesses, accesspattern, access_smaller_than, access_bigger_than, globalvars

	# construct dictionary of statement IDs, and dictionary of accesses
	statements_IDs = {}
	statements_accesses = {}
	j = 0
	for o in model.objects:
		for sm in o.type.statemachines:
			for tr in sm.transitions:
				ap = statement_structure_accesspattern(tr.statements[0], o)
				if ap != [(set([]), set([]))]:
					statements_IDs[str(o.name) + "'ST'" + str(tr._tx_position)] = j
					statements_accesses[j] = (o, "ST'" + str(tr._tx_position), ap)
					j += 1

def accesses_minus(A1, A2):
	"""Set minus for sets of accesses"""
	global pure_static_analysis
	if not pure_static_analysis:
		return A1 - A2
	else:
		newA1 = set([])
		for a in A1:
			if "(*)" in a:
				newA1.add(a)
			elif a not in A2:
				newA1.add(a)
		return newA1

def access_is_in_set(a, A):
	"""Return whether access a is in set A"""
	global pure_static_analysis
	if not pure_static_analysis:
		return a in A
	else:
		if "(*)" in a:
			return False
		else:
			return a in A

def filter_accesses(A, V):
	"""Filter the accesses in A, only keep those that are also in V"""
	newA = set([])
	for a in A:
		# print(a)
		a_splitted = a.split("(")
		if len(a_splitted) > 1:
			a2 = a_splitted[0]
		else:
			a2 = a
		if a2 in V:
			newA.add(a)
	return newA

def analyse_statements():
	"""Analyse the statements occurring in the state space. Construct relations on accesses."""
	global model, model_statespace, statements_IDs, statements_accesses, accesspattern, access_smaller_than, access_bigger_than, globalvars, pure_static_analysis

	accesspattern = {}
	# build access patterns based on the statements_accesses information
	for i in range(0,len(statements_IDs)):
		ap = statements_accesses[i][2]
		V = globalvars.get(statements_accesses[i][0], set([]))
		reads = set([])
		writes = set([])
		j = 0
		if statement_has_guard(i):
			condreads = filter_accesses(ap[0][0], V)
			j = 1
		else:
			condreads = set([])
		while j < len(ap):
			reads |= filter_accesses(accesses_minus(accesses_minus(ap[j][0], condreads), writes), V)
			writes |= filter_accesses(ap[j][1], V)
			j += 1
		accesspattern[i] = (condreads, reads, writes)
		# print(i)
		# print(condreads)
		# print(reads)
		# print(writes)

	# build a dictionary for each instruction, indicating the successor and predecessor accesses of each access
	access_smaller_than = {}
	access_bigger_than = {}
	for i in range(0,len(statements_IDs)):
		read_pred_tmp = {}
		access_predecessors_i = {}
		ap = statements_accesses[i][2]
		#print("access pattern: " + str(statements_accesses[i]))
		j = 0
		condreads = set([])
		if statement_has_guard(i):
			condreads = set([])
			for ra in ap[0][0]:
				condreads.add(ra)
			j = 1
		reads = set([])
		#for ra in condreads:
		#	reads.add(ra[1])
		writes = set([])
		while j < len(ap):
			reads_j = ap[j][0] - writes #- reads
			writes_j = ap[j][1]
			# handle the case that a variable is being read after having been written to: take over dependencies of write for the read
			for ra in reads_j:
				reads.add(ra)
				ra_reads = set([])
				if access_is_in_set(ra, writes):
					R = access_predecessors_i.get(('W',ra), set([]))
					read_pred_tmp[ra] = R
			# handle address dependencies; a read from an array may depend on previous reads to compute the index
			#for ra in reads_j:
				# are we accessing an array?
				#if ra.find("(") != -1:
					# walk through both the current access pattern and a static version of it to retrieve the dependencies
					#print("TODO")
			# handle the write access of an assignment: reads are predecessors, handle special case of read after write occurrences
			for wa in writes_j:
				wa_reads = set([])
				for ra in reads_j:
					if pure_static_analysis:
						if "(*)" in ra:
							wa_reads.add(('R',ra))
							continue
					R = read_pred_tmp.get(ra, set([]))
					if R != set([]):
						wa_reads |= R
					else:
						wa_reads.add(('R',ra))
				# add conditional reads
				#wa_reads |= condreads
				for ra in condreads:
					if pure_static_analysis:
						if "(*)" in ra:
							wa_reads.add(('C',ra))
							continue
					R = read_pred_tmp.get(ra, set([]))
					if R != set([]):
						wa_reads |= R
					else:
						wa_reads.add(('C',ra))
				# if read from before, add this read to dependencies
				if access_is_in_set(wa, reads):
					#if not access_is_in_set(wa, ap[0][0]):
					wa_reads.add(('R',wa))
				W = access_predecessors_i.get(('W',wa), set([]))
				access_predecessors_i[('W',wa)] = W | wa_reads
			j += 1
		# we now have the predecessor relation of the accesses
		# find and replace local variables
		new_access_predecessors_i = {}
		for a in access_predecessors_i.keys():
			if not var_is_local(a[1], i):
				new_access_predecessors_i[a] = get_global_accesses(a, access_predecessors_i, i)
		access_predecessors_i = new_access_predecessors_i
		# build transitive closure of predecessor relation
		access_smaller_than_i = {}
		openset = set([])
		closedset = set([])
		for a in access_predecessors_i.keys():
			V = set([])
			openset = set([a])
			closedset = set([])
			while openset != set([]):
				v = openset.pop()
				closedset.add(v)
				S = access_predecessors_i.get(v, set([]))
				V |= S
				for v2 in S:
					if v2 not in closedset:
						openset.add(v2)
			access_smaller_than_i[a] = V
		access_smaller_than[i] = access_smaller_than_i
		#print("smaller than: " + str(access_smaller_than[i]))

		# reverse
		access_bigger_than_i = {}
		for a1, deps in access_smaller_than_i.items():
			for a2 in deps:
				R = access_bigger_than_i.get(a2, set([]))
				R.add(a1)
				access_bigger_than_i[a2] = R
		access_bigger_than[i] = access_bigger_than_i
	# print("<: " + str(access_smaller_than))
	# print(">: " + str(access_bigger_than))

def get_global_accesses(a, D, i):
	"""From the dependency relation encoded by dictionary D, get the global accesses on which access a depends. i is statement performing a."""
	V = D.get(a, set([]))
	newV = set([])
	for v in V:
		if var_is_local(v[1], i):
			newV |= get_global_accesses(v, D, i)
		else:
			newV.add(v)
	return newV

def access_has_smaller_than(a1, a2, i):
	"""In i, access a2 is indeed executed before a1. 0: no, 1: perhaps (but this should be enforced), 2: yes"""
	global access_smaller_than
	if a2 in access_smaller_than[i].get(a1, set([])):
		return 2
	elif a2[0] == 'R' and ('C',a2[1]) in access_smaller_than[i].get(a1, set([])):
		return 2
	elif a1 not in access_smaller_than[i].get(a2, set([])):
		if a2[0] == 'R':
			if a1 not in access_smaller_than[i].get(('C',a2[1]), set([])):
				return 1
			else:
				return 0
		else:
			return 1
	else:
		return 0

def access_has_read_smaller_than(a1, i):
	"""In i, a read is indeed executed before a1."""
	global access_smaller_than
	if a1[0] == 'R' and access_smaller_than[i].get(a1) == None:
		return access_has_read_smaller_than(('C',a1[1]), i)
	R = access_smaller_than[i].get(a1, set([]))
	if R == set([]):
		return False
	else:
		for a2 in R:
			if a2[0] == 'R' or a2[0] == 'C':
				return True
		return False

def access_has_read_bigger_than(a1, i):
	"""In i, a read is indeed executed after a1."""
	global access_bigger_than
	if a1[0] == 'R' and access_bigger_than[i].get(a1) == None:
		return access_has_read_bigger_than(('C',a1[1]), i)
	R = access_bigger_than[i].get(a1, set([]))
	if R == set([]):
		return False
	else:
		for a2 in R:
			if a2[0] == 'R' or a2[0] == 'C':
				return True
		return False

def access_has_condition_smaller_than(a1, i):
	"""In i, a read is indeed executed before a1."""
	global access_smaller_than
	if a1[0] == 'R' and access_smaller_than[i].get(a1) == None:
		return access_has_condition_smaller_than(('C',a1[1]), i)
	R = access_smaller_than[i].get(a1, set([]))
	if R == set([]):
		return False
	else:
		for a2 in R:
			if a2[0] == 'C':
				return True
		return False

def access_has_condition_bigger_than(a1, i):
	"""In i, a read is indeed executed after a1."""
	global access_bigger_than
	if a1[0] == 'R' and access_bigger_than[i].get(a1) == None:
		return access_has_condition_bigger_than(('C',a1[1]), i)
	R = access_bigger_than[i].get(a1, set([]))
	if R == set([]):
		return False
	else:
		for a2 in R:
			if a2[0] == 'C':
				return True
		return False

def access_has_bigger_than(a1, a2, i):
	"""In i, access a2 is indeed executed after a1. False if not guaranteed."""
	global access_bigger_than
	if a1[0] == 'R' and access_bigger_than[i].get(a1) == None:
		return access_has_bigger_than(('C',a1[1]), a2, i)
	if a2 in access_bigger_than[i].get(a1, set([])):
		return True
	elif a2[0] == 'R' and ('C',a2[1]) in access_bigger_than[i].get(a1, set([])):
		return True
	return False

def update_smaller_than(a1, a2, i):
	"""Extend the P-order in i to execute a2 before a1"""
	global access_smaller_than, access_bigger_than, access_successors, access_predecessors
	R = access_bigger_than[i].get(a2, set([]))
	R.add(a1)
	R |= access_bigger_than[i].get(a1, set([]))
	access_bigger_than[i][a2] = R
	R = access_smaller_than[i].get(a1, set([]))
	R.add(a2)
	R |= access_smaller_than[i].get(a2, set([]))
	access_smaller_than[i][a1] = R

def update_bigger_than(a1, a2, i):
	"""Extend the P-order in i to execute a2 after a1"""
	global access_smaller_than, access_bigger_than, access_successors, access_predecessors
	R = access_bigger_than[i].get(a1, set([]))
	R.add(a2)
	R |= access_bigger_than[i].get(a2, set([]))
	access_bigger_than[i][a1] = R
	R = access_smaller_than[i].get(a2, set([]))
	R.add(a1)
	R |= access_smaller_than[i].get(a1, set([]))
	access_smaller_than[i][a2] = R

def transitive_outgoing(s):
	"""Transitively over tau return outgoing transitions of state s. Precondition: no tau-loops present."""
	global model_statespace
	outgoing = model_statespace[1].get(s,{})
	tr_out = {}
	for a, tgts in outgoing.items():
		if a != '\"tau\"':
			T = tr_out.get(a, set([]))
			T |= tgts
			tr_out[a] = T
		else:
			for t in tgts:
				tr_out.update(transitive_outgoing(t))
	return tr_out

def constructDG():
	"""Construct the dependency graph of the SLCO model"""
	global model, model_statespace, DG_P, DG_C, statements_IDs, statements_accesses, memory_model, SMowner, RR_order, accesspattern, WR_order, StoreBuffering

	# Build dependency graph - P-relation
	DG_P = {}
	for o in model.objects:
		DG_P[o] = {}
	# construct P-relation. Iterate over all states, consider their outgoing transitions, and compare these to the outgoing transitions of successors.
	for i in range(0, int(model_statespace[0][2])):
		s = str(i)
		outgoing = transitive_outgoing(s)
		s_aset = set(outgoing.keys())
		for a, tgts in outgoing.items():
			j = statements_IDs[a]
			for t in tgts:
				t_outgoing = transitive_outgoing(t)
				t_aset = set(t_outgoing.keys())
				enabled = t_aset - s_aset
				tgtset = set([])
				for a2 in enabled:
					tgtset.add(statements_IDs[a2])
				out = DG_P[statements_accesses[j][0]].get(j,set([]))
				DG_P[statements_accesses[j][0]][j] = out | tgtset

	# Build dependency graph - C-relation
	DG_C = {}
	for o in model.objects:
		DG_C[o] = {}
	# construct C-relation. Iterate over all states, consider their outgoing transitions, and compare those from different state machines w.r.t. conflicting accesses.
	parallel_statements = {}
	for i in range(0, int(model_statespace[0][2])):
		s = str(i)
		outgoing = transitive_outgoing(s)
		s_aset = set(outgoing.keys())
		for a in s_aset:
			aid = statements_IDs[a]
			paset = set([])
			for a2 in s_aset:
				a2id = statements_IDs[a2]
				if statements_are_parallel(aid, a2id):
					paset.add(a2id)
			par = parallel_statements.get(aid,set([]))
			parallel_statements[aid] = par | paset
	for aid in parallel_statements.keys():
		for aid2 in parallel_statements[aid]:
			if statements_are_conflicting(aid, aid2):
				# Not yet present in P-relation?
				#if aid2 not in DG_P[statements_accesses[aid][0]].get(aid,set([])) and aid not in DG_P[statements_accesses[aid2][0]].get(aid2,set([])):
				out = DG_C[statements_accesses[aid][0]].get(aid,set([]))
				out.add(aid2)
				DG_C[statements_accesses[aid][0]][aid] = out
	
	# For each guarded statement, obtain the predecessors in P, to identify C conflicts with that guarded statement if execution of those predecessors does not
	# enable the guarded statement
	predecessors = {}
	for i in range(0, len(statements_IDs)):
		if statement_has_guard(i):
			o = statements_accesses[i][0]
			iset = set([])
			for j in range(0, len(statements_IDs)):
				if i in DG_P[o].get(j,set([])) and SMowner[statements_accesses[i][1]] == SMowner[statements_accesses[j][1]]:
					iset.add(j)
			predecessors[i] = iset

	# For each guarded statement, add conflicts with the statements executed in parallel with the former's predecessors
	# Perform a DFS from each guarded statement to identify reads that can be done speculatively. Only relevant if speculative reading is allowed by the memory model
	speculative_reads = {}
	if not RR_order:
		speculative_reads = {}
		for i in predecessors.keys():
			(condreads_i,reads,writes) = accesspattern[i]
			blocked = deepcopy(writes)
			specreads = {}
			if reads != set([]):
				specreads[i] = deepcopy(reads)
			# call stack for DFS
			callstack = []
			onstack = set([])
			blocked_vars = {}
			outgoing = DG_P[o].get(i, set([]))
			# put statement on call stack. Third argument is the set of variables that are blocked from speculative reading
			# (due to having observed a write to those variables on the stack trace)
			callstack.append((i, list(outgoing), blocked))
			onstack.add(i)
			while callstack != []:
				j, tgts, blocked = peek(callstack)
				# stop early if WR is guaranteed
				if WR_order:
					if blocked != set([]):
						callstack.pop()
						continue
				while len(tgts) > 0:
					k = peek(tgts)
					(condreads,reads,writes) = accesspattern[k]
					if not k in onstack and (blocked_vars.get(k) == None or ((blocked | writes) < blocked_vars.get(k, set([])))):
						# - any variables read from by k, not yet written to along the trace, and not read from by i can be done speculatively
						# - any variables written to by k should be blocked from speculative reading further down the trace. If WR is guaranteed
						# by the memory model, further searching is actually not needed.
						kblocked = deepcopy(blocked)
						for a in condreads | reads:
							if (a not in blocked or StoreBuffering):
								accesses = specreads.get(j,set([]))
								accesses.add(a)
								specreads[j] = accesses
						kblocked |= writes

						# put k on call stack
						koutgoing = DG_P[o].get(k, set([]))
						callstack.append((k, list(koutgoing), kblocked))
						onstack.add(k)
						break
					else:
						tgts.pop()
				if len(tgts) == 0:
					# close j
					B = blocked_vars.get(j)
					if B == None:
						blocked_vars[j] = blocked
					else:
						blocked_vars[j] = (B & blocked)
					# pop j off call stack
					callstack.pop()
					onstack.remove(j)
			speculative_reads[i] = specreads

	# identify for each statement s which statements of other state machines are enabled after executing s
	parallel_statements_enabled_after = {}
	for i in range(0, int(model_statespace[0][2])):
		s = str(i)
		outgoing = transitive_outgoing(s)
		s_aset = set(outgoing.keys())
		for a in s_aset:
			aid = statements_IDs[a]
			tgts = outgoing.get(aid,set([]))
			paset = set([])
			for t in tgts:
				t_outgoing = transitive_outgoing(t)
				t_aset = set(t_outgoing.keys())
				for a2 in t_aset:
					a2id = statements_IDs[a2]
					if statements_accesses[aid][0] == statements_accesses[a2id][0] and SMowner[statements_accesses[aid][1]] != SMowner[statements_accesses[a2id][1]]:
						paset.add(a2id)
			par = parallel_statements_enabled_after.get(aid,set([]))
			parallel_statements_enabled_after[aid] = par | paset

	# add additional C-edges to handle guarded statements
	for i in predecessors.keys():
		o = statements_accesses[i][0]
		condreads_i = accesspattern[i][0]
		# construct list of parallel statements
		parallel_instr = set([])
		for j in predecessors[i]:
			parallel_instr |= parallel_statements_enabled_after.get(j, set([]))
		# check for conflicts, and if present, add a C-edge
		for j in parallel_instr:
			if accesspattern_conflicts_with_statement((condreads_i,set([])), j):
				if i not in DG_P[o].get(j, set([])) and j not in DG_P[o].get(i, set([])):
					out = DG_C[o].get(i, set([]))
					out.add(j)
					DG_C[o][i] = out
					out = DG_C[o].get(j, set([]))
					out.add(i)
					DG_C[o][j] = out
			# Take speculative reading into account?
			if not RR_order:
				spreads = speculative_reads[i]
				for k, accs in spreads.items():
					if k not in parallel_statements_enabled_after.get(j, set([])):
						#conflicting_reads_k = set([])
						#conflicting_accesses_j = (set([]), set([]))
						for a in accs:
							accesses = conflicting_accesses(a, False, j)
							#print("conflicting with " + str(k) + ": " + str(accesses))
							if accesses != (set([]),set([])):
								if j not in DG_P[o].get(k, set([])) and k not in DG_P[o].get(j, set([])):
									out = DG_C[o].get(j, set([]))
									out.add(k)
									DG_C[o][j] = out
									out = DG_C[o].get(k, set([]))
									out.add(j)
									DG_C[o][k] = out
									#conflicting_accesses_j[0] |= accesses[0]
									#conflicting_accesses_j[1] |= accesses[1]
									#conflicting_reads_k.add(a)
						# record access conflicts between j and k
						#conflicts = DG_C_spec_conflicts.get(j, {})
						#conflicts[k] = conflicting_accesses_j
						#DG_C_spec_conflicts[j] = conflicts
						#conflicts = DG_C_spec_conflicts.get(k, {})
						#conflicts[j] = (conflicting_reads_k, set([]))
						#DG_C_spec_conflicts[k] = conflicts
	#print("statements: ")
	#for (name, nid) in statements_IDs.items():
	#	print(name + ": " + str(nid))
	#print("P-relation: " + str(DG_P))
	#print("C-relation with spec reads: " + str(DG_C))

def transitive_next_statements(t, trans):
	"""Return set of next statements (transitive w.r.t. local statements)"""
	global statements_accesses
	outgoing = set([])
	out = trans.get(t, set([]))
	for i, tgt in out:
		iid = statements_IDs.get(i)
		if iid == None:
			outgoing |= transitive_next_statements(tgt, trans)
		else:
			outgoing.add(iid)
	return outgoing

def static_constructDG():
	"""Construct the dependency graph of the SLCO model statically"""
	global model, DG_P, DG_C, statements_IDs, statements_accesses, memory_model, SMowner, accesspattern

	# Build dependency graph - P-relation
	DG_P = {}
	for o in model.objects:
		DG_P[o] = {}
		# For each state machine, construct a P-relation.
		for sm in o.type.statemachines:
			# place transitions in dictionary
			trans = {}
			for tr in sm.transitions:
				out = trans.get(tr.source, [])
				out.append((str(o.name) + "'ST'" + str(tr._tx_position), tr.target))
				trans[tr.source] = out
			# now construct P-relation
			for tr in sm.transitions:
				trid = statements_IDs.get(str(o.name) + "'ST'" + str(tr._tx_position))
				if trid != None:
					Pout = transitive_next_statements(tr.target, trans)
					DG_P[o][trid] = Pout

	# Build dependency graph - C-relation
	DG_C = {}
	for o in model.objects:
		DG_C[o] = {}
		# Compare pairs of statements. If conflicting, add a C-edge
		for sm in o.type.statemachines:
			for tr in sm.transitions:
				trid = statements_IDs.get(str(o.name) + "'ST'" + str(tr._tx_position))
				if trid != None:
					for sm2 in o.type.statemachines:
						if sm != sm2:
							for tr2 in sm2.transitions:
								trid2 = statements_IDs.get(str(o.name) + "'ST'" + str(tr2._tx_position))
								if trid2 != None:
									if statements_are_conflicting(trid, trid2):
										# add C-edge
										out = DG_C[o].get(trid, set([]))
										out.add(trid2)
										DG_C[o][trid] = out
	#print("statements: ")
	#for (name, nid) in statements_IDs.items():
	#	print(name + ": " + str(nid))
	#print("P-relation: " + str(DG_P))
	#print("C-relation: " + str(DG_C))

def main(args):
	"""The main function"""
	global modelname, model, memory_model, model_statespace, WR_order, RW_order, RR_order, WW_order, StoreBuffering, WriteAtomicity, critical_cycles, transaction_counter, fence_counter_intra, fence_counter_inter, pure_static_analysis, statements_IDs
	if len(args) == 0:
		print("Missing arguments: SLCO model")
		sys.stdout.flush()
		sys.exit(1)
	else:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 slco2mcrl2")
			print("")
			print("Check given SLCO model for sequentially inconsistent behaviour. For the model, a .aut file containing its state space is required, unless '-s' is used.")
			print(" -w                                    weak memory model to consider (SC,TSO,PSO,ARM)  (default: TSO)")
			print(" -s                                    apply only static analysis (ignore state space) (default: no)")
			sys.stdout.flush()
			sys.exit(0)
		else:
			i = 0
			while i < len(args):
				if args[i] == '-w':
					if args[i+1] == "SC":
						memory_model = MemoryModel.SC
						WR_order = True
						RW_order = True
						RR_order = True
						WW_order = True
						StoreBuffering = False
						WriteAtomicity = True
						print("memory model set to SC")
						sys.stdout.flush()						
					elif args[i+1] == "PSO":
						memory_model = MemoryModel.PSO
						WR_order = False
						RW_order = True
						RR_order = True
						WW_order = False
						StoreBuffering = True
						WriteAtomicity = True
						print("memory model set to PSO")
						sys.stdout.flush()
					elif args[i+1] == "ARM":
						memory_model = MemoryModel.ARM
						WR_order = False
						RW_order = False
						RR_order = False
						WW_order = False
						StoreBuffering = True
						WriteAtomicity = False
						print("memory model set to ARM")
						sys.stdout.flush()
					else:
						memory_model = MemoryModel.TSO
						WR_order = False
						RW_order = True
						RR_order = True
						WW_order = True
						StoreBuffering = True
						WriteAtomicity = True		
						print("memory model set to TSO")
						sys.stdout.flush()
					i += 1
				elif args[i] == '-s':
					pure_static_analysis = True
				else:
					modelname = args[i]
				i += 1

	batch = []
	if modelname.endswith('.slco'):
		batch = [modelname]
	else:
		batch = glob.glob(join(this_folder, modelname, "*.slcotxt"))

	if not batch:
		exit(1)

	#gen_dir = "generated_mcrl2"
	#dir = dirname(batch[0])
	#gen_folder = join(dir, gen_dir)
	#if exists(gen_folder):
	#	rmtree(gen_folder)
	#mkdir(gen_folder)

	for file in batch:
		# read model
		modelname = file
		model = read_SLCO_model(file)
		try:
			print("extracting relevant info from the SLCO model")
			sys.stdout.flush()
			preprocess()
			print_STIDs()
			# read state space
			if not pure_static_analysis:
				print("reading the state space of the SLCO model")
				sys.stdout.flush()
				read_statespace(modelname)
				start = time.time()
				print("analysing the statements occurring in the state space")
				sys.stdout.flush()
				obtain_statements_accesses()
				analyse_statements()
				print("constructing dependency graph")
				sys.stdout.flush()
				constructDG()
			else:
				start = time.time()
				print("statically analysing the statements")
				sys.stdout.flush()
				static_obtain_statements_accesses()
				analyse_statements()
				print("statically constructing dependency graph")
				sys.stdout.flush()
				static_constructDG()
			# flush state space from memory to reduce memory use
			model_statespace = {}
			# print(statements_IDs)
			print("processing critical cycles in dependency graph")
			sys.stdout.flush()
			process_critical_cycles()
			print("report: ")
			print("number of transactions to create: " + str(transaction_counter))
			print("number of fences to place inside statements: " + str(fence_counter_intra))
			print("number of fences to place between statements: " + str(fence_counter_inter))
			sys.stdout.flush()
			end = time.time()
			print("runtime: " + str(end-start))
			sys.stdout.flush()
		except Exception:
			print("failed to process model %s" % basename(file))
			sys.stdout.flush()
			print(traceback.format_exc())
			sys.stdout.flush()

if __name__ == '__main__':
	args = []
	for i in range(1,len(sys.argv)):
		args.append(sys.argv[i])
	main(args)
