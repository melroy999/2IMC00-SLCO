# Race Condition Explorer tool, v0.1, 2017
#
# Update History:
#

# Start of program

import argparse
import logging
import logging.config
import sys
from os.path import dirname, join
import jinja2
from textx.metamodel import metamodel_from_file
from textx.model import get_children_of_type

from help_on_error_argument_parser import HelpOnErrorArgumentParser
from suggestions import *

this_folder = dirname(__file__)

# import libraries
sys.path.append(join(this_folder,'../../libraries'))
from slcolib import *
from SCCTarjan import *
this_folder = dirname(__file__)

# set of actions in the model
actions = set([])

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# *** BEGIN TRANSLATION FILTERS ***

def printstatement(s):
	"""print the given statement"""
	result = ''
	if s.__class__.__name__ == "Assignment":
		result += s.left.var.name
		if s.left.index != None:
			result += "[" + printstatement(s.left.index) + "]"
		result += " := " + printstatement(s.right)
	elif s.__class__.__name__ == "Composite":
		result += "["
		if s.guard != None:
			result += printstatement(s.guard)
			if len(s.assignments) > 1:
				result += ";"
		for i in range(0,len(s.assignments)):
			result += " " + printstatement(s.assignments[i])
			if i < len(s.assignments)-1:
				result += ";"
		result += "]"
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
				result += "[" + printstatement(s.ref.index) + "]"
		else:
			result += '(' + printstatement(s.body) + ')'
	elif s.__class__.__name__ == "VariableRef":
		result += s.var.name
		if s.index != None:
			result += "[" + printstatement(s.index) + "]"
	return result

def variabledeclarations(s, indent):
	"""provide list of variable declarations for given array of variables"""
	previous_base = ""
	previous_size = 0
	output = ""
	if len(s) > 0:
		output = "variables"
	for i in range(0,len(s)):
		# write type?
		if s[i].type.base != previous_base or s[i].type.size != previous_size:
			output += "\n  "
			for j in range (0,indent):
				output += " "
			output += s[i].type.base
			if s[i].type.size > 1:
				output += "[" + str(s[i].type.size) + "]"
			previous_base = s[i].type.base
			previous_size = s[i].type.size
		output += " " + s[i].name
		# add default value?
		if s[i].defvalue != None or s[i].defvalues != []:
			output += " := "
			if s[i].defvalue != None:
				output += str(s[i].defvalue)
			else:
				output += str(s[i].defvalues)
	if output != "":
		for i in range (0,indent):
			output += " "
	return output

# *** END TRANSLATION FILTERS ***

def apply_suggestions(model, suggestions):
	global nr_local_int_vars, nr_local_byte_vars, nr_local_bool_vars

	for o in model.objects:
		for sm in o.type.statemachines:
			nr_local_bool_vars = 0
			nr_local_byte_vars = 0
			nr_local_int_vars = 0
			for tr in sm.transitions:
				# process suggestions
				print(str(tr._tx_position))
				ad = suggestions.get(tuple([o.name, sm.name, "ST'" + str(tr._tx_position)]))
				if ad: # atomicity violations, apply advice
					# first construct program order graph for this statement
					POG = construct_pog(tr.statements[0])
					# next create conflict graph
					CG = construct_cg(ad, POG)
					# determine the SCCs in the union of the two graphs
					uG = merge_graphs(POG, CG)
					SCCs = []
					SCCdict = {}
					identifySCCs(uG, SCCdict, SCCs)
					# create SCC compressed pog and cg
					sccPOG = {}
					for a in POG.keys():
						sccset = sccPOG.get(SCCdict[a], set([]))
						sccPOG[SCCdict[a]] = sccset | set([SCCdict[a1] for a1 in POG[a]]) - set([SCCdict[a]])
					sccCG = {}
					for a in CG.keys():
						sccset = sccPOG.get(SCCdict[a], set([]))
						sccCG[SCCdict[a]] = sccset | set([SCCdict[a1] for a1 in CG[a]]) - set([SCCdict[a]])
					# place the SCCs in a sequence, derived from the compressed graphs. Start with a set of SCCs that do not need
					# to be preceded by others, etc.
					SQ = []
					SCCOpen = set([i for i in range(0, len(SCCs))])
					while len(SCCOpen) > 0:
						tmp = set([])
						for i in SCCOpen:
							if sccPOG.get(i, set([])) & SCCOpen == set([]) and sccCG.get(i, set([])) & SCCOpen == set([]):
								tmp.add(i)
						SQ.append(tmp)
						SCCOpen -= tmp
					print(SQ)
					print(SCCs)
					# break down the statements according to the given access sequence SQ
					for st in tr.statements:
						break_down_statement(st, SQ, SCCs)
				else: # no atomicity violations. If the statement is composite, remove the composite aspect from the model
					newst = []
					for st in tr.statements:
						if st.__class__.__name__ == "Composite":
							if st.guard != None:
								newst.append(st.guard)
							for stt in st.assignments:
								newst.append(stt)
						else:
							newst.append(st)
					tr.statements = newst
			#composites = children_of_type("Composite", t)
			# We assume that a transition only contains one statement.
			# Hence, we can simply assign the new statements
			#if composites:
			#	composite = composites[0]
			#	t.statements = ad.apply(composite)
	
	# We do not support actions
	#model.actions
	# We do not support multiple objects
	#model.objects
	# We do not support channels
	#model.channels
	#globalClass.ports
	return model

def expression(s,primmap):
	"""Rewrites given expression s according to given variable dictionary primmap."""
	output = ''
	# special case: s is a variableref. In this case, it is the left-hand side of an assignment, and s refers to an array
	if s.__class__.__name__ == "Variable":
		output = primmap.get(tuple([s.name]), s)
	if s.__class__.__name__ == "Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2" or s.__class__.__name__ == "ExprPrec1":
		sc = copy.deepcopy(s)
		sc.left = expression(s.left, primmap)
		sc.right = expression(s.right, primmap)
		output = sc
	elif s.__class__.__name__ == "Primary":
		if s.value != None:
			output = s
		elif s.ref != None:
			if s.ref.ref in actions:
				output = s
			else:
				sc = copy.deepcopy(s)
				# is an index to an array given?
				if s.ref.index != None:
					varname = s.ref.ref
					varname_with_index = varname + "[" + expression(s.ref.index,primmap)
					if primmap.get(varname_with_index) != None:
						sc.ref.index = primmap.get(varname_with_index)
				output = primmap.get(tuple([s.ref.ref, sc.ref.index]), s)
		else:
			output = expression(s.body,primmap)
	return output

def merge_graphs(G1, G2):
	"""Merge two graphs G1, G2 into a single one"""
	G3 = G1.copy()
	for v in G2.keys():
		T = G1.get(v, set([]))
		G3[v] = G2[v] | T
	return G3

def construct_pog(s):
	"""Construct program order graph for given statement s"""
	POG = {}
	# obtain sets of variables referenced in statement
	accsets = statement_accesssets(s, False)
	# create mapping between dynamic and static accesses (map dynamic reads to static writes, and dynamic writes to static reads)
	ds_map = {}
	for a1 in accsets[1]:
		aset = set([])
		ca1 = a1
		if ca1[0] == 'r':
			ca1[0] = 'w'
		else:
			ca1[0] = 'r'
		for a2 in accsets[0]:
			if a1 in a2:
				aset.add(a2)
		ds_map[a1] = aset
	# accesses are nodes in graph
	for a in accsets[0]:
		POG[a] = set([])
	for a in accsets[1]:
		POG[a] = set([])
	# derive the edges
	if s.__class__.__name__ == "Assignment":
		wsets = statement_accesssets(s.left, True)
		rsets = statement_accesssets(s.right, False)
		for a1 in wsets[0] | wsets[1]:
			for a2 in rsets[0] | rsets[1]:
				POG[a1].add(a2)
	elif s.__class__.__name__ == "Composite":
		accesses_seen = tuple([set([]), set([])])
		guard_accesses = tuple([set([]), set([])])
		if s.guard != None:
			guard_accesses = statement_accesssets(e.guard, False)
			accesses_seen = mergesets(accesses_seen, guard_accesses)
		for e in s.assignments:
			# ! TO DO! The left can also have reads (to compute indices)
			wsets = statement_accesssets(e.left, True)
			rsets = statement_accesssets(e.right, False)
			# 1. writes are dependent on the reads happening first. if a read accesses a variable previously written to, the current
			# write depends on the same accesses as that previous write
			for a1 in wsets[0] | wsets[1]:
				nset = POG.get(a1, set([]))
				for a2 in rsets[0]:
					# have we seen a previous write to the same variable?
					wa = 'w_' + a2[2:]
					if wa in accesses_seen[0]:
						# current write depends on the same accesses as wa
						aset2 = POG.get(wa, set([]))
						nset |= aset2
					elif 'w_' + a2.split('[')[0] in accesses_seen[1]:
						# current write depends on the same accesses as wa
						wa = 'w_' + a2.split('[')[0]
						aset2 = POG.get(wa, set([]))
						nset |= aset2
					# no write seen. add read access
					else:
						nset.add(a2)
						# 2a. the read also depends on the guard reads
						aset = POG.get(a2, set([]))
						POG[a2] = aset | guard_accesses[0] | guard_accesses[1]
				for a2 in rsets[1]:
					# for a dynamic array read access, we have to assume that earlier dynamic writes coincide, and that this read needs to be performed
					wasets = ds_map.get(a2, set([]))
					isec = wasets & accesses_seen[0]
					for a3 in isec:
						aset3 = POG.get(a3, set([]))
						nset |= aset3
					wa = 'w_' + a2[2:]
					if wa in accesses_seen[1]:
						aset2 = POG.get(wa, set([]))
						nset |= aset2
					# add the access itself
					nset.add(a2)
					# 2b. the access also depends on the guard reads
					aset = POG.get(a2, set([]))
					POG[a2] = aset | guard_accesses[0] | guard_accesses[1]
				POG[a1] = nset
				if a1 in wsets[1]:
					# propagate result also to other matching static accesses
					for ca in ds_map.get(a1, set([])):
						aset = POG.get(ca, set([]))
						aset |= nset
						POG[ca] = aset
			# 3. writes are also dependent on earlier reads from the same variable
			for a1 in wsets[0]:
				if 'r_' + a1[2:] in accesses_seen[0]:
					aset = POG.get(a1, set([]))
					aset.add('r_' + a1[2:])
					POG[a1] = aset
				if 'r_' + a1[2:].split('[')[0] in accesses_seen[1]:
					aset = POG.get(a1, set([]))
					aset.add('r_' + a1[2:].split('[')[0])
					POG[a1] = aset
			for a1 in wsets[1]:
				if 'r_' + a1[2:] in accesses_seen[1]:
					aset = POG.get(a1, set([]))
					aset.add('r_' + a1[2:])
					POG[a1] = aset
				sset = ds_map.get(a1, set([]))
				aset = POG.get(a1, set([]))
				aset |= (sset & accesses_seen1)
				POG[a1] = aset
			# update accesses_seen
			accesses_seen = mergesets(mergesets(accesses_seen, wsets), rsets)
	print("here")
	print(POG)
	return POG

def construct_cg(ad, POG):
	"""Construct a conflict graph based on the given advice/suggestion.
	Use the program order graph to identify allowed nodes."""
	CG = {}

	POGkeys = set([k for k in POG.keys()])

	for a in ad:
		# add access patterns info in ad as cyclic dependencies in graph
		AP = a.get_access_patterns()
		for ap in AP:
			# filter the read set
			newr = set([])
			for ra in ap.r:
				rra = 'r_' + ra
				if rra in POGkeys:
					newr.add(rra)
				if '[' in ra:
					ra_base = 'r_' + ra.split('[')[0]
					if ra_base in POGkeys:
						newr.add(ra_base)
			# filter the write set
			neww = set([])
			for wa in ap.w:
				wwa = 'w_' + wa
				if wwa in POGkeys:
					neww.add(wwa)
				if '[' in wa:
					wa_base = 'w_' + wa.split('[')[0]
					if wa_base in POGkeys:
						neww.add(wa_base)
			for ra in ap.r:
				rra = 'r_' + ra
				if rra in POGkeys:
					aset = CG.get(rra, set([]))
					CG[rra] = aset | neww
				if '[' in ra:
					ra_base = 'r_' + ra.split('[')[0]
					if ra_base in POGkeys:
						aset = CG.get(ra_base, set([]))
						CG[ra_base] = aset | neww
			for wa in ap.w:
				wwa = 'w_' + wa
				if wwa in POGkeys:
					aset = CG.get(wwa, set([]))
					CG[wwa] = aset | newr
				if '[' in wa:
					wa_base = 'w_' + wa.split('[')[0]
					if wa_base in POGkeys:
						aset = CG.get(wa_base, set([]))
						CG[wa_base] = aset | newr
		# add shuffle suggestions as acyclic dependencies in graph
		# to do: fix for dynamic accesses!
		SH = a.get_shuffles()
		for sh in SH:
			# create set of 'before' accesses
			bset = set(['r_' + ra for ra in sh.before.r]) | set(['w_' + wa for wa in sh.before.w])
			for ra in sh.after.r:
				aset = CG.get('r_' + ra, set([]))
				CG['r_' + ra] = aset | bset
			for wa in sh.after.w:
				aset = CG.get('w_' + wa, set([]))
				CG['w_' + wa] = aset | bset
	print(CG)
	return CG

def break_down_statement(st, SQ, SCCs):
	"""Break down the given statement st based on the access sequence SQ"""

	# dictionary mapping object-local to state machine-local variables
	vardict = {}
	# dictionary to look up right-hand side in assignment for a given variable that was written to
	assdict = {}
	# new block of statements
	newblock = []
	# iterate over the sets in SQ:
	for aset in SQ:
		# iterate over the SCCs in aset
		for scc in aset:
			# create statement (possibly composite if size > 1) for scc and add to new block
			newblock.append(create_statement(scc, SCCs, vardict, assdict, st))

def find_variable(vname, V):
	"""Find given variable in list of variables"""
	for v in V:
		if vname == v.name:
			return v
	return null

def create_statement(scc, SCCs, vardict, assdict, st):
	"""Return new statement(s) for the given scc of accesses, based on the given statement st"""
	alist = [a for a in SCCs[scc][1].keys()]
	# sort list to get the reads in front
	alist = sorted(alist)

	c = st.parent.parent.parent
	sm = st.parent.parent
	tr = st.parent

	# link to the appropriate parent
	sparent = tr
	if len(alist) > 1:
		sparent = Composite(tr, None, [])

	for a in alist:
		# if the variable is object-local, find it
		varname = a[2:]
		v = find_variable(varname, c.variables)
		if v != null:
			if a[0] == 'r':
				# create a read instruction
				# create VariableRef to var
				vr = VariableRef(sparent, v, None)
				asright = create_var_expression(sparent, v.name, '')
				if len(alist) > 1:
					sparent.assignments.append(Assignment(sparent, asleft, asright))
				# add association with local variable to vardict
				if v.index != None:
					if RepresentsInt(v.index):
						varname += "[" + str(v.index) + "]"
				vardict[varname] = vl
			else:
				# this is a write access. construct a new assignment reading from local variables and writing to the associated object-local variable
				# first find the last assignment to the given variable
				if st.__class__.__name__ == "Assignment":
					lastas = st
				elif st.__class__.__name__ == "Composite":
					for i in reversed(range(0, len(st.assignments))):
						if st.assignments[i].left.name == varname

# def break_down_statements(tr, sm, o):
# 	"""Break down the statements of the given transition tr. Involves transforming those statements into a new statement block,
# 	in which each statement only accesses a single global variable, and at most once (read or write).
# 	sm and o are the owning state machine and object, respectively"""
# 	global nr_local_int_vars, nr_local_byte_vars, nr_local_bool_vars

# 	livarcount = 0
# 	lbvarcount = 0
# 	lbyvarcount = 0
# 	# build vardict for the use of local variables
# 	vardict = {}
# 	for s in tr.statements:
# 		vars = statement_varset(s)
# 		for v in vars:
# 			if vardict.get(v) == None:
# 				# lookup referenced variable to figure out type
# 				tt = ''
# 				for v2 in o.type.variables:
# 					if v2.name == v:
# 						tt = v2.type
# 						break
# 				# if we have not found a type, the variable is state machine-local, therefore not relevant
# 				if tt == 'Integer':
# 					vardict[v] = "itmp'" + str(livarcount)
# 					livarcount += 1
# 				elif tt == 'Boolean':
# 					vardict[v] = "btmp'" + str(lbvarcount)
# 					lbvarcount += 1
# 				elif tt == 'Byte':
# 					vardict[v] = "bytmp'" + str(lbyvarcount)
# 					lbyvarcount += 1
# 	# now that we have assigned local vars to global vars, create new block of statements
# 	newstat = []
# 	for s in tr.statements:
# 		if s.guard != None:
# 			vars = statement_varset(s.guard)
# 			for v in vars:
# 				ns = Assignment(tr, )

def mergesets(S1, S2):
	"""Merge the two given tuples of sets"""
	return tuple([S1[0]|S2[0], S1[1]|S2[1]])

def mergesets_filterreads(S1, S2):
	"""Merge the two given tuples of sets, filtering out new read accesses if corresponding write accesses already exist"""
	new0 = S1[0]
	for s2 in S2[0]:
		if 'w_' + s2[2:] not in new0:
			new0.add(s2)
	return tuple([new0, S1[1]|S2[1]])

def accesstype(b):
	if b:
		return "w"
	else:
		return "r"

def statement_accesssets(s, wflag):
	"""Produces two sets of variable accesses referenced in given SLCO statement, the first for static accesses and the second for dynamic array accesses.
	wflag indicates whether we are currently checking write accesses or not."""
	global actions
	output = tuple([set([]), set([])])

	if s.__class__.__name__ == "Assignment":
		output = mergesets_filterreads(output, statement_accesssets(s.right, False))
		output = mergesets(output, statement_accesssets(s.left, True))
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			output = mergesets(output, statement_accesssets(s.guard, False))
		for a in s.assignments:
			output = mergesets(output, statement_accesssets(a, False))
	elif s.__class__.__name__ == "ReceiveSignal":
		for p in s.params:
			output = mergesets(output, statement_accesssets(p), True)
		if s.guard != None:
			output = mergesets(output, statement_accesssets(s.guard, False))
	elif s.__class__.__name__ == "SendSignal":
		for p in s.params:
			output = mergesets(output, statement_accesssets(p, False))
	elif s.__class__.__name__ == "VariableRef":
		if s.index != None:
			if RepresentsInt(s.index):
				output[0].add(accesstype(wflag) + "_" + s.var.name + "[" + s.index + "]")
			else:
				output[1].add(accesstype(wflag) + "_" + s.var.name)
				output = mergesets(output, statement_accesssets(s.index, False))
		else:
			output[0].add(accesstype(wflag) + "_" + s.var.name)
	elif s.__class__.__name__ == "Delay":
		output = tuple([set([]), set([])])
	elif s.__class__.__name__ != "Primary":
		output = mergesets(output, statement_accesssets(s.left, False))
		if s.op != '':
			output = mergesets(output, statement_accesssets(s.right, False))
	else:
		if s.ref != None:
			if s.ref.ref not in actions:
				if s.ref.index != None:
					if RepresentsInt(s.ref.index):
						output[0].add(accesstype(wflag) + "_" + s.ref.ref + "[" + s.ref.index + "]")
					else:
						output[1].add(accesstype(wflag) + "_" + s.ref.ref)
						output = mergesets(output, statement_accesssets(s.ref.index, False))
				else:
					output[0].add(accesstype(wflag) + "_" + s.ref.ref)
		if s.body != None:
			output = mergesets(output, statement_accesssets(s.body, False))
	return output

#def combine_varsets(O,N):
#	"""Combine two lists of varsets"""
#	return [O[0] | N[0], O[1] | N[1]]
#
#def statement_varsets(s):
#	"""Produces sets of variables referenced in given SLCO statement, one for read accesses, one for write accesses"""
#	global actions
#	output = [set([]), set([])]
#	if s.__class__.__name__ == "Assignment":
#		output = combine_varsets(output, statement_varsets(s.left))
#		output = combine_varsets(output, statement_varsets(s.right))
#	elif s.__class__.__name__ == "Composite":
#		if s.guard != None:
#			output = combine_varsets(output, statement_varsets(s.guard))
#		for a in s.assignments:
#			output = combine_varsets(output, statement_varsets(a))
#	elif s.__class__.__name__ == "ReceiveSignal":
#		for p in s.params:
#			output = combine_varsets(output, statement_varsets(p))
#		if s.guard != None:
#			output = combine_varsets(output, statement_varsets(s.guard))
#	elif s.__class__.__name__ == "SendSignal":
#		for p in s.params:
#			output = combine_varsets(output, statement_varsets(p))
#	elif s.__class__.__name__ == "VariableRef":
#		# VariableRefs always indicate write accesses
#		output[1].add(s.var.name)
#		if s.index != None:
#			output = combine_varsets(output, statement_varsets(s.index))
#	elif s.__class__.__name__ == "Delay":
#		output |= [set([]), set([])]
#	elif s.__class__.__name__ != "Primary":
#		output = combine_varsets(output, statement_varsets(s.left))
#		if s.op != '':
#			output = combine_varsets(output, statement_varsets(s.right))
#	else:
#		if s.ref != None:
#			if s.ref.ref not in actions:
#				output[0].add(s.ref.ref)
#	return output

def translate(model, suggestions, path):
	"""The translation function"""
	
	# Initialize the template engine.
	jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(join(this_folder,'../../jinja2_templates')), trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols','jinja2.ext.do',])
	print(join(this_folder,'../../jinja2_templates'))

	# Register the filters
	jinja_env.filters['printstatement'] = printstatement
	jinja_env.filters['variabledeclarations'] = variabledeclarations
	
	# Register the tests

	# apply selected transformations
	model = apply_suggestions(model, suggestions)
	# load the SLCO template
	template = jinja_env.get_template('slco.jinja2template')
	out = template.render(model=model)
	# write new SLCO model
	outFile = open(path, 'w')
	outFile.write(out)
	outFile.close()

def main():
	global actions

	# setup logging
	file_handler    = logging.FileHandler(filename='slco_to_al_slco.log', mode='w')
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
	parser.add_argument("-o", "--out", type=str, default="INPUT_MODEL", help="Output AL-SLCO model")
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
		out_path = slco_path[:-5] + '.alslco'
	elif not out_path.endswith('.alslco'):
		out_path = out_path + '.alslco'
	
	logging.info('Starting SLCO Atomicity-Level model transformation')
	logging.info('Input model       : %s', slco_path)
	logging.info('Input suggestions : %s', sugg_path)
	logging.info('Output File       : %s', out_path)

	# read model
	model = read_SLCO_model(slco_path)

	suggestions = parse_suggestions(sugg_path)
	# translate
	out_model = translate(model, suggestions, out_path)
	
	logging.info('Finished, output written to %s', out_path)
	logging.shutdown()


if __name__ == '__main__':
	main()
