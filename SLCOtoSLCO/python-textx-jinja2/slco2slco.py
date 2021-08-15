import sys
from os import mkdir
from os.path import exists, dirname, basename, join, split
import jinja2
from textx.metamodel import metamodel_from_file
from copy import deepcopy, copy
import itertools
import glob
import traceback

modelname = ""
model = ""
this_folder = dirname(__file__)

# import libraries
sys.path.append(join(this_folder,'../../libraries'))
from slcolib import *
from SCCTarjan import *
this_folder = dirname(__file__)

check_variables = False
check_repeated = False
check_sequences = False
check_simple = False

# set of actions in model
actions = set([])
# dictionary to look up state machine owning a given statement
statemachine = {}
# dictionary to look up transition owning a given statement
tr = {}

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
			if len(s.assignments) > 0:
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

# *** BEGIN TRANSLATION TESTS ***

# *** END TRANSLATION TESTS ***

def statement_varset(s):
	"""Produces set of variables referenced in given SLCO statement"""
	global actions
	output = set([])
	if s.__class__.__name__ == "Assignment":
		output |= statement_varset(s.left)
		output |= statement_varset(s.right)
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			output |= statement_varset(s.guard)
		for a in s.assignments:
			output |= statement_varset(a)
	elif s.__class__.__name__ == "ReceiveSignal":
		for p in s.params:
			output |= statement_varset(p)
		if s.guard != None:
			output |= statement_varset(s.guard)
	elif s.__class__.__name__ == "SendSignal":
		for p in s.params:
			output |= statement_varset(p)
	elif s.__class__.__name__ == "VariableRef":
		output.add(s.var.name)
		if s.index != None:
			output |= statement_varset(s.index)
	elif s.__class__.__name__ == "Delay":
		output |= set([])
	elif s.__class__.__name__ != "Primary":
		output |= statement_varset(s.left)
		if s.op != '':
			output |= statement_varset(s.right)
	else:
		if s.ref != None:
			if s.ref.ref not in actions:
				output.add(s.ref.ref)
				if s.ref.index != None:
					output |= statement_varset(s.ref.index)
		if s.body != None:
			output |= statement_varset(s.body)
	return output

def check_vars(m):
	"""check the use of variables in the model, and improve where needed"""
	# build a dictionary providing sets of state machine-local variables
	smvars = {}
	for c in model.classes:
		smvars[c] = {}
		for sm in c.statemachines:
			smvars[c][sm] = set([])
			for v in sm.variables:
				smvars[c][sm].add(v.name)
	# keep track of state machines referencing class variables, and if state machine-local variables are referenced
	classvars_references = {}
	smvars_referenced = {}
	for c in model.classes:
		smvars_referenced[c] = {}
		classvars_references[c] = {}
		for sm in c.statemachines:
			smvars_referenced[c][sm] = set([])
			varset = set([])
			for tr in sm.transitions:
				for stat in tr.statements:
					varset |= statement_varset(stat)
			# process varset
			for v in varset:
				# if the variable is state machine-local, add it to smvars_referenced, otherwise add a reference to classvars_references
				if v in smvars[c][sm]:
					smvars_referenced[c][sm].add(v)
				else:
					smset = classvars_references[c].get(v,set([]))
					smset.add(sm)
					classvars_references[c][v] = smset
	# change model to take result into account
	# remove variables that are not referenced at all
	for c in m.classes:
		newvars = []
		newsmvars = {}
		for v in c.variables:
			refset = classvars_references[c].get(v.name,set([]))
			if len(refset) < 2:
				# changes need to be applied
				if len(refset) == 1:
					# move variable to var list of state machine referencing the variable
					sm = refset.pop()
					varset = newsmvars.get(sm,set([]))
					varset.add(v)
					newsmvars[sm] = varset
			else:
				# keep variable in list
				newvars.append(v)
		# set new variable list
		c.variables = newvars
		for sm in c.statemachines:
			newvars = []
			for v in sm.variables:
				if v.name in smvars_referenced[c][sm]:
					newvars.append(v)
			for v in newsmvars.get(sm,set([])):
				newvars.append(v)
			sm.variables = newvars

def check_repeatedwrites(m):
	"""Check for repeated writes in a composite statement and remove these by using state machine-local variables.
	Note: dynamic accesses to the same array cell may not be detected as equal."""
	newlocalvars = {}
	for c in m.classes:
		for sm in c.statemachines:
			for tr in sm.transitions:
				for st in tr.statements:
					if st.__class__.__name__ == "Composite":
						vardict = {}
						for e in st.assignments:
							varname = e.left.var.name
							if e.left.index != None:
								if RepresentsInt(e.left.index):
									varname += "[" + str(e.left.index) + "]"
							vardict[varname] = vardict.get(varname, 0) + 1
						# local variables needed:
						icounter = 0
						bcounter = 0
						bycounter = 0
						writedict = {}
						for e in st.assignments:
							# check right hand side
							e.right = replace_read_vars(e.right, writedict)
							# check left hand side
							varname = e.left.var.name
							if e.left.index != None:
								if RepresentsInt(e.left.index):
									varname += "[" + str(e.left.index) + "]"
								else:
									# skip this variable
									continue
							if vardict[varname] > 1:
								# create new variable object?
								if e.left.var.type.base == "Integer":
									v = newlocalvars.get("itmp'" + str(icounter))
									if v == None:
										v = create_smlocal_var(sm, "itmp'" + str(icounter), "Integer", 1)
										sm.variables.append(v)
										newlocalvars["itmp'" + str(icounter)] = v
									icounter += 1
								if e.left.var.type.base == "Boolean":
									v = newlocalvars.get("btmp'" + str(bcounter))
									if v == None:
										v = create_smlocal_var(sm, "btmp'" + str(bcounter), "Boolean", 1)
										sm.variables.append(v)
										newlocalvars["btmp'" + str(bcounter)] = v
									bcounter += 1
								if e.left.var.type.base == "Byte":
									v = newlocalvars.get("bytmp'" + str(bycounter))
									if v == None:
										v = create_smlocal_var(sm, "bytmp'" + str(bycounter), "Byte", 1)
										sm.variables.append(v)
										newlocalvars["bytmp'" + str(bycounter)] = v
									bycounter += 1
								# replace write access
								e.left.var = v
								e.left.index = None
								# keep track of the fact that v has been inserted in the statement
								writedict[varname] = v
								# decrement number of writes of this variable
								vardict[varname] -= 1

def replace_read_vars(s, W):
	"""Replace read occurrences of variables in W in s with their associated variable"""
	global actions
	if s.__class__.__name__ == "Assignment":
		s.right = replace_read_vars(s.right, W)
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			s.guard = replace_read_vars(s.guard, W)
		newasgn = []
		for a in s.assignments:
			newasgn.append(replace_read_vars(a, W))
		s.assignments = newasgn
	elif s.__class__.__name__ == "ReceiveSignal":
		if s.guard != None:
			s.guard = replace_read_vars(s.guard, W)
	elif s.__class__.__name__ == "SendSignal":
		newps = []
		for p in s.params:
			newps.append(replace_read_vars(p, W))
	elif s.__class__.__name__ == "VariableRef":
		# varname = s.var.name
		# if s.index != None:
		# 	if RepresentsInt(s.index):
		# 		varname += "[" + str(s.index) + "]"
		# 	else:
		# 		s.index = replace_vars(s.index, W)
		# 		return s
		# if W.get(varname) != None:
		# 	s.var = W.get(varname)
		# 	s.index = None
		return s
	elif s.__class__.__name__ == "Delay":
		return s
	elif s.__class__.__name__ != "Primary":
		s.left = replace_read_vars(s.left, W)
		if s.op != '':
			s.right = replace_read_vars(s.right, W)
	else:
		if s.ref != None:
			if s.ref.ref not in actions:
				varname = s.ref.ref
				if s.ref.index != None:
					if RepresentsInt(s.ref.index):
						varname += "[" + str(s.ref.index) + "]"
					else:
						s.ref.index = replace_read_vars(s.ref.index, W)
						return s
				if W.get(varname) != None:
					s.ref.ref = W.get(varname).name
					s.index = None
		if s.body != None:
			s.body = replace_read_vars(s.body, W)
	return s

def statement_is_blocking(s):
	"""Is the given statement blocking or not?"""
	if s.__class__.__name__ == "Assignment":
		return False
	elif s.__class__.__name__ == "Composite":
		if s.guard == None:
			return False
		else:
			return True
	else:
		return True

def combine_trans(m):
	"""Combine transitions, i.e., statements associated with them, that are placed in sequence without any transitions branching off"""
	"""Also make sure no blocking statements appear in the middle or end of a statement block"""
	# first add additional states and transitions to remove blocking statements from blocks
	for c in m.classes:
		for sm in c.statemachines:
			newstatecount = 0
			for tr in sm.transitions:
				first = True
				newblocks = []
				newblock = []
				for stat in tr.statements:
					if not first:
						if statement_is_blocking(stat):
							newblocks.append(newblock)
							newblock = [stat]
					else:
						newblock.append(stat)
						first = False
				newblocks.append(newblock)
				if len(newblocks) > 1:
					tr.statements = newblocks[0]
					tr1 = tr
					for i in range(1,len(newblocks)):
						tr2 = copy(tr1)
						# new state
						snew = copy(tr1.target)
						snew.name = "NewState" + str(newstatecount)
						newstatecount += 1
						sm.states.append(snew)
						tr2.target = tr1.target
						tr1.target = snew
						tr2.source = snew
						tr2.statements = newblocks[i]
						sm.transitions.append(tr2)
	# next, combine transitions where possible
	for c in m.classes:
		for sm in c.statemachines:
			trcount = {}
			transdict = {}
			singletransstateset = set([])
			# detect which states have a single, non-blocking outgoing transition
			for tr in sm.transitions:
				count = trcount.get(tr.source, 0)
				count += 1
				trcount[tr.source] = count
			# iterate again over the transitions, and store transition blocks linked to transitions that are the only transition of their source
			for tr in sm.transitions:
				if trcount[tr.source] == 1:
					if tr.statements == [] or not statement_is_blocking(tr.statements[0]):
						transdict[tr.source] = tr
						singletransstateset.add(tr.source)
			# combine transitions that have been selected as much as possible
			for s in singletransstateset:
				tr1 = transdict[s]
				while tr1.target in singletransstateset:
					tr1.statements += transdict[tr1.target].statements
					tr1.target = transdict[tr1.target].target
			# next, combine transitions
			for tr in sm.transitions:
				if tr.target in singletransstateset:
					tr.statements += transdict[tr.target].statements
					tr.target = transdict[tr.target].target
			# finally, remove selected states and transitions
			newstates = []
			for s in sm.states:
				if s not in singletransstateset or s == sm.initialstate:
					newstates.append(s)
			sm.states = newstates
			transtoremove = set([])
			for s, tr in transdict.items():
				transtoremove.add(tr)
			newtransitions = []
			for tr in sm.transitions:
				if tr not in transtoremove:
					newtransitions.append(tr)
			sm.transitions = newtransitions

	# for c in m.classes:
	# 	for sm in c.statemachines:
	# 		trcount = {}
	# 		transdict = {}
	# 		singletransstateset = set([])
	# 		for tr in sm.transitions:
	# 			count = trcount.get(tr.source, 0)
	# 			count += 1
	# 			trcount[tr.source] = count
	# 		# iterate again over the transitions, and store transition blocks linked to transitions that are the only transition of their source
	# 		for tr in sm.transitions:
	# 			if trcount[tr.source] == 1:
	# 				transdict[tr.source] = tr
	# 				singletransstateset.add(tr.source)
	# 		# combine transitions in the selected set, for as long as possible
	# 		updated = True
	# 		while updated:
	# 			updated = False
	# 			removestates = set([])
	# 			for s in singletransstateset:
	# 				if transdict[s].target in singletransstateset:
	# 					transdict[s].statements += transdict[transdict[s].target].statements
	# 					removestates.add(transdict[s].target)
	# 					transdict[s].target = transdict[transdict[s].target].target

def make_simple(m):
	"""Make the given SLCO model simple, i.e., up to one statement per transition"""
	for c in m.classes:
		for sm in c.statemachines:
			newstatecount = 0
			for tr in sm.transitions:
				if len(tr.statements) > 1:
					newtransitions = []
					target = tr.target
					snew1 = copy(tr.target)
					snew1.name = "SimpleState" + str(newstatecount)
					newstatecount += 1
					sm.states.append(snew1)
					tr.target = snew1
					for i in range(1,len(tr.statements)):
						trnew = copy(tr)
						if i < len(tr.statements)-1:
							snew2 = copy(tr.target)
							snew2.name = "SimpleState" + str(newstatecount)
							newstatecount += 1
							sm.states.append(snew2)
						else:
							snew2 = target
						trnew.source = snew1
						trnew.target = snew2
						trnew.statements = [tr.statements[i]]
						trnew.priority = 0
						newtransitions.append(trnew)
						snew1 = snew2
					# update statement list of tr
					tr.statements = [tr.statements[0]]
					# update transition list
					sm.transitions += newtransitions

# TODO: removal of 'dead model': when using transition priorities, lower priority alternatives to unguarded transitions will never be fired, and can be safely removed.

def preprocess():
	global model, actions, statemachine, tr

	# build dictionaries providing for a given statement the state machine and transition owning it
	for c in model.classes:
		for stm in c.statemachines:
			for trn in stm.transitions:
				for stat in trn.statements:
					statemachine[stat] = stm
					tr[stat] = trn

def translate():
	"""The translation function"""
	global modelname, model, check_variables, check_sequences, check_repeated, check_simple
	
	path, name = split(modelname)
	if name.endswith('.slco'):
		name = name[:-5]
	else:
		name = name[:-8]
	outFile = open(join(path,name + "_new.slco"), 'w')

	# Initialize the template engine.
	jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(join(this_folder,'../../jinja2_templates')), trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols','jinja2.ext.do',])

	# Register the filters
	jinja_env.filters['printstatement'] = printstatement
	jinja_env.filters['variabledeclarations'] = variabledeclarations
	
	# Register the tests

	# apply selected transformations
	if check_variables:
		check_vars(model)
	if check_repeated:
		check_repeatedwrites(model)
	if check_sequences:
		combine_trans(model)
	if check_simple:
		make_simple(model)
	# load the SLCO template
	template = jinja_env.get_template('slco.jinja2template')
	out = template.render(model=model)
	# write new SLCO model
	outFile.write(out)
	outFile.close()

def main(args):
	"""The main function"""
	global modelname, model, check_variables, check_sequences, check_simple, check_repeated
	if len(args) == 0:
		print("Missing argument: SLCO model")
		sys.exit(1)
	else:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 slco2slco.py [-vrsc]")
			print("")
			print("Transform an SLCO 2.0 model to an SLCO 2.0 model.")
			print("")
			print(" -v                    transform use of variables: remove non-referenced ones and make variables that are not used by multiple state machines state machine-local")
			print(" -r                    transform use of variables: remove repeated writing to variables in composite statements, by introducing state machine-local variables")
			print(" -s                    transform an SLCO model to a 'simple' SLCO model (up to one statement per transition)")
			print(" -c                    combine transitions that model a sequence of statements")
			sys.exit(0)
		else:
			for i in range(0,len(args)):
				if args[i] == '-v':
					check_variables = True
				elif args[i] == '-r':
					check_repeated = True
				elif args[i] == '-c':
					check_sequences = True
				elif args[i] == '-s':
					check_simple = True
				else:
					modelname = args[i]

	batch = []
	if modelname.endswith('.slco'):
		batch = [modelname]
	else:
		batch = glob.glob(join(this_folder, modelname, "*.slco"))

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
		print("processing model %s" % basename(file))
		try:
			preprocess()
			# translate
			translate()
		except Exception:
			print("failed to process model %s" % basename(file))
			print(traceback.format_exc())

if __name__ == '__main__':
	args = []
	for i in range(1,len(sys.argv)):
		args.append(sys.argv[i])
	main(args)
