import sys
from os import mkdir
from os.path import exists, dirname, basename, join, split
import jinja2
from textx.metamodel import metamodel_from_file
from copy import deepcopy
import itertools
import glob
import traceback

this_folder = dirname(__file__)

# import libraries
sys.path.append(join(this_folder,'../../libraries'))
from slcolib import *
from SCCTarjan import identifySCCs
this_folder = dirname(__file__)

modelname = ""
model = ""

# apply only static analysis
static_analysis_only = False

# type of ports
porttypes = {}
# statemachine names used in the model
statemachinenames = set([])
# set of state names used in the model
states = set([])
# dictionary to look up state machine owning a given statement
statemachine = {}
# dictionary to look up class owning a given state machine
smclass = {}
# dictionary to look up transition owning a given statement
trowner = {}
# dictionary providing the type of all model variables (with class-based scopes)
vartypes = {}
# per class / statemachine, give a dictionary, making the scope of variables explicit
scopedvars = {}
# per class / statemachine, give a set of local variables
smlocalvars = {}
# transitions of state machines stored in dictionary per class/state machine/source state
trans = {}
# the summands for the output mCRL2 model
mcrl2summands = []

# Lists of occurring array sizes, both of type Int and Bool
IntArraySizes = []
BoolArraySizes = []

mcrl2varprefix = "var_"

# the type of the involved channels
channeltypes = {}
asynclosslesstypes = set([])
asynclossytypes = set([])
synctypes = set([])

# dictionary to look up channel linked to a given (object) statement
ochannel = {}
# set of pairs of communicating objects
object_sync_commpairs = set([])

# dictionary providing the access patterns of statements
statement_access = {}
statement_condition_access = {}
statement_full_condition_access = {}
statement_structure_access = {}

# statements that require syncing
syncing_statements = {}

# sorted list of variables in the model
sorted_variables = []
# sorted objects
sorted_objects = []
# sorted_statemachines
sorted_statemachines = []

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def hasdynamicaddressing(s, sm):
	"""Returns whether the given statement s contains dynamic array addressing"""
	global actions

	if s.__class__.__name__ == "Composite":
		if s.guard != None:
			result = hasdynamicaddressing(s.guard, sm)
			if result:
				return True
		for e in s.assignments:
			result = hasdynamicaddressing(e.right, sm)
			if result:
				return True
			if e.left.index != None:
				if expression_usedvars(e.left.index, sm, sm.parent, sm.parent) != []:
					return True
	elif s.__class__.__name__ == "Delay":
		return False
	elif s.__class__.__name__ == "SendSignal":
		for p in s.params:
			result = hasdynamicaddressing(p, sm)
			if result:
				return True
	elif s.__class__.__name__ == "ReceiveSignal":
		if s.guard != None:
			result = hasdynamicaddressing(s.guard, sm)
			if result:
				return True
		for p in s.params:
			result = hasdynamicaddressing(p, sm)
			if result:
				return True
	elif s.__class__.__name__ == "Assignment":
		result = hasdynamicaddressing(s.left, sm)
		if result:
			return True
		result = hasdynamicaddressing(s.right, sm)
		if result:
			return True
	elif s.__class__.__name__ == "VariableRef":
		if s.index != None:
			if expression_usedvars(s.index, sm, sm.parent, sm.parent) != []:
				return True
	elif s.__class__.__name__ != "Primary":
		result = hasdynamicaddressing(s.left, sm)
		if result:
			return True
		if s.op != '':
			result = hasdynamicaddressing(s.right, sm)
			if result:
				return True
	else:
		if s.ref != None:
			if s.ref.ref not in actions:
				if s.ref.index != None:
					if expression_usedvars(s.ref.index, sm, sm.parent, sm.parent) != []:
						return True
		if s.body != None:
			result = hasdynamicaddressing(s.body, sm)
			if result:
				return result
	return False

def hascondition(s, o):
	"""Returns if the given statement has a condition. o is the Object owning s.
	A statement has a condition if it is guarded, or there are statements with a higher priority."""
	global ochannel, trans
	result = False
	if s.__class__.__name__ == "Expression":
		result = (not expression_is_actionref(s))
	elif s.__class__.__name__ == "SendSignal":
		result = (ochannel[o][s].synctype == "async")
	elif s.__class__.__name__ == "ReceiveSignal":
		if s.guard != None:
			result = True
		else:
			return ochannel[o][s].synctype == "async"
	elif s.__class__.__name__ == "Composite":
		result = (s.guard != None)
	if not result:
		# Check for higher priority transitions
		prio = s.parent.priority
		outgoing = trans[o.type][s.parent.parent][s.parent.source]
		for t in outgoing:
			if t.priority < prio:
				return True
	return result

def variabledefault(s):
	""" return default value for given variable """
	if s.defvalue != None:
		return s.defvalue
	elif s.defvalues != []:
		if s.type.base == 'Integer' or s.type.base == 'Byte':
			defv = "AI"
		else:
			defv = "AB"
		defv += str(s.type.size) + "'("
		first = True
		for v in s.defvalues:
			if not first:
				defv += ','
			else:
				first = False
			defv += str(v)
		defv += ')'
		return defv
	elif s.type.base == 'Integer' or s.type.base == 'Byte':
		if s.type.size < 2:
			return '0'
		else:
			type = "AI" + str(s.type.size) + "'("
			first = True
			for i in range(0,s.type.size):
				if not first:
					type += ','
				else:
					first = False
				type += '0'
			type += ')'
			return type
	elif s.type.base == 'Boolean':
		if s.type.size < 2:
			return 'true'
		else:
			type = "AB" + str(s.type.size) + "'("
			first = True
			for i in range(0,s.type.size):
				if not first:
					type += ','
				else:
					first = False
				type += 'true'
			type += ')'
			return type

def hasvariables(o):
	""" Test to check whether given object has object variables"""
	return len(o.type.variables) > 0

def hasindex(v):
	""" Test to check whether given (variable / variable scope name) pair has an index"""
	return v[0].type.size > 1

def hasnoindex(v):
	""" Test to check whether given (variable / variable scope name) pair has no index"""
	return v[0].type.size <= 1

def datatypeacronym(s):
	""" return acronym for given data type """
	if s == 'Int':
		return 'I'
	elif s == 'Bool':
		return 'B'
	elif "Int" in s and len(s) > 3:
		return 'LI'
	elif "Bool" in s and len(s) > 4:
		return 'LB'

def mcrl2type(s):
	"""Maps type names from SLCO to mCRL2"""
	if s.base == 'Integer' or s.base == 'Byte':
		if s.size < 2:
			return 'Int'
		else:
			return 'Int' + str(s.size)
	elif s.base == 'Boolean':
		if s.size < 2:
			return 'Bool'
		else:
			return 'Bool' + str(s.size)

def mcrl2value(s):
	"""Maps values from SLCO to mCRL2 values"""
	if s == "True" or s == "False" or s == True or s == False:
		return str(s).lower()
	else:
		return s

def mcrl2typetuple(s):
	"""Maps list of type names from SLCO to mCRL2"""
	newtype = []
	for t in s:
		newtype.append(mcrl2type(t))
	return tuple(newtype)

def operator(s):
	"""Maps SLCO expression operators to mCRL2 operators"""
	if s == '=':
		return '=='
	elif s == '!=':
		return '!='
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

def mcrl2_model_vartypes(m, startsep):
	"""Return a string containing the types of the variables in the given model, mCRL2 style.
	startsep indicates whether we should start with a separator or not"""
	result = ""
	first = True
	for o in model.objects:
		for v in o.type.variables:
			if not first or startsep:
				result += " # "
			else:
				first = False
			result += mcrl2type(v.type)
		for sm in o.type.statemachines:
			for v in sm.variables:
				if not first or startsep:
					result += " # "
				else:
					first = False
				result += mcrl2type(v.type)
	for ch in model.channels:
		if ch.synctype == 'async':
			if not first or startsep:
				result += " # "
			else:
				first = False
			result += "List(M'"
			for t in mcrl2typetuple(ch.type):
				result += datatypeacronym(t)
			result += ") # Int"
	return result

def mcrl2_model_vars(m, startsep):
	"""Return a string containing the names of the variables in the given model, mCRL2 style.
	startsep indicates whether we should start with a separator or not"""
	result = ""
	first = True
	for o in model.objects:
		for v in o.type.variables:
			if not first or startsep:
				result += ", "
			else:
				first = False
			result += o.name + "'" + v.name
		for sm in o.type.statemachines:
			for v in sm.variables:
				if not first or startsep:
					result += ", "
				else:
					first = False
				result += o.name + "'" + sm.name + "'" + v.name
	for ch in model.channels:
		if ch.synctype == 'async':
			if not first or startsep:
				result += ", "
			else:
				first = False
			result += ch.name + "'buffer, " + ch.name + "'size"
	return result

def mcrl2_model_vars_with_types(m, startsep):
	"""Return a string containing the names of the variables in the given model with their types, mCRL2 style.
	startsep indicates whether we should start with a separator or not"""
	result = ""
	first = True
	for o in model.objects:
		for v in o.type.variables:
			if not first or startsep:
				result += ", "
			else:
				first = False
			result += o.name + "'" + v.name + ": " + mcrl2type(v.type)
		for sm in o.type.statemachines:
			for v in sm.variables:
				if not first or startsep:
					result += ", "
				else:
					first = False
				result += o.name + "'" + sm.name + "'" + v.name + ": " + mcrl2type(v.type)
	for ch in model.channels:
		if ch.synctype == 'async':
			if not first or startsep:
				result += ", "
			else:
				first = False
			result += ch.name + "'buffer"
			result += ": List(M'"
			for t in mcrl2typetuple(ch.type):
				result += datatypeacronym(t)
			result += "), " + ch.name + "'size: Int"
	return result

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
					varname_with_index = varname + "[" + expression(s.ref.index,stm,c,primmap,owner) + "]"
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

def expression_usedvars(s,stm,c,owner):
	"""Produces list of variables occurring in the given expression. This method is different from 'expression_varset', which is used for variable references w.r.t. atomicity violation checking.
	The current function is used to get a complete overview of the used variables, not ignoring any that are irrelevant for atomicity violation checking.
	The function can be called with 'owner' being a class or an object. Depending on this, the names of the variables are given with class-based or object-based scope."""
	output = expression_usedvars_scan(s,stm,c)
	olist = sorted(list(output))

	olist_scoped = []
	for v in olist:
		olist_scoped.append(owner.name + "'" + v)
	return olist_scoped

def expression_usedvars_scan(s,stm,c):
	"""Scanning method to actually recursively detect variable references in a given expression"""
	global actions, scopedvars
	output = set([])
	if s.__class__.__name__ == "VariableRef":
		# is the variable not local to state machine?
		output.add(scopedvars[c.name + "'" + stm.name][s.var.name])
		if s.index != None:
			output |= expression_usedvars_scan(s.index,stm,c)
	elif s.__class__.__name__ != "Primary":
		output |= expression_usedvars_scan(s.left,stm,c)
		if s.op != '':
			output |= expression_usedvars_scan(s.right,stm,c)
	else:
		if s.ref != None:
			if s.ref.ref not in actions:
				output.add(scopedvars[c.name + "'" + stm.name][s.ref.ref])
				if s.ref.index != None:
					output |= expression_usedvars_scan(s.ref.index,stm,c)
		if s.body != None:
			output |= expression_usedvars_scan(s.body,stm,c)
	return output

def expression_varset(s,stm,c,primmap,owner,ignore_indices):
	"""Produce set of variables minus the local ones"""
	return expression_variables(s,stm,c,primmap,owner,ignore_indices,False)

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
			indices = set([])
			if s.var.name in smlocalvars.get(c.name + "'" + stm.name,set([])):
				varname = owner.name + "'" + stm.name + "'" + s.var.name
			else:
				varname = owner.name + "'" + s.var.name
			if s.index != None:
				# varname += "(index'(" + expression(s.index,stm,c,primmap,owner) + "))"
				varname += "(" + expression(s.index,stm,c,primmap,owner) + ")"
				if not ignore_indices:
					indices = expression_variables(s.index,stm,c,primmap,owner,ignore_indices,add_local)
					output |= indices
			output.add((varname, tuple(indices)))
	elif s.__class__.__name__ != "Primary":
		output |= expression_variables(s.left,stm,c,primmap,owner,ignore_indices,add_local)
		if s.op != '':
			output |= expression_variables(s.right,stm,c,primmap,owner,ignore_indices,add_local)
	else:
		if s.ref != None:
			if s.ref.ref not in actions:
				# IGNORE STATE MACHINE LOCAL VARS?
				if add_local or (s.ref.ref not in smlocalvars.get(c.name + "'" + stm.name,set([]))):
					indices = set([])
					if s.ref.ref in smlocalvars.get(c.name + "'" + stm.name,set([])):
						varname = owner.name + "'" + stm.name + "'" + s.ref.ref
					else:
						varname = owner.name + "'" + s.ref.ref
					if s.ref.index != None:
						# varname += "(index'(" + expression(s.ref.index,stm,c,primmap,owner) + "))"
						varname += "(" + expression(s.ref.index,stm,c,primmap,owner) + ")"
						if not ignore_indices:
							indices = expression_variables(s.ref.index,stm,c,primmap,owner,ignore_indices,add_local)
							output |= indices
					output.add((varname, tuple(indices)))
		if s.body != None:
			output |= expression_variables(s.body,stm,c,primmap,owner,ignore_indices,add_local)
	return output

def expression_mcrl2type(s,stm,c):
	"""Provide the mCRL2 type of the given expression"""
	global vartypes
	vlist = expression_usedvars(s,stm,c,c)
	type = ""
	first = True
	for v in vlist:
		if not first:
			type += "# "
		else:
			first = False
		type += mcrl2type(vartypes[v])
	type += " -> Bool"
	return type

def statement_condition(s, owner):
	"""Provide the condition associated to the given statement. Owner is the Object owner of statement s."""
	global statemachine, smclass, porttypes, ochannel, trans
	result = ""
	if s.__class__.__name__ == "Expression":
		if not expression_is_actionref(s):
			result += expression(s, statemachine[s], smclass[statemachine[s]], {}, owner)
	elif s.__class__.__name__ == "SendSignal" or s.__class__.__name__ == "ReceiveSignal":
		if ochannel[owner][s].synctype == "async":
			if s.__class__.__name__ == "SendSignal":
				result += ochannel[owner][s].name + "'size < " + ochannel[owner][s].name + "'max"
			if s.__class__.__name__ == "ReceiveSignal":
				result += ochannel[owner][s].name + "'size != 0 && el'0(head(" + ochannel[owner][s].name + "'buffer)) == " + s.signal
				if s.guard != None:
					# create dictionary to translate primaries to correct parts of the next message on the channel
					newdict = {}
					for i in range(0, len(s.params)):
						varname = s.params[i].var.name
						if s.params[i].index != None:
							varname += "[" + expression(s.params[i].index,statemachine[s],smclass[statemachine[s]],{},owner) + "]"
						newdict[varname] = "el'" + str(i+1) + "(head(" + ochannel[owner][s].name + "'buffer))"
					result += " && " + expression(s.guard, statemachine[s], smclass[statemachine[s]], newdict, owner)
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			result += expression(s.guard, statemachine[s], smclass[statemachine[s]], {}, owner)
	# Add the negation of conditions of any alternatives with higher priority than s
	tr = s.parent
	sm = tr.parent
	c = sm.parent
	src = tr.source
	T = trans[c][sm][src]
	for t in T:
		if t.priority < tr.priority:
			if result != "":
				result += " && "
			result += "!(" + statement_condition(t.statements[0], owner) + ")"
	return result

def sync_statement_condition(s_rcv, rcv_owner, s_snd, snd_owner):
	"""Provide condition for the syncing of the given send and receive statements. Precondition is that s_snd is of type SendSignal, and s_rcv is of type ReceiveSignal.
	Furthermore, it is assumed that their Signals match."""
	global statemachine, smclass
	result = ""
	if s_rcv.guard != None:
		# create dictionary to translate primaries to correct parts of the next message on the channel
		newdict = {}
		for i in range(0, len(s_rcv.params)):
			varname = s_rcv.params[i].var.name
			if s_rcv.params[i].index != None:
				varname += "[" + s_rcv.params[i].index + "]"
			varname2 = ""
			if s_snd.params[i].index != None:
				varname2 += "get'("
			varname2 += s_snd.params[i].var.name
			if s_snd.params[i].index != None:
				varname2 += ", " + expression(s_snd.params[i].index,statemachine[s_snd],smclass[statemachine[s_snd]],{},snd_owner) + ")"
			newdict[varname] = varname2
		result += " && " + expression(s_rcv.guard, statemachine[s_rcv], smclass[statemachine[s_rcv]], newdict, rcv_owner)
	return result

def statementstatechanges(s,o):
	"""Maps SLCO statement effects to mCRL2 state changes. Object o owning the statement is also given"""
	global porttypes, vardict, scopedvars, statemachine, ochannel
	output = ''
	if s.__class__.__name__ == "Assignment":
		varname = scopedvars[o.type.name + "'" + statemachine[s].name][s.left.var.name]
		# an assignment to an array cell should be handled differently from other cases
		if s.left.index != None:
			output = ", " + o.name + "'" + varname + "=set'(" + o.name + "'" + varname + ", " + expression(s.left.index,statemachine[s],o.type,{},o) + ", "
			# in case we are updating a Byte, restrict the new value
			eright = expression(s.right,statemachine[s],o.type,{},o)
			if s.left.var.type.base == 'Byte':
				eright = "(" + eright + ") mod 256"
			output += eright + ")"
		else:
			output = ", " + o.name + "'" + varname
			eright = expression(s.right,statemachine[s],o.type,{},o)
			if s.left.var.type.base == 'Byte':
				eright = "(" + eright + ") mod 256"
			output += "=" + eright
	elif s.__class__.__name__ == "Composite":
		# first build vardict for sequence of assignments
		vardict = {}
		for e in s.assignments:
			newright = expression(e.right,statemachine[s],o.type,vardict,o)
			# in case we are updating a Byte, restrict the new value
			if e.left.var.type.base == 'Byte':
				newright = "(" + newright + ") mod 256"
			varname = scopedvars[o.type.name + "'" + statemachine[s].name][e.left.var.name]
			if e.left.index != None:
				vardict[varname] = "set'(" + expression(e.left.var,statemachine[s],o.type,vardict,o) + ", " + expression(e.left.index,statemachine[s],o.type,vardict,o) + ", " + newright + ")"
			else:
				vardict[varname] = "(" + newright + ")"
		# add assignments to state changes
		for key, value in vardict.items():
			output += ", " + o.name + "'" + key + "=" + value
	elif s.__class__.__name__ == "Delay":
		output = ""
	elif s.__class__.__name__ == "SendSignal":
		output = ", " + ochannel[o][s].name + "'buffer=" + ochannel[o][s].name + "'buffer ++ [m'"
		for t in mcrl2typetuple(ochannel[o][s].type):
			output += datatypeacronym(t)
		output += "(" + s.signal
		for p in s.params:
			output += ", "
			output += expression(p,statemachine[s],o.type,{},o)
		output += ")], " + ochannel[o][s].name + "'size=" + ochannel[o][s].name + "'size+1"
	elif s.__class__.__name__ == "ReceiveSignal":
		vardict = {}
		for i in range(0,len(s.params)):
			p = s.params[i]
			newright = "el'" + str(i+1) + "(head(" + ochannel[o][s].name + "'buffer))"
			# in case we are updating a Byte, restrict the new value
			if p.var.type.base == 'Byte':
				newright = "(" + newright + ") mod 256"
			varname = scopedvars[o.type.name + "'" + statemachine[s].name][p.var.name]
			if p.index != None:
				vardict[varname] = "set'(" + expression(p.var,statemachine[s],o.type,vardict,o) + ", " + expression(p.index,statemachine[s],o.type,vardict,o) + ", " + newright + ")"
			else:
				vardict[varname] = "(" + newright + ")"
		# add assignments
		for key, value in vardict.items():
			output += ", " + o.name + "'" + key + "=" + value
		output += ", " + ochannel[o][s].name + "'buffer=tail(" + ochannel[o][s].name + "'buffer), " + ochannel[o][s].name + "'size=" + ochannel[o][s].name + "'size-1"
	elif s.__class__.__name__ == "Expression":
		output = ""
	return output

def sync_statementstatechanges(s_rcv,o_rcv,s_snd,o_snd):
	"""Maps SLCO syncing statements effects to mCRL2 state changes. Objects o owning the statements are also given.
	Preconditions: the statements are of the right types, and the Signals match."""
	global porttypes, scopedvars, statemachine
	pindex = 0
	output = ""
	vardict = {}
	for i in range(0,len(s_rcv.params)):
		p = s_rcv.params[i]
		newright = expression(s_snd.params[i],statemachine[s_snd],o_snd.type,{},o_snd)
		# in case we are updating a Byte, restrict the new value
		if p.var.type.base == 'Byte':
			newright = "(" + newright + ") mod 256"
		varname = scopedvars[o_rcv.type.name + "'" + statemachine[s_rcv].name][p.var.name]
		if p.index != None:
			vardict[varname] = "set'(" + expression(p.var,statemachine[s_rcv],o_rcv.type,vardict,o_rcv) + ", " + expression(p.index,statemachine[s_rcv],o_rcv.type,vardict,o_rcv) + ", " + newright + ")"
		else:
			vardict[varname] = "(" + newright + ")"
	# add assignments
	for key, value in vardict.items():
		output += ", " + o_rcv.name + "'" + key + "=" + value
	return output

def statement_accesspattern(s, o):
	"""Provide the access pattern of the given statement. o is Object owning statement s."""
	global statemachine, smclass

	if s.__class__.__name__ == "Assignment":
		readset = expression_full_varset(s.right,statemachine[s],smclass[statemachine[s]],{},o,False)
		if s.left.index != None:
			readset |= expression_full_varset(s.left.index,statemachine[s],smclass[statemachine[s]],{},o,False)
		writeset = expression_full_varset(s.left,statemachine[s],smclass[statemachine[s]],{},o,True)
		return tuple([readset, writeset])
	elif s.__class__.__name__ == "Composite":
		vardict = {}
		readset = set([])
		writeset = set([])
		if s.guard != None:
			readset |= expression_full_varset(s.guard,statemachine[s],smclass[statemachine[s]],vardict,o,False)
		for st in s.assignments:
			# read accesses to variables that have already been written to do not need to be added, therefore '- writeset'
			readset |= (expression_full_varset(st.right,statemachine[s],smclass[statemachine[s]],vardict,o,False) - writeset)
			if st.left.index != None:
				readset |= (expression_full_varset(st.left.index,statemachine[s],smclass[statemachine[s]],vardict,o,False) - writeset)
			writeset |= expression_full_varset(st.left,statemachine[s],smclass[statemachine[s]],vardict,o,True)
			# update vardict (to correctly handle possible array index use in subsequent assignments)
			newright = expression(st.right,statemachine[s],smclass[statemachine[s]],vardict,o)
			varname = scopedvars[smclass[statemachine[s]].name + "'" + statemachine[s].name][st.left.var.name]
			if st.left.index != None:
				vardict[varname] = "set'(" + expression(st.left.var,statemachine[s],smclass[statemachine[s]],vardict,o) + ", " + expression(st.left.index,statemachine[s],smclass[statemachine[s]],vardict,o) + ", " + newright + ")"
			else:
				vardict[varname] = "(" + newright + ")"
		return tuple([readset, writeset])
	elif s.__class__.__name__ == "Delay":
		return tuple([set([]),set([])])
	elif s.__class__.__name__ == "SendSignal":
		readset = set([])
		for st in s.params:
			readset |= expression_full_varset(st,statemachine[s],smclass[statemachine[s]],{},o,False)
		return tuple([readset, set([])])
	elif s.__class__.__name__ == "ReceiveSignal":
		writeset = set([])
		for st in s.params:
			writeset |= expression_full_varset(st,statemachine[s],smclass[statemachine[s]],{},o,True)
		readset = set([])
		if s.guard != None:
			readset = expression_full_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False)
		# in SLCO ReceiveSignal, it is not possible to refer to the old value of a variable to which you are reading. Hence, reading AND writing to the same variable cannot occur
		readset = readset - writeset
		return tuple([readset, writeset])
	elif s.__class__.__name__ == "Expression":
		if expression_is_actionref(s):
			return tuple([set([]), set([])])
		else:
			readset = expression_full_varset(s,statemachine[s],smclass[statemachine[s]],{},o,False)
			return tuple([readset, set([])])

def statement_condition_accesspattern(s, o):
	return statement_condition_accesspattern_2(s, o, False)

def statement_full_condition_accesspattern(s, o):
	return statement_condition_accesspattern_2(s, o, True)

def statement_condition_accesspattern_2(s, o, withlocal):
	"""Provide the access pattern of the condition of the given statement. o is Object owning statement s."""
	global statemachine, smclass

	if s.__class__.__name__ == "Assignment":
		return tuple([set([]), set([])])
	elif s.__class__.__name__ == "Composite":
		readset = set([])
		if s.guard != None:
			if withlocal:
				readset |= expression_full_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False)
			else:
				readset |= expression_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False)
		return tuple([readset, set([])])
	elif s.__class__.__name__ == "Delay":
		return tuple([set([]),set([])])
	elif s.__class__.__name__ == "SendSignal":
		return tuple([set([]),set([])])
	elif s.__class__.__name__ == "ReceiveSignal":
		writeset = set([])
		for st in s.params:
			if withlocal:
				writeset |= expression_full_varset(st,statemachine[s],smclass[statemachine[s]],{},o,False)
			else:
				writeset |= expression_varset(st,statemachine[s],smclass[statemachine[s]],{},o,False)
		readset = set([])
		if s.guard != None:
			if withlocal:
				readset = expression_full_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False)
			else:
				readset = expression_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False)
		# in SLCO ReceiveSignal, it is not possible to refer to the old value of a variable to which you are writing. Hence, reading AND writing to the same variable cannot occur
		readset = readset - writeset
		return tuple([readset, set([])])
	elif s.__class__.__name__ == "Expression":
		if expression_is_actionref(s):
			return tuple([set([]), set([])])
		else:
			if withlocal:
				readset = expression_full_varset(s,statemachine[s],smclass[statemachine[s]],{},o,False)
			else:
				readset = expression_varset(s,statemachine[s],smclass[statemachine[s]],{},o,False)
			return tuple([readset, set([])])

def statement_structure_accesspattern(s, o):
	"""Provide the access pattern of the given statement. o is Object owning statement s."""
	global statemachine, smclass

	if s.__class__.__name__ == "Assignment":
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
			alist.append(tuple([readset2 - readset - writeset, writeset2]))
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

def compute_accesspatterns(m):
	"""Compute the access patterns for all statements in the system"""
	adict = {}
	for o in m.objects:
		adict[o] = {}
		for stm in o.type.statemachines:
			for tr in stm.transitions:
				for st in tr.statements:
					access = statement_accesspattern(st,o)
					adict[o][st] = access
	return adict

def compute_condition_accesspatterns(m):
	"""Compute the access patterns for all statements in the system"""
	adict = {}
	for o in m.objects:
		adict[o] = {}
		for stm in o.type.statemachines:
			for tr in stm.transitions:
				for st in tr.statements:
					access = statement_condition_accesspattern(st,o)
					adict[o][st] = access
	return adict

def compute_full_condition_accesspatterns(m):
	"""Compute the access patterns for all statements in the system"""
	adict = {}
	for o in m.objects:
		adict[o] = {}
		for stm in o.type.statemachines:
			for tr in stm.transitions:
				for st in tr.statements:
					access = statement_full_condition_accesspattern(st,o)
					adict[o][st] = access
	return adict

def compute_structure_accesspatterns(m):
	"""Compute the access patterns for all statements in the system"""
	adict = {}
	for o in m.objects:
		adict[o] = {}
		for stm in o.type.statemachines:
			for tr in stm.transitions:
				for st in tr.statements:
					access = statement_structure_accesspattern(st,o)
					adict[o][st] = access
	return adict

def mcrl2_accesspattern(s, o, b):
	"""Return the access pattern of the given statement for use in an mCRL2 specification. Owner o is also given.
	Boolean b indicates whether the statement should be considered as enabled or not."""
	global statement_access, statement_condition_access
	if b:
		access = statement_access[o][s]
	else:
		access = statement_condition_access[o][s]
	return mcrl2_accesspattern_to_string(access)

def mcrl2_structure_accesspattern(s, o, enabled):
	"""Return the access pattern structure of the given statement, for use in an mCRL2 specification. Owner o is also given.
	Enabled is a boolean indicating whether all accesses (true) or only those to check the condition of s (false) should be returned."""
	global statement_structure_access, statement_full_condition_access
	if enabled:
		alist = statement_structure_access[o][s]
	else:
		alist = [statement_full_condition_access[o][s]]
	return mcrl2_accesspatterns_to_string(alist)

def mcrl2_accesslist_to_string(L):
	"""Convert given list of accesses to string"""
	slist = sorted(list(mcrl2_accesslist_to_string_2(L)))
	output = ""
	for s in slist:
		if output != "":
			output += ", "
		output += s
	return output

def mcrl2_accesslist_to_string_2(L):
	"""Convert given list of accesses to string"""
	sset = set([])
	for v in L:
		sset.add(mcrl2varprefix + v[0])
		if v[1] != ():
			sset |= mcrl2_accesslist_to_string_2(v[1])
	return sset

def mcrl2_accesspattern_to_string(access):
	"""Convert given access pattern to string"""
	output = "A'("
	sorted_access = list(access[0]) #sorted(list(access[0]))
	# if the list requires dynamic sorting (it contains more than one dynamic accesses to an array), indicate this
	# occurrences = set([])
	# found = False
	# for v in sorted_access:
	# 	v_splitted = v.split("(index\'(")
	# 	if len(v_splitted) > 1:
	# 		if not RepresentsInt(v_splitted[1][:-2]):
	# 			if v_splitted[0] in occurrences:
	# 				found = True
	# 				break
	# 			else:
	# 				occurrences.add(v_splitted[0])
	# if found:
	# 	output += "sorted_array_accesses("
	output += "["
	first = True
	for v in sorted_access:
		if not first:
			output += ", "
		else:
			first = False
		if v[1] != ():
			output += "lcdp'(" + mcrl2varprefix + v[0] + ", [" + mcrl2_accesslist_to_string(v[1]) + "])"
		else:
			output += "lc'(" + mcrl2varprefix + v[0] + ")"
	output += "]"
	# if found:
	# 	output += ")"
	output += ", "
	sorted_access = list(access[1]) #sorted(list(access[1]))
	# occurrences = set([])
	# found = False
	# for v in sorted_access:
	# 	v_splitted = v.split("(index\'(")
	# 	if len(v_splitted) > 1:
	# 		if not RepresentsInt(v_splitted[1][:-2]):
	# 			if v_splitted[0] in occurrences:
	# 				found = True
	# 				break
	# 			else:
	# 				occurrences.add(v_splitted[0])
	# if found:
	# 	output += "sorted_array_accesses("
	output += "["
	first = True
	for v in sorted_access:
		if not first:
			output += ", "
		else:
			first = False
		if v[1] != ():
			output += "lcdp'(" + mcrl2varprefix + v[0] + ", [" + mcrl2_accesslist_to_string(v[1]) + "])"
		else:
			output += "lc'(" + mcrl2varprefix + v[0] + ")"
	output += "]"
	# if found:
	# 	output += ")"
	output += ")"
	return output

def mcrl2_accesspatterns_to_string(alist):
	output = "["
	first = True
	for a in alist:
		if first:
			first = False
		else:
			output += ", "
		output += mcrl2_accesspattern_to_string(a)
	output += "]"
	return output

def preprocess():
	"""preprocessing method"""
	global model, statemachine, statemachinenames, statemachine_states, trowner, smlocalvars, states, actions, class_sync_receives, class_sync_sends, trans, smclass, channeltypes, asynclosslesstypes, asynclossytypes, synctypes, porttypes, ochannel, vartypes, object_sync_commpairs, syncing_statements, sorted_variables, sorted_objects, sorted_statemachines, filtered_statement_access, filtered_statemachine_access, IntArraySizes, BoolArraySizes
	# build dictionaries providing for a given statement the state machine and transition owning it, and for a given state machine, the class owning it
	for c in model.classes:
		for stm in c.statemachines:
			smclass[stm] = c
			for trn in stm.transitions:
				for stat in trn.statements:
					statemachine[stat] = stm
					trowner[stat] = trn
	# build a set of used statemachine names
	statemachinenames = set([])
	for c in model.classes:
		for stm in c.statemachines:
			statemachinenames.add(stm.name)
	# build a dictionary providing the type of messages that can be sent over a given port
	for c in model.channels:
		porttypes[str(c.source.type.name) + "'" + str(c.ports[0].name)] = c.type
		porttypes[str(c.target.type.name) + "'" + str(c.ports[1].name)] = c.type
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
	# construct dictionary of variables (with class-based scopes) and their types
	vartypes = {}
	for c in model.classes:
		for v in c.variables:
			vartypes[c.name + "'" + v.name] = v.type
			for stm in c.statemachines:
				for v in stm.variables:
					vartypes[c.name + "'" + stm.name + "'" + v.name] = v.type
	# make a set of state names
	states = set([])
	for c in model.classes:
		for stm in c.statemachines:
			for s in stm.states:
				states.add(s.name)
	# make a dictionary of channel types, where for each type we keep track of the encountered signal names
	channeltypes = {}
	
	for c in model.channels:
		type = mcrl2typetuple(c.type)
		channeltypes[type] = set([])
		# add the type to the right set
		if c.synctype == 'async':
			if c.losstype == 'lossless':
				asynclosslesstypes.add(type)
			else:
				asynclossytypes.add(type)
		else:
			synctypes.add(tuple(type))
	# lookup all signal names
	for c in model.classes:
		for stm in c.statemachines:
			for trn in stm.transitions:
				for stat in trn.statements:
					if stat.__class__.__name__ == "ReceiveSignal" or stat.__class__.__name__ == "SendSignal":
						# get type
						ctype = mcrl2typetuple(porttypes[c.name + "'" + stat.target.name])
						signalset = channeltypes[ctype]
						signalset.add(stat.signal)
	# construct a dictionary that provides the channel linked to a given object/statement pair
	ochannel = {}
	for o in model.objects:
		ochannel[o] = {}
		for sm in o.type.statemachines:
			for tr in sm.transitions:
				for st in tr.statements:
					if st.__class__.__name__ == "ReceiveSignal" or st.__class__.__name__ == "SendSignal":
						for ch in model.channels:
							if ch.source == o and st.target == ch.ports[0]:
								ochannel[o][st] = ch
							elif ch.target == o and st.target == ch.ports[1]:
								ochannel[o][st] = ch
	# construct a dictionary that provides a list of (object, send statement) pairs linked to a given channel
	channelsends = {}
	for o in model.objects:
		for sm in o.type.statemachines:
			for tr in sm.transitions:
				for st in tr.statements:
					if st.__class__.__name__ == "SendSignal":
						ch = ochannel[o][st]
						if ch.synctype == 'sync':
							pset = channelsends.get(ch, set([]))
							pset.add(tuple([o, st]))
							channelsends[ch] = pset
	# construct dictionary of statements, indicating with which other statements synchronous communication can be done
	syncing_statements = {}
	for o in model.objects:
		syncing_statements[o] = {}
		for sm in o.type.statemachines:
			for tr in sm.transitions:
				for st in tr.statements:
					syncing_statements[o][st] = set([])
	for o in model.objects:
		for sm in o.type.statemachines:
			for tr in sm.transitions:
				for st in tr.statements:
					if st.__class__.__name__ == "ReceiveSignal":
						ch = ochannel[o][st]
						if ch.synctype == "sync":
							if o == ch.source and st.target == ch.ports[0]:
								snd_object = ch.target
								snd_port = ch.ports[1]
							else:
								snd_object = ch.source
								snd_port = ch.ports[0]
							sends = channelsends[ch]
							for snd in sends:
								if snd_object == snd[0] and snd_port == snd[1].target and st.signal == snd[1].signal:
									syncing_statements[snd[0]][snd[1]].add(tuple([o,st]))

	# construct the transition relation of the state machines in a dictionary
	trans = {}
	for c in model.classes:
		ctrans = {}
		for sm in c.statemachines:
			smtrans = {}
			for s in sm.states:
				smtrans[s] = []
			for tr in sm.transitions:
				smtrans[tr.source].append(tr)
			ctrans[sm] = smtrans
		trans[c] = ctrans
	# construct set of object pairs that are involved in communication, i.e., a pair (o1, o2) indicates that there is a channel from o1 to o2
	object_sync_commpairs = set([])
	for ch in model.channels:
		if ch.synctype == 'sync':
			pair = tuple([ch.source, ch.target])
			object_sync_commpairs.add(pair)
			pair = tuple([ch.target, ch.source])
			object_sync_commpairs.add(pair)
	# create sorted list of object and statemachine variables
	tmp_sorted_variables = []
	for o in model.objects:
		for v in o.type.variables:
			v_size = 1
			if v.type.size > 1:
				v_size = v.type.size
			tmp_sorted_variables.append(tuple([v, o.name + "'" + v.name, v_size]))
		for sm in o.type.statemachines:
			for v in sm.variables:
				v_size = 1
				if v.type.size > 1:
					v_size = v.type.size
				tmp_sorted_variables.append(tuple([v, o.name + "'" + sm.name + "'" + v.name, v_size]))
	sorted_variables = sorted(tmp_sorted_variables, key=lambda vartuple: vartuple[1])
	# create sorted list of objects and statemachines
	tmp_objects = []
	for o in model.objects:
		tmp_objects.append(o.name)
	sorted_objects = sorted(tmp_objects)
	tmp_statemachines = []
	for c in model.classes:
		for sm in c.statemachines:
			tmp_statemachines.append(sm.name)
	sorted_statemachines = sorted(tmp_statemachines)
	# create lists of array sizes occurring in the model
	isizes = set([])
	bsizes = set([])
	for c in model.classes:
		for v in c.variables:
			if v.type.size > 1:
				if v.type.base == "Boolean":
					bsizes.add(v.type.size)
				elif v.type.base == "Integer" or v.type.base == "Byte":
					isizes.add(v.type.size)
		for sm in c.statemachines:
			for v in sm.variables:
				if v.type.size > 1:
					if v.type.base == "Boolean" or v.type.base == "Byte":
						bsizes.add(v.type.size)
					elif v.type.base == "Integer":
						isizes.add(v.type.size)
	IntArraySizes = list(isizes)
	BoolArraySizes = list(bsizes)

def translate_sa():
	"""The translation function for pure static analysis"""
	global model, modelname, trans, statement_access, statement_condition_access, statement_full_condition_access, statement_structure_access

	path, name = split(modelname)
	if name.endswith('.slco'):
		name = name[:-4]
	else:
		name = name[:-7]
	outFile = open(join(path,name + "instructions"), 'w')

	# compute statement access patterns
	statement_access = compute_accesspatterns(model)
	statement_condition_access = compute_condition_accesspatterns(model)
	statement_full_condition_access = compute_full_condition_accesspatterns(model)
	statement_structure_access = compute_structure_accesspatterns(model)

	print("Writing statically derived model instruction information")
	# Dump the instruction descriptions in the file
	for o in model.objects:
		for sm in o.type.statemachines:
			for tr in sm.transitions:
				outFile.write("rw(" + o.name + ", " + sm.name + ", ST'" + str(tr._tx_position) + ", " + mcrl2_structure_accesspattern(tr.statements[0], o, True) + "\n")
	print("Writing statically derived PR-relation at instruction level")
	for c in model.classes:
		for sm in c.statemachines:
			for s in sm.states:
				outgoing_s = trans[c][sm][s]
				for tr in outgoing_s:
					t = tr.target
					outgoing_t = trans[c][sm][t]
					for tr2 in outgoing_t:
						outFile.write("ST'" + str(tr._tx_position) + " ST'" + str(tr2._tx_position) + "\n")
	outFile.close

def translate():
	"""The translation function"""
	global model, modelname, statemachinenames, statemachine, tr, smlocalvars, states, actions, class_receives, class_sends, vartypes, mcrl2varprefix, channeltypes, asynclossytypes, asynclosslesstypes, synctypes, statement_access, statement_full_condition_access, statement_condition_access, statement_structure_access, object_sync_commpairs, syncing_statements, sorted_variables, IntArraySizes, BoolArraySizes
	
	path, name = split(modelname)
	if name.endswith('.slco'):
		name = name[:-4]
	else:
		name = name[:-7]
	outFile = open(join(path,name + "mcrl2"), 'w')

	# Initialize the template engine.
	jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(join(this_folder,'../../jinja2_templates')), trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols','jinja2.ext.do',])

	# Register the filters
	jinja_env.filters['mcrl2value'] = mcrl2value
	# jinja_env.filters['mcrl2type'] = mcrl2type
	# jinja_env.filters['mcrl2typetuple'] = mcrl2typetuple
	jinja_env.filters['datatypeacronym'] = datatypeacronym
	jinja_env.filters['variabledefault'] = variabledefault
	jinja_env.filters['expression'] = expression
	jinja_env.filters['expression_usedvars'] = expression_usedvars
	# jinja_env.filters['statement_condition_mcrl2type'] = statement_condition_mcrl2type
	# jinja_env.filters['statement_condition_varlist'] = statement_condition_varlist
	jinja_env.filters['statement_condition'] = statement_condition
	jinja_env.filters['statementstatechanges'] = statementstatechanges
	# jinja_env.filters['complete_newstate'] = complete_newstate
	jinja_env.filters['sync_statement_condition'] = sync_statement_condition
	jinja_env.filters['sync_statementstatechanges'] = sync_statementstatechanges
	# jinja_env.filters['sync_complete_newstate'] = sync_complete_newstate
	# jinja_env.filters['hasobjectvariables'] = hasobjectvariables
	jinja_env.filters['hascondition'] = hascondition
	# jinja_env.filters['mcrl2_state_accesspattern'] = mcrl2_state_accesspattern
	# jinja_env.filters['mcrl2_sourcestate_accesspattern'] = mcrl2_sourcestate_accesspattern
	jinja_env.filters['mcrl2_accesspattern'] = mcrl2_accesspattern
	jinja_env.filters['mcrl2_structure_accesspattern'] = mcrl2_structure_accesspattern
	# jinja_env.filters['mcrl2_read_accesspattern'] = mcrl2_read_accesspattern
	# jinja_env.filters['mcrl2_write_accesspattern'] = mcrl2_write_accesspattern
	jinja_env.filters['mcrl2_model_vartypes'] = mcrl2_model_vartypes
	jinja_env.filters['mcrl2_model_vars'] = mcrl2_model_vars
	jinja_env.filters['mcrl2_model_vars_with_types'] = mcrl2_model_vars_with_types
	
	# Register the tests
	# jinja_env.tests['hasvariables'] = hasvariables
	jinja_env.tests['hasindex'] = hasindex
	jinja_env.tests['hasnoindex'] = hasnoindex
	jinja_env.tests['hasdynamicaddressing'] = hasdynamicaddressing

	# compute statement access patterns
	statement_access = compute_accesspatterns(model)
	statement_condition_access = compute_condition_accesspatterns(model)
	statement_full_condition_access = compute_full_condition_accesspatterns(model)
	statement_structure_access = compute_structure_accesspatterns(model)

	# Produce (guard, action, effect) tuples for the translation
	# produce_summands(model)
	# load the mCRL2 template
	template = jinja_env.get_template('mcrl2-seqcon.jinja2template')
	out = template.render(model=model, statemachinenames=statemachinenames, states=states, vartypes=vartypes, mcrl2varprefix=mcrl2varprefix, channeltypes=channeltypes, asynclosslesstypes=asynclosslesstypes, asynclossytypes=asynclossytypes, synctypes=synctypes, trans=trans, ochannel=ochannel, object_sync_commpairs=object_sync_commpairs, syncing_statements=syncing_statements, transowner=trowner, statemachine=statemachine, sorted_variables=sorted_variables, sorted_objects=sorted_objects, sorted_statemachines=sorted_statemachines, IntArraySizes=IntArraySizes, BoolArraySizes=BoolArraySizes)
	# write mCRL2 spec
	outFile.write(out)
	outFile.close()

def main(args):
	"""The main function"""
	global modelname, model, static_analysis_only
	if len(args) == 0:
		print("Missing argument: SLCO model")
		sys.exit(1)
	else:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 slco2mcrl2 [-rc]")
			print("")
			print("Transform an SLCO 2.0 model to an mCRL2 model for sequential consistency violation checking.")
			print(" -s                 instead of an mCRL2 model, produce a text file containing all instructions and the PR-relation for pure static analysis")
			sys.exit(0)
		else:
			i = 0
			while i < len(args):
				if args[i] == '-s':
					static_analysis_only = True
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
		# produce statistics
		countosm = 0
		countvars = 0
		countstats = 0
		for o in model.objects:
			for v in o.type.variables:
				if v.type.size > 1:
					countvars += v.type.size
				else:
					countvars += 1
			for sm in o.type.statemachines:
				countosm += 1
				for tr in sm.transitions:
					countstats += len(tr.statements)
		print("model contains " + str(countosm) + " object/statemachine pairs, " + str(countvars) + " shared variables, and " + str(countstats) + " statements")
		print("processing model %s" % basename(file))
		try:
			# preprocess model
			preprocess()
			# translate
			if static_analysis_only:
				translate_sa()
			else:
				translate()
		except Exception:
			print("failed to process model %s" % basename(file))
			print(traceback.format_exc())

if __name__ == '__main__':
	args = []
	for i in range(1,len(sys.argv)):
		args.append(sys.argv[i])
	main(args)
