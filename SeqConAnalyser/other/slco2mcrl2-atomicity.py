import sys
from os import mkdir
from os.path import exists, dirname, basename, join, split
import jinja2
from textx.metamodel import metamodel_from_file
from copy import deepcopy
import itertools
import glob
import traceback
from SCCTarjan import identifySCCs

this_folder = dirname(__file__)

# import libraries
sys.path.append(join(this_folder,'../../libraries'))
from slcolib import *
from SCCTarjan import *
this_folder = dirname(__file__)

# on-the-fly atomicity violation checking
check_onthefly = False
# lock on-the-fly
lock_onthefly = False
# apply por
apply_por = False
# default search without static analysis
default_search = False

modelname = ""
model = ""

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
# per class, give a list of shared variables
accessed_sharedvars = []
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

# partitioning of the statements into safe and unsafe statements
unsafe_statements = set([])
safe_statements = set([])

# unsafe variables are those referenced in unsafe statements
unsafe_variables = []

# safe statements for POR
safe_por_statements = set([])
# conditionally safe statements for POR
conditionally_safe_por_statements = set([])

# dictionary to look up channel linked to a given (object) statement
ochannel = {}
# set of pairs of communicating objects
object_sync_commpairs = set([])

# dictionary providing the access patterns of statements
statement_access = {}
statement_condition_access = {}

# statements that require syncing
syncing_statements = {}

# sorted list of variables in the model
sorted_variables = []
# sorted objects
sorted_objects = []
# sorted_statemachines
sorted_statemachines = []

# filtered accesses to reason about single and array variable accesses
filtered_statement_access = {}
filtered_statemachine_access = {}

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def hascondition(s, o):
	"""Returns if the given statement has a condition. o is the Object owning s."""
	global ochannel
	if s.__class__.__name__ == "Expression":
		return not expression_is_actionref(s)
	elif s.__class__.__name__ == "SendSignal":
		return ochannel[o][s].synctype == "async"
	elif s.__class__.__name__ == "ReceiveSignal":
		if s.guard != None:
			return True
		else:
			return ochannel[o][s].synctype == "async"
	elif s.__class__.__name__ == "Composite":
		return s.guard != None
	return False

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

def hasobjectvariables(m):
	""" Filter to check whether the given model have object variables"""
	found = False
	for c in m.classes:
		if len(c.variables) > 0:
			found = True
			break
	return found

def hasvariables(o):
	""" Test to check whether given object has object variables"""
	return len(o.type.variables) > 0

def hasindex(v):
	""" Test to check whether given (variable / variable scope name) pair has an index"""
	return v[0].type.size > 1

def hasnoindex(v):
	""" Test to check whether given (variable / variable scope name) pair has no index"""
	return v[0].type.size <= 1

def o_hassharedaccesses(o):
	""" Test to check whether given object has at least one state machine accessing o's shared variables"""
	global accessed_sharedvars
	return accessed_sharedvars.get(o.type) != None

def sm_hassharedaccesses(sm, o):
	""" Test to check whether given object / statemachine accesses o's shared variables"""
	global accessed_sharedvars
	return accessed_sharedvars[o.type].get(sm) != None

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
	return str(s).lower()

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
			print(s.ref.ref)
			if s.ref.ref not in actions:
				output.add(scopedvars[c.name + "'" + stm.name][s.ref.ref])
				if s.ref.index != None:
					output |= expression_usedvars_scan(s.ref.index,stm,c)
		if s.body != None:
			output |= expression_varset(s.body,stm,c,primmap,owner)
	return output

def expression_varset(s,stm,c,primmap,owner,ignore_indices):
	"""Produces set of variables referenced in given SLCO expression or variableref. Statemachine stm and Class c owning the statement are also given. Primmap is a dictionary for rewriting primaries, which is relevant when array indices are used.
	The owner of the statement is also given. ignore_indices is a Boolean flag indicating whether array indices should be ignored or not."""
	global actions, scopedvars, smlocalvars
	output = set([])
	if s.__class__.__name__ == "VariableRef":
		# is the variable not local to state machine?
		if s.var.name not in smlocalvars.get(c.name + "'" + stm.name,set([])):
			varname = owner.name + "'" + s.var.name
			if s.index != None:
				varname += "(Int2Nat(" + expression(s.index,stm,c,primmap,owner) + "))"
				if not ignore_indices:
					output |= expression_varset(s.index,stm,c,primmap,owner,ignore_indices)
			output.add(varname)
	elif s.__class__.__name__ != "Primary":
		output |= expression_varset(s.left,stm,c,primmap,owner,ignore_indices)
		if s.op != '':
			output |= expression_varset(s.right,stm,c,primmap,owner,ignore_indices)
	else:
		if s.ref != None:
			if s.ref.ref not in actions:
				# IGNORE STATE MACHINE LOCAL VARS!
				if s.ref.ref not in smlocalvars.get(c.name + "'" + stm.name,set([])):
					varname = owner.name + "'" + s.ref.ref
					if s.ref.index != None:
						varname += "(Int2Nat(" + expression(s.ref.index,stm,c,primmap,owner) + "))"
						if not ignore_indices:
							output |= expression_varset(s.ref.index,stm,c,primmap,owner,ignore_indices)
					output.add(varname)
		if s.body != None:
			output |= expression_varset(s.body,stm,c,primmap,owner,ignore_indices)
	return output

def expression_is_actionref(s):
	"""Determine whether the given expression is an action reference"""
	global actions
	if s.__class__.__name__ == "Expression":
		if s.op == '':
			snext = s.left
			if snext.op == '':
				snext = snext.left
				if snext.op == '':
					snext = snext.left
					if snext.op == '':
						snext = snext.left
						if snext.op == '':
							snext = snext.left
							if snext.ref != None and snext.sign == '':
								snext = snext.ref
								if snext.ref in actions:
									return True
	return False

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

def statement_condition_mcrl2type(s):
	"""Provide the type of the condition associated to the given statement"""
	global statemachine, smclass, porttypes
	result = ""
	if s.__class__.__name__ == "Expression":
		result += expression_mcrl2type(s, statemachine[s], smclass[statemachine[s]])
	elif s.__class__.__name__ == "SendSignal" or s.__class__.__name__ == "ReceiveSignal":
		ctype = porttypes[smclass[statemachine[s]].name + "'" + s.target.name]
		acronym = ""
		for t in ctype:
			acronym += datatypeacronym(mcrl2type(t))
		result += "List(M'" + acronym + ")"
		if s.__class__.__name__ == "SendSignal":
			result += " # Int # Int"
		if s.__class__.__name__ == "ReceiveSignal":
			if s.guard != None:
				types = expression_mcrl2type(s.guard, statemachine[s], smclass[statemachine[s]])
				if types != "":
					result += " # " + types
			else:
				result += " -> Bool"
		else:
			result += " -> Bool"
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			result += expression_mcrl2type(s.guard, statemachine[s], smclass[statemachine[s]])
	return result

def statement_condition_varlist(s):
	"""Provide the list of variables of the condition associated to the given statement"""
	global statemachine, smclass, porttypes, vartypes
	result = ""
	if s.__class__.__name__ == "Expression":
		vlist = expression_usedvars(s, statemachine[s], smclass[statemachine[s]], smclass[statemachine[s]])
		first = True
		for v in vlist:
			if not first:
				result += ", "
			else:
				first = False
			result += v + ": " + mcrl2type(vartypes[v])
	elif s.__class__.__name__ == "SendSignal" or s.__class__.__name__ == "ReceiveSignal":
		ctype = porttypes[smclass[statemachine[s]].name + "'" + s.target.name]
		acronym = ""
		for t in ctype:
			acronym += datatypeacronym(mcrl2type(t))
		result += "ch'buffer: List(M'" + acronym + ")"
		if s.__class__.__name__ == "SendSignal":
			result += ", ch'size: Int, ch_max: Int"
		if s.__class__.__name__ == "ReceiveSignal":
			if s.guard != None:
				vlist = expression_usedvars(s.guard, statemachine[s], smclass[statemachine[s]], smclass[statemachine[s]])
				for v in vlist:
					result += " # " + v + ": " + mcrl2type(vartypes[v]) 
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			result +=  statement_condition_varlist(s.guard)
	return result

def statement_condition(s, owner):
	"""Provide the condition associated to the given statement. Owner is the owner of statement s."""
	global statemachine, smclass, porttypes, ochannel
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

def complete_newstate(s, o):
	"""Produce the complete new state after executing the given statement s, with owner o"""
	global porttypes, vardict, scopedvars, statemachine, ochannel
	# dictionary with changes
	statechanges = {}
	if s.__class__.__name__ == "Assignment":
		varname = scopedvars[o.type.name + "'" + statemachine[s].name][s.left.var.name]
		# an assignment to an array cell should be handled differently from other cases
		if s.left.index != None:
			output = "set'(" + o.name + "'" + varname + ", " + expression(s.left.index,statemachine[s],o.type,{},o) + ", "
			# in case we are updating a Byte, restrict the new value
			eright = expression(s.right,statemachine[s],o.type,{},o)
			if s.left.var.type.base == 'Byte':
				eright = "(" + eright + ") mod 256"
			output += eright + ")"
		else:
			eright = expression(s.right,statemachine[s],o.type,{},o)
			if s.left.var.type.base == 'Byte':
				eright = "(" + eright + ") mod 256"
			output = eright
		statechanges[o.name + "'" + varname] = output
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
			statechanges[o.name + "'" + key] = value
	elif s.__class__.__name__ == "Delay":
		statechanges = {}
	elif s.__class__.__name__ == "SendSignal":
		output = ochannel[o][s].name + "'buffer ++ [m'"
		for t in mcrl2typetuple(ochannel[o][s].type):
			output += datatypeacronym(t)
		output += "(" + s.signal
		for p in s.params:
			output += ", "
			output += expression(p,statemachine[s],o.type,{},o)
		output += ")]"
		statechanges[ochannel[o][s].name + "'buffer"] = output
		statechanges[ochannel[o][s].name + "'size"] = ochannel[o][s].name + "'size+1"
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
			statechanges[o.name + "'" + key] = value
		statechanges[ochannel[o][s].name + "'buffer"] = "tail(" + ochannel[o][s].name + "'buffer)"
		statechanges[ochannel[o][s].name + "'size"] = ochannel[o][s].name + "'size-1"
	elif s.__class__.__name__ == "Expression":
		statechanges = {}
	# now return the complete new state
	stateoutput = ""
	for o2 in model.objects:
		for v in o2.type.variables:
			newvalue = statechanges.get(o2.name + "'" + v.name)
			if newvalue == None:
				stateoutput += ", " + o2.name + "'" + v.name
			else:
				stateoutput += ", " + newvalue
		for sm in o.type.statemachines:
			for v in sm.variables:
				newvalue = statechanges.get(o2.name + "'" + sm.name + "'" + v.name)
				if newvalue == None:
					stateoutput += ", " + o2.name + "'" + sm.name + "'" + v.name
				else:
					stateoutput += ", " + newvalue
	for c in model.channels:
		newvalue = statechanges.get(c.name + "'buffer")
		if newvalue == None:
			stateoutput += ", " + c.name + "'buffer"
		else:
			stateoutput += ", " + newvalue
		newvalue = statechanges.get(c.name + "'size")
		if newvalue == None:
			stateoutput += ", " + c.name + "'size"
		else:
			stateoutput += ", " + newvalue
	return stateoutput

def sync_complete_newstate(s_rcv,o_rcv,s_snd,o_snd):
	"""Produces complete new state after sync of two statements"""
	global porttypes, scopedvars, statemachine
	pindex = 0
	output = ""
	vardict = {}
	statechanges = {}
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
		statechanges[o_rcv.name + "'" + key] = value
	# now return the complete new state
	stateoutput = ""
	for o2 in model.objects:
		for v in o2.variables:
			newvalue = statechanges.get(o2.name + "'" + v.name)
			if newvalue == None:
				stateoutput += ", " + o2.name + "'" + v.name
			else:
				stateoutput += ", " + newvalue
		for sm in o.type.variables:
			for v in sm.variables:
				newvalue = statechanges.get(o2.name + "'" + sm.name + "'" + v.name)
				if newvalue == None:
					stateoutput += ", " + o2.name + "'" + sm.name + "'" + v.name
				else:
					stateoutput += ", " + newvalue
	for c in model.channels:
		stateoutput += ", " + c.name + "'buffer"
		stateoutput += ", " + c.name + "'size"
	return stateoutput

def statement_accesspattern(s, o):
	"""Provide the access pattern of the given statement. o is Object owning statement s."""
	global statemachine, smclass

	if s.__class__.__name__ == "Assignment":
		readset = expression_varset(s.right,statemachine[s],smclass[statemachine[s]],{},o,False)
		if s.left.index != None:
			readset |= expression_varset(s.left.index,statemachine[s],smclass[statemachine[s]],{},o,False)
		writeset = expression_varset(s.left,statemachine[s],smclass[statemachine[s]],{},o,True)
		return tuple([readset, writeset])
	elif s.__class__.__name__ == "Composite":
		vardict = {}
		readset = set([])
		writeset = set([])
		if s.guard != None:
			readset |= expression_varset(s.guard,statemachine[s],smclass[statemachine[s]],vardict,o,False)
		for st in s.assignments:
			# read accesses to variables that have already been written to do not need to be added, therefore '- writeset'
			readset |= (expression_varset(st.right,statemachine[s],smclass[statemachine[s]],vardict,o,False) - writeset)
			if st.left.index != None:
				readset |= (expression_varset(st.left.index,statemachine[s],smclass[statemachine[s]],vardict,o,False) - writeset)
			writeset |= expression_varset(st.left,statemachine[s],smclass[statemachine[s]],vardict,o,True)
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
			readset |= expression_varset(st,statemachine[s],smclass[statemachine[s]],{},o,False)
		return tuple([readset, set([])])
	elif s.__class__.__name__ == "ReceiveSignal":
		writeset = set([])
		for st in s.params:
			writeset |= expression_varset(st,statemachine[s],smclass[statemachine[s]],{},o,True)
		readset = set([])
		if s.guard != None:
			readset = expression_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False)
		# in SLCO ReceiveSignal, it is not possible to refer to the old value of a variable to which you are reading. Hence, reading AND writing to the same variable cannot occur
		readset = readset - writeset
		return tuple([readset, writeset])
	elif s.__class__.__name__ == "Expression":
		if expression_is_actionref(s):
			return tuple([set([]), set([])])
		else:
			readset = expression_varset(s,statemachine[s],smclass[statemachine[s]],{},o,False)
			return tuple([readset, set([])])

def statement_condition_accesspattern(s, o):
	"""Provide the access pattern of the condition of the given statement. o is Object owning statement s."""
	global statemachine, smclass

	if s.__class__.__name__ == "Assignment":
		return tuple([set([]), set([])])
	elif s.__class__.__name__ == "Composite":
		readset = set([])
		if s.guard != None:
			readset |= expression_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False)
		return tuple([readset, set([])])
	elif s.__class__.__name__ == "Delay":
		return tuple([set([]),set([])])
	elif s.__class__.__name__ == "SendSignal":
		return tuple([set([]),set([])])
	elif s.__class__.__name__ == "ReceiveSignal":
		writeset = set([])
		for st in s.params:
			writeset |= expression_varset(st,statemachine[s],smclass[statemachine[s]],{},o,False)
		readset = set([])
		if s.guard != None:
			readset = expression_varset(s.guard,statemachine[s],smclass[statemachine[s]],{},o,False)
		# in SLCO ReceiveSignal, it is not possible to refer to the old value of a variable to which you are reading. Hence, reading AND writing to the same variable cannot occur
		readset = readset - writeset
		return tuple([readset, set([])])
	elif s.__class__.__name__ == "Expression":
		if expression_is_actionref(s):
			return tuple([set([]), set([])])
		else:
			readset = expression_varset(s,statemachine[s],smclass[statemachine[s]],{},o,False)
			return tuple([readset, set([])])

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

def mcrl2_accesspattern(s, o, b):
	"""Return the access pattern of the given statement for use in an mCRL2 specification. Owner o is also given.
	Boolean b indicates whether the statement should be considered as enabled or not."""
	global statement_access, statement_condition_access
	if b:
		access = statement_access[o][s]
	else:
		access = statement_condition_access[o][s]
	output = "A'("
	sorted_access = sorted(list(access[0]))
	# if the list requires dynamic sorting (it contains more than one dynamic accesses to an array), indicate this
	occurrences = set([])
	found = False
	for v in sorted_access:
		v_splitted = v.split("(Int2Nat(")
		if len(v_splitted) > 1:
			if not RepresentsInt(v_splitted[1][:-2]):
				if v_splitted[0] in occurrences:
					found = True
					break
				else:
					occurrences.add(v_splitted[0])
	if found:
		output += "sorted_array_accesses("
	output += "["
	first = True
	for v in sorted_access:
		if not first:
			output += ", "
		else:
			first = False
		output += mcrl2varprefix + v
	output += "]"
	if found:
		output += ")"
	output += ", "
	sorted_access = sorted(list(access[1]))
	occurrences = set([])
	found = False
	for v in sorted_access:
		v_splitted = v.split("(Int2Nat(")
		if len(v_splitted) > 1:
			if not RepresentsInt(v_splitted[1][:-2]):
				if v_splitted[0] in occurrences:
					found = True
					break
				else:
					occurrences.add(v_splitted[0])
	if found:
		output += "sorted_array_accesses("
	output += "["
	first = True
	for v in sorted_access:
		if not first:
			output += ", "
		else:
			first = False
		output += mcrl2varprefix + v
	output += "]"
	if found:
		output += ")"
	output += ")"
	return output

def mcrl2_read_accesspattern(s, o, b):
	"""Return the read access pattern of the given statement for use in an mCRL2 specification. Owner o is also given.
	Boolean b indicates whether the statement should be considered as enabled or not."""
	global statement_access, statement_condition_access
	if b:
		access = statement_access[o][s]
	else:
		access = statement_condition_access[o][s]
	output = "{"
	first = True
	sorted_access = sorted(list(access[0]))
	for v in sorted_access:
		if not first:
			output += ", "
		else:
			first = False
		output += mcrl2varprefix + v
	output += "}"
	return output

def mcrl2_write_accesspattern(s, o, b):
	"""Return the write access pattern of the given statement for use in an mCRL2 specification. Owner o is also given.
	Boolean b indicates whether the statement should be considered as enabled or not."""
	global statement_access, statement_condition_access
	if b:
		access = statement_access[o][s]
	else:
		access = statement_condition_access[o][s]
	output = "{"
	first = True
	sorted_access = sorted(list(access[1]))
	for v in sorted_access:
		if not first:
			output += ", "
		else:
			first = False
		output += mcrl2varprefix + v
	output += "}"
	return output

def statement_is_por_safe(s, b):
	"""Indicate whether the given statement for use in an mCRL2 specification.
	Boolean b indicates whether the statement should be considered as enabled or not."""
	global safe_por_statements, conditionally_safe_por_statements
	if b:
		return s in safe_por_statements
	else:
		return s in conditionally_safe_por_statements

def peek(stack):
	"""Return but do not pop top element of stack"""
	return stack[len(stack) - 1]

def identify_por_safe_statements(model):
	global statemachine, smclass, safe_por_statements, conditionally_safe_por_statements, trans, default_search

	# in case of the default search, no statement is por safe
	safe_por_statements = set([])
	conditionally_safe_por_statements = set([])
	for c in model.classes:
		for sm in c.statemachines:
			for tr in sm.transitions:
				for st in tr.statements:
					access = statement_accesspattern(st, c)
					if len(access[0]) == 0 and len(access[1]) == 0:
						safe_por_statements.add(st)
					access = statement_condition_accesspattern(st, c)
					if len(access[0]) == 0:
						conditionally_safe_por_statements.add(st)

	# explore each state machine, trying to detect cycles consisting of safe transitions. Each time such a cycle is found,
	# we mark the final transition as 'sticky', by removing it from the set of safe transitions. This, in turn, ensures
	# a cycle proviso when applying POR.
	for c in model.classes:
		for sm in c.statemachines:
			states = set([])
			for s in sm.states:
				states.add(s)
			# call stack
			callstack = []
			callstack_set = set([])
			# closed set
			closed = set([])
			for s in states:
				if s not in closed:
					toutgoing = trans.get(s, [])
					toutgoing_filtered = []
					for tr in toutgoing:
						if tr.statements == None or tr.statements[0] in safe_por_statements:
							toutgoing_filtered.append(tr) 
					callstack.append((s, toutgoing_filtered))
					callstack_set.add(s)
					while len(callstack) > 0:
						s, transitions = peek(callstack)
						while len(transitions) > 0:
							t = peek(transitions)
							if t.target not in closed and t.target not in callstack_set:
								# add target state to call stack
								toutgoing = trans.get(tr.target, [])
								toutgoing_filtered = []
								for tr in toutgoing:
									if tr.statements == None or tr.statements[0] in safe_por_statements:
										toutgoing_filtered.append(tr)
								callstack.append((tr.target, toutgoing_filtered))
								callstack_set.add(tr.target)
								break
							elif t.target in callstack_set:
								# we are closing a cycle. Mark this transition as unsafe (sticky)
								if len(t.statements) > 0:
									safe_por_statements.remove(t.statements[0])
							transitions.pop()
						if len(transitions) == 0:
							callstack.pop()
							callstack_set.remove(s)
							closed.add(s)

def statement_is_safe(s):
	"""Returns whether the given statement is safe (w.r.t. atomicity violations) or not"""
	global safe_statements
	return s in safe_statements

def count_conflicts2(s1, s2, limit):
	"""Compare two access patterns. Both consist of a set and two dictionaries, the set providing all simple accesses and one dictionary providing exact (statically clear) array accesses.
	The other provides representations of all imprecise array accesses.
	The question is whether at least 'limit' access matches can be made from s1 to s2. As soon as this limit is reached, the limit is returned, otherwise the number of matches."""
	# first count size of intersection of the first sets
	count = len(s1[0] & s2[0])
	if count >= limit:
		return limit
	# determine the largest number of possible combinations of precise/imprecise array accesses
	# collect the arrays that are accessed by the first pattern
	aset = set([a for a in s1[1].keys()]) | set([a for a in s1[2].keys()])
	for a in aset:
		# first compare precise array accesses
		precise_match = len(s1[1].get(a,set([])) & s2[1].get(a,set([])))
		count += precise_match
		if count >= limit:
			return limit
		# how many precise accesses are left?
		remainder1 = max(0, len(s1[1].get(a,set([]))) - precise_match)
		remainder2 = max(0, len(s2[1].get(a,set([]))) - precise_match)
		# match remaining precise accesses with imprecise accesses
		imprecise_match1 = min(s2[2].get(a,0), remainder1)
		imprecise_match2 = min(s1[2].get(a,0), remainder2)
		count += imprecise_match1 + imprecise_match2
		if count >= limit:
			return limit
		# finally, match remaining imprecise accesses with each other
		imprecise_remainder1 = max(0,s1[2].get(a,0) - imprecise_match2)
		imprecise_remainder2 = max(0,s2[2].get(a,0) - imprecise_match1)
		count += min(imprecise_remainder1, imprecise_remainder2)
		if count >= limit:
			return limit
	return count

def compare_accesses2(s1, s2):
	"""Compare two access patterns. Both consist of a set and two dictionaries, the set providing all simple accesses and one dictionary providing exact (statically clear) array accesses.
	The other provides representations of all imprecise array accesses.
	The question is which accesses in s2 conflict with s1. This subset of s2 is returned."""
	conf_accesses = [set([]), {}, {}]
	conf_accesses[0] = s1[0] & s2[0]
	# first compare precise array accesses
	for a in s1[1].keys():
		# first compare precise array accesses
		if s2[1].get(a) != None:
			conf_accesses[1][a] = s1[1][a] & s2[1][a]
		# compare with imprecise array accesses
		if s2[2].get(a) != None:
			conf_accesses[2][a] = 1
	# next compare imprecise array accesses
	for a in s1[2].keys():
		if s2[1].get(a) != None:
			conf_accesses[1][a] = s2[1][a]
		if s2[2].get(a) != None:
			conf_accesses[2][a] = 1
	return conf_accesses

def number_of_accesses(s):
	"""Count the number of accesses in the given filtered_access"""
	count = len(s[0][0])
	for a in s[0][1].keys():
		count += len(s[0][1][a])
	for a in s[0][2].keys():
		count += s[0][2][a]
	count += len(s[1][0])
	for a in s[1][1].keys():
		count += len(s[1][1][a])
	for a in s[1][2].keys():
		count += s[1][2][a]
	return count

def compute_filtered_statemachine_access(o, sts, filtered_statement_access, in_access):
	"""Combine the filtered_accesses of the statements of a statemachine with the given in_access, producing a single filtered access. o is the owning object"""
	setr = set([])
	setw = set([])
	keysetr0 = set([])
	keysetr1 = set([])
	keysetw0 = set([])
	keysetw1 = set([])
	for st in sts:
		fa = filtered_statement_access[o][st]
		setr |= fa[0][0]
		setw |= fa[1][0]
		keysetr0 |= set(fa[0][1].keys())
		keysetr1 |= set(fa[0][2].keys())
		keysetw0 |= set(fa[1][1].keys())
		keysetw1 |= set(fa[1][2].keys())
	setr |= in_access[0][0]
	setw |= in_access[1][0]
	keysetr0 |= set(in_access[0][1].keys())
	keysetr1 |= set(in_access[0][2].keys())
	keysetw0 |= set(in_access[1][1].keys())
	keysetw1 |= set(in_access[1][2].keys())
	dictr0 = {}
	for k in keysetr0:
		occset = set([])
		for st in sts:
			occset |= filtered_statement_access[o][st][0][1].get(k,set([]))
		occset |= in_access[0][1].get(k,set([]))
		dictr0[k] = occset
	dictr1 = {}
	for k in keysetr1:
		count = 0
		for st in sts:
			count += filtered_statement_access[o][st][0][2].get(k,0)
		dictr1[k] = count
		count += in_access[0][2].get(k,0)
	dictw0 = {}
	for k in keysetw0:
		occset = set([])
		for st in sts:
			occset |= filtered_statement_access[o][st][1][1].get(k,set([]))
		occset |= in_access[1][1].get(k,set([]))
		dictw0[k] = occset
	dictw1 = {}
	for k in keysetw1:
		count = 0
		for st in sts:
			count += filtered_statement_access[o][st][1][2].get(k,0)
		count += in_access[1][2].get(k,0)
		dictw1[k] = count
	return [[setr, dictr0, dictr1], [setw, dictw0, dictw1]]

def accesses_conflict(ac1, ac2):
	"""Return whether the two given filtered accesses conflict on at least one variable"""
	if ac1[0][0] & ac2[1][0]:
		return True
	if ac1[1][0] & ac2[0][0]:
		return True
	if ac1[1][0] & ac2[1][0]:
		return True
	for k in set(ac1[0][1].keys()) & set(ac2[1][1].keys()):
		if ac1[0][1][k] & ac2[1][1][k] != set([]):
			return True
	for k in set(ac1[1][1].keys()) & set(ac2[0][1].keys()):
		if ac1[1][1][k] & ac2[0][1][k] != set([]):
			return True
	for k in set(ac1[1][1].keys()) & set(ac2[1][1].keys()):
		if ac1[1][1][k] & ac2[1][1][k] != set([]):
			return True
	if set(ac1[0][1].keys()) & set(ac2[1][2].keys()) != set([]):
		return True
	if set(ac1[1][1].keys()) & set(ac2[0][2].keys()) != set([]):
		return True
	if set(ac1[1][1].keys()) & set(ac2[1][2].keys()) != set([]):
		return True
	if set(ac1[1][2].keys()) & set(ac2[0][1].keys()) != set([]):
		return True
	if set(ac1[0][2].keys()) & set(ac2[1][1].keys()) != set([]):
		return True
	if set(ac1[1][2].keys()) & set(ac2[1][1].keys()) != set([]):
		return True
	return False

def build_dep_graph(o, sm, smfadict):
	"""Build dependency graph between statemachines in o, excluding statemachine sm"""
	depgraph = {}
	for sm1 in o.type.statemachines:
		if sm1 != sm:
			depgraph[sm1] = set([])
	for i in range(0,len(o.type.statemachines)):
		sm1 = o.type.statemachines[i]
		if sm1 != sm:
			for j in range(i+1,len(o.type.statemachines)):
				sm2 = o.type.statemachines[j]
				if sm2 != sm:
					if accesses_conflict(smfadict[o.type][sm1], smfadict[o.type][sm2]):
						depgraph[sm1].add(sm2)
						depgraph[sm2].add(sm1)
	return depgraph

def compute_filtered_statement_accesses():
	"""Compute filtered accesses to reason about single and array variable accesses"""
	global filtered_statement_access, statement_access
	# filter statement_access to remove references to array cells that cannot be (or are hard to be) evaluated statically
	# for each access set (read and write) we actually build a set and two dictionaries:
	# - the first set consists of accesses to simple, i.e., not array, variables
	# - the first dictionary produces per accessed array a set of concrete indices (integers)
	# - the second dictionary produces per accessed array the number of accesses where the location (cell index) could not be determined statically.
	filtered_statement_access = {}
	for o in statement_access.keys():
		filtered_statement_access[o] = {}
		for st in statement_access[o].keys():
			newaccess = [[set([]), {}, {}], [set([]), {}, {}]]
			for i in range(0,2):
				imprecise_seen = set([])
				for v in statement_access[o][st][i]:
					varname = v
					if v.find('(Int2Nat(') != -1:
						varname, varindex = v.split('(Int2Nat(')
						# get index
						varindex = varindex[:-2]
						if RepresentsInt(varindex):
							occset = newaccess[i][1].get(varname,set([]))
							occset.add(varindex)
							newaccess[i][1][varname] = occset
						elif (varname + "[" + varindex + "]") not in imprecise_seen:
							imprecise_seen.add(varname + "[" + varindex + "]")
							counter = newaccess[i][2].get(varname,0)
							counter += 1
							newaccess[i][2][varname] = counter
					else:
						newaccess[i][0].add(varname)
			filtered_statement_access[o][st] = tuple(newaccess)

def compute_filtered_statemachine_accesses(m):
	"""Filtered accesses on statemachine level"""
	global filtered_statement_access, filtered_statemachine_access
	# create list of objects, one at most per class
	classobjects = []
	represented_classes = set([])
	for o in m.objects:
		if o.type not in represented_classes:
			classobjects.append(o)
			represented_classes.add(o.type)
	# create dictionary of statements
	sdict = {}
	for c in m.classes:
		cdict = {}
		for sm in c.statemachines:
			stset = set([])
			for tr in sm.transitions:
				for st in tr.statements:
					stset.add(st)
			cdict[sm] = stset
		sdict[c] = cdict
	filtered_statemachine_access = {}

	for o in m.objects:
		smdict = {}
		for sm in o.type.statemachines:
			smdict[sm] = compute_filtered_statemachine_access(o, sdict[o.type][sm], filtered_statement_access, [[set(),{},{}],[set(),{},{}]])
		filtered_statemachine_access[o.type] = smdict

def build_accessed_sharedvars(m):
	"""build a dictionary providing a list of shared variables accessed by each statemachine"""
	global accessed_sharedvars
	accessed_sharedvars = {}
	for c in m.classes:
		smdict = {}
		for sm in c.statemachines:
			vsets = [set([]),set([])]
			fa = filtered_statemachine_access[c][sm]
			for i in range(0,2):
				vsets[i] |= fa[i][0]
				aset = set([])
				for a in fa[i][2].keys():
					aset.add(a)
					# look up array, and add all its cells
					o1, v1 = a.split("'")
					for v2 in c.variables:
						if v2.name == v1:
							for j in range(0,v2.type.size):
								vsets[i].add(a + "(Int2Nat(" + str(j) + "))")
							found = True
							break
				for a in fa[i][1].keys():
					if a not in aset:
						for j in fa[i][1][a]:
							vsets[i].add(a + "(Int2Nat(" + str(j) + "))")
			if vsets != [set([]),set([])]:
				smdict[sm] = [list(vsets[0]),list(vsets[1])]
		if smdict != {}:
			accessed_sharedvars[c] = smdict

def identify_safe_unsafe_statements(m):
	"""Partition the statements of the model into safe and unsafe statements"""
	global safe_statements, unsafe_statements, statement_access, statemachine, smclass, unsafe_variables, default_search, filtered_statement_access, filtered_statemachine_access

	# Distribute statements over the sets safe and unsafe
	safe_statements = set([])
	unsafe_statements = set([])

	# unsafe accesses
	unsafe_filtered_accesses = list()

	# create list of objects, one at most per class
	classobjects = []
	represented_classes = set([])
	for o in model.objects:
		if o.type not in represented_classes:
			classobjects.append(o)
			represented_classes.add(o.type)

	# build a list of unsafe variables
	unsafe_variables = set([])

	# do we create a list of unsafe statements and variables with simply all statements / variables?
	if default_search:
		for c in model.classes:
			for sm in c.statemachines:
				for tr in sm.transitions:
					for st in tr.statements:
						unsafe_statements.add(st)
		# create complete list of global variables
		for o in model.objects:
			for v in o.type.variables:
				varname = o.name + "'" + v.name
				if v.type.size > 1:
					for i in range(0,v.type.size):
						varindex = "(Int2Nat(" + str(i) + "))"
						unsafe_variables.add(varname + varindex)
				else:
					unsafe_variables.add(varname)		
	else:
		# create dictionary of statements
		sdict = {}
		for o in classobjects:
			c = o.type
			cdict = {}
			for sm in c.statemachines:
				stset = set([])
				for tr in sm.transitions:
					for st in tr.statements:
						stset.add(st)
				cdict[sm] = stset
			sdict[c] = cdict

		# check possibility for atomicity violations
		for o in classobjects:
			c = o.type
			for i in range(0,len(c.statemachines)):
				sm = c.statemachines[i]
				depgraph = {}
				SCC_filtered_accesses = list()
				SCCs = list()
				stset1 = sdict[c][sm]
				for st1 in stset1:
					# is st1 trivially safe?
					if number_of_accesses(filtered_statement_access[o][st1]) < 2:
						safe_statements.add(st1)
					else:
						# if SCCs indicating connections between other statemachines in o are not yet identified, do so
						# and compute combined filtered accesses for the SCCs
						if SCCs == []:
							depgraph = build_dep_graph(o, sm, filtered_statemachine_access)
							identifySCCs(depgraph, {}, SCCs)
							for scc in SCCs:
								combined_access = [[set(),{},{}],[set(),{},{}]]
								for sm1 in scc[1].keys():
									combined_access = compute_filtered_statemachine_access(o, sdict[c][sm1], filtered_statement_access, combined_access)
								SCC_filtered_accesses.append(combined_access)
						# check if any combined filtered access conflicts with st1
						marked_unsafe = False
						for fa in SCC_filtered_accesses:
							count = 0
							count += count_conflicts2(fa[1], filtered_statement_access[o][st1][0], 2)
							if count >= 2:
								unsafe_statements.add(st1)
								unsafe_filtered_accesses.append(fa)
								marked_unsafe = True
							else:
								count += count_conflicts2(fa[0], filtered_statement_access[o][st1][1], 2)
								if count >= 2:
									unsafe_statements.add(st1)
									unsafe_filtered_accesses.append(fa)
									marked_unsafe = True
								else:
									count += count_conflicts2(fa[1], filtered_statement_access[o][st1][1], 2)
									if count >= 2:
										unsafe_statements.add(st1)
										unsafe_filtered_accesses.append(fa)
										marked_unsafe = True
							if count >= 2:
								conflict = [[],[]]
								conflict[0] = compare_accesses2(fa[1], filtered_statement_access[o][st1][0])
								tmp_conflict1 = compare_accesses2(fa[0], filtered_statement_access[o][st1][1])
								tmp_conflict2 = compare_accesses2(fa[1], filtered_statement_access[o][st1][1])
								tmp_conflict1[0] |= tmp_conflict2[0]
								for a in tmp_conflict2[1].keys():
									aset = tmp_conflict1[1].get(a,set([]))
									tmp_conflict1[1][a] = aset | tmp_conflict2[1][a]
								for a in tmp_conflict2[2].keys():
									anumber = tmp_conflict1[2].get(a,0)
									tmp_conflict1[2][a] = anumber + tmp_conflict2[2][a]
								conflict[1] = tmp_conflict1
								unsafe_filtered_accesses.append(conflict)
						if not marked_unsafe:
							safe_statements.add(st1)

	# for st in unsafe_statements:
	# 	# get the class owning st, and then all objects of that type
	# 	c = smclass[statemachine[st]]
	# 	olist = []
	# 	for o in model.objects:
	# 		if o.type == c:
	# 			olist.append(o)
	# 	# use the representative object to obtain the access pattern of st
	# 	for o in olist:
	# 		if o in classobjects:
	# 			ap = statement_access[o][st]
	# 			break
	# 	# add all elements of the read and write set, making sure to make dynamic array accesses static (to all cells in the array)
	# 	for i in range(0,2):
	# 		for v in ap[i]:
	# 			v_splitted = v.split("(Int2Nat(")
	# 			if len(v_splitted) > 1:
	# 				if not RepresentsInt(v_splitted[1][:-2]):
	# 					# look up array, and add all its cells
	# 					o1, v1 = v_splitted[0].split("'")
	# 					found = False
	# 					for o in model.objects:
	# 						if o.name == o1:
	# 							for v2 in o.type.variables:
	# 								if v2.name == v1:
	# 									for j in range(0,v2.type.size):
	# 										unsafe_variables.add(v_splitted[0] + "(Int2Nat(" + str(j) + "))")
	# 									found = True
	# 									break
	# 						if found:
	# 							break
	# 				else:
	# 					unsafe_variables.add(v)
	# 			else:
	# 				unsafe_variables.add(v)

	for ac in unsafe_filtered_accesses:
		for a in ac[0][0]:
			unsafe_variables.add(a)
		for a in ac[1][0]:
			unsafe_variables.add(a)
		for a in ac[0][1].keys():
			for i in ac[0][1][a]:
				unsafe_variables.add(a + "(Int2Nat(" + str(i) + "))")
		for a in ac[1][1].keys():
			for i in ac[1][1][a]:
				unsafe_variables.add(a + "(Int2Nat(" + str(i) + "))")		
		for a in ac[0][2].keys():
			# look up array, and add all its cells
			o1, v1 = a.split("'")
			found = False
			for o in model.objects:
				if o.name == o1:
					for v2 in o.type.variables:
						if v2.name == v1:
							for j in range(0,v2.type.size):
								unsafe_variables.add(a + "(Int2Nat(" + str(j) + "))")
							found = True
							break
				if found:
					break
		for a in ac[1][2].keys():
			# look up array, and add all its cells
			o1, v1 = a.split("'")
			found = False
			for o in model.objects:
				if o.name == o1:
					for v2 in o.type.variables:
						if v2.name == v1:
							for j in range(0,v2.type.size):
								unsafe_variables.add(a + "(Int2Nat(" + str(j) + "))")
							found = True
							break
				if found:
					break
	# make unsafe_variables a sorted list
	unsafe_variables = sorted(list(unsafe_variables))
	print(unsafe_variables)

def preprocess():
	"""preprocessing method"""
	global model, statemachinenames, statemachine, trowner, smlocalvars, accessed_sharedvars, states, actions, class_sync_receives, class_sync_sends, trans, smclass, channeltypes, asynclosslesstypes, asynclossytypes, synctypes, porttypes, ochannel, vartypes, object_sync_commpairs, syncing_statements, sorted_variables, sorted_objects, sorted_statemachines, filtered_statement_access, filtered_statemachine_access, IntArraySizes, BoolArraySizes
	# build dictionaries providing for a given statement the state machine and transition owning it, and for a given state machine, the class owning it
	for c in model.classes:
		for stm in c.statemachines:
			smclass[stm] = c
			for trn in stm.transitions:
				for stat in trn.statements:
					statemachine[stat] = stm
					trowner[stat] = trn
	# add tau action to all transitions without statements
	# for c in model.classes:
	# 	for stm in c.statemachines:
	# 		for trn in stm.transitions:
	# 			if len(trn.statements) == 0:
	# 				trn.statements.append("tau'")
	# fix wrong references from transitions to states (scope errors)
	for c in model.classes:
		for sm in c.statemachines:
			sdict = {}
			for s in sm.states:
				sdict[s.name] = s
			for tr in sm.transitions:
				if tr.source != sdict[tr.source.name]:
					tr.source = sdict[tr.source.name]
				if tr.target != sdict[tr.target.name]:
					tr.target = sdict[tr.target.name]
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
	# give transitions without a priority a high enough priority value, such that they actually get the lowest priority
	prios = {}
	for c in model.classes:
		for stm in c.statemachines:
			prios = {}
			for trn in stm.transitions:
				if prios.get(trn.source.name,0) == 0:
					prios[trn.source.name] = trn.priority
				else:
					if trn.priority > prios[trn.source.name]:
						prios[trn.source.name] = trn.priority
			# assign new priorities
			for trn in stm.transitions:
				if trn.priority == 0:
					trn.priority = prios[trn.source.name]+1
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
	# create sorted list of object variables
	tmp_sorted_variables = []
	for o in model.objects:
		for v in o.type.variables:
			tmp_sorted_variables.append(tuple([v, o.name + "'" + v.name, 0]))
	tmp_sorted_variables = sorted(tmp_sorted_variables, key=lambda vartuple: vartuple[1])
	v_index = 0
	sorted_variables = []
	for v in tmp_sorted_variables:
		vlist = list(v)
		vlist[2] = v_index
		sorted_variables.append(tuple(vlist))
		if v[0].type.size > 1:
			v_index += v[0].type.size
		else:
			v_index += 1
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

def translate():
	"""The translation function"""
	global model, modelname, statemachinenames, statemachine, tr, smlocalvars, accessed_sharedvars, states, actions, class_receives, class_sends, vartypes, mcrl2varprefix, channeltypes, asynclossytypes, asynclosslesstypes, synctypes, statement_access, statement_condition_access, unsafe_statements, object_sync_commpairs, syncing_statements, check_onthefly, lock_onthefly, apply_por, sorted_variables, unsafe_variables, IntArraySizes, BoolArraySizes
	
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
	jinja_env.filters['mcrl2type'] = mcrl2type
	jinja_env.filters['mcrl2typetuple'] = mcrl2typetuple
	jinja_env.filters['datatypeacronym'] = datatypeacronym
	jinja_env.filters['variabledefault'] = variabledefault
	jinja_env.filters['expression'] = expression
	jinja_env.filters['expression_usedvars'] = expression_usedvars
	jinja_env.filters['statement_condition_mcrl2type'] = statement_condition_mcrl2type
	jinja_env.filters['statement_condition_varlist'] = statement_condition_varlist
	jinja_env.filters['statement_condition'] = statement_condition
	jinja_env.filters['statementstatechanges'] = statementstatechanges
	jinja_env.filters['complete_newstate'] = complete_newstate
	jinja_env.filters['sync_statement_condition'] = sync_statement_condition
	jinja_env.filters['sync_statementstatechanges'] = sync_statementstatechanges
	jinja_env.filters['sync_complete_newstate'] = sync_complete_newstate
	jinja_env.filters['hasobjectvariables'] = hasobjectvariables
	jinja_env.filters['hascondition'] = hascondition
	jinja_env.filters['statement_is_safe'] = statement_is_safe
	jinja_env.filters['statement_is_por_safe'] = statement_is_por_safe
	jinja_env.filters['mcrl2_accesspattern'] = mcrl2_accesspattern
	jinja_env.filters['mcrl2_read_accesspattern'] = mcrl2_read_accesspattern
	jinja_env.filters['mcrl2_write_accesspattern'] = mcrl2_write_accesspattern
	jinja_env.filters['mcrl2_model_vartypes'] = mcrl2_model_vartypes
	jinja_env.filters['mcrl2_model_vars'] = mcrl2_model_vars
	jinja_env.filters['mcrl2_model_vars_with_types'] = mcrl2_model_vars_with_types
	
	# Register the tests
	jinja_env.tests['hasvariables'] = hasvariables
	jinja_env.tests['hasindex'] = hasindex
	jinja_env.tests['hasnoindex'] = hasnoindex
	jinja_env.tests['o_hassharedaccesses'] = o_hassharedaccesses
	jinja_env.tests['sm_hassharedaccesses'] = sm_hassharedaccesses

	# compute statement access patterns
	statement_access = compute_accesspatterns(model)
	statement_condition_access = compute_condition_accesspatterns(model)
	# compute filtered accesses per statement and on state machine level
	compute_filtered_statement_accesses()
	compute_filtered_statemachine_accesses(model)
	# build dictionary of accesses to shared variables done by each statemachine
	build_accessed_sharedvars(model)
	# compute which statements are safe and unsafe
	identify_safe_unsafe_statements(model)

	# special case: if no unsafe statements are present, the result is that all variables can remain unlocked
	if len(unsafe_statements) == 0:
		print("No variables require locking!")
		exit(0)

	# identify POR safe statements
	identify_por_safe_statements(model)

	# Produce (guard, action, effect) tuples for the translation
	# produce_summands(model)
	# load the mCRL2 template
	template = jinja_env.get_template('mcrl2-atomicity.jinja2template')
	out = template.render(model=model, statemachinenames=statemachinenames, states=states, vartypes=vartypes, mcrl2varprefix=mcrl2varprefix, channeltypes=channeltypes, asynclosslesstypes=asynclosslesstypes, asynclossytypes=asynclossytypes, synctypes=synctypes, trans=trans, ochannel=ochannel, object_sync_commpairs=object_sync_commpairs, syncing_statements=syncing_statements, transowner=trowner, statemachine=statemachine, check_onthefly=check_onthefly, lock_onthefly=lock_onthefly, apply_por=apply_por, sorted_variables=sorted_variables, unsafe_variables=unsafe_variables, sorted_objects=sorted_objects, sorted_statemachines=sorted_statemachines, accessed_sharedvars=accessed_sharedvars, IntArraySizes=IntArraySizes, BoolArraySizes=BoolArraySizes)
	# write mCRL2 spec
	outFile.write(out)
	outFile.close()

def main(args):
	"""The main function"""
	global modelname, model, check_onthefly, apply_por, lock_onthefly, default_search
	if len(args) == 0:
		print("Missing argument: SLCO model")
		sys.exit(1)
	else:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 slco2mcrl2 [-rc]")
			print("")
			print("Transform an SLCO 2.0 model to an mCRL2 model for atomicity violation checking (reduced state space).")
			print(" -d                                    default search. do NOT apply static analysis")
			print(" -o                                    apply on-the-fly atomicity violation checking")
			print(" -l                                    lock on-the-fly (applies checking on-the-fly and acts on result)")
			print(" -r                                    apply partial-order reduction")
			sys.exit(0)
		else:
			for i in range(0,len(args)):
				if args[i] == '-d':
					default_search = True
					check_onthefly = False
					lock_onthefly = False
					apply_por = False
				elif args[i] == '-o':
					check_onthefly = True
				elif args[i] == '-r':
					apply_por = True
				elif args[i] == '-l':
					lock_onthefly = True
					check_onthefly = True
				else:
					modelname = args[i]

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
		print("processing model %s" % basename(file))
		try:
			# preprocess model
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
