# SLCO 2.0 to multi-threaded Java transformation, limited to SLCO models consisting of a single object

import sys
import os
import jinja2
from textx.metamodel import metamodel_from_file
import re

# Set of all state names in the model
states = set([])
# number of elementary global variables in the model
numberofelemvariables = 0
# names of state machine local variables
smlocalvars = set([])
# dictionary providing lock ids (offset) for a given variable
varids = {}
# locking dictionary (provides variables to lock per object)
lockingdict = {}
# should a transition counter be added to the code? (to make program executions finite)
add_counter = False

this_folder = os.path.dirname(__file__)

def RepresentsInt(s):

    try: 
        int(s)
        return True
    except ValueError:
        return False

def operator(s):
	"""Provide characters to display given operator s"""
	if s == '>':
		return '>'
	elif s == '<':
		return '<'
	elif s == '>=':
		return '>='
	elif s == '<=':
		return '<='
	elif s == '=':
		return '=='
	elif s == '!=':
		return '!='
	elif s == 'and':
		return '&&'
	elif s == 'or':
		return '||'
	elif s == '<>':
		return '!='
	else:
		return s

def sign(s):
	if s == 'not':
		return '!'
	else:
		return s

def getlabel(s):
	"""Get the label for the given statement s"""
	result = ''
	if s.__class__.__name__ == "AL_Assignment":
		result += s.left.var.name
		if s.left.index != None:
			result += "[" + getlabel(s.left.index) + "]"
		result += " := " + getlabel(s.right)
	elif s.__class__.__name__ == "Composite":
		result += "["
		if s.guard != None:
			result += getlabel(s.guard)
			result += ";"
		for i in range(0,len(s.assignments)):
			result += " " + getlabel(s.assignments[i])
			if i < len(s.assignments)-1:
				result += ";"
		result += "]"
	elif s.__class__.__name__ == "Delay":
		result += "<b>after </b>" + str(s.length) + "<b> ms</b>"
	elif s.__class__.__name__ == "MemoryFence":
		result += "#"
	elif s.__class__.__name__ == "SendSignal":
		result += "<b>send </b>" + s.signal + "("
		first = True
		for p in s.params:
			if not first:
				result += ","
			else:
				first = False
			result += getlabel(p)
		result += ") <b>to </b>" + s.target.name
	elif s.__class__.__name__ == "ReceiveSignal":
		result += "<b>receive </b>" + s.signal + "("
		first = True
		for p in s.params:
			if not first:
				result += ","
			else:
				first = False
			result += getlabel(p)
		if s.guard != None:
			result += " | " + getlabel(s.guard)
		result += ") <b>from </b>" + s.target.name
	elif s.__class__.__name__ == "AL_Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2" or s.__class__.__name__ == "ExprPrec1":
		if s.op != '':
			result += getlabel(s.left) + " " + operator(s.op) + " " + getlabel(s.right)
		else:
			result += getlabel(s.left)
	elif s.__class__.__name__ == "Primary":
		result += s.sign
		if s.sign == "not":
			result += " "
		if s.value != None:
			newvalue = s.value
			result += str(newvalue)
		elif s.ref != None:
			result += s.ref.ref
			if s.ref.index != None:
				result += "[" + getlabel(s.ref.index) + "]"
		else:
			result += '(' + getlabel(s.body) + ')'
	elif s.__class__.__name__ == "VariableRef":
		result += s.var.name
		if s.index != None:
			result += "[" + getlabel(s.index) + "]"
	elif s.__class__.__name__ == "DoAction":
		result += "<b>do</b>" + str(s.act)
	return result

def getinstruction(s):
	"""Get the Java instruction for the given statement s"""
	global smlocalvars
	result = ''
	if s.__class__.__name__ == "AL_Assignment":
		result += s.left.var.name
		if s.left.index != None:
			result += "[" + getinstruction(s.left.index) + "]"
		result += " = "
		if s.left.var.type.base == 'Byte':
			result += "(byte) ("
		result += getinstruction(s.right)
		if s.left.var.type.base == 'Byte':
			result += ")"
	elif s.__class__.__name__ == "Composite":
		result += "["
		if s.guard != None:
			result += getinstruction(s.guard)
		if len(s.assignments) > 1:
			result += ";"
		for i in range(0,len(s.assignments)):
			result += " " + getinstruction(s.assignments[i])
			if i < len(s.assignments)-1:
				result += ";"
		result += "]"
	elif s.__class__.__name__ == "AL_Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2" or s.__class__.__name__ == "ExprPrec1":
		if s.op != '':
			result += getinstruction(s.left) + " " + operator(s.op) + " " + getinstruction(s.right)
		else:
			result += getinstruction(s.left)
	elif s.__class__.__name__ == "Primary":
		result += sign(s.sign)
		if s.sign == "not":
			result += "("
		if s.value != None:
			newvalue = s.value
			result += str(newvalue).lower()
		elif s.ref != None:
			result += s.ref.ref
			if s.ref.index != None:
				result += "[" + getinstruction(s.ref.index) + "]"
		else:
			result += '(' + getinstruction(s.body) + ')'
		if s.sign == "not":
			result += ")"
	elif s.__class__.__name__ == "VariableRef":
		result += s.var.name
		if s.index != None:
			result += "[" + getinstruction(s.index) + "]"
	return result

def javatype(s):
	"""Maps type names from SLCO to Java"""
	if s.base == 'Integer':
		if s.size < 2:
			return 'int'
		else:
			return 'int[]'
	elif s.base == 'Boolean':
		if s.size < 2:
			return 'boolean'
		else:
			return 'boolean[]'
	elif s.base == 'Byte':
		if s.size < 2:
			return 'byte'
		else:
			return 'byte[]'

def javastatement(s,indent,nondet,o):
	"""Translates SLCO statement s to Java code. indent indicates how much every line needs to be indented, nlocks indicates how many locks need to be acquired (optional). nondet indicates whether this statement is at the head of a statement block and in a non-deterministic choice; it affects how expressions are translated.
	o is Object owning s."""
	global add_counter

	output = ""
	if s.__class__.__name__ == "AL_Assignment":
		output += getinstruction(s) + ";"
	elif s.__class__.__name__ == "MemoryFence":
		output += "fence_var = true;"
	elif s.__class__.__name__ == "AL_Expression":
		if not nondet:
#			if statement_readsfromlocked(s, o):
			output += "if(!(" + getinstruction(s) + ")) {"
			if add_counter:
				output += " java_transcounter++;"
		else:
			output += "if (!(" + getinstruction(s) + ")) {"
			if add_counter:
				output += "java_transcounter++; "
			#output += "if (!(" + getinstruction(s) + ")) { java_kp.unlock(java_lockIDs, "
			#output += str(nlocks)
			#output += "); "
			#if add_counter:
				#output += "java_transcounter++; "
			#output += "break; }"
	elif s.__class__.__name__ == "Composite":
		indentspace = ""
		for i in range(0,indent):
			indentspace += " "
		if s.guard != None:
			output += javastatement(s.guard,indent,nondet,o)
			if len(s.assignments) > 0:
				output += "\n" + indentspace + indentspace 
		first = True
		for e in s.assignments:
			if not first:
				output += "\n" + indentspace + indentspace
			else:
				first = False
			output += getinstruction(e) + ";"
	elif s.__class__.__name__ == "DoAction":
		output += 'System.out.println("%s");' % s.act
	return output

def expression(s,primmap):
	"""Maps SLCO expression to mCRL2 expression. Primmap is a dictionary for rewriting primaries."""
	output = ''
	# special case: s is a variableref. In this case, it is the left-hand side of an assignment, and s refers to an array
	if s.__class__.__name__ == "Variable":
		output = primmap.get(s.name, s.name)
	if s.__class__.__name__ == "AL_Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2":
		#if s.op != '':
		#	output = '('
		if s.op != '' and s.op != 'xor':
			output += expression(s.left,primmap) + " " + operator(s.op) + " " + expression(s.right,primmap)
		elif s.op == 'xor':
			output += operator(s.op) + '(' + expression(s.left,primmap) + ', ' + expression(s.right,primmap) + ')'
		else:
			output += expression(s.left,primmap)
	elif s.__class__.__name__ == "ExprPrec1":
		if s.op != '':
			output += operator(s.op) + '(' + expression(s.left,primmap) + ', ' + expression(s.right,primmap) + ')'
		else:
			output += expression(s.left,primmap)
	elif s.__class__.__name__ == "Primary":
		output = operator(s.sign)
		if s.value != None:
			newvalue = s.value
			output += str(newvalue).lower()
		elif s.ref != None:
			# is an index to an array given?
			key = s.ref.ref
			if s.ref.index != None:
				key += "[" + expression(s.ref.index,primmap) + "]"
			output += primmap.get(key, key)
		else:
			output += '(' + expression(s.body,primmap) + ')'
	return output

def statement_varids(s,primmap):
	"""Produce for the given statement a list of sorted lock ids to be acquired"""
	global varids
	output = []
	if s.__class__.__name__ == "VariableRef":
		# is the variable not local to state machine?
		if varids.get(s.var.name) != None:
			varid = varids[s.var.name]
			if s.index != None:
				varid += " + " + expression(s.index,primmap)
			output.append(varid)
	elif s.__class__.__name__ == "Composite":
		# update primmap
		if s.guard != None:
			output += statement_varids(s.guard,primmap)
		for e in s.assignments:
			output += statement_varids(e.left,primmap)
			output += statement_varids(e.right,primmap)
			newright = expression(e.right,primmap)
			varname = e.left.var.name
			if e.left.index != None:
				varname += "[" + expression(e.left.index,primmap) + "]"
			primmap[varname] = "(" + newright + ")"
	elif s.__class__.__name__ == "AL_Assignment":
		output += statement_varids(s.left,primmap)
		output += statement_varids(s.right,primmap)
	elif s.__class__.__name__ != "Primary":
		output += statement_varids(s.left,primmap)
		if s.op != '':
			output += statement_varids(s.right,primmap)
	else:
		if s.ref != None:
			# IGNORE STATE MACHINE LOCAL VARS!
			if varids.get(s.ref.ref) != None:
				varid = varids[s.ref.ref]
				if s.ref.index != None:
					varid += " + " + expression(s.ref.index,primmap)
				output.append(varid)
	# remove duplicates
	output = list(set(output))
	# sort output
	sortedoutput = sorted(output)
	return sortedoutput

def statement_writestolocked(s, o):
	"""Indicate whether the given statement writes to shared variables that require locking. o is owner Object of statement s."""
	global varids, lockingdict
	output = []
	if s.__class__.__name__ == "VariableRef":
		# is the variable not local to state machine?
		if varids.get(s.var.name) != None:
			if s.var.name in set(lockingdict[o.name]):
				return True
			elif s.index != None:
				index = expression(s.index,{})
				varname = s.var.name + index
				if varname in set(lockingdict[o.name]):
					return True
				elif not RepresentsInt(index):
					for v in lockingdict[o.name]:
						if s.var.name == (v.split("("))[0]:
							return True
					return False
				else:
					return False
			else:
				return False
		else:
			return False
	elif s.__class__.__name__ == "Composite":
		for e in s.assignments:
			result = statement_writestolocked(e.left, o)
			if result:
				return result
		return result
	elif s.__class__.__name__ == "AL_Assignment":
		return statement_writestolocked(s.left, o)
	else:
		return False

def statement_readsfromlocked(s, o):
	"""Indicate whether the given statement reads for its guard from shared variables that require locking. o is owner Object of statement s."""
	global varids, lockingdict
	output = []
	if s.__class__.__name__ == "VariableRef":
		# is the variable not local to state machine?
		if varids.get(s.var.name) != None:
			if s.var.name in set(lockingdict[o.name]):
				return True
			elif s.index != None:
				index = expression(s.index,{})
				varname = s.var.name + index
				if varname in set(lockingdict[o.name]):
					return True
				elif not RepresentsInt(index):
					for v in lockingdict[o.name]:
						if s.var.name == (v.split("("))[0]:
							return True
					return False
				else:
					return False
			else:
				return False
		else:
			return False
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			return statement_readsfromlocked(s.guard, o)
		else:
			return False
	elif s.__class__.__name__ == "AL_Assignment":
		return False
	elif s.__class__.__name__ != "Primary":
		result = statement_readsfromlocked(s.left, o)
		if result:
			return True
		elif s.op != '':
			return statement_readsfromlocked(s.right, o)
	else:
		if s.ref != None:
			# IGNORE STATE MACHINE LOCAL VARS!
			if varids.get(s.ref.ref) != None:
				if s.ref.ref in set(lockingdict[o.name]):
					return True
				elif s.ref.index != None:
					index = expression(s.ref.index,{})
					varname = s.ref.ref + index
					if varname in set(lockingdict[o.name]):
						return True
					elif not RepresentsInt(index):
						for v in lockingdict[o.name]:
							if s.ref.ref == (v.split("("))[0]:
								return True
						return False
					else:
						return False
				else:
					return False

def getvarids(s):
	return statement_varids(s,{})

def maxnumbervarids(s):
	"""For the given state machine s, determine max number of varids required for a statement contained in it"""
	maxnumber = 0
	for tr in s.transitions:
		for st in tr.statements:
			varids = statement_varids(st,{})
			if len(varids) > maxnumber:
				maxnumber = len(varids)
	return maxnumber

def variabledefault(s):
	""" return default value for given variable """
	if s.defvalue != None:
		return s.defvalue
	elif s.defvalues != []:
		defv = '['
		first = True
		for v in s.defvalues:
			if not first:
				defv += ','
			else:
				first = False
			defv += v
		defv += ']'
		return defv
	elif s.type.base == 'Integer' or s.type.base == 'Byte':
		if s.type.size < 2:
			return '0'
		else:
			type = '{'
			first = True
			for i in range(0,s.type.size):
				if not first:
					type += ','
				else:
					first = False
				type += '0'
			type += '}'
			return type
	elif s.type.base == 'Boolean':
		if s.type.size < 2:
			return 'true'
		else:
			type = '{'
			first = True
			for i in range(0,s.type.size):
				if not first:
					type += ','
				else:
					first = False
				type += 'true'
			type += '}'
			return type

def initialvalue(s,o):
	""" return initial value for given global variable in given object o """
	# try to find variable in initial assignments
	value = ''
	values = []
	for vinit in o.assignments:
		if s.name == vinit.left.name:
			if vinit.right != None:
				value = getinstruction(vinit.right)
			else:
				for v in vinit.rights:
					values.append(str(v).lower())
	if value != '':
		return value
	elif values != []:
		output = '{'
		first = True
		for v in values:
			if not first:
				output += ','
			else:
				first = False
			output += v
		output += "}"
		return output
	elif s.defvalue != None:
		return s.defvalue
	elif s.defvalues != []:
		defv = '{'
		first = True
		for v in s.defvalues:
			if not first:
				defv += ','
			else:
				first = False
			defv += str(v)
		defv += '}'
		return defv
	elif s.type.base == 'Integer' or s.type.base == 'Byte':
		if s.type.size < 2:
			return '0'
		else:
			type = '{'
			first = True
			for i in range(0,s.type.size):
				if not first:
					type += ','
				else:
					first = False
				type += '0'
			type += '}'
			return type
	elif s.type.base == 'Boolean':
		if s.type.size < 2:
			return 'true'
		else:
			type = '{'
			first = True
			for i in range(0,s.type.size):
				if not first:
					type += ','
				else:
					first = False
				type += 'true'
			type += '}'
			return type

def outgoingtrans(s,t):
	"""Return the set of transitions with s as source"""
	tlist = []
	for tr in t:
		if tr.source.name == s.name:
			tlist.append(tr)
	return tlist

def hasoutgoingtrans(s,t):
	"""Check whether state s has outgoing transitions in t"""
	for tr in t:
		if tr.source.name == s.name:
			return True
	return False

def preprocess(model):
	"""Preprocess the model"""
	global states, numberofelemvariables, smlocalvars, varids
	# for each state machine, add the initial state to its list of states
	for c in model.classes:
		for stm in c.statemachines:
			stm.states = [stm.initialstate] + stm.states

	# fill in types of variables
	for c in model.classes:
		for i in range(0,len(c.variables)):
			if c.variables[i].type == None:
				c.variables[i].type = c.variables[i-1].type
		for sm in c.statemachines:
			for i in range(0,len(sm.variables)):
				if sm.variables[i].type == None:
					sm.variables[i].type = sm.variables[i-1].type
	# fill set of state names
	states  = set([])
	for c in model.classes:
		for sm in c.statemachines:
			for s in sm.states:
				states.add(s.name)
	# compute number of elementary variables
	numberofelemvariables = 0
	for c in model.classes:
		for v in c.variables:
			if v.type.size > 1:
				numberofelemvariables += v.type.size
			else:
				numberofelemvariables += 1
	# build set of state machine local variable names
	smlocalvars = set([])
	for c in model.classes:
		for sm in c.statemachines:
			for v in sm.variables:
				smlocalvars.add(v.name)
	# build dictionary of global variables with ids
	varids = {}
	# first create list of global variables and sort it
	varlist = []
	for c in model.classes:
		for v in c.variables:
			varlist.append(v)
	sortedlist = sorted(varlist, key=lambda v: v.name)
	# create dict
	count = 0
	for v in sortedlist:
		varids[v.name] = str(count)
		if v.type.size > 1:
			count += v.type.size
		else:
			count += 1
	return model

lockscanner=re.Scanner([
	(r"[A-Za-z0-9]+:",											lambda scanner,token:("OBJECT", token)),
	(r"[A-Za-z0-9]+(\([0-9]+\))?",			lambda scanner,token:("VARIABLE", token)),
	(r"[\s\(\),]+", None), # None == skip token.
])

def read_locking_file(model,lockingfilename):
	"""Read locking file and return dict of vars to lock"""
	global varids
	lockdict = {}
	if lockingfilename != '':
		try:
			inFile = open(lockingfilename,'r')
		except IOError:
			print("ERROR: locking file does not exist!")
			exit(1);
		inFile.close()
		key = ''
		for line in open(lockingfilename):
			remainder = line
			while remainder != "":
				results, remainder = lockscanner.scan(remainder)
			if results[0][0] == 'OBJECT':
				key = results[0][1][:-1]
				# lock all variables in case we see 'all_objects'
				if key == "all_objects":
					lockingfilename = '';
					break
			else:
				varlist = []
				for i in range(0,len(results)):
					if results[i][0] == 'VARIABLE':
						varlist.append(results[i][1])
				oldlist = lockdict.get(key,[])
				varlist += oldlist
				lockdict[key] = varlist
	if lockingfilename == '':
		# fill lockdict with all variables in the system
		for o in model.objects:
			varlist = []
			for v in o.type.variables:
				varname = v.name
				if v.type.size > 1:
					for i in range(0,v.type.size):
						varlist.append(varname + "(" + str(i) + ")")
				else:
					varlist.append(varname)
			lockdict[o.name] = varlist
	return lockdict

def slco_to_java(modelfolder,modelname,model,lockingfilename):
	"""The translation function"""
	global states, varids, numberofelemvariables, lockingdict, add_counter
	outFile = open(os.path.join(modelfolder,model.name + ".java"), 'w')

	# prepare lockingdict information for code generator
	# read locking file
	lockingdict = read_locking_file(model,lockingfilename)
	lockids = []
	for o in model.objects:
		locklist = lockingdict.get(o.name,[])
		for v in locklist:
			vsplit = v.split('(')
			vid = int(varids.get(vsplit[0],"0"))
			if len(vsplit) > 1:
				vx = vsplit[1].rstrip(')')
				vid = vid + int(vx)
			lockids.append(vid)
	lockids = set(lockids)
	lockneeded = []
	for i in range(0,numberofelemvariables):
		lockneeded.append(i in lockids)

	# Initialize the template engine.
	#jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(this_folder), trim_blocks=True, lstrip_blocks=True,
	#	extensions=['jinja2.ext.loopcontrols','jinja2.ext.do',])
	jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.join(this_folder, '../../jinja2_templates')),
		trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols', 'jinja2.ext.do', ])

	# Register the filters
	jinja_env.filters['getvarids'] = getvarids
	jinja_env.filters['statement_writestolocked'] = statement_writestolocked
	jinja_env.filters['statement_readsfromlocked'] = statement_readsfromlocked
	jinja_env.filters['maxnumbervarids'] = maxnumbervarids
	jinja_env.filters['getlabel'] = getlabel
	jinja_env.filters['variabledefault'] = variabledefault
	jinja_env.filters['initialvalue'] = initialvalue
	jinja_env.filters['javatype'] = javatype
	jinja_env.filters['outgoingtrans'] = outgoingtrans
	jinja_env.filters['javastatement'] = javastatement

	# Register the tests
	jinja_env.tests['hasoutgoingtrans'] = hasoutgoingtrans

	# load the Java template
	template = jinja_env.get_template('java_transmem.jinja2template')

	# write the program
	outFile.write(template.render(model=model,states=states,numberofelemvariables=numberofelemvariables,lockneeded=lockneeded,add_counter=add_counter))
	outFile.close()

def main(args):
	"""The main function"""
	global numberofelemvariables, lockingdict, add_counter
	lockingfilename = ''
	if len(args) == 0:
		print("Missing argument: SLCO model")
		sys.exit(1)
	else:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 slco2java")
			print("")
			print("Transform an SLCO 2.0 model to a Java program.")
			print("-l <file>                            provide locking file for smart locking")
			print("-c                                   produce a transition counter in the code, to make program executions finite")
			sys.exit(0)
		else:
			i = 0
			while i < len(args):
				if args[i] == '-l':
					i += 1
					if i >= len(args):
						print("ERROR: lock file name missing!")
						exit(1);
					lockingfilename = args[i];
				elif args[i] == '-c':
					add_counter = True
				else:
					modelfolder, modelname = os.path.split(args[i])
				i += 1

	# create meta-model
	slco_mm = metamodel_from_file(os.path.join(this_folder,'../../textx_grammars/slco2-al.tx'))
	# read model
	model = slco_mm.model_from_file(os.path.join(modelfolder,modelname))
	# preprocess
	model = preprocess(model)
	# translate
	slco_to_java(modelfolder,modelname,model,lockingfilename)

	for o in model.objects:
		lockedvars = lockingdict.get(o.name,[])
		lockedvars = set(lockedvars)
	print("SLCO model translated to Java")
	print("Ratio of locked variables/total number: " + str(len(lockedvars)) + "/" + str(numberofelemvariables))

if __name__ == '__main__':
	args = []
	for i in range(1,len(sys.argv)):
		args.append(sys.argv[i])
	main(args)
