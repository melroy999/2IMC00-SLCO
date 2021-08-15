import sys
from os import mkdir
from os.path import exists, dirname, basename, join, split
import jinja2
from textx.metamodel import metamodel_from_file
from copy import deepcopy
import itertools
import glob
import traceback

# powerset function
def powerset(iterable):
		"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
		s = list(iterable)
		return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

# flag to indicate whether race condition checking will be conducted on resulting mCRL2 model
check_rc = False

modelname = ""
model = ""
this_folder = dirname(__file__)

# import libraries
sys.path.append(join(this_folder,'../../libraries'))
from slcolib import *
from SCCTarjan import identifySCCs
this_folder = dirname(__file__)

porttypes = {}
# statemachine names used in the model
statemachinenames = set([])
# dictionary to look up state machine owning a given statement
statemachine = {}
# dictionary to look up transition owning a given statement
tr = {}
# per class / statemachine, give a set of local variables
smlocalvars = {}
# per class / statemachine, give a dictionary, making the scope of variables explicit
scopedvars = {}
# set of variables (with scopes) appearing in the model
modelvars = set([])
vardict = {}
states = set([])
# dictionary providing the objects of a particular class
classobjects = {}
# dictionary providing for given (send and receive) statement labels the corresponding pair of labels consisting of the synchronising action and the resulting action
syncactionlabelsdict = {}

channeltypes = {}
asynclosslesstypes = set([])
asynclossytypes = set([])
synctypes = set([])

# actions and variables defined in the model
actions = set([])

# structures to handle transition priorities
guard_priority = set([])
guard_priority_in_construction = set([])
# dictionary of guarded statements for each class
sync_guarded_statements = {}
# dictionary of sids for each guarded action label
sync_guarded_sids = {}
# stores most recently assigned sid (in list, in order to allow updating on Jinja2 template side)
previous_sid = ['']
sync_guard_rules = []

# number of receive statements appearing in each class specification
class_receives = {}

# currently recorded guard actions (not conditional receives!) for each object of a given class
# key: object
# value: set of action labels
sync_guard_actions = {}
# the same as sync_guard_actions, but it records dependencies between objects
sync_guard_objects = {}
# currently recorded guard conditional receives for each object of a given class
sync_guard_receives_actions = {}
# the same, recording the objects to which the receives are connected
sync_guard_receives_objects = {}
# per class keep track of the conditional receives, and all combinations of them
class_sguard_creceives = []
class_sguard_creceives_combs = []
# object to object sync summands to be added to mCRL2 processes to handle prioritised transitions involving direct syncs between objects
# key: class for which summands are recorded
# value: dict with key: label of the statement, and value: tuple consisting of 1: one of the Statement objects for which summand is recorded, and 2: guard of the summand
o2o_sync_summands = {}
current_priority_guard = 0
current_priority_sync = 0
current_sm = ''
current_state = ''

# definitions of action labels to send/receive messages to/from a channel in mCRL2 (object side)
mcrl2objectsend = "snd"
mcrl2objectreceive = "rec"
mcrl2objectpeek = "pk"
mcrl2objectshow = "show"
# the actions for successful communication between an object and a channel (asynchronous)
mcrl2commsend = "send"
mcrl2commreceive = "receive"
mcrl2commpeek = "peek"
mcrl2readwrite = "rw"
# the action for successful communication between objects (synchronous)
mcrl2comm = "comm"
# the action to enable all receive actions from a state again
mcrl2resetrecs = "resetrecs"
# prefix to refer to variable names in mCRL2 model (for race condition checking)
mcrl2varprefix = "var_"
# visible actions used in the different classes
visibleactions = {}
# a dictionary to build up transition guards (to handle transition priorities in SLCO)
guarddict = {}

# *** BEGIN TRANSLATION FILTERS ***

def checkforsynccondition(s,c,i,firstinblock):
	""" Check whether for statement s, an additional sync condition (in the form of an extra mCRL2 summand) should be added. This is to handle prioritised transitions. Class c owning the statement is also given. i is the current index of the statement as considered in the template. firstinblock is a Boolean indicating whether the given statement is the first in a statement block"""
	global classobjects, syncactionlabelsdict, o2o_sync_summands, guard_priority, statemachine, tr
	is_synchronous = False
	for o in classobjects.get(c,set([])):
		action = statementactionlabel(s,o,False)
		if syncactionlabelsdict[action][2]:
			is_synchronous = True
			break
	if is_synchronous:
		summands = o2o_sync_summands.get(c,{})
		label = statementactionlabel(s,c,False)
		if s.__class__.__name__ == 'ReceiveSignal':
			label = statementactionlabel(s,c,True)
		summandtuple = summands.get(label,tuple([s,set([])]))
		stat = summandtuple[0]
		guard = summandtuple[1]
		newguard = "!(" + statemachine[s].name + "_state == " + tr[s].source.name + " && " + statemachine[s].name + "_stindex == " + str(i)
		if s.__class__.__name__ == 'ReceiveSignal':
			newguard += " && rec_enabled" + str(s._tx_position)
		if firstinblock:
			for g in guard_priority:
				newguard += " && " + g
		newguard += ")"
		guard.add(newguard)
		summands[label] = tuple([stat,guard])
		o2o_sync_summands[c] = summands

def statementguard(s,c,i,firstinblock,negated):
	"""Maps SLCO statement guards to mCRL2 action guards, possibly negated, if negated == True. Class c owning the statement is also given. i is the current index of the statement as considered in the template. firstinblock is a Boolean indicating whether the given statement is the first in a statement block"""
	global porttypes, guard_priority, guard_priority_in_construction, current_priority_guard, current_sm, current_state, o2o_sync_summands, syncactionlabelsdict, classobjects, sync_guard_actions, sync_guard_objects, sync_guard_receives_actions, sync_guard_receives_objects, class_sguard_creceives, class_sguard_creceives_combs, scopedvars, statemachine, tr
	output = ''
	# possibly reset the priority guards
	if firstinblock and not negated:
		if statemachine[s] != current_sm or current_state != tr[s].source.name:
			current_sm = statemachine[s]
			current_state = tr[s].source.name
			# reset guards
			guard_priority = set([])
			guard_priority_in_construction = set([])
			# reset sync actions and objects
			sync_guard_actions = {}
			sync_guard_objects = {}
			sync_guard_receives_actions = {}
			sync_guard_receives_objects = {}
			class_sguard_creceives[:] = []
			class_sguard_creceives_combs[:] = []	# possibly reset the priority guards
			# update priority
			current_priority_guard = tr[s].priority
		elif current_priority_guard < tr[s].priority:
			guard_priority |= guard_priority_in_construction
			guard_priority_in_construction = set([])
			current_priority_guard = tr[s].priority
		# add current priority guard to output
		first = True
		for g in guard_priority:
			if not first:
				output += " && "
			else:
				first = False
			output += g
	if s == "tau'":
		if negated:
			if output != '':
				output += " && "
			output += "False"
		# add negation of guard to priority guard under construction, if statement first in block
		if firstinblock and not negated:
			guard_priority_in_construction.add("!(true)")
	elif s.__class__.__name__ == "Assignment":
		if negated:
			if output != '':
				output += " && "
			output += "False"
		# add negation of guard to priority guard under construction, if statement first in block
		if firstinblock and not negated:
			guard_priority_in_construction.add("!(true)")
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			if output != '':
				output += " && "
			guard = expression(s.guard,statemachine[s],c,{})
			if negated:
				output += "!(" + guard + ")"
			else:
				output += guard
			# add negation of guard to priority guard under construction, if statement first in block
			if firstinblock and not negated:
				guard_priority_in_construction.add("!(" + guard + ")")
	elif s.__class__.__name__ == "Delay":
		if output != '':
			output += " && "
		if negated:
			output += "False"
		# add negation of guard to priority guard under construction, if statement first in block
		if firstinblock and not negated:
			guard_priority_in_construction.add("!(true)")
	elif s.__class__.__name__ == "SendSignal":
		if output != '':
			output += " && "
		if negated:
			output += "False"
		# to handle priorities, check for all objects of type c whether this statement is synchronous (object to object). If so, record an additional summand encoding the inability to perform this statement
		checkforsynccondition(s,c,i,firstinblock)
	elif s.__class__.__name__ == "ReceiveSignal":
		if output != '':
			output += " && "
		if negated:
			output += "False && "
		# to handle priorities, check for all objects of type c whether this statement is synchronous (object to object). If so, record an additional summand encoding the inability to perform this statement
		checkforsynccondition(s,c,i,firstinblock)
		# add the requirement that the corresponding Boolean flag must be true
		output += "rec_enabled" + str(s._tx_position)
	elif s.__class__.__name__ == "Expression":
		if not expression_is_actionref(s):
			if output != '':
				output += " && "
			guard = expression(s,statemachine[s],c,scopedvars[c.name + "'" + statemachine[s].name])
			if negated:
				output += "!(" + guard + ")"
			else:
				output += guard
			# add negation of guard to priority guard under construction, if statement first in block
			if firstinblock and not negated:
				guard_priority_in_construction.add("!(" + guard + ")")
	if output == '':
		return output
	else:
		return " && (" + output + ")"

def statementsummation(s,c):
	"""Produce mCRL2 variable summations for the given SLCO statement. Class c owns the statement"""
	global check_rc, vardict, porttypes, scopedvars
	output = ''
	if check_rc:
		if s == "tau'":
			output = ""
		elif s.__class__.__name__ == "Assignment":
			output = ""
		elif s.__class__.__name__ == "Composite":
			output = ""
		elif s.__class__.__name__ == "Delay":
			output = ""
		elif s.__class__.__name__ == "SendSignal":
			output = "sum SM': Statemachine, R': Set(Var), W': Set(Var). "
		elif s.__class__.__name__ == "ReceiveSignal":
			vardict = {}
			pindex = 0
			output += "sum SM': Statemachine, R': Set(Var), W': Set(Var), "
			# add a variable for the signal
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += "s': Signal'"
			for d in type:
				output += datatypeacronym(d)
			for p in s.params:
				output += ", x'" + str(pindex) + ": " + type[pindex]
				# add this variable to the vardict
				varname = scopedvars[c.name + "'" + statemachine[s].name][p.var.name]
				vardict[varname] = "x'" + str(pindex)
				pindex += 1
			output += ". "
		elif s.__class__.__name__ == "Expression":
			output = ""
	else:
		if s == "tau'":
			output = ""
		elif s.__class__.__name__ == "Assignment":
			output = ""
		elif s.__class__.__name__ == "Composite":
			output = ""
		elif s.__class__.__name__ == "Delay":
			output = ""
		elif s.__class__.__name__ == "SendSignal":
			output = ""
		elif s.__class__.__name__ == "ReceiveSignal":
			vardict = {}
			pindex = 0
			output += "sum "
			# add a variable for the signal
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += "s': Signal'"
			for d in type:
				output += datatypeacronym(d)
			for p in s.params:
				output += ", x'" + str(pindex) + ": " + type[pindex]
				# add this variable to the vardict
				vardict[p.var.name] = "x'" + str(pindex)
				pindex += 1
			output += ". "
		elif s.__class__.__name__ == "Expression":
			output = ""
	return output

def statementactionlabel(s,c,includesignal):
	"""Maps SLCO statements to mCRL2 action labels, Class (or Object) c owns the statement. includesignal is a Boolean indicating whether in case of Send and ReceiveSignals, the signal name should be included in the label"""
	global check_rc, statemachine
	output = ''
	readwrite = False
	if check_rc:
		# produce output for race condition checking
		if s == "tau'":
			output = "tau'"
		elif s.__class__.__name__ == "Assignment":
			output = mcrl2readwrite + "_" + c.name
			readwrite = True
		elif s.__class__.__name__ == "Composite":
			output = mcrl2readwrite + "_" + c.name
			readwrite = True
		elif s.__class__.__name__ == "Delay":
			output = "tau'"
		elif s.__class__.__name__ == "SendSignal":
			output += mcrl2objectsend + "_"
			if includesignal:
				output += s.signal + "_"
			output += c.name + "'" + s.target.name
		elif s.__class__.__name__ == "ReceiveSignal":
			output += mcrl2objectreceive + "_"
			if includesignal:
				output += s.signal + "_"
			output += c.name + "'" + s.target.name
		elif s.__class__.__name__ == "Expression":
			if expression_is_actionref(s):
				#output += expression(s,statemachine[s],c,{})
				output += "tau'"
			else:
				output = mcrl2readwrite + "_" + c.name
				readwrite = True
		# add the object name to the label, in case we call this function with c being an object
		if (output == "tau'" or expression_is_actionref(s)) and c.__class__.__name__ == "Object":
			output += "_" + c.name
	else:
		if s == "tau'":
			output = "tau'"
		elif s.__class__.__name__ == "Assignment":
			output = "tau'"
		elif s.__class__.__name__ == "Composite":
			output = "tau'"
		elif s.__class__.__name__ == "Delay":
			output = "delay"
		elif s.__class__.__name__ == "SendSignal":
			output += mcrl2objectsend + "_"
			if includesignal:
				output += s.signal + "_"
			output += c.name + "'" + s.target.name
		elif s.__class__.__name__ == "ReceiveSignal":
			output += mcrl2objectreceive + "_"
			if includesignal:
				output += s.signal + "_"
			output += c.name + "'" + s.target.name
		elif s.__class__.__name__ == "Expression":
			if expression_is_actionref(s):
				output += expression(s,statemachine[s],c,{})
			else:
				output += "tau'"
		# add the object name to the label, in case we call this function with c being an object
		if (output == "tau'" or expression_is_actionref(s)) and c.__class__.__name__ == "Object":
			output += "_" + c.name
	return output

def statementparameters(s,c):
	"""Produce mCRL2 action parameters for the given SLCO statement. Class c owns the statement"""
	global check_rc, statemachine, scopedvars
	output = ''
	if check_rc:
		if s == "tau'":
			output = ""
		elif s.__class__.__name__ == "Assignment":
			output = "(" + statemachine[s].name + ",{"
			varlist = list(expression_varset(s.right,statemachine[s],c,{}))
			varlist = sorted(varlist)
			first = True
			for v in varlist:
				if not first:
					output += ","
				else:
					first = False
				output += mcrl2varprefix + v
			output += "},{"
			writeset = expression_varset(s.left,statemachine[s],c,{})
			for v in writeset:
				output += mcrl2varprefix + v
			output += "})"
			readwrite = True
		elif s.__class__.__name__ == "Composite":
			output = "(" + statemachine[s].name + ",{"
			vardict = {}
			readset = set([])
			writeset = set([])
			if s.guard != None:
				readset |= expression_varset(s.guard,statemachine[s],c,vardict)
			for st in s.assignments:
				readset |= expression_varset(st.right,statemachine[s],c,vardict)
				writeset |= expression_varset(st.left,statemachine[s],c,vardict)
				# update vardict (to correctly handle possible array index use in subsequent assignments)
				newright = expression(st.right,statemachine[s],c,vardict)
				varname = scopedvars[c.name + "'" + statemachine[s].name][st.left.var.name]
				if st.left.index != None:
					vardict[varname] = "update(" + expression(st.left.var,statemachine[s],c,vardict) + ",Int2Nat(" + expression(st.left.index,statemachine[s],c,vardict) + ")," + newright + ")"
				else:
					vardict[varname] = "(" + newright + ")"
			varlist = sorted(list(readset))
			first = True
			for v in varlist:
				if not first:
					output += ","
				else:
					first = False
				output += mcrl2varprefix + v
			output += "},{"
			varlist = sorted(list(writeset))
			first = True
			for v in varlist:
				if not first:
					output += ","
				else:
					first = False
				output += mcrl2varprefix + v
			output += "})"
		elif s.__class__.__name__ == "Delay":
			output = ""
		elif s.__class__.__name__ == "SendSignal":
			output = "(" + statemachine[s].name + ",{"
			varset = set([])
			for st in s.params:
				varset |= expression_varset(st,statemachine[s],c,{})
			varlist = sorted(list(varset))
			first = True
			for v in varlist:
				if not first:
					output += ","
				else:
					first = False
				output += mcrl2varprefix + v
			output += "},{},SM',R',W'," + s.signal
			for p in s.params:
				output += "," + expression(p,statemachine[s],c,{})
			output += ")"
		elif s.__class__.__name__ == "ReceiveSignal":
			pindex = 0
			output = "(SM',R',W'," + statemachine[s].name + ",{"
			writeset = set([])
			for st in s.params:
				writeset |= expression_varset(st,statemachine[s],c,{})
			readset = set([])
			if s.guard != None:
				readset = expression_varset(s.guard,statemachine[s],c,{})
			# in SLCO ReceiveSignal, it is not possible to refer to the old value of a variable to which you are reading. Hence, reading AND writing to the same variable cannot occur
			readset = readset - writeset
			varlist = sorted(list(readset))
			first = True
			for v in varlist:
				if not first:
					output += ","
				else:
					first = False
				output += mcrl2varprefix + v
			output += "},{"
			varlist = sorted(list(writeset))
			first = True
			for v in varlist:
				if not first:
					output += ","
				else:
					first = False
				output += mcrl2varprefix + v
			output += "},s'"
			for p in s.params:
				output += ",x'" + str(pindex)
			output += ")"
		elif s.__class__.__name__ == "Expression":
			if expression_is_actionref(s):
				output = ""
			else:
				output += "(" + statemachine[s].name + ",{"
				varlist = list(expression_varset(s,statemachine[s],c,{}))
				varlist = sorted(varlist)
				first = True
				for v in varlist:
					if not first:
						output += ","
					else:
						first = False
					output += mcrl2varprefix + v
				output += "},{})"
	else:
		if s == "tau'":
			output = ""
		elif s.__class__.__name__ == "Assignment":
			output = ""
		elif s.__class__.__name__ == "Composite":
			output = ""
		elif s.__class__.__name__ == "Delay":
			output = "(" + str(s.length) + ")"
		elif s.__class__.__name__ == "SendSignal":
			output = "(" + s.signal
			for p in s.params:
				output += ", " + expression(p,statemachine[s],c,{})
			output += ")"
		elif s.__class__.__name__ == "ReceiveSignal":
			pindex = 0
			output = "(s'"
			for p in s.params:
				output += ", x'" + str(pindex)
			output += ")"
		elif s.__class__.__name__ == "Expression":
			output = ""
	return output

def statementactiontype(s,c):
	"""Produce the type of the given statement. Class c owns the statement"""
	global porttypes, check_rc
	output = ''
	if check_rc:
		if s == "tau'":
			output = ""
		elif s.__class__.__name__ == "Assignment":
			output = ": Statemachine # Set(Var) # Set(Var)"
		elif s.__class__.__name__ == "Composite":
			output = ": Statemachine # Set(Var) # Set(Var)"
		elif s.__class__.__name__ == "Delay":
			output = ""
		elif s.__class__.__name__ == "SendSignal":
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += ": Statemachine # Set(Var) # Set(Var) # Statemachine # Set(Var) # Set(Var) # Signal'"
			params = ""
			for t in type:
				output += datatypeacronym(t)
				params += " # " + str(t)
			output += params
		elif s.__class__.__name__ == "ReceiveSignal":
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += ": Statemachine # Set(Var) # Set(Var) # Statemachine # Set(Var) # Set(Var) # Signal'"
			params = ""
			for t in type:
				output += datatypeacronym(t)
				params += " # " + str(t)
			output += params
		elif s.__class__.__name__ == "Expression":
			if expression_is_actionref(s):
				output = ""
			else:
				output = ": Statemachine # Set(Var) # Set(Var)"
	else:
		if s == "tau'":
			output = ""
		elif s.__class__.__name__ == "Assignment":
			output = ""
		elif s.__class__.__name__ == "Composite":
			output = ""
		elif s.__class__.__name__ == "Delay":
			output = "Int"
		elif s.__class__.__name__ == "SendSignal":
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += ": Signal'"
			params = ""
			for t in type:
				output += datatypeacronym(t)
				params += " # " + str(t)
			output += params
		elif s.__class__.__name__ == "ReceiveSignal":
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += ": Signal'"
			params = ""
			for t in type:
				output += datatypeacronym(t)
				params += " # " + str(t)
			output += params
		elif s.__class__.__name__ == "Expression":
			output = ""
	return output

def statementlabel(s,c,firstinblock,elseindex):
	"""Maps SLCO statements to mCRL2 behaviour. Class (or Object) c owning this statement is also given. Flag firstinblock indicates whether the statement is the first in a statement block. elseindex is an index for the statement in a block of else alternatives. If it is not set to 0, it should be added to the statement label"""
	global porttypes, vardict, sync_guard_rules, sync_guard_actions, sync_guard_objects, current_priority_sync, sync_guarded_statements, sync_guarded_sids, syncactionlabelsdict, classobjects, sync_guard_receives_actions, sync_guard_receives_objects, class_sguard_creceives, class_sguard_creceives_combs, previous_sid, statemachine, tr
	output = ''
	statement_id = ''
	if elseindex != 0:
		statement_id = "_sid" + str(s._tx_position)
	if firstinblock:
		if current_priority_sync < tr[s].priority:
			# if appropriate, create new sync rules
			if sync_guard_actions != {} or class_sguard_creceives != []:
				# iterate over all possible subsets of conditional receives
				i = 0
				class_sguard_creceives_combs[:] = []
				for crecs in powerset(class_sguard_creceives):
					# store combination
					if len(crecs) > 0:
						class_sguard_creceives_combs.append(set(crecs))
					# get new statement_id, add it to current statement label
					statement_id = "_sid" + str(s._tx_position)
					# first add this statement to the list of sync guarded statements of the current class
					statset = sync_guarded_statements.get(c,set([]))
					if i == 0:
						statset.add(tuple([s, statement_id]))
					else:
						statset.add(tuple([s, statement_id + "_" + str(i)]))
					sync_guarded_statements[c] = statset
					# for all objects of the current class, create a sync rule
					for o in classobjects.get(c,set([])):
						oset = sync_guard_actions.get(o,set([]))
						objset = sync_guard_objects.get(o,set([]))
						newset = deepcopy(oset)
						newobjset = objset
						# add subset of conditional receive 'denials'
						for stat in set(class_sguard_creceives):
							if stat not in set(crecs):
								newset.add(sync_guard_receives_actions[o][stat])
								newobjset.add(sync_guard_receives_objects[o][stat])
						# add current statement
						action = statementactionlabel(s,o,False)
						if s.__class__.__name__ == "ReceiveSignal":
							action = peekstatementlabel(s,o)
						# if appropriate, add action and object with which current statement needs to synchronise
						syncaction = ''
						if syncactionlabelsdict.get(action) != None:
							syncaction = syncactionlabelsdict[action][0][0] + syncactionlabelsdict[action][0][1]
							newobjset.add(syncactionlabelsdict[action][0][2])
						postfix = statement_id
						if i != 0:
							postfix += "_" + str(i)
						renamedaction = deepcopy(action)
						if statementactionlabel(s,o.type,False) == "tau'":
							renamedaction = "tau'"
						# remember the use of the current postfix for the current action
						sidlist = sync_guarded_sids.get(action,[])
						sidlist.append(postfix)
						sync_guarded_sids[action] = sidlist
						# add the current object itself to newobjset
						newobjset.add(o)
						sync_guard_rules.append(tuple([tuple([action + postfix, renamedaction]), tuple(newset), syncaction, newobjset]))
						# create another version of this rule for a show action, in case the current statement is a SendSignal
						if s.__class__.__name__ == "SendSignal":
							showaction = peekstatementlabel(s,o)
							syncaction = syncactionlabelsdict[showaction][0][0] + syncactionlabelsdict[showaction][0][1]
							# remember the use of the current postfix for the current action
							sidlist = sync_guarded_sids.get(showaction,[])
							sidlist.append(postfix)
							sync_guarded_sids[showaction] = sidlist
							sync_guard_rules.append(tuple([tuple([showaction + postfix, "show_" + o.name + "'" + s.target.name]), tuple(newset), syncaction, newobjset]))
					i += 1
			current_priority_guard = tr[s].priority
	if s == "tau'":
		output = statementactionlabel(s,c,False) + statement_id
		if elseindex != 0:
			output += "_" + str(elseindex)
	elif s.__class__.__name__ == "Assignment":
		output = statementactionlabel(s,c,False) + statement_id
		if elseindex != 0:
			output += "_" + str(elseindex)
	elif s.__class__.__name__ == "Composite":
		output = statementactionlabel(s,c,False) + statement_id
		if elseindex != 0:
			output += "_" + str(elseindex)
	elif s.__class__.__name__ == "Delay":
		output = statementactionlabel(s,c,False) + statement_id
		if elseindex != 0:
			output += "_" + str(elseindex)
	elif s.__class__.__name__ == "SendSignal":
		output += statementactionlabel(s,c,False) + statement_id
		if elseindex != 0:
			output += "_" + str(elseindex)
		# possibly add a new sync action to priority guard
		if firstinblock:
			# async or sync?
			for o in classobjects.get(c,set([])):
				sset = sync_guard_actions.get(o,set([]))
				oset = sync_guard_objects.get(o,set([]))
				if syncactionlabelsdict[statementactionlabel(s,o,False)][2]:
					# add connected action
					sset.add("cannot'" + syncactionlabelsdict[statementactionlabel(s,o,False)][0][0] + s.signal + "_" + syncactionlabelsdict[statementactionlabel(s,o,False)][0][1])
					oset.add(syncactionlabelsdict[statementactionlabel(s,o,False)][0][2])
				else:
					sset.add("cannot'insert_" + o.name + "'" + s.target.name)
					oset.add(o)
				sync_guard_actions[o] = sset
				sync_guard_objects[o] = oset
	elif s.__class__.__name__ == "ReceiveSignal":
		# construct action parameters
		pindex = 0
		params = "(s'"
		for p in s.params:
			params += ", x'" + str(pindex)
			pindex += 1
		params += ")"
		# add a peek action
		output += peekstatementlabel(s,c) + statement_id
		if elseindex != 0:
			output += "_" + str(elseindex)
		output += peekstatementparameters(s,c) + ".((s'==" + s.signal
		sguard = expression(s.guard,statemachine[s],c,vardict)
		if sguard != '':
			output += " && " + expression(s.guard,statemachine[s],c,vardict)
		output += ") -> "
		output += statementactionlabel(s,c,False)
		output += statement_id
		if elseindex != 0:
			output += "_" + str(elseindex)
		# possibly add a new sync action to priority guard
		if firstinblock:
			# async or sync?
			for o in classobjects.get(c,set([])):
				newaction = ''
				newobject = ''
				if syncactionlabelsdict[statementactionlabel(s,o,False)][2]:
					# add connected action and object
					newaction = "cannot'" + syncactionlabelsdict[statementactionlabel(s,o,False)][0][0] + syncactionlabelsdict[statementactionlabel(s,o,False)][0][1]
					newobject = syncactionlabelsdict[statementactionlabel(s,o,False)][0][2]
				else:
					newaction = "cannot'remove_" + o.name + "'" + s.target.name
					newobject = o
				# store this action in list of receives
				if s not in set(class_sguard_creceives):
					class_sguard_creceives.append(s)
				sdict = sync_guard_receives_actions.get(o,{})
				sdict[s] = newaction
				sync_guard_receives_actions[o] = sdict
				odict = sync_guard_receives_objects.get(o,{})
				odict[s] = newobject
				sync_guard_receives_objects[o] = odict
	elif s.__class__.__name__ == "Expression":
		output += statementactionlabel(s,c,False) + statement_id
		if elseindex != 0:
			output += "_" + str(elseindex)
	# set previous sid
	pid = statement_id
	if elseindex != 0:
		pid += "_" + str(elseindex)
	previous_sid[0] = pid
	return output

def statementstatechanges(s,c):
	"""Maps SLCO statement effects to mCRL2 state changes. Class c owning the statement is also given"""
	global porttypes, vardict, scopedvars, statemachine
	output = ''
	if s == "tau'":
		output = ""
	elif s.__class__.__name__ == "ActionRef":
		output = ""
	elif s.__class__.__name__ == "Assignment":
		varname = scopedvars[c.name + "'" + statemachine[s].name][s.left.var.name]
		#if s.left.var.name in smlocalvars.get(c.name + "'" + statemachine[s].name,set([])):
		#	varname = statemachine[s].name + "'" + s.left.var.name
		#else:
		#	varname = s.left.var.name
		# an assignment to an array cell should be handled differently from other cases
		if s.left.index != None:
			output = ", " + varname + "=update(" + varname + ",Int2Nat(" + expression(s.left.index,statemachine[s],c,{}) + "),"
			# in case we are updating a Byte, restrict the new value
			eright = expression(s.right,statemachine[s],c,{})
			if s.left.var.type.base == 'Byte':
				eright = "(" + eright + ") mod 256"
			output += eright + ")"
		else:
			output = ", " + varname
			eright = expression(s.right,statemachine[s],c,{})
			if s.left.var.type.base == 'Byte':
				eright = "(" + eright + ") mod 256"
			output += "=" + eright
	elif s.__class__.__name__ == "Composite":
		# first build vardict for sequence of assignments
		vardict = {}
		for e in s.assignments:
			newright = expression(e.right,statemachine[s],c,vardict)
			# in case we are updating a Byte, restrict the new value
			if e.left.var.type.base == 'Byte':
				newright = "(" + newright + ") mod 256"
			varname = scopedvars[c.name + "'" + statemachine[s].name][e.left.var.name]
			if e.left.index != None:
				vardict[varname] = "update(" + expression(e.left.var,statemachine[s],c,vardict) + ",Int2Nat(" + expression(e.left.index,statemachine[s],c,vardict) + ")," + newright + ")"
			else:
				vardict[varname] = "(" + newright + ")"
		# add assignments to state changes
		for key, value in vardict.items():
			output += ", " + key + "=" + value
	elif s.__class__.__name__ == "Delay":
		output = ""
	elif s.__class__.__name__ == "SendSignal":
		output = ""
	elif s.__class__.__name__ == "ReceiveSignal":
		pindex = 0
		for p in s.params:
			output += ", "
			vname = p.var.name
			output += scopedvars[c.name + "'" + statemachine[s].name][vname]
			if p.index != None:
				output += ".Int2Nat(" + expression(p.index,statemachine[s],c,{})
			# in case we are updating a Byte, restrict the new value
			type = porttypes[c.name + "'" + s.target.name]
			output += "=x'" + str(pindex)
			if type[pindex] == 'Byte':
				output += " mod 256"
			pindex += 1
	elif s.__class__.__name__ == "Expression":
		output = ""
	return output

def peekstatementlabel(s,c):
	"""Produce a peek/show action label for s, which is supposed to be a receive or send statement"""
	if s.__class__.__name__ == "ReceiveSignal":
		output = mcrl2objectpeek + "_" + c.name + "'" + s.target.name
	else:
		output = mcrl2objectshow + "_" + c.name + "'" + s.target.name
	return output

def peekstatementparameters(s,c):
	"""Produce mCRL2 action parameters for the oeek action of a given ReceiveSignal or SendSignal SLCO statement. Class c owns the statement"""
	global check_rc, statemachine
	output = ''
	if check_rc:
		if s.__class__.__name__ == "ReceiveSignal":
			output += "(SM',R',W',"
			pindex = 0
			output += "s'"
			for p in s.params:
				output += ",x'" + str(pindex)
			output += ")"
		else:
			output += "(" + statemachine[s].name + ", {"
			varset = set([])
			for st in s.params:
				varset |= expression_varset(st,statemachine[s],c,{})
			varlist = sorted(list(varset))
			first = True
			for v in varlist:
				if not first:
					output += ","
				else:
					first = False
				output += v
			output += "},{},"
			output += s.signal
			for p in s.params:
				output += ", " + expression(p,statemachine[s],c,{})
			output += ")"
	else:
		if s.__class__.__name__ == "ReceiveSignal":
			pindex = 0
			output += "(s'"
			for p in s.params:
				output += ",x'" + str(pindex)
			output += ")"
		else:
			output += "(" + s.signal
			for p in s.params:
				output += ", " + expression(p,statemachine[s],c,{})
			output += ")"
	return output

def peekstatementactiontype(s,c):
	"""Produce the type for the peek action of the given ReceiveSignal or SendSignal statement. Class c owns the statement"""
	global porttypes, check_rc
	output = ''
	if check_rc:
		if s.__class__.__name__ == "ReceiveSignal":
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += ": Statemachine # Set(Var) # Set(Var) # Signal'"
			params = ""
			for t in type:
				output += datatypeacronym(t)
				params += " # " + str(t)
			output += params
		else:
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += ": Statemachine # Set(Var) # Set(Var) # Signal'"
			params = ""
			for t in type:
				output += datatypeacronym(t)
				params += " # " + str(t)
			output += params
	else:
		if s.__class__.__name__ == "ReceiveSignal":
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += ": Signal'"
			params = ""
			for t in type:
				output += datatypeacronym(t)
				params += " # " + str(t)
			output += params
		else:
			type = mcrl2typetuple(porttypes[c.name + "'" + s.target.name])
			output += ": Signal'"
			params = ""
			for t in type:
				output += datatypeacronym(t)
				params += " # " + str(t)
			output += params
	return output

def expression(s,stm,c,primmap):
	"""Maps SLCO expression to mCRL2 expression. Statemachine stm and Class c owning the statement are also given. Primmap is a dictionary for rewriting primaries."""
	global actions, scopedvars
	output = ''
	# special case: s is a variableref. In this case, it is the left-hand side of an assignment, and s refers to an array
	if s.__class__.__name__ == "Variable":
		output = primmap.get(s.name, scopedvars[c.name + "'" + stm.name][s.name])
	if s.__class__.__name__ == "Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2":
		#if s.op != '':
		#	output = '('
		if s.op != '' and s.op != 'xor':
			output += expression(s.left,stm,c,primmap) + " " + operator(s.op) + " " + expression(s.right,stm,c,primmap)
		elif s.op == 'xor':
			output += operator(s.op) + '(' + expression(s.left,stm,c,primmap) + ', ' + expression(s.right,stm,c,primmap) + ')'
		else:
			output += expression(s.left,stm,c,primmap)
	elif s.__class__.__name__ == "ExprPrec1":
		if s.op != '':
			output += operator(s.op) + '(' + expression(s.left,stm,c,primmap) + ', ' + expression(s.right,stm,c,primmap) + ')'
		else:
			output += expression(s.left,stm,c,primmap)
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
					output += "("
				output += primmap.get(scopedvars[c.name + "'" + stm.name][s.ref.ref], scopedvars[c.name + "'" + stm.name][s.ref.ref])
				if s.ref.index != None:
					output += ".Int2Nat(" + expression(s.ref.index,stm,c,primmap) + "))"
		else:
			output += '(' + expression(s.body,stm,c,primmap) + ')'
	return output

def expression_varset(s,stm,c,primmap):
	"""Produces set of variables referenced in given SLCO expression or variableref. Statemachine stm and Class c owning the statement are also given. Primmap is a dictionary for rewriting primaries, which is relevant when array indices are used."""
	global actions, scopedvars, smlocalvars
	output = set([])
	if s.__class__.__name__ == "VariableRef":
		# is the variable not local to state machine?
		if s.var.name not in smlocalvars.get(c.name + "'" + stm.name,set([])):
			varname = s.var.name
			if s.index != None:
				varname += "(Int2Nat(" + expression(s.index,stm,c,primmap) + "))"
			output.add(varname)
	elif s.__class__.__name__ != "Primary":
		output |= expression_varset(s.left,stm,c,primmap)
		if s.op != '':
			output |= expression_varset(s.right,stm,c,primmap)
	else:
		if s.ref != None:
			if s.ref.ref not in actions:
				# IGNORE STATE MACHINE LOCAL VARS!
				if s.ref.ref not in smlocalvars.get(c.name + "'" + stm.name,set([])):
					varname = s.ref.ref
					if s.ref.index != None:
						varname += "(Int2Nat(" + expression(s.ref.index,stm,c,primmap) + "))"
					output.add(varname)
	return output

# Communication actions (given a channel object)

def class_sendactionlabel(s):
	"""Produces mCRL2 send action label for a class"""
	return mcrl2objectsend + "_" + s.source.type.name + "'" + s.ports[0].name

def class_receiveactionlabel(s):
	"""Produces mCRL2 receive action label for a class"""
	return mcrl2objectreceive + "_" + s.target.type.name + "'" + s.ports[1].name

def class_sendpeekactionlabel(s):
	"""Produces mCRL2 send peek action label for a class"""
	return "show_" + s.source.type.name + "'" + s.ports[0].name

def class_peekactionlabel(s):
	"""Produces mCRL2 peek action label for a class"""
	return mcrl2objectpeek + "_" + s.target.type.name + "'" + s.ports[1].name

def sendactionlabel(s):
	"""Produces mCRL2 send action label for object connected to given channel"""
	return mcrl2objectsend + "_" + s.source.name + "'" + s.ports[0].name

def receiveactionlabel(s):
	"""Produces mCRL2 receive action label from channel connected to given channel"""
	return mcrl2objectreceive + "_" + s.target.name + "'" + s.ports[1].name

def sendpeekactionlabel(s):
	"""Produces mCRL2 show action label for a peek between objects connected to given channel"""
	return "show_" + s.source.name + "'" + s.ports[0].name

def peekactionlabel(s):
	"""Produces mCRL2 send action label for object connected to given channel"""
	return mcrl2objectpeek + "_" + s.target.name + "'" + s.ports[1].name

def commsendactionlabel(s):
	"""Produces mCRL2 successful communication send action label for object connected to given channel"""
	return mcrl2commsend + "_" + s.source.name + "_" + s.ports[0].name

def commreceiveactionlabel(s):
	"""Produces mCRL2 successful communication receive action label from channel connected to given channel"""
	return mcrl2commreceive + "_" + s.target.name + "_" + s.ports[1].name

def commpeekactionlabel(s):
	"""Produces mCRL2 successful communication receive action label from channel connected to given channel"""
	return mcrl2commpeek + "_" + s.target.name + "_" + s.ports[1].name

def syncactionlabel(s):
	"""Produces mCRL2 successful communication action label between objects, given the channel"""
	return mcrl2comm + "_" + s.source.name + "_" + s.ports[0].name + "_" + s.target.name + "_" + s.ports[1].name

def peeksyncactionlabel(s):
	"""Produces mCRL2 successful peek action label between objects, given the channel"""
	return mcrl2commpeek + "_" + s.source.name + "_" + s.ports[0].name + "_" + s.target.name + "_" + s.ports[1].name

# Communication types (given a channel object)

def sendactiontype(s):
	"""Produces mCRL2 send action type for object connected to given channel"""
	if check_rc:
		type = ": Statemachine # Set(Var) # Set(Var) # Statemachine # Set(Var) # Set(Var) # Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	else:
		type = ": Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	return type

def receiveactiontype(s):
	"""Produces mCRL2 receive action type from channel connected to given channel"""
	if check_rc:
		type = ": Statemachine # Set(Var) # Set(Var) # Statemachine # Set(Var) # Set(Var) # Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	else:
		type = ": Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	return type

def sendpeekactiontype(s):
	"""Produces mCRL2 show action type for a peek between objects connected to given channel"""
	if check_rc:
		type = ": Statemachine # Set(Var) # Set(Var) # Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	else:
		type = ": Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	return type

def peekactiontype(s):
	"""Produces mCRL2 send action type for object connected to given channel"""
	if check_rc:
		type = ": Statemachine # Set(Var) # Set(Var) # Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	else:
		type = ": Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	return type

def commsendactiontype(s):
	"""Produces mCRL2 successful communication send action type for object connected to given channel"""
	if check_rc:
		type = ": Statemachine # Set(Var) # Set(Var) # Statemachine # Set(Var) # Set(Var) # Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	else:
		type = ": Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	return type

def commreceiveactiontype(s):
	"""Produces mCRL2 successful communication receive action type from channel connected to given channel"""
	if check_rc:
		type = ": Statemachine # Set(Var) # Set(Var) # Statemachine # Set(Var) # Set(Var) # Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	else:
		type = ": Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	return type

def commpeekactiontype(s):
	"""Produces mCRL2 successful communication receive action type from channel connected to given channel"""
	if check_rc:
		type = ": Statemachine # Set(Var) # Set(Var) # Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	else:
		type = ": Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	return type

def syncactiontype(s):
	"""Produces mCRL2 successful communication action type between objects, given the channel"""
	if check_rc:
		type = ": Statemachine # Set(Var) # Set(Var) # Statemachine # Set(Var) # Set(Var) # Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	else:
		type = ": Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	return type

def peeksyncactiontype(s):
	"""Produces mCRL2 successful peek action type between objects, given the channel"""
	if check_rc:
		type = ": Statemachine # Set(Var) # Set(Var) # Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	else:
		type = ": Signal'"
		for t in s.type:
			type += datatypeacronym(mcrl2type(t))
		for t in s.type:
			type += " # " + mcrl2type(t)
	return type

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
			defv += str(v)
		defv += ']'
		return defv
	elif s.type.base == 'Integer' or s.type.base == 'Byte':
		if s.type.size < 2:
			return '0'
		else:
			type = '['
			first = True
			for i in range(0,s.type.size):
				if not first:
					type += ','
				else:
					first = False
				type += '0'
			type += ']'
			return type
	elif s.type.base == 'Boolean':
		if s.type.size < 2:
			return 'true'
		else:
			type = '['
			first = True
			for i in range(0,s.type.size):
				if not first:
					type += ','
				else:
					first = False
				type += 'true'
			type += ']'
			return type

def datatypeacronym(s):
	""" return acronym for given data type """
	if s == 'Int':
		return 'I'
	elif s == 'Bool':
		return 'B'
	elif s == 'List(Int)':
		return 'LI'
	elif s == 'List(Bool)':
		return 'LB'

def mcrl2type(s):
	"""Maps type names from SLCO to mCRL2"""
	if s.base == 'Integer' or s.base == 'Byte':
		if s.size < 2:
			return 'Int'
		else:
			return 'List(Int)'
	elif s.base == 'Boolean':
		if s.size < 2:
			return 'Bool'
		else:
			return 'List(Bool)'

def mcrl2value(s):
	"""Maps values from SLCO to mCRL2 values"""
	return str(s).lower()

def mcrl2typetuple(s):
	"""Maps list of type names from SLCO to mCRL2"""
	newtype = []
	for t in s:
		newtype.append(mcrl2type(t))
	return tuple(newtype)

def sync_guard_hideset(s):
	"""Compute the hide set of the given sync_guard_rules"""
	rset = set([])
	for rule in s:
		for r in rule[1]:
			rset.add(r)
	return rset

# *** END TRANSLATION FILTERS ***

# *** BEGIN TRANSLATION TESTS ***

def outgoingfromstate(t,s):
	"""Check whether transition t has source state s"""
	return t.source.name == s

def hasoutgoingtrans(s,t):
	"""Check whether state s has outgoing transitions in t"""
	for tr in t:
		if tr.source.name == s.name:
			return True
	return False

def isasync(s):
	"""Test whether the given channel is asynchronous"""
	return s.synctype == 'async'

def hassyncguards(s):
	"""Test if the given object has sync guards"""
	global sync_guarded_statements
	return sync_guarded_statements.get(s.type) != None

def syncstatwithsyncguards(s):
	"""Test if the given object has sync guards for at least one sync statement"""
	global sync_guarded_statements, syncactionlabelsdict
	orules = sync_guarded_statements.get(s.type,set([]))
	for sref in orules:
		if syncactionlabelsdict.get(statementactionlabel(sref[0],s,False)) != None:
			return True
	return False

def syncstatement(s):
	"""Test if s is a sync statement. s is a tuple, with the first element being a Statement object"""
	return s[0].__class__.__name__ == "SendSignal" or s[0].__class__.__name__ == "ReceiveSignal"

def o2o_sync(s,c):
	"""Test if s is a sync statement that for at least one object in the model, directly synchronises with another object. Class c owning the statement is also given"""
	global syncactionlabelsdict, classobjects
	for o in classobjects.get(c,set([])):
		if syncactionlabelsdict.get(statementactionlabel(s,o,False),tuple(['','',False]))[2]:
			return True
	return False

def hasreceivestatements(s):
	"""Test if class s has receive statements"""
	global class_receives
	return len(class_receives[s]) > 0

def nonemptydict(s):
	"""Check whether there are values stored in dict of dict s"""
	for s, slist in s.items():
		if len(slist) > 0:
			return True
	return False

# *** END TRANSLATION TESTS ***

def preprocess():
	"""preprocessing method"""
	global model, porttypes, scopedvars, smlocalvars, signaltypes, states, channeltypes, asynclosslesstypes, asynclossytypes, synctypes, actions, visibleactions, syncactionlabelsdict, classobjects, class_receives, statemachine, tr, check_rc, modelvars, statemachinenames
	# build dictionaries providing for a given statement the state machine and transition owning it
	for c in model.classes:
		for stm in c.statemachines:
			for trn in stm.transitions:
				for stat in trn.statements:
					statemachine[stat] = stm
					tr[stat] = trn
	# build a set of used statemachine names
	statemachinenames = set([])
	for c in model.classes:
		for stm in c.statemachines:
			statemachinenames.add(stm.name)
	# build a dictionary providing the objects of a given class
	classobjects = {}
	for o in model.objects:
		oset = classobjects.get(o.type,set([]))
		oset.add(o)
		classobjects[o.type] = oset
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
	# if required, construct set of variables (with scopes) appearing in the model)
	# also store the type of each variable
	if check_rc:
		modelvars = set([])
		for c in model.classes:
			for v in c.variables:
				modelvars.add(tuple([v.name,v.type]))
				for stm in c.statemachines:
					for v in stm.variables:
						modelvars.add(tuple([stm.name + "_" + v.name,v.type]))
	# make a set of state names
	states = set([])
	for c in model.classes:
		for stm in c.statemachines:
			for s in stm.states:
				states.add(s.name)
	# make a dictionary of channel types, where for each type we keep track of the encountered signal names
	channeltypes = {}
	asynclosslesstypes = set([])
	asynclossytypes = set([])
	synctypes = set([])
	
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

	# construct the set of actions as they appear in the model
	actions = set([])
	for a in model.actions:
		actions.add(a.name)

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

	# build map between synchronisation actions. also record whether the synchronisation is 'direct' (between objects) or indirect 'via channels'
	for c in model.channels:
		if c.synctype == 'async':
			syncactionlabelsdict[sendactionlabel(c)] = tuple([tuple(["insert_", c.source.name + "'" + c.ports[0].name, c.source]), commsendactionlabel(c), False])
			syncactionlabelsdict[receiveactionlabel(c)] = tuple([tuple(["remove_", c.target.name + "'" + c.ports[1].name, c.target]), commreceiveactionlabel(c), False])
			syncactionlabelsdict[peekactionlabel(c)] = tuple([tuple(["show_", c.target.name + "'" + c.ports[1].name, c.target]), commpeekactionlabel(c), False])
		else:
			syncactionlabelsdict[sendactionlabel(c)] = tuple([tuple([mcrl2objectreceive + "_", c.target.name + "'" + c.ports[1].name, c.target]), syncactionlabel(c), True])
			syncactionlabelsdict[receiveactionlabel(c)] = tuple([tuple([mcrl2objectsend + "_", c.source.name + "'" + c.ports[0].name, c.source]), syncactionlabel(c), True])
			syncactionlabelsdict[sendpeekactionlabel(c)] = tuple([tuple([mcrl2objectpeek + "_", c.target.name + "'" + c.ports[1].name, c.target]), peeksyncactionlabel(c), True])
			syncactionlabelsdict[peekactionlabel(c)] = tuple([tuple(["show_", c.source.name + "'" + c.ports[0].name, c.source]), peeksyncactionlabel(c), True])

	# create a list of statements representing receives for each class
	class_receives = {}
	for c in model.classes:
		slist = []
		for stm in c.statemachines:
			for trn in stm.transitions:
				for stat in trn.statements:
					if stat.__class__.__name__ == "ReceiveSignal":
						slist.append(stat)
		# record list
		class_receives[c] = slist
	# construct a dictionary of visible actions per class
	visibleactions = {}
	# temporary dict to ensure that no two statements are stored in visibleactions with the same label
	visibleactionstrings = {}
	if not check_rc:
		for c in model.classes:
			for stm in c.statemachines:
				for trn in stm.transitions:
					for stat in trn.statements:
						if expression_is_actionref(stat):
							stringset = visibleactionstrings.get(c, set([]))
							if statementactionlabel(stat,c,False) not in stringset:
								stringset.add(statementactionlabel(stat,c,False))
								visibleactionstrings[c] = stringset
								actset = visibleactions.get(c, set([]))
								actset.add(stat)
								visibleactions[c] = actset
	# add other visible actions, in case of race condition checking
	else:
		for c in model.classes:
			for stm in c.statemachines:
				for trn in stm.transitions:
					for stat in trn.statements:
						if stat.__class__.__name__ == "Assignment" or stat.__class__.__name__ == "Composite" or (stat.__class__.__name__ == "Expression" and not expression_is_actionref(stat)):
							stringset = visibleactionstrings.get(c, set([]))
							if statementactionlabel(stat,c,False) not in stringset:
								stringset.add(statementactionlabel(stat,c,False))
								visibleactionstrings[c] = stringset
								actset = visibleactions.get(c, set([]))
								actset.add(stat)
								visibleactions[c] = actset

def translate():
	"""The translation function"""
	global modelname, model, porttypes, states, channeltypes, asynclosslesstypes, asynclossytypes, synctypes, syncactionlabelsdict, o2o_sync_summands, class_receives, check_rc
	
	path, name = split(modelname)
	if name.endswith('.slco'):
		name = name[:-4]
	else:
		name = name[:-7]
	outFile = open(join(path,name + "mcrl2"), 'w')

	# Initialize the template engine.
	jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(join(this_folder,'../../jinja2_templates')), trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols','jinja2.ext.do',])

	# Register the filters
	jinja_env.filters['mcrl2type'] = mcrl2type
	jinja_env.filters['mcrl2value'] = mcrl2value
	jinja_env.filters['mcrl2typetuple'] = mcrl2typetuple
	jinja_env.filters['statementactionlabel'] = statementactionlabel
	jinja_env.filters['statementactiontype'] = statementactiontype
	jinja_env.filters['statementsummation'] = statementsummation
	jinja_env.filters['statementlabel'] = statementlabel
	jinja_env.filters['statementparameters'] = statementparameters
	jinja_env.filters['statementstatechanges'] = statementstatechanges
	jinja_env.filters['statementguard'] = statementguard
	jinja_env.filters['peekstatementlabel'] = peekstatementlabel
	jinja_env.filters['peekstatementactiontype'] = peekstatementactiontype
	jinja_env.filters['peekstatementparameters'] = peekstatementparameters
	jinja_env.filters['variabledefault'] = variabledefault
	jinja_env.filters['datatypeacronym'] = datatypeacronym
	jinja_env.filters['class_sendactionlabel'] = class_sendactionlabel
	jinja_env.filters['class_receiveactionlabel'] = class_receiveactionlabel
	jinja_env.filters['class_peekactionlabel'] = class_peekactionlabel
	jinja_env.filters['class_sendpeekactionlabel'] = class_sendpeekactionlabel
	jinja_env.filters['sendactionlabel'] = sendactionlabel
	jinja_env.filters['receiveactionlabel'] = receiveactionlabel
	jinja_env.filters['peekactionlabel'] = peekactionlabel
	jinja_env.filters['sendpeekactionlabel'] = sendpeekactionlabel
	jinja_env.filters['commsendactionlabel'] = commsendactionlabel
	jinja_env.filters['commreceiveactionlabel'] = commreceiveactionlabel
	jinja_env.filters['commpeekactionlabel'] = commpeekactionlabel
	jinja_env.filters['syncactionlabel'] = syncactionlabel
	jinja_env.filters['peeksyncactionlabel'] = peeksyncactionlabel
	jinja_env.filters['sendactiontype'] = sendactiontype
	jinja_env.filters['receiveactiontype'] = receiveactiontype
	jinja_env.filters['peekactiontype'] = peekactiontype
	jinja_env.filters['sendpeekactiontype'] = sendpeekactiontype
	jinja_env.filters['commsendactiontype'] = commsendactiontype
	jinja_env.filters['commreceiveactiontype'] = commreceiveactiontype
	jinja_env.filters['commpeekactiontype'] = commpeekactiontype
	jinja_env.filters['syncactiontype'] = syncactiontype
	jinja_env.filters['peeksyncactiontype'] = peeksyncactiontype
	jinja_env.filters['expression'] = expression
	jinja_env.filters['sync_guard_hideset'] = sync_guard_hideset
	
	# Register the tests
	jinja_env.tests['outgoingfromstate'] = outgoingfromstate
	jinja_env.tests['hasoutgoingtrans'] = hasoutgoingtrans
	jinja_env.tests['isasync'] = isasync
	jinja_env.tests['hassyncguards'] = hassyncguards
	jinja_env.tests['hasreceivestatements'] = hasreceivestatements
	jinja_env.tests['nonemptydict'] = nonemptydict
	jinja_env.tests['syncstatwithsyncguards'] = syncstatwithsyncguards
	jinja_env.tests['syncstatement'] = syncstatement
	jinja_env.tests['o2o_sync'] = o2o_sync
	jinja_env.tests['actionref'] = expression_is_actionref

	# load the mCRL2 template
	template = jinja_env.get_template('mcrl2.jinja2template')
	out = template.render(model=model, states=states, channeltypes=channeltypes,
	                asynclosslesstypes=asynclosslesstypes, asynclossytypes=asynclossytypes, synctypes=synctypes,
	                visibleactions=visibleactions, actions=actions, sync_guarded_statements=sync_guarded_statements,
	                sync_guard_rules=sync_guard_rules, guard_priority=guard_priority,
	                syncactionlabelsdict=syncactionlabelsdict, mcrl2objectsend=mcrl2objectsend,
	                mcrl2objectreceive=mcrl2objectreceive, mcrl2resetrecs=mcrl2resetrecs, classobjects=classobjects,
	                o2o_sync_summands=o2o_sync_summands, class_receives=class_receives,
	                class_sguard_creceives=class_sguard_creceives,
	                class_sguard_creceives_combs=class_sguard_creceives_combs, previous_sid=previous_sid,
	                sync_guarded_sids=sync_guarded_sids, check_rc=check_rc, modelvars=modelvars,
	                statemachinenames=statemachinenames, mcrl2varprefix=mcrl2varprefix)
	# write mCRL2 spec
	outFile.write(out)
	outFile.close()

def main(args):
	"""The main function"""
	global modelname, model, check_rc
	if len(args) == 0:
		print("Missing argument: SLCO model")
		sys.exit(1)
	else:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 slco2mcrl2 [-rc]")
			print("")
			print("Transform an SLCO 2.0 model to an mCRL2 model.")
			print("")
			print(" -rc                    produce an mCRL2 model suitable for race condition checking")
			sys.exit(0)
		else:
			for i in range(0,len(args)):
				if args[i] == '-rc':
					check_rc = True
				else:
					modelname = args[i]

	# create meta-model
	slco_mm = metamodel_from_file(join(this_folder,'../../textx_grammars/slco2.tx'))

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
