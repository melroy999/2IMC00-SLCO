import codecs
import sys
from os import mkdir
import re
from os.path import exists, dirname, join
from textx.metamodel import metamodel_from_file
if sys.version < '3':
	text = unicode
else:
	text = str

modelname = ""
model = ""
this_folder = dirname(__file__)
# layout of the drawing
layout = []

HEADER = '''
	digraph xtext {
	fontname = "Bitstream Vera Sans"
	fontsize = 25
	node[
		shape=rectangle,
		style=rounded
	]
	edge[dir=black,arrowsize=0.6]


'''

def operator(s):
	"""Provide characters to display given operator s"""
	if s == '>':
		return '&gt;'
	elif s == '<':
		return '&lt;'
	elif s == '>=':
		return '&ge;'
	elif s == '<=':
		return '&le;'
	else:
		return s

def getlabel(s):
	"""Get the label for the given statement s"""
	result = ''
	if s.__class__.__name__ == "Assignment":
		result += s.left.var.name
		if s.left.index != None:
			result += "[" + getlabel(s.left.index) + "]"
		result += " := " + getlabel(s.right)
	elif s.__class__.__name__ == "Composite":
		result += "["
		if s.guard != None:
			result += getlabel(s.guard)
		if len(s.assignments) > 1:
			result += ";"
		for i in range(0,len(s.assignments)):
			result += " " + getlabel(s.assignments[i])
			if i < len(s.assignments)-1:
				result += ";"
		result += "]"
	elif s.__class__.__name__ == "Delay":
		result += "<b>after </b>" + str(s.length) + "<b> ms</b>"
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
	elif s.__class__.__name__ == "Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2" or s.__class__.__name__ == "ExprPrec1":
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
	return result

layoutscanner=re.Scanner([
	(r"[0-9]+",														lambda scanner,token:("SM", token)),
	(r"\[",																lambda scanner,token:("START", token)),
	(r"\]",																lambda scanner,token:("END", token)),
	(r"[\s,]+", None), # None == skip token.
])

def processLayout(e):
	"""Process the given layout expression, and store result in layout global variable"""
	global layout
	# scan the given expression
	results, remainder = layoutscanner.scan(e)
	# store layout
	layout = []
	newlist = []
	in_nested_list = False
	for label,token in results:
		if label == "SM":
			newlist.append(int(token))
			if not in_nested_list:
				layout.append(newlist)
				newlist = []
		elif label == "START":
			in_nested_list = True
		elif label == "END":
			layout.append(newlist)
			newlist = []
			in_nested_list = False

def slco_to_dot(model):
	"""Transforms given model to dot str"""
	global layout

	dot_str = HEADER
	
	# create map between state machines and IDs
	smids = {}
	i = 0
	for o in model.objects:
		for sm in o.type.statemachines:
			smids[i] = sm
			i += 1
	# create map between state machines and their objects
	smobj = {}
	for o in model.objects:
		for sm in o.type.statemachines:
			smobj[sm] = o

	if layout == []:
		i = 0
		for o in model.objects:
			for sm in o.type.statemachines:
				layout.append([i])
				i += 1

	ocounter = 0
	for smlist in layout:
		# possibly add a structure to put state machines on top of each other
		if len(smlist) > 1:
			dot_str += '{\n'
			for i in range(0,len(smlist)):
				dot_str += '  ' + str(i+ocounter) + ' [style = invis];\n'
			for i in range(0,len(smlist)-1):
				dot_str += '  ' + str(i+ocounter) + ' -> ' + str(i+ocounter+1) + ' [style = invis];\n'
			dot_str += '}\n\n'
		for smid in smlist:
			# Obtain state machine
			sm = smids[smid]
			# Obtain object
			o = smobj[sm]
			# Create subgraph
			dot_str += 'subgraph cluster{}'.format(
				o.name + "_" + sm.name)
			dot_str += ' {\n'
			dot_str += 'fontsize = 25;\n'
			dot_str += 'label = <<b>{}</b>>\n'.format(o.name + "." + sm.name)
			dot_str += 'subgraph {}_init'.format(
				o.name + "_" + sm.name)
			dot_str += ' {\n'
			# Initial state
			dot_str += '{}[shape=point,width=0.1];\n'.format(
				o.name + "_" + sm.name + "_" + sm.initialstate.name + "_init")
			dot_str += '{}[label="{}",fontsize=25];\n'.format(
				o.name + "_" + sm.name + "_" + sm.initialstate.name, sm.initialstate.name)
			# Transition into initial state
			dot_str += '{} -> {} [minlen="0.01"];\n'.format(
				o.name + "_" + sm.name + "_" + sm.initialstate.name + "_init", o.name + "_" + sm.name + "_" + sm.initialstate.name)
			dot_str += '}\n'
			# Render states
			for s in sm.states:
				dot_str += '{}[label="{}",fontsize=25];\n'.format(
					o.name + "_" + sm.name + "_" + s.name, s.name)
			
			# Render transitions
			for tr in sm.transitions:
				# create transition label
				trlabel = "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"-9\">"
				for i in range(0,len(tr.statements)):
					trlabel += "<TR><TD ALIGN=\"LEFT\">      </TD><TD ALIGN=\"LEFT\">" + getlabel(tr.statements[i])
					if i < len(tr.statements)-1:
							trlabel += ";"
					trlabel += "     "
					trlabel += "</TD></TR>"
				trlabel += "</TABLE>"
				dot_str += '{} -> {} [fontsize=25, label=<{}>];\n'.format(
				o.name + "_" + sm.name + "_" + tr.source.name, o.name + "_" + sm.name + "_" + tr.target.name, trlabel)
			# End of current subgraph
			dot_str += '}\n'
		# Connect the subgraphs to place them above each other
		if len(smlist) > 1:
			dot_str += '\n'
			for i in range(0,len(smlist)):
				dot_str += str(i+ocounter) + ' -> ' + o.name + "_" + sm.name + "_" + sm.initialstate.name + "_init[style = invis, minlen = \"0.01\"];\n"
			# update ocounter
			ocounter += len(smlist)

	dot_str += '\n}\n'
	return dot_str

def main(args):
	"""The main function"""
	global modelname, model, layout
	layoutexpression = ""
	if len(args) == 0:
		print("Missing argument: SLCO model")
		sys.exit(1)
	else:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 slco2dot [-l]")
			print("")
			print("Transform an SLCO 2.0 model to a dot file.")
			print("")
			print(" -l <layout>                                display the state machines in the given layout. E.g. '[0,1],2' indicates that state machines 0 and 1 should be printed on top of each other and state machine 2 should be printed to the right of them")
			sys.exit(0)
		else:
			i = 0
			while i < len(args):
				if args[i] == '-sm':
					if i > len(args)-2:
						print("Missing arguments to select state machine")
						sys.exit(1)
					viewobject = args[i+1]
					viewsm = args[i+2]
					i += 2
				elif args[i] == '-l':
					if i > len(args)-1:
						print("Missing layout")
						sys.exit(1)
					layoutexpression = args[i+1]
					i += 1
				else:
					modelname = args[i]
				i += 1

	# create meta-model
	slco_mm = metamodel_from_file(join(this_folder,'../../textx_grammars/slco2.tx'))
	# read model
	model = slco_mm.model_from_file(join(this_folder,modelname))
	# optionally process given layout
	processLayout(layoutexpression)
	with codecs.open("{}.dot".format(modelname), 'w', encoding='utf-8') as f:
		f.write(slco_to_dot(model))

if __name__ == '__main__':
	args = []
	for i in range(1,len(sys.argv)):
		args.append(sys.argv[i])
	main(args)
