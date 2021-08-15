import sys
from os.path import dirname, basename, join, exists
from os import mkdir
from textx.metamodel import metamodel_from_file
from shutil import rmtree
import glob

dve_mm = None
slco_mm = None
modelname = ""
this_folder = dirname(__file__)
use_composite_statement = False

slco_template = """model dve_%s {
classes
	GlobalClass {
		variables
			%s
		state machines
			%s
	}

objects
	globalObject : GlobalClass()
}
"""

class TranslateException(Exception):
    pass

keywords = ['not', 'or', 'and', 'xor']
changed_words = {}

typemap = {'byte'   : 'Byte',
           'int'    : 'Integer',
           'boolean': 'Boolean'}


def dve_primitive_type2slco_type(dve_type, length=None):
	type = typemap[dve_type.base]
	if length:
		type += '[%d]' % length
	return type

def dve_varlists2slco_variablelist(varlists):
	slco_vars = []
	for varlist in varlists:
		for var in varlist.variables:
			slco_vars.append(dve_var2slco_variable(var, varlist.type))
	return slco_vars


def dve_const_array2slco_const_array(dve_array):
	values = ','.join([dve_expr2slco_expr(x) for x in dve_array.values])
	return "[%s]" % values

def fix_for_keywords(s):
	global changed_words
	if s in changed_words:
		return changed_words[s]

	s_new = s
	changed = False
	if '_' in s_new:
		s_new = s_new.replace('_', '')
		changed = True
	for k in keywords:
		if s_new.startswith(k):
			s_new = 'dve' + s_new
			break
	if changed:
		changed_words[s] = s_new
	return s_new

def dve_var2slco_variable(dve_var, dve_type):
	slco_var = None
	name = fix_for_keywords(dve_var.name)
	if dve_var.length and dve_var.length > 0:
		if dve_var.initialValue and\
		   dve_var.initialValue.__class__.__name__ != 'ArrayConstantExpression':
			raise TranslateException('Unsupported initialValue "'+dve_var.initialValue.__class__.__name__+'" on variable "'+dve_var.name+'"')
		slco_var = dve_primitive_type2slco_type(dve_type, dve_var.length) + " " + name
		if dve_var.initialValue:
			slco_var += " := " + dve_const_array2slco_const_array(dve_var.initialValue)
	else:
		slco_var = dve_primitive_type2slco_type(dve_type) + " " + name
		if dve_var.initialValue:
			slco_var += " := " + dve_expr2slco_expr(dve_var.initialValue)
	return slco_var


def dve_expr2slco_expr(dve_expr):
	i = 0
	expr = dve_mm.parser.input[dve_expr._tx_position:dve_expr._tx_position_end]
	expr = expr.replace('==', '=')
	expr = expr.replace('!=', '<>')
	expr = expr.replace(' && ', ' and ')
	expr = expr.replace(' &&', ' and ')
	expr = expr.replace('&&', ' and ')
	expr = expr.replace(' || ', ' or ')
	expr = expr.replace(' ||', ' or ')
	expr = expr.replace('||', ' or ')
	for old, new in changed_words.items():
		expr = expr.replace(old, new)
	return expr
'''
	if dve_expr.__class__.__name__ == 'BooleanConstantExpression'\
		or dve_expr.__class__.__name__ == 'IntegerConstantExpression':
		return dve_expr.value
	elif dve_expr.__class__.__name__ == 'ArrayConstantExpression':
		intvals = [x for x in dve_expr.values if isinstance(x, int)]
		if len(intvals) > 0:
			return intvals
		boolvals = [x for x in dve_expr.values if isinstance(x, bool)]
		if len(boolvals) > 0:
			return boolvals
		return None
	elif dve_expr.__class__.__name__ == 'BracketExpression':
		return Primary(body = dve_expr2slco_expr(dve_expr.expression))
	elif dve_expr.__class__.__name__ == 'VariableExpression':
		expr_ref = ExpressionRef(dve_expr.variable, index=dve_expr.index)
		return Primary(ref=expr_ref)
	elif dve_expr.__class__.__name__ == 'Expression':
		return Expression(
			left=dve_expr2slco_expr(dve_expr.left),
			right=dve_expr2slco_expr(dve_expr.right)
		)
	elif dve_expr.__class__.__name__ == 'ExprPrec3':
		return ExprPrec3(
			left=dve_expr2slco_expr(dve_expr.left),
			right=dve_expr2slco_expr(dve_expr.right)
		)
	elif dve_expr.__class__.__name__ == 'ExprPrec2':
		return ExprPrec2(
			left=dve_expr2slco_expr(dve_expr.left),
			right=dve_expr2slco_expr(dve_expr.right)
		)
	elif dve_expr.__class__.__name__ == 'ExprPrec1':
		return ExprPrec1(
			left=dve_expr2slco_expr(dve_expr.left),
			right=dve_expr2slco_expr(dve_expr.right)
		)
	elif dve_expr.__class__.__name__ == 'Primary':
		raise TranslateException('DVE Primary not supported')
'''


def dve_state2slco_state(dve_state):
	return fix_for_keywords(dve_state.name)


def dve_process2slco_state_machine(dve_process):
	statemachine = '''%s {
				variables
					%s
				initial %s
				states  %s
				transitions
					%s
				}
	''' % (dve_process.name,
	       "\n\t\t\t\t\t".join(dve_varlists2slco_variablelist(dve_process.varlists)),
	       dve_state2slco_state(dve_process.init),
	       " ".join([dve_state2slco_state(x) for x in dve_process.states if x.name != dve_process.init.name]),
	       "\n\t\t\t\t\t".join([dve_transition2slco_transition(x) for x in dve_process.transitions])
	       )
	return statemachine

def dve_transition2slco_transition(dve_trans):
	statements = ""
	if dve_trans.guard:
		statements += dve_expr2slco_expr(dve_trans.guard)
	if dve_trans.effects:
		if statements:
			statements += "; "
		statement_list = [dve_statement2slco_statement(x) for x in dve_trans.effects]
		statements += "; ".join(statement_list)
	trans = '''from %s to %s {
						%s
					}\n'''
	if use_composite_statement and dve_trans.effects:
			statements = '[%s]' % statements

	trans = trans % (dve_state2slco_state(dve_trans.source),
	                 dve_state2slco_state(dve_trans.target),
	                 "%s" % statements)
	return trans

def dve_statement2slco_statement(dve_statement):
	if dve_statement.__class__.__name__ == 'AssignmentSequence':
		assignments = ";".join([dve_assignment2slco_assignment(x) for x in dve_statement.assignments])
		return assignments

	raise TranslateException('Statement "%s" not supported!' %dve_statement.__class__.__name__)


def dve_assignment2slco_assignment(dve_assign):
	assign = "%s := %s" % (dve_var_ref2slco_var_ref(dve_assign.variable,dve_assign.index),
	                       dve_expr2slco_expr(dve_assign.expression))
	return assign


def dve_var_ref2slco_var_ref(dve_var, index_expr):
	var = fix_for_keywords(dve_var.name)
	if index_expr:
		var += "[%s]" % dve_expr2slco_expr(index_expr)
	return var


def translate(dve_model, mname):
	"""The translation function"""
	dve_global_vars = [x for x in dve_model.declarations
	                   if x.__class__.__name__ == "VarList"]
	global_vars = []
	for varlist in dve_global_vars:
		for var in varlist.variables:
			global_vars.append(dve_var2slco_variable(var, varlist.type))

	statemachines = [dve_process2slco_state_machine(x) for x in dve_model.declarations
	                   if x.__class__.__name__ == "Process"]

	return slco_template % (mname, "\n\t\t\t".join(global_vars), "\n\t\t\t".join(statemachines))


def main(args):
	"""The main function"""
	global dve_mm, slco_mm, use_composite_statement
	if len(args) == 0:
		print("Missing argument: DVE model")
		sys.exit(1)
	else:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 dve2slco <file_in>")
			print("")
			print("Transform an DVE model to an SLCO 2.0 model.")
			print("-composite       use composite statement for dve transitions")
			sys.exit(0)
		else:
			for i in range(0,len(args)):
				if args[i] == '-composite':
					use_composite_statement = True
				else:
					modelname = args[i]
					break

	# create meta-model
	dve_mm = metamodel_from_file(join(this_folder,'../../textx_grammars/dve.tx'))
	slco_mm = metamodel_from_file(join(this_folder,'../../textx_grammars/slco2.tx'))

	batch = []
	if modelname.endswith('.dve'):
		batch = [modelname]
	else:
		batch = glob.glob(join(this_folder,modelname,"*.dve"))

	gen_dir = "generated_slco"
	dir = dirname(batch[0])
	gen_folder = join(dir,gen_dir)
	if exists(gen_folder):
		rmtree(gen_folder)
	mkdir(gen_folder)
	# read model
	for file in batch:
		model = dve_mm.model_from_file(file)
		mname = basename(file)
		print("processing model %s" % mname)
		mname = mname[:-4]
		mname = mname.replace('.', '_')
		# translate
		out_model = translate(model, mname)
		# write to file
		if len(batch) == 1:
			print(out_model)

		outFile = open(join(gen_folder,basename(file)[:-3] + "slco"), 'w')
		outFile.write(out_model)
		outFile.close()


if __name__ == '__main__':
	args = []
	for i in range(1,len(sys.argv)):
		args.append(sys.argv[i])
	main(args)
