import sys
from os.path import dirname, basename, join, exists
from os import mkdir
from shutil import rmtree
import json
import random
import igraph  # pip install python-igraph
import math
from collections import defaultdict, Counter

modelname = ""
this_folder = dirname(__file__)

local_var_idx = "local"


def cap(n, low, high):
	return min(high, max(low, n))


def rand_exp_int(lambd):
	return round(random.expovariate(lambd))


class MinMax:
	min = 0
	max = 0

	def __init__(self, min, max):
		self.min = min
		self.max = max

	def randint_cap(self, low, high):
		mi = max(self.min, low)
		ma = min(self.max, high)
		if (ma <= mi+1):
			return mi
		return random.randint(mi, ma)

	def randint(self):
		if (self.max <= self.min):
			return self.min
		return random.randint(self.min, self.max)

	def randuniform(self):
		return random.uniform(self.min, self.max)


class GraphGenerator:
	def generate(self):
		pass


class DegreeSequence:
	degree_range = MinMax(1, 1)
	num_nodes = MinMax(1, 1)

	def generate_degree_sequence(self):
		pass

	def to_json(self):
		return {'__class__': 'DegreeSequence',
				'degree_range': to_json(self.degree_range),
				'num_nodes': to_json(self.num_nodes)}

	def set_from_json(self, js):
		self.degree_range = js['degree_range']
		self.num_nodes = js['num_nodes']


class DegreeSequenceExponential(DegreeSequence):
	median = MinMax(2, 2)

	def generate_degree_sequence(self):
		med = self.median.randint()
		if med < 2:
			med = 2
		lambd = math.log(2) / med

		n = self.num_nodes.randint()
		low = max(1, self.degree_range.min)
		high = min(n - 1, self.degree_range.max)

		out_degree = []
		while n > 1:
			d = cap(rand_exp_int(lambd), low, min(n - 1, high))
			n -= d
			out_degree.append(d)
		random.shuffle(out_degree)
		return out_degree

	def to_json(self):
		ret = super().to_json()
		ret['__class__'] = 'DegreeSequenceExponential'
		ret['median'] = to_json(self.median)
		return ret

	def set_from_json(self, js):
		super().set_from_json(js)
		self.median = from_json(js['median'])


class DegreeSequenceLinear(DegreeSequence):
	def generate_degree_sequence(self):
		n = self.num_nodes.randint()
		out_degree = []
		while n > 1:
			d = self.degree_range.randint_cap(1, n - 1)
			n -= d
			out_degree.append(d)
		random.shuffle(out_degree)
		return out_degree

	def to_json(self):
		ret = super().to_json()
		ret['__class__'] = 'DegreeSequenceLinear'
		return ret


class GraphGenerator_DegreeSequenceSpanningTree(GraphGenerator):
	dseq = DegreeSequenceExponential()

	def generate(self):
		out_degree = self.dseq.generate_degree_sequence()
		n = sum(out_degree) + 1
		g = igraph.Graph()
		g.add_vertices(n)
		vertex_id = 0
		vertex_selection = [vertex_id]
		for d in out_degree:
			idx = random.randint(0, len(vertex_selection)-1)
			v = vertex_selection.pop(idx)
			while d > 0:
				vertex_id += 1
				g.add_edge(v, vertex_id)
				vertex_selection.append(vertex_id)
				d -= 1
		return g


class GraphGenerator_Connected(GraphGenerator_DegreeSequenceSpanningTree):
	directed = False
	num_edges = MinMax(1, 1)
	loops = False

	def max_edges(self, n):
		max_e = n * (n - 1)
		if not self.directed:
			max_e /= 2
		return max_e

	def generate(self):
		class EdgeSelection:
			edges_out = defaultdict(list)
			edges_in = defaultdict(list)
			selection = Counter()
			n = 0
			
			def count(self):
				len(selection)

			def __init__(innerSelf, n):
				innerSelf.n = n

			def extend(innerSelf, list):
				innerSelf.selection.update(list)

			def select2(innerSelf):
				selectables = innerSelf.selection.copy()
				# remove vertices that can no longer be sources
				selectables_list = \
					[x for x in selectables.elements()] if self.directed \
					else [x for x in selectables.elements() if len(innerSelf.edges_out[x]) < innerSelf.n - 1]
				a = random.choice(selectables_list)
				if not self.directed:
					# do no select edges that are already present
					for val in innerSelf.edges_out[a]:
						selectables[val] = 0
					for val in innerSelf.edges_in[a]:
						selectables[val] = 0
				if not self.loops:
					# do not create self-loops
					selectables[a] = 0
				b = random.choice([x for x in selectables.elements()])
				return a, b

			def add_edge(innerSelf, g, source, target):
				g.add_edge(source, target)
				innerSelf.selection.update([source, target])
				innerSelf.edges_out[source].append(target)
				innerSelf.edges_in[target].append(source)

		g = super().generate()
		actual_n_edges = g.ecount()
		n_edges = self.num_edges.randint_cap(actual_n_edges, self.max_edges(g.vcount()))
		if actual_n_edges < n_edges:
			# we want to favor vertices with high degree
			# we shall select vertices from a list in which a vertex occurs as many times as its degree
			selection = EdgeSelection(g.vcount())
			for v in g.vs:
				d = g.degree(v, loops=False)
				selection.extend([v.index] * d)
			# add edges until the desired number of edges is reached
			while actual_n_edges < n_edges: # also stop when selection is empty (IndexError)
				try:
					source, target = selection.select2()
				except IndexError:
					break
				selection.add_edge(g, source, target)
				actual_n_edges += 1
		return g


class GraphGeneratorMixed(GraphGenerator):
	gen_list = [GraphGenerator_Connected()]

	def generate(self):
		idx = random.randint(0, len(self.gen_list)-1)
		return self.gen_list[idx].generate()


class Config:
	num_models_to_generate = 1
	model_name = "gen_"
	statement_size = MinMax(1, 1)
	vars_per_edge = MinMax(1, 1)
	local_var_chance = MinMax(0.5, 0.5)
	num_vars_shared_by_more_than_two_sms = MinMax(0, 0)
	sm_gen_method = GraphGeneratorMixed()
	topology_gen_method = GraphGenerator_Connected()



	@staticmethod
	def fromFile(path):
		file = open(path, 'r')
		config = json.load(file, object_hook=from_json)
		file.close()
		return config

	def writeToFile(self, path):
		file = open(path, 'w')
		tmp = self.num_models_to_generate
		json.dump(self, file, default=to_json, indent=4)
		file.close()


def to_json(obj):
	if isinstance(obj, Config):
		return {'__class__': 'Config',
		        'num_models_to_generate': obj.num_models_to_generate,
		        'model_name_prefix': obj.model_name,
		        'statement_size': to_json(obj.statement_size),
		        'vars_per_edge' : to_json(obj.vars_per_edge),
		        'local_var_chance': to_json(obj.local_var_chance),
		        'sm_gen_method': to_json(obj.sm_gen_method),
		        'topology_gen_method': to_json(obj.topology_gen_method),
		        'num_vars_shared_by_more_than_two_sms': to_json(obj.num_vars_shared_by_more_than_two_sms)}
	elif isinstance(obj, MinMax):
		return {'__class__': 'MinMax',
		        'min': obj.min,
		        'max': obj.max}
	elif isinstance(obj, GraphGeneratorMixed):
		return {'__class__': 'GraphGeneratorMixed',
		        'generator_list': obj.gen_list}
	elif isinstance(obj, DegreeSequence):
		return obj.to_json()
	elif isinstance(obj, GraphGenerator_Connected):
		return {'__class__': 'GraphGenerator_Connected',
		        'directed': obj.directed,
		        'num_edges': obj.num_edges,
		        'loops': obj.loops,
		        'degree_sequence': obj.dseq}
	elif isinstance(obj, GraphGenerator_DegreeSequenceSpanningTree):
		return {'__class__': 'GraphGenerator_DegreeSequenceSpanningTree',
				'degree_sequence': obj.dseq}
	raise TypeError(repr(obj) + ' is not JSON serializable')


def from_json(obj):
	if not isinstance(obj, dict):
		return obj

	if '__class__' in obj:
		if obj['__class__'] == 'Config':
			cfg = Config()
			cfg.num_models_to_generate = obj['num_models_to_generate']
			cfg.model_name = obj['model_name_prefix']
			cfg.statement_size = from_json(obj['statement_size'])
			cfg.vars_per_edge = from_json(obj['vars_per_edge'])
			cfg.local_var_chance = from_json(obj['local_var_chance'])
			cfg.sm_gen_method = from_json(obj['sm_gen_method'])
			cfg.topology_gen_method = from_json(obj['topology_gen_method'])
			cfg.num_vars_shared_by_more_than_two_sms = from_json(obj['num_vars_shared_by_more_than_two_sms'])
			return cfg
		elif obj['__class__'] == 'MinMax':
			return MinMax(obj['min'], obj['max'])
		elif obj['__class__'] == 'GraphGeneratorMixed':
			gen = GraphGeneratorMixed()
			gen.gen_list = from_json(obj['generator_list'])
			return gen
		elif obj['__class__'] == 'DegreeSequenceExponential':
			seq = DegreeSequenceExponential()
			seq.set_from_json(obj)
			return seq
		elif obj['__class__'] == 'DegreeSequenceLinear':
			seq = DegreeSequenceLinear()
			seq.set_from_json(obj)
			return seq
		elif obj['__class__'] == 'GraphGenerator_Connected':
			gen = GraphGenerator_Connected()
			gen.directed = from_json(obj['directed'])
			gen.num_edges = from_json(obj['num_edges'])
			gen.loops = from_json(obj['loops'])
			gen.dseq = from_json(obj['degree_sequence'])
			return gen
		elif obj['__class__'] == 'GraphGenerator_DegreeSequenceSpanningTree':
			gen = GraphGenerator_DegreeSequenceSpanningTree()
			gen.dseq = from_json(obj['degree_sequence'])
			return gen
	return obj


def make_statement(vars, var, chance_for_local):
	read = random.choice([True, False])
	write = random.choice([True, False])
	if not write:
		read = True
	if write:
		write_var = var
	else:
		write_var = local_var_idx if (random.uniform(0, 1) > chance_for_local) else random.choice(vars)

	if read:
		read_var = var
	else:
		read_var = local_var_idx if (random.uniform(0, 1) > chance_for_local) else random.choice(vars)

	return "var_%s := var_%s" % (write_var, read_var)


def generate_statement(vars, idx, local_var_chance):
	var = random.choice(vars) if idx >= len(vars) else vars[idx]
	return make_statement(vars, var, local_var_chance)


def add_generated_statements(g, vars, statement_size, local_var_chance):
	vars_counter = 0
	for e in g.es:
		size = statement_size.randint()
		statement = []
		for i in range(0, size):
			statement.append(generate_statement(vars, vars_counter, local_var_chance))
			vars_counter += 1
		statement_txt = "; ".join(statement)
		if size > 1:
			statement_txt = "[%s]" % statement_txt
		e["statement"] = statement_txt
	return g


slco_template = """model %s {
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

statemachine = '''%s {
				variables
					%s
				initial %s
				states  %s
				transitions	%s
				} '''


def graph_to_slco(g, sm_idx):
	sm_name = "StateMachine_%s" % sm_idx
	local_vars = "Boolean var_%s" % local_var_idx
	init_state = "S0"
	states = " ".join(["S" + str(i) for i in range(1, g.vcount())])
	transitions = ""
	for e in g.es:
		transitions += "\n\t\t\t\t\tfrom S%s to S%s { %s }" % (e.source, e.target, e["statement"])
	return statemachine % (sm_name, local_vars, init_state, states, transitions)


def generate_model(config, name):
	sm_network = config.topology_gen_method.generate()
	num_sm = sm_network.vcount()
	# assign a variable to each edge (to be shared by two state machines)
	var_count = 0
	for e in sm_network.es:
		num_vars = max(1, config.vars_per_edge.randint())
		e["vars"] = [var_count+x for x in range(0, num_vars)]
		var_count += num_vars + 1

	if num_sm > 2:
		# randomly generate a number of variables shared by more than two state machines
		num_shared_vars = config.num_vars_shared_by_more_than_two_sms.randint()
		for x in range(0, num_shared_vars):
			var_count += 1
			num_shared = random.randint(2, num_sm-1)
			selected_edges = random.sample(list(sm_network.es), num_shared)
			for e in selected_edges:
				e["vars"].append(var_count)
	var_count += 1

	state_machines = []
	for sm in range(0, num_sm):
		g = config.sm_gen_method.generate()
		variables = [sm_network.es[e]["vars"] for e in sm_network.incident(sm, igraph.ALL)]
		flat_vars = [item for sublist in variables for item in sublist]
		g = add_generated_statements(g, flat_vars, config.statement_size, config.local_var_chance.randuniform())
		state_machines.append(graph_to_slco(g, sm))

	vars_text = "Boolean var_0\n"
	for i in range(1, var_count):
		vars_text += "\t\t\tBoolean var_%s\n" % i

	sm_text = state_machines[0] + "\n\n"
	for i in range(1, len(state_machines)):
		sm_text += "\t\t\t%s\n\n" % state_machines[i]

	return slco_template % (name, vars_text, sm_text)


def generate_models(config, gen_folder):
	for i in range(0, config.num_models_to_generate):
		name = config.model_name + "_model_" + str(i)
		outFile = open(join(gen_folder, name + ".slco"), 'w')
		out_model = generate_model(config, name)
		outFile.write(out_model)
		outFile.close()


def main(args):
	"""The main function"""
	dump_config = False
	if len(args) > 0:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 generator [<config_file> | -d <file_name>]")
			print("")
			print("Generate an SLCO 2.0 model.")
			print("-d <file_name>       Dumps a default config file named <file_name>")
			sys.exit(0)
		elif args[0] == '-d':
			dump_config = True

	if dump_config:
		config = Config()
		config.writeToFile(args[1])
		return

	config_file = args[0]
	config = Config.fromFile(config_file)

	if config_file.endswith(".cfg"):
		config_file = config_file[:-4]
	gen_dir = config_file
	gen_folder = join(this_folder, gen_dir)
	if not exists(gen_folder):
		mkdir(gen_folder)

	generate_models(config, gen_folder)


if __name__ == '__main__':
	args = []
	for i in range(1, len(sys.argv)):
		args.append(sys.argv[i])
	main(args)
