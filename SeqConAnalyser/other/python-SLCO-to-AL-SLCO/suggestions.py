
from transformation_utils import *
from utils import stack
from collections import defaultdict

STATEMENT_MEMORY_FENCE = '#'

def parse_var(var):
	return var.split("'")[1]

def parse_suggestions(sugg_path):
	file = open(sugg_path, 'r')
	lines = file.readlines()
	file.close()

	suggestions = defaultdict(AD)
	for line in lines:
		start = line.find("(") + 1
		end = line.rfind("'") - 1
		line = line[start:end]
		sm = line.split(", ")
		statement = tuple([sm[0], sm[1], sm[2]])
		line = ""
		first = True
		for i in range(3,len(sm)):
			if not first:
				line += ", "
			else:
				first = False
			line += sm[i]
		if line.startswith('AD'):
			ad = suggestions.get(statement, [])
			ad.append(AD.parse(line))
			suggestions[statement] = ad

	#for ad in suggestions.values():
	#	ad.fix_conflicting_shuffles()
	total = 0
	for ad in suggestions.values():
		total += len(ad)
	return suggestions

class SuggestionSyntaxException(Exception):
	def __init__(self, value):
		self.parameter = value

	def __str__(self):
		return repr(self.parameter)


class AccessPattern:
	r = set()
	w = set()

	def __init__(self, r, w):
		self.r = r
		self.w = w

	def __hash__(self):
		return hash((frozenset(self.r), frozenset(self.w)))

	@staticmethod
	def parse(text):
		text = text[3:-1]
		splt = text.split('], [')
		rtxt = splt[0][1:]
		wtxt = splt[1][:-1]
		rset = {x.split("'")[1] for x in rtxt.split(', ')} if rtxt else set()
		wset = {x.split("'")[1] for x in wtxt.split(', ')} if wtxt else set()
		return AccessPattern(rset, wset)

	def find_assignment_full(self, statement_list, start):
		for i in range(start, len(statement_list)):
			statement = statement_list[i]
			if statement.__class__.__name__ == "str":
				continue
			b_write_found = len(self.w) == 0 or (len(self.w) == 1 and statement.left.var.name in self.w)
			reads = set()
			get_read_vars_from_expression(statement.right, reads)
			b_reads_found = not (self.r - reads)  # reads found if result is empty set
			# return if required elements are found
			if b_write_found and b_reads_found:
				return i
		return -1

	def match_partially(self, statement):
		write = None
		expression = None
		reads_found = set()
		if statement.__class__.__name__ == "Assignment":
			if self.w:
				if statement.left.var.name in self.w:
					write = statement.left.var.name

			expression = statement.right
		elif statement.__class__.__name__ == "Expression":
			expression = statement

		if expression:
			get_read_vars_from_expression(expression, reads_found)
			reads_found = reads_found & self.r  # set of reads found
		return write, reads_found

	def find_sub_composite(self, statements, guard):
		# -2 indicates not found, -1 indicates guard partially satisfies pattern
		first = -2
		last = -2

		_, reads = self.match_partially(guard)
		if reads:
			last = first = -1

		total_writes = set()
		total_reads = set()

		for i in range(0, len(statements)):
			write, reads = self.match_partially(statements[i])
			if write or reads:
				if write:
					total_writes.add(write)
				total_reads = total_reads | reads
				if first == -2:
					last = first = i
				else:
					last = i

		# return_statements = [composite]
		# when self.r and self.w are fully covered then we have a sub composite
		read_remainder = (self.r - total_reads)
		write_remainder = (self.w - total_writes)
		# if remainders are empty then self.r and self.w are fully covered
		if not read_remainder and not write_remainder:
			return first, last + 1  # range is [low, last + 1), i.e. last+1 is not selected by the range
		return -2, -2

	def __repr__(self):
		return "R: " + str(self.r) + ", W: " + str(self.w)

	def __str__(self):
		return "R: " + str(self.r) + ", W: " + str(self.w)

class AD:
	access_patterns = set()
	shuffles = set()

	def __init__(self):
		self.access_patterns = set()
		self.shuffles = set()

	@staticmethod
	def parse(text):
		text = text[4:-1]
		if text.startswith("A'("):
			ap, text = get_and_remove_up_to_first(text, ')', 3)
			access_pattern = AccessPattern.parse(ap + ')')
		else:
			raise SuggestionSyntaxException("An AD requires at least one access pattern, it may not be empty")
		# access_pattern != None otherwise exception is raised

		text = text[1:-1]
		shuffles = set()
		while text.startswith("SH'("):
			text = text[4:]
			shuffle, text = get_and_remove_up_to_first(text, "SH'", 0)
			shuffle = shuffle if shuffle.endswith(')') else shuffle[:-2]
			shuffles.add(SH.parse("SH'(" + shuffle))

		ad = AD()
		ad.access_patterns.add(access_pattern)
		ad.shuffles = shuffles
		#ad.merge_overlapping_access_patterns()
		return ad
	
	def size(self):
		total = 0
		for ac in self.access_patterns:
			total += len(ac.r) + len(ac.w)
		return total

	def get_access_patterns(self):
		return self.access_patterns

	def get_shuffles(self):
		return self.shuffles

	# def merge(self, ad):
	# 	self.access_patterns = self.access_patterns | ad.access_patterns
	# 	self.shuffles = self.shuffles | ad.shuffles

	# def merge_overlapping_access_patterns(self):
	# 	# merge access paterns that have reads or writes in common
	# 	for ap1 in self.access_patterns:
	# 		for ap2 in self.access_patterns:
	# 			if ap1 == ap2:
	# 				continue
				
	# 			if ap1.r & ap2.r or ap1.w & ap2.w:
	# 				# merge the two access_patterns
	# 				ap1.r = ap1.r | ap2.r
	# 				ap1.w = ap1.w | ap2.w
	# 				ap2.r = ap2.w = set()
	# 	# remove redundant access_patterns
	# 	self.access_patterns = {x for x in self.access_patterns if x.r}
		
	# def apply_shuffling(self, assignments, guard):
	# 	graph = ShuffleGraph(self.shuffles, assignments, guard)
	# 	return graph.apply()
				
	# def apply_atomics(self, statements, guard, atomic_sets, composite_class):
	# 	# move composite statement over the part that requires it
	# 	# first get matching ranges for each access pattern ap
	# 	ranges = []
	# 	for ap in self.access_patterns:
	# 		low, high = ap.find_sub_composite(statements, guard)
	# 		if low != -2:
	# 			ranges.append((low, high))
	# 	# add ranges from atomic_sets
	# 	for s in atomic_sets:
	# 		low = 99999999999
	# 		high = -1
	# 		for a in s:
	# 			if a == guard:
	# 				low = -1
	# 			else:
	# 				i = statements.index(a)
	# 				high = max(high, i)
	# 				low = min(low, i)
	# 		ranges.append((low, high+1))
	# 	# merge overlapping ranges
	# 	for i in range(0, len(ranges)):
	# 		for j in range(0, len(ranges)):
	# 			if i == j:
	# 				continue
	# 			lowi, highi = ranges[i]
	# 			lowj, highj = ranges[j]
	# 			if lowj <= lowi < highj or lowj < highi <= highj:
	# 				# merge range
	# 				new_low = lowi if lowi < lowj else lowj
	# 				new_high = highi if highi > highj else highj
	# 				ranges[i] = (new_low, new_high)
	# 				# 'disable' redundant range
	# 				ranges[j] = (-2, -2)
	# 	ranges = [(low, high) for (low, high) in ranges if low != -2]
	# 	# ranges are now disjoint
	# 	# sort ranges by lowest
	# 	sorted(ranges, key=lambda x: x[0])
	# 	return_statements = []
	# 	asgn_idx = 0
	# 	for r in ranges:
	# 		low, high = r
	# 		if asgn_idx < low:
	# 			return_statements.extend(statements[asgn_idx:low])
	# 		asgn_idx = high  # next index of assignments
			
	# 		# create new composite class
	# 		new_composite = composite_class.__new__(composite_class)
	# 		composite_class._tx_metamodel._init_obj_attrs(new_composite)
			
	# 		if low == -1:
	# 			new_composite.guard = guard
	# 			low = 0
	# 		new_composite.assignments = [x for x in statements[low:high] if x != '#']
	# 		return_statements.append(new_composite)
	# 	asgn_size = len(statements)
	# 	if asgn_size > asgn_idx:
	# 		return_statements.extend(statements[asgn_idx:asgn_size])
	# 	return return_statements
	
	# def apply(self, composite):
	# 	statements = composite.assignments
	# 	statements, atomics = self.apply_shuffling(statements, composite.guard)
	# 	statements = self.apply_atomics(statements, composite.guard, atomics, composite.__class__)
	# 	return statements


SH_NONE     = 0
SH_BEFORE   = 1
SH_AFTER    = 2
SH_BOTH     = 3

class SH:
	before = None
	after = None

	def __init__(self, before, after):
		self.before = before
		self.after = after

	def __hash__(self):
		return hash((hash(self.before), hash(self.after)))

	@staticmethod
	def parse(text):
		splt = text.split("A'")
		before = AccessPattern.parse("A'" + splt[1][:-2])
		after = AccessPattern.parse("A'" + splt[2][:-1])
		return SH(before, after)

	def apply(self, assignment_list):
		i_after = -1
		shuffled = False
		while True:  # termination condition at first if
			i_after = self.after.find_assignment_full(assignment_list, i_after + 1)
			i_before = self.before.find_assignment_full(assignment_list, i_after + 1)
			
			if i_before == -1 or i_after == -1:
				break
			if i_before > i_after:  # need to swap
				# TODO check whether swap is allowed by conflict-serialisability
				temp = assignment_list[i_before]
				assignment_list[i_before] = assignment_list[i_after]
				assignment_list[i_after] = temp
				assignment_list.insert(i_before, STATEMENT_MEMORY_FENCE)
				shuffled = True
		return shuffled

	def match(self, assignment):
		reads_before, writes_before = self.before.match_partially(assignment)
		reads_after , writes_after  = self.after.match_partially(assignment)
		before = reads_before or writes_before
		after = reads_after or writes_after
		if before and after:
			return SH_BOTH
		elif before:
			return SH_BEFORE
		elif after:
			return SH_AFTER
		return SH_NONE
	
	@staticmethod
	def get_shuffle_to(i_before, i_after, assignments):
		if i_before == i_after:
			return i_before, i_before
		
		i_before_new = i_before
		i_after_new = i_after
		
		# TODO:
		# Simple approach, for now we move afters towards end only
		
		# reads = set()
		# get_read_vars_from_expression(assignments[i_before].right, reads)
		# write = assignments[i_before].left.var.name
		#
		# # attempt to move i_before_new before i_after
		# while i_before_new >= i_after:
		# 	i_before_new -= 1
		# 	reads_a = set()
		# 	get_read_vars_from_expression(assignments[i_before_new].right, reads_a)
		# 	write_a = assignments[i_before_new].left.var.name
		# 	if {write} & (reads_a | {write_a}) or {write_a} & (reads | {write}):
		# 		# Two assignments have accesses on a variable in common of which at least one is a write
		# 		i_before_new += 1
		# 		break
		
		
		reads = set()
		get_read_vars_from_expression(assignments[i_after].right, reads)
		write = assignments[i_after].left.var.name

		while i_after_new <= i_before_new:
			i_after_new += 1
			reads_a = set()
			get_read_vars_from_expression(assignments[i_after_new].right, reads_a)
			write_a = assignments[i_after_new].left.var.name
			if {write} & (reads_a | {write_a}) or {write_a} & (reads | {write}):
				# Two assignments have accesses on a variable in common of which at least one is a write
				i_after_new -= 1
				break
		
		return i_before_new, i_after_new
				

class Node:
	parents = []
	children = []
	value = -2
	
	def __init__(self, value):
		self.value = value
		
		

class ShuffleGraph:
	nodes = []
	node_map = {}
	guard_node = None
	
	atomic_sets = []
	graph = defaultdict(list)  # graph[n] = list of nodes before node n
	guard = None
	assignments = []
	
	def __init__(self, shuffles, assignments, guard):
		self.assignments = assignments
		self.guard = guard
		# find matches for shuffles
		# matches: for a shuffle and SH_... give the statement indices (with -1 = guard)
		matches = {SH_BEFORE: defaultdict(list), SH_AFTER: defaultdict(list),
			SH_BOTH: defaultdict(list), SH_NONE: defaultdict(list)}
		for sh in shuffles:
			m = sh.match(guard)
			matches[m][sh].append(-1)
			for i in range(0, len(assignments)):
				m = sh.match(assignments[i])
				matches[m][sh].append(i)
		# all shuffles sh with matches[sh,SH_BOTH] = [...,a,...]form a cycle on a
		# these have to be protected
		self.atomic_sets = \
			[{assignments[x]} for lst in matches[SH_BOTH].values() for x in lst.values()]
		# these statements have to be removed from matches[SH_BEFORE or AFTER]
		for sh in matches[SH_BEFORE]:
			matches[SH_BEFORE][sh] = [x for x in matches[SH_BEFORE][sh] if x not in self.atomic_sets]
		for sh in matches[SH_AFTER]:
			matches[SH_AFTER][sh] = [x for x in matches[SH_AFTER][sh] if x not in self.atomic_sets]
		# build shuffle graph to find cycles in order of statements dictated by shuffles
		for sh in shuffles:
			for node in matches[SH_AFTER][sh]:
				n = self.node_map.get(node)
				if not n:
					n = Node(node)
					self.nodes.append(n)
				if node == -1:
					self.guard_node = n
				for value in matches[SH_BEFORE][sh]:
					child = self.node_map.get(value)
					if not child:
						child = Node(value)
						self.nodes.append(child)
					child = Node(value)
					child.parents.append(n)
					n.children.append(child)
				# draw edges: node --before--> match
				self.graph[node].extend(matches[SH_BEFORE][sh])
		# nothing may be shuffled before the guard
		if self.guard_node:#graph[-1]:
			self.b_cycle = True
		# prune graph from illegal shuffles
		# TODO self.resolve_illegal_shuffles(assignments)
		# resolve cycles
		self.b_cycle = self.resolve_cycles(assignments)
	
	def resolve_cycles(self, assignments):
		visited = set()
		for s in self.nodes:
			if s in visited:
				continue

			node_stack = stack()
			node_stack.append(s)
			while node_stack:
				node = node_stack.peek()
				if node in visited:
					# node was already visited
					node_stack.pop()
					continue
				
				visited.add(node)
				targets = s.children
				for target in targets:
					if target not in visited:
						node_stack.append(target)
					elif target in node_stack:
						# if target is visited and in state_stack, then there is a cycle!
						# NOTE: it would be sufficient to find all elements in the cycle
						# and put these together in a composite
						# for now we keep the original composite intact
						return True
		return False
	
	def resolve_illegal_shuffles(self, assignments):
		for n in self.nodes:
			i_after = n.value
			befores = n.children
			to_remove = []
			for b in befores:
				i_before = b.value
				i_before_new, i_after_new = SH.get_shuffle_to(i_before, i_after, assignments)
				if i_before_new > i_after_new:
					# the shuffle edges cannot be applied (due to a conflict)
					self.atomic_sets.append({assignments[i_before], assignments[i_after]})
					to_remove.append(b)
			for b in to_remove:
				b.parents.remove(n)
				n.children.remove(b)
				
	def apply(self):
		if self.b_cycle:
			return self.assignments, [set(self.assignments)]
		else:
			return self.assignments, []
		
		new_assignments = list(self.assignments)
		# NOTE: we move afters towards end only to avoid creating new conflicts
		# while graph not empty
		# find leaves
		leaves = set()
		for n in self.nodes:
			if not n.children:
				leaves.add(n)
		# traverse tree upwards
		while leaves:
			for n in list(leaves):
				leaves.remove(n)
				ni = new_assignments.index(n)
				for p in n.parents:
					pi = new_assignments.index(p)
					if ni < pi:
						continue
					item = new_assignments.pop(pi)
					new_assignments.insert(ni + 1, '#')
					new_assignments.insert(ni + 2, item)
					p.remove_child(n)
					if not p.children:
						leaves.add(p)
		
		return new_assignments, self.atomic_sets
