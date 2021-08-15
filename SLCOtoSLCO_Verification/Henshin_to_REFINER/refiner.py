
class LTS:
	transitions = set()

	def __init__(self, transitions):
		self.transitions = transitions

	def add_transition(self, source, action, target):
		self.transitions.add((source, action, target))

	def rename_states(self, mapping):
		new_transitions = set()
		for s, a, t in self.transitions:
			new_transitions.add((mapping[s], a, mapping[t]))
		self.transitions = new_transitions

	@staticmethod
	def get_states_in_transition_set(trans_set):
		states = set()
		for s, _, t in trans_set:
			states.add(s)
			states.add(t)
		return states

	def get_states(self):
		return LTS.get_states_in_transition_set(self.transitions)

	def to_string(self, num_glue_states, num_out_states):
		num_states = len(self.get_states())
		# TODO; what are the 3rd and 4th elements of des header?
		header = "des(%s,%s,%s,%s)\n" % (len(self.transitions), num_states, num_glue_states, num_out_states)
		trans = ['(%s,"%s",%s)' % (s, a, t) for s, a, t in self.transitions]
		return header + "\n".join(trans)


class RuleSystem:
	rules = []
	sync_left = []
	sync_right = []

	def to_string(self):
		str_rules = [r.to_string() for r in self.rules]
		str_sync_left = [l.to_string() for l in self.sync_left]
		str_sync_right = [l.to_string() for l in self.sync_right]
		return "transform\n%s\n\nadd\nsynch left\n\n%ssynch right\n%s" % \
			("\n\n".join(str_rules), "\n".join(str_sync_left), "\n".join(str_sync_right))


class Rule:
	L = None
	R = None
	glue_states = set()

	def __init__(self, L, R):
		self.L = L
		self.R = R
		l_states = self.L.get_states()
		r_states = self.R.get_states()
		self.glue_states = l_states & r_states
		# rename state numbers so that glue_states appear first
		mapping = {}
		counter = 0
		for s in self.glue_states:
			mapping[s] = counter
			counter += 1
		remaining_states = (l_states | r_states) - self.glue_states
		for s in remaining_states:
			mapping[s] = counter
			counter += 1
		self.L.rename_states(mapping)
		self.R.rename_states(mapping)

	def to_string(self):
		l_str = self.L.to_string(len(self.glue_states), 0)
		r_str = self.R.to_string(len(self.glue_states), 0)
		return "%s\n->\n%s" % (l_str, r_str)


class Law:
	t = set()
	a = ""

	def to_string(self):
		return "%s -> %s" % (" ".join(self.t), self.a)

