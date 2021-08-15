from .lts import LTS, ISuportLabelRename, ISupportLabelHide, ISupportMinimise, _lts_factories
from . import aut
from . import mcrl2
import os
import re
import io


class AutLTS(   LTS,
			    ISupportMinimise,
				ISupportLabelHide,
				ISuportLabelRename):
	_init_state = 0
	_transitions = dict()
	_action_labels = list()
	
	def __init__(self, init_state, transitions, action_labels):
		self._init_state    = init_state
		self._transitions   = transitions
		self._action_labels = action_labels
	
	@property
	def num_states(self):
		states = set(self._transitions.keys())
		for trans in self._transitions.values():
			for tgt in trans.values():
				states |= tgt
		return len(states)
	
	@property
	def num_transitions(self):
		counter = 0
		for trans in self._transitions.values():
			for tgt in trans.values():
				counter += len(tgt)
		return counter
	
	@property
	def transition_dict(self):
		return self._transitions
	
	@property
	def action_labels(self):
		return self._action_labels
	
	def hide_action_labels(self,hiding_set):
		hidden = False
		hiding_res = [re.compile(x) for x in hiding_set]
		for src, trans in self._transitions.items():
			for a in list(trans.keys()):
				for rex in hiding_res:
					m = rex.match(a)
					if m:
						tau_tgts = trans.get('tau', set())
						tau_tgts |= trans[a]
						trans['tau'] = tau_tgts
						del trans[a]
						hidden = True
						break
		return hidden
	
	
	def rename_action_labels(self, rename_dict):
		changed = False
		for src, trans in self._transitions.items():
			for a in list(trans.keys()):
				if a not in self._action_labels:
					continue
				self._action_labels.remove(a)
				new_a = rename_dict.get(a)
				if new_a:
					targets = trans.get(new_a, set())
					trans[new_a] = targets | trans[a]
					del trans[a]
					self._action_labels.add(new_a)
					changed = True
		return changed
	
	def write_to_file(self, path):
		folder, name = os.path.split(path)
		header = [self._init_state, str(self.num_transitions), str(self.num_states)]
		aut.write_to_file(folder, header, self._transitions, name)
	
	
	def minimise(self,equivalence):
		header = [self._init_state, str(self.num_transitions), str(self.num_states)]
		lts_stream = aut.to_string(header, self._transitions)
		lts_str = ''.join(list(lts_stream))
		out = mcrl2.minimize(lts_str, equivalence, 'aut', 'aut')
		header, trans, labels = aut.read(io.StringIO(out))
		return AutLTS(header[0], trans, labels)
	
	
	@staticmethod
	def _read_from_file(path):
		folder, name = os.path.split(path)
		header, trans, labels = aut.read_from_file(folder,name)
		return AutLTS(header[0], trans, labels)


# register AutLTS as read/writer for .aut files
_lts_factories['.aut'] = AutLTS._read_from_file
