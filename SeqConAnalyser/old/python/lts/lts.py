from abc import ABC, abstractmethod
from enum import Enum
import os

_lts_factories = dict()

class ISupportMinimise(ABC):
	@abstractmethod
	def minimise(self,equivalence):
		pass


class ISupportLabelHide(ABC):
	
	# @return:  True iff some action label has been hidden
	@abstractmethod
	def hide_action_labels(self, hiding_set):
		pass



class ISuportLabelRename(ABC):
	# @return:  True iff some action label has been renamed
	@abstractmethod
	def rename_action_labels(self, rename_dict):
		pass



# Abstract LTS class
# instatiate via read_from_file
class LTS(ABC):
	
	class Equivalence(Enum):
		WEAK_BISIM = 0
		BRANCHING_BISIM = 1
		
	_folder = ''
	_filename = ''
	_ext = ''
	
	@property
	def folder(self):
		return self._folder
	
	
	@property
	def filename(self):
		return self._filename
	
	
	def ext(self):
		return self._ext
	
	
	@property
	@abstractmethod
	def num_states(self):
		pass
	
	@property
	@abstractmethod
	def num_transitions(self):
		pass

	@property
	@abstractmethod
	def transition_dict(self):
		pass
	
	@property
	@abstractmethod
	def action_labels(self):
		pass
	
#	@property
#	@abstractmethod
#	def states(self):
#		pass
	
#	@abstractmethod
#	def get_transitions(self):
#		pass
	
#	@abstractmethod
#	def get_transitions_from_state(self,state):
#		pass

	#TODO:
	@abstractmethod
	def write_to_file(path):
		pass
	
	@staticmethod
	def create(path):
		_, ext = os.path.splitext(path)
		create_method = _lts_factories[ext]
		lts = create_method(path)
		lts._folder, filename = os.path.split(path)
		lts._filename, lts._ext = os.path.splitext(filename)
		return lts
	
	
	@staticmethod
	@abstractmethod
	def _read_from_file(path):
		pass

