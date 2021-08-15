
class stack(list):
	# use append instead of push
	#def push(self, item):
	#	self.append(item)
	
	def isEmpty(self):
		return not self
	
	def peek(self):
		return self[-1]