import re

def stripInterfacecounter(action):
	"""Strip the interface counter off the given action"""
	index = action.find('_itf')
	if index == -1:
		return action
	else:
		return action[:-(len(action)-index)]

def stripParams(action):
	"""Strip the parameters and quotes off the action string"""
	# We strip both LOTOS style and mCRL/mCRL2 style, i.e. everything following
	# '(' and ' !', respectively
	l1 = action.split('!')
	l2 = l1[0].split('#')
	l1 = l2[0].split('(')
	#newa = l1[0].rstrip(' ')
	newa = l1[0].replace('"','')
	return newa

def stripPlaceholder(action):
	"""Strip the parameters and quotes off the action string"""
	# We strip placeholders
	l1 = action.split('#')
	#newa = l1[0].rstrip(' ')
	newa = l1[0].replace('"','')
	return newa

def getParams(action):
	"""Strip the action off the params"""
	params = ''
	l1 = action.split('!')
	if len(l1) > 1:
		for i in range(1,len(l1)):
			params += '!' + l1[i]
	else:
		l1 = action.split('#')
		if len(l1) > 1:
			params = '#' + l1[1]
		else:
			l1 = action.split('(')
			if len(l1) > 1:
				for i in range(1,len(l1)):
					params += '(' + l1[i]
	params = params.rstrip('"')
	return params

def getPlaceholder(action):
	"""Return the placeholder in action"""
	params = ''
	l1 = action.split('#')
	if len(l1) > 1:
		params = '#' + l1[1]
	#params = params.rstrip('"')
	return params

def isMatch(action1, action2, placeholderdict):
	"""Does action1 match action2? Comparison is sensitive for placeholders in action1"""
	prefix = stripPlaceholder(action1)
	prefix2 = action2[:-(len(action2)-len(prefix))+1].replace('"','')
	if prefix[0] == '\'':
		areg = re.compile('"' + prefix + '"')
		if areg.match(prefix2) is not None:
			prefixesEqual = True
		else:
			prefixesEqual = False
	else:
		prefixesEqual = (prefix == prefix2)
	if not prefixesEqual:
		return False
	params = getPlaceholder(action1)
	if params != '':
		# try to obtain the placeholder value from placeholderdict
		value = placeholderdict.get(params)
		if value is not None:
			return (prefix1 + value == action2)
		else:
			return True
	else:
		return getParams(action2) == ''

def getPlaceholdervalue(action1, action2):
	"""PRECONDITION: action1 matches action2. Return, if present, the value for the placeholder in action1"""
	params = getPlaceholder(action1)
	prefix = stripPlaceholder(action1)
	#index = action1.find(params)
	if params == '':
		return ''
	else:
		# get the placeholder value
		#return action2[index:].rstrip('"')
		return action2[len(prefix)+1:]
		#.rstrip('"')

def rename(action, placeholderdict):
	"""Renames the action by replacing its placeholder, if present, by the concrete value in the given dict"""
	prefix = stripPlaceholder(action)
	if prefix == action:
		return action
	params = getPlaceholder(action)
	value = placeholderdict.get(params)
	if value is None:
		return action
	return prefix + value
