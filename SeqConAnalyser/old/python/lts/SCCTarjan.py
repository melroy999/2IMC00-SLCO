#Tarjan’s SCC detection algorithm

def peek(stack):
	"""Return but do not pop top element of stack"""
	return stack[len(stack) - 1]


def identifySCCs(L, initialstates, SCCdict, SCCs):
	"""Identify the SCC in L, keep track for each state to which SCC it belongs in 'SCCdict', and store the SCCs in 'SCCs'"""
	# list of all states
	states = set(L.keys())
	states |= {tgt for outgoing in L.values() for tgt in outgoing.keys()}
	# call stack
	callstack = []
	# scc stack
	sccstack = []
	sccstackset = set()
	# numbers
	number = {}
	scc_index = 0
	scc_count = 0
	# iterate over all states
	for s in initialstates:
		if number.get(s) == None:
			outgoing = L.get(s, {})
			targets = outgoing.keys()
			callstack.append((s, list(targets)))
			
			while len(callstack) > 0:
				s, targets = peek(callstack)
				if number.get(s) == None:
					number[s] = scc_index
					SCCdict[s] = scc_index
					scc_index += 1
					sccstack.append(s)
					sccstackset.add(s)
				# walk over targets
				while len(targets) > 0:
					t = peek(targets)
					if number.get(t) == None:
						# add successor to call stack
						toutgoing = L.get(t, {})
						ttargets = toutgoing.keys()
						callstack.append((t, list(ttargets)))
						break
					elif t in sccstackset:
						r = SCCdict.get(s)
						SCCdict[s] = min(r, number.get(t))
					targets.pop()
				if len(targets) == 0:
					# is s a root state?
					if SCCdict[s] == number[s]:
						# a new SCC has been found
						scc_count += 1
						if len(sccstack) > 0:
							t = sccstack.pop()
							sccstackset.remove(t)
						while t != s and len(sccstack) > 0:
							SCCdict[t] = SCCdict[s]
							t = sccstack.pop()
							sccstackset.remove(t)
					callstack.pop()
					# update predecessor
					if len(callstack) > 0:
						t, _ = peek(callstack)
						SCCdict[t] = min(SCCdict[t], SCCdict[s])
	# a hash table to map SCC id's to consecutive id's
	scchtable = {}
	scc_newidcount = 0
	# update SCCdict
	for s in states:
		sSCC = SCCdict.get(s)
		# lookup the new SCC id
		scc_newid = scchtable.get(sSCC)
		if scc_newid == None:
			scchtable[sSCC] = scc_newidcount
			scc_newid = scc_newidcount
			scc_newidcount += 1
		SCCdict[s] = scc_newid
	# construct the SCCs in 'SCCs'
	for j in range(0, scc_newidcount):
		SCCs.append([0, {}])
	for s in states:
		sSCC = SCCdict.get(s)
		# update state counter of the corresponding SCC
		SCCs[sSCC][0] += 1
		# identify relevant outgoing transitions
		outgoing = L.get(s, {})
		sccoutgoing = {t: a for t,a in outgoing.items() if SCCdict[t] == sSCC}
		SCCs[sSCC][1][s] = sccoutgoing
