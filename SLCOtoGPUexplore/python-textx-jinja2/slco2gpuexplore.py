import sys
from os import mkdir
from os.path import exists, dirname, basename, join, split
import jinja2
from textx.metamodel import metamodel_from_file
from copy import deepcopy, copy
import itertools
import glob
import traceback
import math
import importlib

# Variable to bind the pyCUDA library to in case it is needed. If it is not, the dynamic import means that it is not required that pyCUDA is installed.
cuda = ''

modelname = ""
model = ""
# experimental settings:
# - gpuexplore2_succdist applies successor generation work distribution over threads in the style of GPUexplore 2.0 (vector groups)
# - no_regsort disables warp-based in register sorting, thereby reducing the amount of expected regularity in successor generation.
# - no_smart_fetching disables smart fetching of vectortrees from the global memory hash table.
# - compact_hash_table enables compact storage of states in the global memory hash table.
# - with_cuckoo enables Cuckoo hashing.
gpuexplore2_succdist = False
no_regsort = False
no_smart_fetching = False
compact_hash_table = True
with_cuckoo = True
# The maximum number of allowed Cuckoo hashing evictions in one chain.
max_evictions = 10

# Will the GPU be queried, using pycuda, to use GPU properties during code generation? This requires the presence of a CUDA capable GPU
# in the machine
gpu_querying = True
# if GPU querying is enabled, it will be used to pre-determine the size of shared memory caches. This variable will store the number
# of elements that a cache can contain.
nr_cache_elements = 0

# number of blocks
nrblocks = 3120
# number of threads per block
nrthreadsperblock = 512

# the size of a warp
warpsize = 32

this_folder = dirname(__file__)

# import libraries
sys.path.append(join(this_folder,'../../libraries'))
from slcolib import *
this_folder = dirname(__file__)

# nr of elements per thread, for intra-warp regsort of tile elements
regsort_nr_el_per_thread = 0
# nr of warps for a tile
nr_warps_per_tile = 0

# nr of bits per shared memory element
nr_bits_shared_mem_element = 32

actions = set([])

# state vector size (in bits)
vectorsize = 0
# max state size (in bits)
max_statesize = 0
# max array size (in bits)
max_arrayindexsize = 0

# dictionary to look up signal size (in bits) for each channel
signalsize = {}
# provide number encoding for each signal, per channel
signalnr = {}

# structure of vector
vectorstructure = []
vectorstructure_string = ""
# dictionary to identify vector elements
elements_strings = {}
# vector tree, to navigate from root to leaves
vectortree = {}
# transposed vector tree
vectortree_T = {}
# the node ids (from the left) in each level of the vectortree
vectortree_level_ids = {}
# dictionary indicating for each level how many leaves are present
vectortree_level_nr_of_leaves = {}
# dictionary indicating for each level how many nodes have two children
vectortree_level_nr_of_nodes_with_two_children = {}
# dictionary indicating for each level how many state parts are reachable
vectortree_nr_reachable_state_parts = {}
# assign threads to vectortree nodes
vectortree_node_thread = {}
# size of a vectortree group (groups of threads needed to fetch a vector tree from the global hash table)
# vectortree_group_size is a power of two, sufficiently large to contain vectortree_size (the actually needed number of threads)
vectortree_group_size = 0
vectortree_size = 0
# depth of a vectortree
vectortree_depth = 0
# dictionary to map node ids to part ids
vectorpart_id_dict = {}
# dictionary in the other direction, i.e., from part id to node id
vectornode_id_dict = {}

# dictionary of bitmasks for smart vectortree fetching
smart_vectortree_fetching_bitmask = {}

# dictionary indicating in which part of the vector (which integer) a given vector element can be found
vectorelem_in_structure_map = {}
# state order: given a number, the name of the state can be retrieved
state_order = {}
# state id: given a state object, retrieve its order number in the list of states of that state machine
state_id = {}
# map smnames to smobjects
smname_to_object = {}
# maximum allocations needed to process a transition block
max_buffer_allocs = 0
# sizes of buffers needed to process dynamic array indexing of arrays in model
all_arrayindex_allocs_sizes = []
# variables to keep track of fetched vectorparts
fetched = {0: -1, 1: -1}

# tile size (nr of vectortrees to be processed by a block in one iteration)
tilesize = 0

# constant representing 'no state'; high enough to not coincide with an existing state id
no_state_constant = 0
# constant representing no priority; higher than the highest (lowest in interpretation) priority
no_prio_constant = 0

# set of statemachine names in the system
smnames = set([])

# action alphabet of each state machine
alphabet = {}
# set of actions, per class, requiring synchronisation
syncactions = {}
# dictionary providing target states when an action is performed in a given state
actiontargets = {}

# dictionary indicating which (Object,StateMachine) pairs are potentially receiving messages from a given synchronous channel
syncreccomm = {}

# dictionary of array names in model (with types and sizes)
arraynames = {}

# dictionary for arrays that are dynamically accessed in the model. For each such array, an item (name,range) is provided, with name the scopename
# of the array, and range the range of vector parts in which the array is stored in vectors.
dynamic_access_arrays = {}
# dictionary for arrays, providing how the array elements are distributed over the vectorparts. A tuple (p, (i0,j0,s0,z0,r0), (i1,j1,s1,z1,r1), ...) is given,
# where p is the ID of the first vectorpart containing x[0] and further, and the (ik,jk,sk,zk,rk) tuples indicate that in vectorpart p+k, array elements
# ik to (at least left part of) jk are stored, with ik being stored at position sk. In case jk is also partly stored in part p+k+1, zk indicates the number
# of bits stored in that part, and rk is the starting position in that part. 
array_in_structure_map = {}

# dictionary for an (asynchronous channel, vectorpart) pair the range of buffer elements of that channel stored in that part
async_channel_vectorpart_buffer_range = {}

# dictionary providing the channels to which a given port is connected
connected_channel = {}

# LTL property
property_file = ""
# check for deadlocks?
deadlock_check = False

# Configuration of Cleary tables
# Size of GPU global memory in GB
global_memsize = 24

# Bits needed for root element and internal element addressing
def nr_bits_address_root():
	global global_memsize, compact_hash_table, vectorsize
	if not compact_hash_table:
		if vectorsize > 62:
			return 31
		else:
			if global_memsize <= 24:
				if vectorsize <= 31:
					return 64
				else:
					return 32
			else:
				return 64
	elif global_memsize == 24:
		return 32
	elif global_memsize == 48:
		return 33

def nr_bits_address_internal():
	global global_memsize, compact_hash_table
	if not compact_hash_table:
		if vectorsize > 62:
			return 31
		else:
			if global_memsize <= 24:
				if vectorsize <= 31:
					return 64
				else:
					return 32
			else:
				return 64
	elif global_memsize == 24:
		return 29
	elif global_memsize == 48:
		return 30

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# Jinja2 filter to compute the log 2 of a number
def log2(i):
	return int(math.log(i, 2))

# Jinja2 filter to compute 2 power of a number
def pow2(i):
	return int(math.pow(2, i))

# Jinja2 filter to compute bitwise xor of two integers
def xor(i, j):
	return i^j

# Jinja2 filter to perform a one position bit right shift
def bitshift_one_right(i):
	return i >> 1

# Jinja2 filter to perform a bit left shift
def bitshift_left(i, n):
	return i << n;

# Jinja2 filter to provide the hexadecimal version of a given integer
def hexa(i):
	return hex(i)

# Jinja2 filter to check list membership
def in_list(i, L):
	return i in L

def cudatype(s, ignoresize):
	"""Maps type names from SLCO to CUDA"""
	if s.base == 'Integer':
		if s.size < 1 or ignoresize:
			return 'elem_inttype'
		else:
			return 'elem_inttype[]'
	elif s.base == 'Boolean':
		if s.size < 1 or ignoresize:
			return 'elem_booltype'
		else:
			return 'elem_booltype[]'
	elif s.base == 'Byte':
		if s.size < 1 or ignoresize:
			return 'elem_chartype'
		else:
			return 'elem_chartype[]'

def operator(s):
	"""Provide characters to display given operator s"""
	if s == '>':
		return '>'
	elif s == '<':
		return '<'
	elif s == '>=':
		return '>='
	elif s == '<=':
		return '<='
	elif s == '=':
		return '=='
	elif s == '!=':
		return '!='
	elif s == 'and':
		return '&&'
	elif s == 'or':
		return '||'
	elif s == '<>':
		return '!='
	else:
		return s

def sign(s):
	if s == 'not':
		return '!'
	else:
		return s

def outgoingtrans(s,t):
	"""Return the set of transitions with s as source in a list sorted by transition priority"""
	tlist = []
	for tr in t:
		if tr.source == s:
			tlist.append(tr)
	tlist = sorted(tlist, key=(lambda x: x.priority))
	return tlist

def object_trans_to_be_processed_by_sm_thread(s,o):
	"""Return the set of transitions with s as source that must be processed by the thread assigned to state machine sm, in a list sorted by transition priority. o is Object owning s."""
	global state_order
	tlist = []
	# lookup thread id
	for i in range(0,len(state_order)):
		if state_order[i] == o.name + "'" + s.parent.name:
			break
	t = s.parent.transitions
	for tr in t:
		if tr.source == s and must_be_processed_by(tr, i, o):
			tlist.append((o,tr))
	tlist = sorted(tlist, key=(lambda x: x[1].priority))
	return tlist

def gettypesize(t):
	"""Return size of base of given type t, or integer if an integer is passed"""
	if isinstance(t, int):
		return t
	add = 0
	if t.base == 'Integer':
		add = 32
	elif t.base == 'Boolean':
		add = 1
	else:
		add = 8
	return add

def getlogarraysize(t):
	"""Return the log 2 of the size of array type t"""
	if t.size == 0:
		return 0
	else:
		return int(max(1,math.ceil(math.log(t.size, 2))))

def scopename(v,i,o):
	"""Return full name of variable v, possibly with index i, if constant. in case a Port is given, return name of connected Channel"""
	if v.__class__.__name__ == "Channel":
		if i == None:
			name = v.name
		elif v.synctype == 'async':
			if RepresentsInt(i):
				name = v.name + "[" + str(i) + "][0]"
			else:
				name = v.name + str(i)
		else:
			name = i
	else:
		name = o.name
		if v.parent.__class__.__name__ == "StateMachine":
			name += "'" + v.parent.name
		name += "'" + v.name
		if v.__class__.__name__ != "StateMachine":
			if i != None:
				i_str = getinstruction(i, o, {})
				if RepresentsInt(i_str):
					name += "[" + i_str + "]"
	return name

def vectorpart_is_combined_with_nonleaf_node(p):
	"""Return whether or not the given part (number) is combined with a non-leaf node in the vector tree"""
	global vectorstructure, compact_hash_table, smnames, vectorsize
	if p < len(vectorstructure)-1:
		return False
	if compact_hash_table:
		# # the allowed size of the part is influenced by whether or not we try to store it in the root.
		# # we try to insert it in the root if we have either two state parts (parts with state machine states)
		# # and no data parts (parts without state machine states), or we have exactly one data part.
		# nrstateparts = 0
		# nrdataparts = 0
		# for t in vectorstructure:
		# 	if t[0][0] in smnames:
		# 		nrstateparts += 1
		# 	else:
		# 		nrdataparts += 1
		# if (nrstateparts == 2 and nrdataparts == 0) or (nrdataparts == 1):
		# 	return vectorstructure_part_size(vectorstructure[p]) <= nr_bits_address_internal()
		# else:
		# 	return vectorstructure_part_size(vectorstructure[p]) <= (64-1-nr_bits_address_internal())
		return vectorstructure_part_size(vectorstructure[p]) < nr_bits_address_internal()
	elif vectorsize > 62:
		return vectorstructure_part_size(vectorstructure[p]) < 31
	else:
		return False

def vector_has_nonstate_parts():
	"""Return whether the vectorstructure contains parts without state machine states"""
	global vectorstructure, smnames

	for t in vectorstructure:
		if t[0][0] not in smnames:
			return True
	return False

def get_vector_tree_to_node_navigation(p):
	"""For the given tree node (number), provide how to navigate to it from the root node"""
	global vectortree, vectortree_T

	trace = []
	# find path from p to root first
	current = p
	while current != 0:
		parent = vectortree_T[current]
		trace.append(parent)
		current = parent
	# reverse list
	trace = trace[::-1]
	# construct navigation instructions
	nav = []
	for i in range(0,len(trace)):
		if i < len(trace)-1:
			nextnode = trace[i+1]
		else:
			nextnode = p
		children = vectortree[trace[i]]
		nav.append(children[0] != nextnode)
	return nav

def get_vector_tree_to_part_navigation(p):
	"""For the given vector part (number), provide how to navigate to it from a root node"""
	return get_vector_tree_to_node_navigation(vectornode_id(p))

# def get_bitmask(s, write):
# 	"""Return bitmask to extract s from statevector (flag False) or set s in statevector (flag True)"""
# 	global vectorstructure, vectorsize

# 	# first find s
# 	e = -1
# 	for t in vectorstructure:
# 		for (s1,i) in t:
# 			if s1 == s:
# 				e = t
# 				break
# 		if e != -1:
# 			break
# 	# compute lowerbound
# 	lowerbound = 2
# 	upperbound = 0
# 	for i in range(0,len(e)):
# 		if e[i][0] != s:
# 			lowerbound += e[i][1]
# 		else:
# 			upperbound = lowerbound + e[i][1]
# 			break
# 	# construct bitstring for bitmask
# 	bitstring = ""
# 	intsize = 32
# 	if vectorsize > 30:
# 		intsize = 64
# 	if not write:
# 		for i in range(0,intsize):
# 			if i >= lowerbound and i < upperbound:
# 				bitstring += "1"
# 			else:
# 				bitstring += "0"
# 		# convert to hex bitmask
# 		bitmask = hex(int(bitstring, 2))
# 	else:
# 		bitmask = "(x << " + str(intsize - upperbound) + ")"
# 	return str(bitmask)

def get_compact_thread_condition(level):
	"""For the given tree level, produce a compact condition for the thread IDs (for use in the FETCH function)"""
	global vectortree_level_ids, vectortree_node_thread
	L = vectortree_level_ids.get(level, [])
	nodes = []
	for n in L:
		nodes.append(vectortree_node_thread[n])
	output = ""
	# sort the nodes
	nodes = sorted(nodes)
	# group the node ids together in ranges
	prev = -2
	rg = [-1,-1]
	R = []
	for i in range(0, len(nodes)):
		if nodes[i]-1 != prev:
			if prev != -2:
				rg[1] = prev
				R.append(rg)
			rg = [nodes[i],-1]
		prev = nodes[i]
	# prepare the final range
	rg[1] = nodes[len(nodes)-1]
	R.append(rg)
	# now construct the condition
	first = True
	for p in R:
		if not first:
			output += " || "
		else:
			first = False
		if p[0] == p[1]:
			output += "gid == " + str(p[0])
		elif p[0] == 0:
			output += "gid <= " + str(p[1])
		else:
			output += "(gid >= " + str(p[0]) + " && gid <= " + str(p[1]) + ")"
	return output

def get_compact_leaf_thread_condition(level):
	"""For the given tree level, produce a compact condition for the thread IDs, restricted to leaf nodes (for use in the FETCH function)"""
	global vectortree_level_ids, vectortree
	L = vectortree_level_ids.get(level, [])
	nodes = []
	for n in L:
		if vectortree[n] == []:
			nodes.append(vectortree_node_thread[n])
	output = ""
	# sort the nodes
	nodes = sorted(nodes)
	# group the node ids together in ranges
	prev = -2
	rg = [-1,-1]
	R = []
	for i in range(0, len(nodes)):
		if nodes[i]-1 != prev or i == len(nodes)-1:
			if prev != -2:
				rg[1] = prev
				R.append(rg)
			rg = [nodes[i],-1]
			if nodes[i]-1 != prev and i == len(nodes)-1:
				rg[1] = nodes[i]
				R.append(rg)
		prev = nodes[i]
	# now construct the condition
	first = True
	for p in R:
		if not first:
			output += " || "
		else:
			first = False
		if p[0] == p[1]:
			output += "gid == " + str(p[0])
		elif p[0] == 0:
			output += "gid <= " + str(p[1])
		else:
			output += "(gid >= " + str(p[0]) + " && gid <= " + str(p[1]) + ")"
	return output

def get_compact_nonleaf_thread_condition(level):
	"""For the given tree level, produce a compact condition for the thread IDs, restricted to non-leaf nodes (for use in the FETCH function)"""
	global vectortree_level_ids, vectortree
	L = vectortree_level_ids.get(level, [])
	nodes = []
	for n in L:
		if vectortree[n] != []:
			nodes.append(vectortree_node_thread[n])
	output = ""
	# sort the nodes
	nodes = sorted(nodes)
	# group the node ids together in ranges
	prev = -2
	rg = [-1,-1]
	R = []
	for i in range(0, len(nodes)):
		if nodes[i]-1 != prev or i == len(nodes)-1:
			if prev != -2:
				rg[1] = prev
				R.append(rg)
			rg = [nodes[i],-1]
			if nodes[i]-1 != prev and i == len(nodes)-1:
				rg[1] = nodes[i]
				R.append(rg)
		prev = nodes[i]
	# now construct the condition
	first = True
	for p in R:
		if not first:
			output += " || "
		else:
			first = False
		if p[0] == p[1]:
			output += "gid == " + str(p[0])
		elif p[0] == 0:
			output += "gid <= " + str(p[1])
		else:
			output += "(gid >= " + str(p[0]) + " && gid <= " + str(p[1]) + ")"
	return output

def get_startbit(o):
	"""Return starting bit of object o tuple ((object, index) pair) in its corresponding part of the statevector"""
	global vectorstructure, vectorsize

def vectorstructure_part_size(t):
	"""Return the size (in bits) of the given vectorstructure part"""
	size = 0
	for (s,i) in t:
		size += i
	return size

def vectorstructure_to_string(D):
	"""Convert vectorstructure to string. D is dictionary to look up strings for individual elements."""
	global vectorstructure, compact_hash_table

	vs = ""
	tfirst = True
	nr_of_parts = len(vectorstructure)
	for t in vectorstructure:
		if not tfirst:
			vs += ",\n// "
		else:
			tfirst = False
		if nr_of_parts == 1 or ((not compact_hash_table) and vectorstructure_part_size(t) > (64-2-nr_bits_address_root())):
			vs += "[ two bits reserved, "
		elif nr_of_parts > 1 and (compact_hash_table and vectorstructure_part_size(t) > nr_bits_address_internal()):
			vs += "[ one bit reserved, "
		else:
			vs += "Combined with a non-leaf vector tree node: [ "
		first = True
		newline_counter = 0
		for (s,i) in t:
			newline_counter += 1
			if not first:
				vs += ", "
			else:
				first = False
			if newline_counter == 5:
				vs += "\n//   "
				newline_counter = 0
			vs += D[s] + " " + s + ": " + str(i) + " bit(s)"
		vs += " ]"
	return vs

def cuda_xor_lr(a, nrbits, ic):
	"""Produce CUDA code to compute an XOR + left & right bit shift operation on a variable node1."""
	result = "node1 = xor_shft2_" + str(nrbits) + "(node1, " + str(a) + ", " + str(nrbits-a) + ");"
	return result

def cuda_xor_lr_inv(a, nrbits, ic):
	"""Produce CUDA code to compute the inverse of an XOR + left & right bit shift operation on a variable node1."""
	result = "node2 = node1;\n" + indentspace(ic)
	i = nrbits-a
	while True:
		result += "node2 = (node1 ^ rshft(node2, " + str(nrbits-a) + "));\n" + indentspace(ic)
		if i >= nrbits:
			break
		i += nrbits-a
	result += "node1 = node2;\n" + indentspace(ic)
	i = a
	while True:
		result += "node1 = (node2 ^ lshft_" + str(nrbits) + "(node1, " + str(a) + "));\n" + indentspace(ic)
		if i >= nrbits:
			break
		i += a
	result += "node1 &= " + hexa(pow2(nrbits)-1) + ";"
	return result

def cuda_xor_r_inv(a, nrbits, ic):
	"""Produce CUDA code for the xor_rshft_inv function for 64-bits."""
	result = "node2 = node1;\n" + indentspace(ic)
	i = a
	while True:
		result += "node2 = (node1 ^ rshft(node2, " + str(a) + "));\n" + indentspace(ic)
		if i >= nrbits:
			break
		i += a
	result += "node1 = node2;"
	return result

# def cuda_xor_lr_inv(a, nrbits, ic):
# 	"""Produce CUDA code to compute the inverse of an XOR + left & right bit shift operation on a variable node1."""
# 	result = ""
# 	i = a
# 	while i < nrbits:
# 		result += "node1 ^= (node1 << " + str(i) + ");\n" + indentspace(ic)
# 		i += i
# 	if nrbits < 64:
# 		result += "node1 &= " + str(hex(int(math.pow(2,nrbits))-1)) + "L;\n" + indentspace(ic)
# 	i = nrbits-a
# 	while i < nrbits:
# 		result += "node1 ^= (node1 >> " + str(i) + ");"
# 		if i+i < nrbits:
# 			result += "\n" + indentspace(ic)
# 		i += i
# 	return result

def cuda_xor_r3(a, b, c, ic):
	"""Produce CUDA code to compute an XOR + 3 right bit shifts on a variable node1."""
	result = "node1 ^= rshft(node1, " + str(a) + ");\n" + indentspace(ic)
	result += "node1 ^= rshft(node1, " + str(b) + ");\n" + indentspace(ic)
	result += "node1 ^= rshft(node1, " + str(c) + ");"
	return result

def cuda_xor_r3_inv(a, b, c, nrbits, ic):
	"""Produce CUDA code to compute an XOR + 3 right bit shifts on a variable node1."""
	result = "node2 = node1;\n" + indentspace(ic)
	i = a
	while True:
		result += "node2 = (node1 ^ rshft(node2, " + str(a) + "));\n" + indentspace(ic)
		if i >= nrbits:
			break
		i += a
	result += "node1 = node2;\n" + indentspace(ic)
	i = b
	while True:
		result += "node1 = (node2 ^ rshft(node1, " + str(b) + "));\n" + indentspace(ic)
		if i >= nrbits:
			break
		i += b
	result += "node2 = node1;\n" + indentspace(ic)
	i = c
	while True:
		result += "node2 = (node1 ^ rshft(node2, " + str(c) + "));\n" + indentspace(ic)
		if i >= nrbits:
			break
		i += c
	result += "node1 = node2;"
	return result

# def cuda_xor_r3_inv(a, b, c, nrbits, ic):
# 	"""Produce CUDA code to compute the inverse of an XOR + 3 right bit shifts on a variable node1."""
# 	result = ""
# 	i = c
# 	while i < nrbits:
# 		result += "node1 ^= (node1 >> " + str(i) + ");\n" + indentspace(ic)
# 		i += i
# 	i = b
# 	while i < nrbits:
# 		result += "node1 ^= (node1 >> " + str(i) + ");\n" + indentspace(ic)
# 		i += i
# 	i = a
# 	while i < nrbits:
# 		result += "node1 ^= (node1 >> " + str(i) + ");"
# 		if i+i < nrbits:
# 			result += "\n" + indentspace(ic)
# 		i += i
# 	return result

def cuda_xor_r(a, ic):
	"""Produce CUDA code to compute an XOR + 1 right bit shift on a variable node1."""
	result = "node1 ^= rshft(node1, " + str(a) + ");"
	return result

def cudarecsizeguard(s, D, o):
	"""Given a ReceiveSignal statement s, return a guard referring to the size of the buffer, in case the connected channel is asynchronous. o is Object owning s."""
	global connected_channel
	c = connected_channel[(o, s.target)]
	if c.synctype == 'async':
		sizevar = D.get((c, "_size"))
		if sizevar != None:
			if c.size > 1:
				return sizevar[0] + "_" + str(sizevar[1]) + " > 0"
			else:
				return sizevar[0] + "_" + str(sizevar[1])
	else:
		return ""

def cudaguard(s,D,o):
	"""Returns the guard of the given statement s. D is a dictionary mapping variable refs to variable names. o is Object owning s."""
	global connected_channel, signalsize, signalnr, smnames, smname_to_object, alphabet, syncactions, state_id
	if statement_is_actionref(s):
		a = getlabel(s)
		if a in syncactions.get(o.type,set([])):
			sm = s.parent.parent
			guard = ""
			# obtain list of state machines on which this action depends (with which it synchronises)
			S = []
			for m in smnames:
				(o2,sm2) = smname_to_object[m]
				if o == o2 and sm != sm2:
					if a in alphabet[sm2]:
						if guard != "":
							guard += " && "
						guard += "get_target_" + sm2.name + "_" + a + "((statetype) " + D[(sm2,"src")][0] + "_" + str(D[(sm2,"src")][1]) + ", (statetype) NO_STATE) != NO_STATE"
			return guard
	elif s.__class__.__name__ == "Expression":
		return getinstruction(s, o, D)
	elif s.__class__.__name__ == "Composite":
		if s.guard:
			return getinstruction(s.guard, o, D)
		else:
			return ""
	elif s.__class__.__name__ == "SendSignal":
		c = connected_channel[(o, s.target)]
		if c.synctype == 'async':
			sizevar = D.get((c, "_size"))
			if sizevar != None:
				if c.size > 1:
					return sizevar[0] + "_" + str(sizevar[1]) + " < " + str(c.size)
				else:
					return "!" + sizevar[0] + "_" + str(sizevar[1])
		# else:
		# 	guard = ""
		# 	for (o2,sm2) in get_syncrec_sms(o, c, s.signal):
		# 		if guard != "":
		# 			guard += " && "
		# 		guard += "get_target_" + o2.name + "_" + sm2.name + "_" + s.signal + "((statetype) " + D[(c, o2.name + "'" + sm2.name)][0] + "_" + str(D[(c, o2.name + "'" + sm2.name)][1]) + ", (statetype) -1) != -1"
		# 	return guard
	elif s.__class__.__name__ == "ReceiveSignal":
		c = connected_channel[(o, s.target)]
		guard = ""
		if c.synctype == 'async':
			# check message signal
			if signalsize[c] > 0:
				guard += D[(c,0)][0] + "_" + str(D[(c,0)][1]) + " == " + str(signalnr[(c,s.signal)])
			return guard
		else:
			# check current state of state machine
			guard += D[(c,"state")][0] + "_" + str(D[(c,"state")][1]) + " == " + str(state_id[s.parent.source])
			return guard
	return ""

def is_vectorpart(pid):
	"""Return whether the given vectortree node with id pid is a vectorpart or not"""
	global vectorstructure, vectortree
	size = len(vectorstructure)
	if len(vectortree[pid]) == 0:
		return True
	if len(vectortree[pid]) == 1 and vectorpart_is_combined_with_nonleaf_node(len(vectorstructure)-1):
		return True
	return False

def vectorpart_id(pid):
	"""For the given vectortree node with id pid, return its vectorpart id"""
	"""Precondition: node pid is a vectorpart"""
	global vectorpart_id_dict
	# # determine number of non-leaf nodes
	# nfnr = len(vectorstructure)-1
	# if vectorpart_is_combined_with_nonleaf_node(len(vectorstructure)-1):
	# 	nfnr -= 1
	# n = len(vectorstructure)
	# i = pid - (n-1)
	# if i < 0:
	# 	i = n-1
	# return i
	return vectorpart_id_dict.get(pid, -1)

def vectornode_id(pid):
	"""For the given vectorpart with id pid, return its vectortree node id"""
	"""Precondition: part pid is a valid vectorpart"""
	global vectornode_id_dict
	return vectornode_id_dict.get(pid, -1)

def is_non_leaf(pid):
	"""Return whether the given vectortree node with id pid is a non-leaf node"""
	global vectortree
	return len(vectortree[pid]) > 0

def indentspace(i):
	"""Provide a string consisting of i tabs"""
	result = ""
	for j in range(0,i):
		result += "\t"
	return result

def variabledefault(s, i):
	"""Return default value for given variable, possibly with index i"""
	if s.defvalue != None:
		return int(s.defvalue)
	elif s.defvalues != []:
		# we assume an index i has been given
		return int(s.defvalues[i])
	elif s.type.base == 'Integer' or s.type.base == 'Byte':
		return 0
	elif s.type.base == 'Boolean':
		return 1

def update_parts(name, value, vectorparts):
	"""Update the given vectorparts with the name, value pair"""
	global model, vectorelem_in_structure_map

	p = vectorelem_in_structure_map[name]
	if len(p) == 2:
		value = (value << p[1][1])
		vectorparts[p[1][0]] |= value
	else:
		# lowest bits
		f = 0
		for j in range(0, p[2][2]):
			f |= (1 << j)
		lovalue = value & f
		lovalue = (lovalue << p[2][1])
		vectorparts[p[2][0]] |= lovalue
		# highest bits
		value = (value >> p[2][2])
		value = (value << p[1][1])
		vectorparts[p[1][0]] |= value

def reset_left_pointer(node):
	"""Reset the left pointer in the given vectortree node."""
	"""Precondition: the given node has a left pointer."""
	resetvalue = ((pow2(31)-1) << 31)
	node |= resetvalue
	return node

def reset_right_pointer(node):
	"""Reset the right pointer in the given vectortree node."""
	"""Precondition: the given node has a right pointer."""
	resetvalue = pow2(31)-1
	node |= resetvalue
	return node

def set_left_cache_pointer(nodes_cachepointers, value):
	"""Set the left cache pointer"""
	hivalue = (value << 15)
	return ((nodes_cachepointers & 0xC0007FFF) | hivalue)

def set_right_cache_pointer(nodes_cachepointers, value):
	"""Set the right cache pointer"""
	lovalue = value
	return ((nodes_cachepointers & 0xFFFF8000) | lovalue)

def cudastore_initial_vector():
	"""Construct CUDA code to put the initial state in the global hash table"""
	global vectorstructure, model, state_id, vectortree, vectorsize

	# create the initial vectorparts
	vectorparts = [0 for i in range(0, len(vectorstructure))]
	for o in model.objects:
		for v in o.type.variables:
			name = scopename(v,None,o)
			if v.type.size > 1:
				for i in range(0,v.type.size):
					iname = name + "[" + str(i) + "]"
					update_parts(iname, variabledefault(v,i), vectorparts)
			else:
				update_parts(name, variabledefault(v,None), vectorparts)
		for sm in o.type.statemachines:
			update_parts(o.name + "'" + sm.name, state_id[sm.initialstate], vectorparts)
			for v in sm.variables:
				name = scopename(v,None,o)
				if v.type.size > 1:
					for i in range(0,v.type.size):
						iname = name + "[" + str(i) + "]"
						update_parts(iname, variabledefault(v,i), vectorparts)
				else:
					update_parts(name, variabledefault(v,None), vectorparts)
	# next create a list of nodes representing the initial state
	nrnodes = 2*len(vectorstructure) - 1
	# compensate for a final vector part integrated into a non-leaf node
	if vectorpart_is_combined_with_nonleaf_node(len(vectorstructure)-1):
		nrnodes -= 1
	nodes = [0 for i in range(0, nrnodes)]
	nodes_cachepointers = [0 for i in range(0, nrnodes)]
	# explore the vectortree structure to construct the list of nodes
	Open = [0]
	while Open != []:
		current = Open.pop(0)
		children = vectortree[current]
		if is_vectorpart(current):
			# set p_cpointers to NEW LEAF value
			p_cpointers = "CACHE_POINTERS_NEW_LEAF"
			part_id = vectorpart_id(current)
			p = vectorparts[part_id]
			if vectorpart_is_combined_with_nonleaf_node(part_id):
				p_cpointers = 0xBFFFFFFF
				p = reset_left_pointer(p)
				p_cpointers = set_left_cache_pointer(p_cpointers, children[0])
			nodes[current] = p
			nodes_cachepointers[current] = p_cpointers
		else:
			p_cpointers = 0xBFFFFFFF
			p = 0
			p = reset_left_pointer(p)
			p_cpointers = set_left_cache_pointer(p_cpointers, children[0])
			if len(children) == 2:
				p = reset_right_pointer(p)
				p_cpointers = set_right_cache_pointer(p_cpointers, children[1])
			nodes[current] = p
			nodes_cachepointers[current] = p_cpointers
		Open += children
	# set the root node to new root
	if vectorsize <= 62:
		if vectorsize < 31:
			nodes[0] |= 0xC0000000
		else:
			nodes[0] |= 0xC000000000000000
	else:
		# mark the root node as being root.
		nodes[0] = (nodes[0] & 0x3FFFFFFFFFFFFFFF) | 0x4000000000000000;
	# construct code
	output = "\t\tif (GLOBAL_THREAD_ID < " + str(nrnodes) + ") {\n"
	output += "\t\t\tswitch (GLOBAL_THREAD_ID) {\n"
	for i in range(0, nrnodes):
		output += "\t\t\t\tcase " + str(i) + ":\n"
		if vectorsize <= 62:
			addr = str(i)
		else:
			addr = "(" + str(i) + "*3)"
		output += "\t\t\t\t\tshared[CACHEOFFSET+" + addr + "] = "
		if vectorsize > 62:
			output += "get_left(" + hexa(nodes[i]) + ");\n"
			output += "\t\t\t\t\tshared[CACHEOFFSET+" + addr + "+1] = get_right(" + hexa(nodes[i]) + ");\n"
		else:
			output += hexa(nodes[i]) + ";\n"
		if vectorsize > 62:
			output += "\t\t\t\t\tshared[CACHEOFFSET+" + addr + "+2] = "
			if isinstance(nodes_cachepointers[i], str):
				output += nodes_cachepointers[i]
			else:
				output += hexa(nodes_cachepointers[i])
			output += ";\n"
		output += "\t\t\t\t\tbreak;\n"
	output += "\t\t\t}\n"
	output += "\t\t}"
	return output

def cudastore_new_vectortree_nodes(nodes_done, nav, pointer_cnt, W, s, o, D, indent):
	"""Construct CUDA code to produce and store new vectortree nodes. nodes_done is a list containing node ids that have been processed before. nav is a list of nodes still to be processed.
	   W is a dictionary defining for all vectorparts which values need to be written to it."""
	global vectortree, vectortree_T, connected_channel, dynamic_access_arrays, state_id, vectorsize, array_in_structure_map
	ic = indent
	output = ""
	if nav != []:
		(p,f) = nav.pop(0)
		# p is the ID of a vector node, f indicates for a vector part whether or not it also needs to update a (left) pointer (true), and for a non-part
		# whether it needs to update one pointer (the left) (false) or two pointers (true).
		if is_vectorpart(p):
			refs = W.get(p,[])
			if refs != []:
				if vectorsize > 62:
					output += "get_vectortree_node(&part1, &part_cachepointers, node_index, " + str(p) + ");\n" + indentspace(ic)
				output += "// Store new values.\n" + indentspace(ic)
				output += "part2 = part1;\n" + indentspace(ic)
				for (v,i,isnotfirstpart) in refs:
					if i != '*':
						# handle the case of the target state(s) of SMs
						if v.__class__.__name__ == "Object":
							if i.__class__.__name__ == "StateMachine":
								target = "target"
								if s.__class__.__name__ == "ReceiveSignal":
									c = connected_channel[(o,s.target)]
									if c.synctype == 'sync':
										# synchronous communication over a channel. Not one, but two SMs change state.
										if v == o and i == s.parent.parent:
											# the current SM changes to a state stored in the thread buffer variables.
											target = str(state_id[s.parent.target])
								elif statement_is_actionref(s):
									if i != s.parent.parent:
										target = D[(i,"tgt")][0] + "_" + str(D[(i,"tgt")][1])
								if not isnotfirstpart:
									output += "set_left_" + v.name + "_" + i.name + "(&part2, (statetype) "
									output += target
									output += ");\n" + indentspace(ic)
								else:
									output += "set_right_" + v.name + "_" + i.name + "(&part2, (statetype) "
									output += target
									output += ");\n" + indentspace(ic)
								# else:
								# 	# v,i is an (Object,Statemachine) pair that needs to change state
								# 	(vname,offset) = D.get((i,"tgt"), (v.name, None))
								# 	result = vname
								# 	if offset != None:
								# 		result += "_" + str(offset)
								# 	if not isnotfirstpart:
								# 		output += "set_left_" + v.name + "_" + i.name + "(&part2, " + result + ");\n" + indentspace(ic)
								# 	else:
								# 		output += "set_right_" + v.name + "_" + i.name + "(&part2, " + result + ");\n" + indentspace(ic)
						# handle the case of communication over a channel
						elif v.__class__.__name__ == "Channel":
							# if v.synctype == 'sync':
							# 	# i is an (Object,Statemachine) pair that needs to change state
							# 	(vname,offset) = D.get((v,"state"), (v.name, None))
							# 	result = vname
							# 	if offset != None:
							# 		result += "_" + str(offset)
							# 	if not isnotfirstpart:
							# 		output += "set_left_" + i[0].name + "_" + i[1].name + "(&part2, " + result + ");\n" + indentspace(ic)
							# 	else:
							# 		output += "set_right_" + i[0].name + "_" + i[1].name + "(&part2, " + result + ");\n" + indentspace(ic)
							# else:

							# Channel must be asynchronous
							(vname,offset) = D.get((v,"_size"), (v.name, None))
							result = vname
							if offset != None:
								result += "_" + str(offset)
							if not isnotfirstpart:
								output += "set_left_" + v.name + "_size(&part2, " + result + ");\n" + indentspace(ic)
							else:
								output += "set_right_" + v.name + "_size(&part2, " + result + ");\n" + indentspace(ic)
						else:
							result = ""
							# look for a match on s.ref.ref in D
							for r in D.keys():
								if not isinstance(r, tuple):
									if v.name == r.name:
										vname = D[r][0]
										offset = D[r][1]
										break
							offsetcnt = -1
							if offset != None:
								offsetcnt = offset
							if i != None:
								indexresult = getinstruction(i, o, D)
								if has_dynamic_indexing(v, v.name, s.parent, o):
									allocs = get_buffer_arrayindex_allocs(s.parent, o)
									size = allocs[(o,v)]
									tpsize = str(gettypesize(v.type))
									result += "A_LD_" + str(size) + "("
									for j in range(0,size):
										result += "idx_" + str(j+D[v][2]) + ", "
									for j in range(0,size):
										result += "buf" + tpsize + "_" + str(j+offset) + ", "
									result += indexresult + ")"
								else:
									indexdict = get_constant_indices(v, v.name, s.parent, o)
									offsetcnt += indexdict[indexresult]
							if result == '':
								result += vname
								if offsetcnt != -1:
									result += "_" + str(offsetcnt)
							# code to update the current vector part
							set_methodname = scopename(v,None,o)
							set_methodname = set_methodname.replace("'","_")
							if i != None:
								set_methodname += "_" + getinstruction(i, o, {})
							if not isnotfirstpart:
								output += "set_left_" + set_methodname + "(&part2, " + result + ");\n" + indentspace(ic)
							else:
								output += "set_right_" + set_methodname + "(&part2, " + result + ");\n" + indentspace(ic)
					else:
						# dynamic indexing into an array or channel buffer. use a special set function for this.
						if v.__class__.__name__ != "Channel":
							output += "// Write array buffer content.\n" + indentspace(ic)
							set_methodname = scopename(v,None,o)
							set_methodname = set_methodname.replace("'","_")
							e = D.get(v, (v.name, None))
							vname = e[0]
							offset = e[1]
							idx_offset = 0
							if len(e) > 2:
								idx_offset = e[2]
							#output += "set_" + set_methodname + "(&part2, idx_" + o.name + "_" + v.name + ", " + vname + ", " + str(offset) + ", " + str(vectorpart_id(p)) + ");\n" + indentspace(ic)
							part_id = str(vectorpart_id(p))
							output += "if (" + part_id + " >= " + str(dynamic_access_arrays[v][1]) + " && " + part_id + " <= " + str(dynamic_access_arrays[v][2]) + ") {\n" + indentspace(ic)
							allocs = get_buffer_arrayindex_allocs(s.parent, o)
							indentnew = "\t"
							# range over the number of elements that may be stored in the buffer for the current array
							for i in range(0,allocs[(o,v)]):
								output += indentnew + "if (idx_" + str(i+idx_offset) + " != EMPTY_INDEX) {\n" + indentspace(ic)
								indentnew = indentnew + "\t"
								output += indentnew + "if (array_element_is_in_vectorpart_" + str(dynamic_access_arrays[v][0]).replace("'","_") + "(idx_" + str(i+idx_offset) + ", " + part_id + ")) {\n" + indentspace(ic)
								indentnew = indentnew + "\t"
								output += indentnew + "if (is_left_vectorpart_for_array_element_" + str(dynamic_access_arrays[v][0]).replace("'","_") + "(idx_" + str(i+idx_offset) + ", " + part_id + ")) {\n" + indentspace(ic)
								indentnew = indentnew + "\t"
								output += indentnew + "set_left_" + str(dynamic_access_arrays[v][0]).replace("'","_") + "(&part2, idx_" + str(i+idx_offset) + ", " + vname + "_" + str(offset+i) + ", " + part_id + ");\n" + indentspace(ic)
								indentnew = indentnew[:-1]
								output += indentnew + "}\n" + indentspace(ic)
								arrayname = scopename(v,None,o)
								arraystructure = array_in_structure_map[arrayname]
								if vectorpart_id(p)-1 >= 0 and vectorpart_id(p)-1 <= arraystructure[0]+(len(arraystructure)-2) and arraystructure[vectorpart_id(p)][3] != 0:
									output += indentnew + "else {\n" + indentspace(ic)
									indentnew = indentnew + "\t"
									output += indentnew + "set_right_" + str(dynamic_access_arrays[v][0]).replace("'","_") + "(&part2, idx_" + str(i+idx_offset) + ", " + vname + "_" + str(offset+i) + ", " + part_id + ");\n" + indentspace(ic)
									indentnew = indentnew[:-1]
									output += indentnew + "}\n" + indentspace(ic)
							for i in range(0,allocs[(o,v)]):
								indentnew = indentnew[:-1]
								output += indentnew + "}\n" + indentspace(ic)
								indentnew = indentnew[:-1]
								output += indentnew + "}\n" + indentspace(ic)
							output += "}\n" + indentspace(ic)
						else:
							c = connected_channel[(o, s.target)]
							if s.__class__.__name__ == "SendSignal":
								first = 0
								if signalsize[v] == 0:
									first = 1
								size = D[(v,"_size")]
								sizeref = size[0] + "_" + str(size[1])
								for i in range(first,len(v.type)+1):
									var = D[(v,i)]
									varref = var[0] + "_" + str(var[1])
									output += "set_buffer_tail_element_" + v.name + "_" + str(i) + "(&part2, "
									if c.size > 1:
										output += sizeref + "-1, "
									else:
										output += "0, "
									output += varref + ", " + str(vectorpart_id(p)) + ");\n" + indentspace(ic)
							else:
								# ReceiveSignal: shift the buffer content one position towards the head
								size = D[(v,"_size")]
								sizeref = size[0] + "_" + str(size[1])
								output += "shift_buffer_tail_elements_" + v.name + "(node_index, &part2, "
								if c.size == 1:
									output += "1, "
								else:
									output += sizeref + "+1, "
								output += str(vectorpart_id(p)) + ");\n" + indentspace(ic)
			if is_non_leaf(p):
				# node is also a non-leaf in the vectortree. update pointers.
				if f:
					ic += 1
					output += "if (bufaddr_" + str(pointer_cnt) + " != EMPTY_HASH_POINTER) {\n" + indentspace(ic)
					if refs == []:
						output += "get_vectortree_node(&part1, &part_cachepointers, node_index, " + str(p) + ");\n" + indentspace(ic)
						output += "part2 = part1;\n" + indentspace(ic)
					ic += 1
					output += "if (mode == TO_CACHE) {\n" + indentspace(ic)
					output += "set_left_cache_pointer(&part_cachepointers, bufaddr_" + str(pointer_cnt) + ");\n" + indentspace(ic)
					output += "reset_left_in_vectortree_node(&part2);\n" + indentspace(ic)
					ic -= 1
					output += "}\n" + indentspace(ic)
					ic += 1
					output += "else {\n" + indentspace(ic)
					ic -= 1
					output += "set_left_in_vectortree_node(&part2, bufaddr_" + str(pointer_cnt) + ");\n" + indentspace(ic)
					if refs != []:
						ic -= 1
					output += "}\n" + indentspace(ic)
					# output += "}\n" + indentspace(ic)
					if refs != []:
						output += "}\n" + indentspace(ic)
						ic += 1
						output += "if (part2 != part1) {\n" + indentspace(ic)
			elif refs != []:
				ic += 1
				output += "if (part2 != part1) {\n" + indentspace(ic)
			if is_non_leaf(p) or refs != []:
				output += "// This part has been altered. Store it and remember address of new part.\n" + indentspace(ic)
				ic += 1
				output += "if (mode == TO_CACHE) {\n" + indentspace(ic)
				if vectorsize > 62:
					if p == 0:
						output += "mark_root(&part2);\n" + indentspace(ic)
					if is_non_leaf(p):
						output += "mark_cached_node_new_nonleaf(&part_cachepointers);\n" + indentspace(ic)
					else:
						output += "part_cachepointers = CACHE_POINTERS_NEW_LEAF;\n" + indentspace(ic)
					output += "bufaddr_" + str(pointer_cnt) + " = STOREINCACHE(part2, part_cachepointers"
					if is_non_leaf(p):
						output += ", false"
					else:
						output += ", true"
					output += ");\n" + indentspace(ic)
				else:
					output += "part2 = mark_new(part2);\n" + indentspace(ic)
					output += "bufaddr_" + str(pointer_cnt) + " = STOREINCACHE(part2);\n" + indentspace(ic)
				ic += 1
				output += "if (bufaddr_" + str(pointer_cnt) + " == CACHE_FULL) {\n" + indentspace(ic)
				output += "// Construct the vector again, and store it directly in the global hash table.\n" + indentspace(ic)
				output += "mode = TO_GLOBAL;\n" + indentspace(ic)
				ic -= 1
				output += "continue;\n" + indentspace(ic)
				ic -= 1
				output += "}\n" + indentspace(ic)
				output += "}\n"  + indentspace(ic)
				ic += 1
				output += "else {\n" + indentspace(ic)
				output += "// Store the node directly in the global hash table.\n" + indentspace(ic)
				if vectorsize > 62:
					#if p == 0 and not compact_hash_table:
					#	output += "mark_root(&part2);\n" + indentspace(ic)
					if compact_hash_table:
						if p == 0:
							output += "bufaddr_" + str(pointer_cnt) + " = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));\n" + indentspace(ic)
						else:
							output += "bufaddr_" + str(pointer_cnt) + " = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, false, (ITERATIONS == d_kernel_iters-1));\n" + indentspace(ic)
					else:
						output += "bufaddr_" + str(pointer_cnt) + " = FINDORPUT_SINGLE(d_q, part2, d_newstate_flags, EMPTY_CACHE_POINTER, (ITERATIONS == d_kernel_iters-1));\n" + indentspace(ic)						
				else:
					output += "part2 = mark_new(part2);\n" + indentspace(ic)
					output += "bufaddr_" + str(pointer_cnt) + " = FINDORPUT_SINGLE(d_q, part2, d_newstate_flags, EMPTY_CACHE_POINTER, (ITERATIONS == d_kernel_iters-1));\n" + indentspace(ic)						
				ic += 1
				output += "if (bufaddr_" + str(pointer_cnt) + " == HASHTABLE_FULL) {\n" + indentspace(ic)
				output += "// Hash table is considered full. Report this back.\n" + indentspace(ic)
				ic -= 1
				output += "return HASH_TABLE_FULL;\n" + indentspace(ic)
				ic -= 1
				output += "}\n" + indentspace(ic)
				ic -= 1
				output += "}\n" + indentspace(ic)
				output += "}\n"  + indentspace(ic)
				ic += 1
				if nav != []:
					output += "else {\n" + indentspace(ic)
					ic -= 1
					output += "bufaddr_" + str(pointer_cnt) + " = EMPTY_HASH_POINTER;\n" + indentspace(ic)
					output += "}\n"  + indentspace(ic)
			if nav != [] and not is_non_leaf(nav[0][0]):
				pointer_cnt += 1
			output += cudastore_new_vectortree_nodes(nodes_done + [p], nav, pointer_cnt, W, s, o, D, ic)
		elif is_non_leaf(p):
			if not f:
				ic += 1
				output += "if (bufaddr_" + str(pointer_cnt) + " != EMPTY_HASH_POINTER) {\n" + indentspace(ic)
				output += "get_vectortree_node(&part1, &part_cachepointers, node_index, " + str(p) + ");\n" + indentspace(ic)
				output += "part2 = part1;\n" + indentspace(ic)
				children = vectortree[p]
				ic += 1
				output += "if (mode == TO_CACHE) {\n" + indentspace(ic)
				if nodes_done[len(nodes_done)-1] == children[0]:
					output += "set_left_cache_pointer(&part_cachepointers, bufaddr_" + str(pointer_cnt) + ");\n" + indentspace(ic)
					output += "reset_left_in_vectortree_node(&part2);\n" + indentspace(ic)
				else:
					output += "set_right_cache_pointer(&part_cachepointers, bufaddr_" + str(pointer_cnt) + ");\n" + indentspace(ic)
					output += "reset_right_in_vectortree_node(&part2);\n" + indentspace(ic)
			else:
				ic += 1
				output += "if (bufaddr_" + str(pointer_cnt-1) + " != EMPTY_HASH_POINTER || bufaddr_" + str(pointer_cnt) + " != EMPTY_HASH_POINTER) {\n" + indentspace(ic)
				output += "get_vectortree_node(&part1, &part_cachepointers, node_index, " + str(p) + ");\n" + indentspace(ic)
				output += "part2 = part1;\n" + indentspace(ic)
				ic += 1
				output += "if (bufaddr_" + str(pointer_cnt-1) + " != EMPTY_HASH_POINTER) {\n" + indentspace(ic)
				ic += 1
				output += "if (mode == TO_CACHE) {\n" + indentspace(ic)
				output += "set_left_cache_pointer(&part_cachepointers, bufaddr_" + str(pointer_cnt-1) + ");\n" + indentspace(ic)
				ic -= 1
				output += "reset_left_in_vectortree_node(&part2);\n" + indentspace(ic)
				output += "}\n" + indentspace(ic)
				ic += 1
				output += "else {\n" + indentspace(ic)
				ic -= 1
				output += "set_left_in_vectortree_node(&part2, bufaddr_" + str(pointer_cnt-1) + ");\n" + indentspace(ic)
				ic -= 1
				output += "}\n" + indentspace(ic)
				output += "}\n" + indentspace(ic)
				ic += 1
				output += "if (bufaddr_" + str(pointer_cnt) + " != EMPTY_HASH_POINTER) {\n" + indentspace(ic)
				ic += 1
				output += "if (mode == TO_CACHE) {\n" + indentspace(ic)
				output += "set_right_cache_pointer(&part_cachepointers, bufaddr_" + str(pointer_cnt) + ");\n" + indentspace(ic)
				ic -= 1
				output += "reset_right_in_vectortree_node(&part2);\n" + indentspace(ic)
				output += "}\n" + indentspace(ic)
				ic += 1
				output += "else {\n" + indentspace(ic)
				ic -= 1
				output += "set_right_in_vectortree_node(&part2, bufaddr_" + str(pointer_cnt) + ");\n" + indentspace(ic)
				ic -= 1
				output += "}\n" + indentspace(ic)
				output += "}\n" + indentspace(ic)
				ic += 1
				output += "if (mode == TO_CACHE) {\n" + indentspace(ic)
				pointer_cnt -= 1
				output += "// This part has been altered. Store it and remember address of new part.\n" + indentspace(ic)
			if p == 0:
				output += "mark_root(&part2);\n" + indentspace(ic)
			output += "mark_cached_node_new_nonleaf(&part_cachepointers);\n" + indentspace(ic)
			output += "bufaddr_" + str(pointer_cnt) + " = STOREINCACHE(part2, part_cachepointers, false);\n" + indentspace(ic)
			ic += 1
			output += "if (bufaddr_" + str(pointer_cnt) + " == CACHE_FULL) {\n" + indentspace(ic)
			output += "// Construct the vector again, and store it directly in the global hash table.\n" + indentspace(ic)
			output += "mode = TO_GLOBAL;\n" + indentspace(ic)
			ic -= 1
			output += "continue;\n" + indentspace(ic)
			ic -= 1
			output += "}\n" + indentspace(ic)
			output += "}\n" + indentspace(ic)
			ic += 1
			output += "else {\n" + indentspace(ic)
			if not f:
				if nodes_done[len(nodes_done)-1] == children[0]:
					output += "set_left_in_vectortree_node(&part2, bufaddr_" + str(pointer_cnt) + ");\n" + indentspace(ic)
				else:
					output += "set_right_in_vectortree_node(&part2, bufaddr_" + str(pointer_cnt) + ");\n" + indentspace(ic)
			#if p == 0 and not compact_hash_table:
			#	output += "mark_root(&part2);\n" + indentspace(ic)
			if compact_hash_table:
				if p == 0:
					output += "bufaddr_" + str(pointer_cnt) + " = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, true, (ITERATIONS == d_kernel_iters-1));\n" + indentspace(ic)
				else:
					output += "bufaddr_" + str(pointer_cnt) + " = FINDORPUT_SINGLE(d_q, d_q_i, d_dummy, part2, d_newstate_flags, EMPTY_CACHE_POINTER, false, (ITERATIONS == d_kernel_iters-1));\n" + indentspace(ic)
			else:
				output += "bufaddr_" + str(pointer_cnt) + " = FINDORPUT_SINGLE(d_q, part2, d_newstate_flags, EMPTY_CACHE_POINTER, (ITERATIONS == d_kernel_iters-1));\n" + indentspace(ic)						
			ic += 1
			output += "if (bufaddr_" + str(pointer_cnt) + " == HASHTABLE_FULL) {\n" + indentspace(ic)
			output += "// Hash table is considered full. Report this back.\n" + indentspace(ic)
			ic -= 1
			output += "return HASH_TABLE_FULL;\n" + indentspace(ic)
			ic -= 1
			output += "}\n" + indentspace(ic)
			ic -= 1
			output += "}\n" + indentspace(ic)
			output += "}\n" + indentspace(ic)
			if nav != []:
				ic += 1
				output += "else {\n" + indentspace(ic)
				ic -= 1
				output += "bufaddr_" + str(pointer_cnt) + " = EMPTY_HASH_POINTER;\n" + indentspace(ic)
				output += "}\n"  + indentspace(ic)
			if nav != [] and not is_non_leaf(nav[0][0]):
				pointer_cnt += 1
			output += cudastore_new_vectortree_nodes(nodes_done + [p], nav, pointer_cnt, W, s, o, D, ic)
	# remove trailing tabs
	while output[-1:] == '\t':
		output = output[:-1]
	return output

def cudastore_new_vector(s,indent,o,D, sender_o='', sender_sm='', lossy=False):
	"""Return CUDA code to store new vector resulting from executing statement s. D is a dictionary mapping variable refs to variable names. o is Object owning s.
	optional arguments involve an Object and Statemachine of a (synchronous) sender SM that needs to change state, lossy indicates lossy communication."""
	global connected_channel, scopename, vectorstructure, vectortree, vectortree_T, vectorelem_in_structure_map, vectornode_id_dict

	indentspace = ""
	for i in range(0,indent):
		indentspace += "\t"
	output = ""

	W = get_write_vectorparts_info(s,o,sender_o=sender_o,sender_sm=sender_sm,lossy=lossy)
	if len(W) != 0:
		output += "mode = (mode == STORED ? TO_CACHE : TO_GLOBAL);\n" + indentspace
		output += "while (mode != STORED && mode != GLOBAL_STORED) {\n" + indentspace + "\t"
		output += "// Store new state vector in the cache or the global hash table.\n" + indentspace + "\t"
		# obtain list of nodes in the tree to update
		L = list(W.keys())
		L = sorted(L)
		Lnew = []
		Wnew = {}
		for v in L:
			nid = vectornode_id_dict[v]
			Lnew.append(nid)
			Wnew[nid] = W[v]
		L = Lnew
		# print("update" + str(L))
		# n = len(vectorstructure)-1
		# Lnew = []
		# Wnew = {}
		# extranode = -1
		# for v in L:
		# 	if vectorpart_is_combined_with_nonleaf_node(v):
		# 		extranode = n-1
		# 		Wnew[n-1] = W[v]
		# 	else:
		# 		Lnew.append(v+n)
		# 		Wnew[v+n] = W[v]
		# L = sorted(Lnew)
		# if extranode != -1:
		# 	L.append(extranode)
		# explore vectortree to construct list of nodes to be updated
		navcounters = {}
		waiting = set([])
		Lcount = deepcopy(L)
		seen = set([])
		while Lcount != []:
			v = Lcount.pop(0)
			if v != 0:
				nextnode = vectortree_T[v]
				C = navcounters.get(nextnode, 0)
				navcounters[nextnode] = C+1
				if nextnode not in seen:
					Lcount.append(nextnode)
				seen.add(nextnode)
		nav = []
		if len(L) > 0:
			added = set([])
			current = L.pop(0)
			if current not in added:
				nav.append((current,False))
				added.add(current)
			while True:
				parent = vectortree_T.get(current)
				if parent == None:
					break
				navcounters[parent] -= 1
				if navcounters[parent] == 0:
					current = parent
					if is_non_leaf(parent) and is_vectorpart(parent):
						# this is the tail vectorpart, and it is integrated into a non-leaf node
						if parent not in added:
							nav.append((parent,True))
							added.add(parent)
					elif parent in waiting:
						if parent not in added:
							nav.append((parent,True))
							added.add(parent)
						waiting.remove(parent)
					else:
						if parent not in added:
							nav.append((parent,False))
							added.add(parent)
					if current == 0:
						break
				else:
					waiting.add(parent)
					current = L.pop(0)
					if current not in added:
						nav.append((current,False))
						added.add(current)
		# list of processed nodes
		nodes_done = []
		# process the nav list of nodes
		output += cudastore_new_vectortree_nodes(nodes_done, nav, 0, Wnew, s, o, D, indent+1) + indentspace + "\t"
		output += "mode = (mode == TO_CACHE ? STORED : GLOBAL_STORED);\n" + indentspace
		output += "}\n"
	return output

def cudastatement(s,indent,o,D,sender_o='',sender_sm='',senderparams=[]):
	"""Translates the unguarded part of SLCO statement s to CUDA code. indent indicates how much every line needs to be indented. o is Object owning s. D is a dictionary mapping variable refs to variable names
	'sender_o', 'sender_sm' and 'senderparams' provide extra arguments. In particular, when s is a ReceiveSignal statement connected to a synchronous channel, the parameters of a corresponding SendSignal statement must be provided via these parameters."""
	global connected_channel, syncactions, alphabet, smnames, smname_to_object, state_id

	indentspace = ""
	for i in range(0,indent):
		indentspace += "\t"
	output = ""
	if statement_is_actionref(s):
		output += "target = " + str(state_id[s.parent.target]) + ";\n" + indentspace
		sm = s.parent.parent
		a = getlabel(s)
		if a in syncactions.get(o.type,set([])):
			# construct list of statemachines with which synchronisation is required
			SMs = []
			for m in smnames:
				(o2,sm2) = smname_to_object[m]
				if o == o2 and sm != sm2:
					if a in alphabet[sm2]:
						SMs.append(sm2)
			for m in SMs:
				output += D[(m,"tgt")][0] + "_" + str(D[(m,"tgt")][1]) + " = NO_STATE;\n" + indentspace
			output += "uint8_t i = 1;\n" + indentspace
			output += "while (i > 0) {\n" + indentspace
			output += "\tswitch (i) {\n" + indentspace
			for i in range(1,len(SMs)+1):
				output += "\t\tcase " + str(i) + ":\n" + indentspace
				output += "\t\t\t" + D[(SMs[i-1],"tgt")][0] + "_" + str(D[(SMs[i-1],"tgt")][1]) + " = get_target_" + SMs[i-1].name + "_" + a + "((statetype) " + D[(SMs[i-1],"src")][0] + "_" + str(D[(SMs[i-1],"src")][1]) + ", (statetype) " + D[(SMs[i-1],"tgt")][0] + "_" + str(D[(SMs[i-1],"tgt")][1]) + ");\n" + indentspace
				output += "\t\t\tif (((statetype) " + D[(SMs[i-1],"tgt")][0] + "_" + str(D[(SMs[i-1],"tgt")][1]) + ") != NO_STATE) {\n" + indentspace
				output += "\t\t\t\ti++;\n" + indentspace
				output += "\t\t\t}\n" + indentspace
				output += "\t\t\telse {\n" + indentspace
				output += "\t\t\t\t" + D[(SMs[i-1],"tgt")][0] + "_" + str(D[(SMs[i-1],"tgt")][1]) + " = NO_STATE;\n" + indentspace
				output += "\t\t\t\ti--;\n" + indentspace
				output += "\t\t\t}\n" + indentspace
				output += "\t\t\tbreak;\n" + indentspace
			output += "\t\tdefault:\n" + indentspace
			output += "\t\t\tbreak;\n" + indentspace
			output += "\t}\n" + indentspace
			output += "\tif (i == " + str(len(SMs)+1) + ") {\n" + indentspace + "\t\t"
			output += cudastore_new_vector(s,indent+2,o,D) + indentspace + "\t\t"
			output += "i--;\n" + indentspace
			output += "\t}\n" + indentspace
			output += "}\n"
		else:
			output += cudastore_new_vector(s,indent,o,D)
	elif s.__class__.__name__ == "Assignment":
		output += "target = " + str(state_id[s.parent.target]) + ";\n" + indentspace
		output += getinstruction(s, o, D) + ";\n" + indentspace
		output += cudastore_new_vector(s,indent,o,D)
	elif s.__class__.__name__ == "Composite":
		output += "target = " + str(state_id[s.parent.target]) + ";"
		for e in s.assignments:
			output += "\n" + indentspace
			# if e.left.index != None:
			# 	if has_dynamic_indexing(e.left.var, e.left.var.name, e.parent.parent, o):
			# 		# add line to obtain index offset
			# 		output += "add_idx(idx_" + o.name + "_" + e.left.var.name + ", " + getinstruction(e.left.index, o, D) + ");\n" + indentspace
			output += getinstruction(e, o, D) + ";"
		output += "\n" + indentspace
		output += cudastore_new_vector(s,indent,o,D)
	elif s.__class__.__name__ == "SendSignal":
		c = connected_channel[(o, s.target)]
		if c.synctype == 'async':
			output += "target = " + str(state_id[s.parent.target]) + ";\n" + indentspace
			if connected_channel[(o, s.target)].losstype == 'lossy':
				output += "// Handle lossy case in which message is lost.\n" + indentspace
				# pass a 'lossy' keyword to restrict new vector writing to updating the target state
				output += cudastore_new_vector(s,indent,o,D, lossy=True) + indentspace
				output += "// Handle lossy case in which message is not lost.\n" + indentspace
			for i in range(0,len(s.params)):
				p = s.params[i]
				if i > 0:
					output += "\n" + indentspace
				output += D[(c, i+1)][0] + "_" + str(D[(c, i+1)][1]) + " = " + getinstruction(p, o, D) + ";\n" + indentspace
				# increment buffer size counter
				if c.size > 1:
					output += D[(c, "_size")][0] + "_" + str(D[(c, "_size")][1]) + "++;\n" + indentspace
				else:
					output += D[(c, "_size")][0] + "_" + str(D[(c, "_size")][1]) + " = true;\n" + indentspace
			output += cudastore_new_vector(s,indent,o,D)
		else:
			# synchronous communication. iterate over all matching ReceiveSignal statements
			# original mapping for the SendSignal statement
			allocs = get_buffer_allocs(object_trans_to_be_processed_by_sm_thread(s.parent.source,o))
			M = map_variables_on_buffer(s.parent, o, allocs)
			for (o2,t2) in get_syncrec_sm_trans(o, c, s.signal):
				# the additional mapping for the ReceiveSignal statement
				Mr = map_variables_on_buffer(t2, o2, allocs, prevM=M)
				# fetch the required data
				for st2 in t2.statements:
					output += "// Consider synchronous communication with object " + o2.name + ", state machine " + t2.parent.name + ",\n" + indentspace
					output += "// transition " + t2.source.name + " --{ " + getlabel(st2) + " }--> " + t2.target.name + "\n" + indentspace
					output += "\n" + indentspace
					output += cudafetchdata(st2, indent, o2, Mr, True, False)
					output += "if (" + cudaguard(st2, Mr, o2) + ") {\n" + indentspace + "\t"
					paramlist = []
					# put object and parameters in paramlist
					paramlist.append(o)
					for p in s.params:
						paramlist.append(p)
					output += cudastatement(st2, indent + 1, o2, Mr, sender_o=o, sender_sm=s.parent.parent, senderparams=paramlist) + indentspace
					output += "}\n"
		# output += "// Set target state.\n" + indentspace
		# output += "target = " + str(state_id[(s.parent.parent,s.parent.target)]) + ";\n" + indentspace

		# else:
		# 	for (o2,sm2) in get_syncrec_sms(o, c, s.signal):
		# 		output += D[(c,"state")][0] + "_" + str(D[(c,"state")][1]) + " = get_target_" + o2.name + "_" + sm2.name + "_" + s.signal + "((statetype) " + D[(c, o2.name + "'" + sm2.name)][0] + "_" + str(D[(c, o2.name + "'" + sm2.name)][1]) + ", (statetype) -1);\n" + indentspace
		# 		output += "while (" + D[(c,"state")][0] + "_" + str(D[(c,"state")][1]) + " != -1) {\n" + indentspace
		# 		output += "\t" + cudastore_new_vector(s,indent+1,o,D,[(o2,sm2)]) + indentspace
		# 		output += "\t" + D[(c,"state")][0] + "_" + str(D[(c,"state")][1]) + " = get_target_" + o2.name + "_" + sm2.name + "_" + s.signal + "((statetype) " + D[(c, o2.name + "'" + sm2.name)][0] + "_" + str(D[(c, o2.name + "'" + sm2.name)][1]) + ", (statetype) " + D[(c,"state")][0] + "_" + str(D[(c,"state")][1]) + ");\n" + indentspace
		# 		output += "}\n"
	elif s.__class__.__name__ == "ReceiveSignal":
		c = connected_channel[(o, s.target)]
		if c.synctype == 'async':
			# decrement buffer size counter
			if c.size > 1:
				output += D[(c, "_size")][0] + "_" + str(D[(c, "_size")][1]) + "--;\n" + indentspace
			else:
				output += D[(c, "_size")][0] + "_" + str(D[(c, "_size")][1]) + " = false;\n" + indentspace
		else:
			# store elements from a matching SendSignal statement in the parameter variables of s.
			# the SendSignal parameters are given in the extra, named, parameters
			output += cudafetchdata(s, indent, o, D, False, False)
			output += "// Statement computation.\n" + indentspace
			output2 = ''
			for i in range(0,len(s.params)):
				e = D.get(s.params[i].var, (s.params[i].var.name, None))
				vname = e[0]
				offset = e[1]
				idx_offset = 0
				if len(e) > 2:
					idx_offset = e[2]
				offsetcnt = -1
				if offset != None:
					offsetcnt = offset
				if s.params[i].index != None:
					indexresult = getinstruction(s.params[i].index, o, D)
					if has_dynamic_indexing(s.params[i].var, s.params[i].var.name, s.parent, o):
						allocs = get_buffer_arrayindex_allocs(s.parent, o)
						size = allocs[(o,s.params[i].var)]
						tpsize = str(gettypesize(s.params[i].var.type))
						output2 += "A_STR_" + str(size) + "("
						for j in range(0,size):
							output2 += "&idx_" + str(j+idx_offset) + ", "
						for j in range(0,size):
							output2 += "&buf" + tpsize + "_" + str(j+offset) + ", "
						output2 += indexresult + ", (" + cudatype(s.params[i].var.type, True) + ") " + getinstruction(senderparams[i+1],sender_o,D) + ");\n" + indentspace
					else:
						indexdict = get_constant_indices(s.params[i].var, s.params[i].var.name, s.parent, o)
						offsetcnt += indexdict[indexresult]
				if output2 == '':
					output2 += vname
					if offsetcnt != -1:
						output2 += "_" + str(offsetcnt)
					output2 += " = "
					if s.params[i].var.type.base == 'Byte':
						output2 += "(elem_chartype) ("
					output2 += getinstruction(senderparams[i+1],sender_o,D)
					if s.params[i].var.type.base == 'Byte':
						output2 += ")"
					output2 += ";\n" + indentspace
			output += output2
		if s.guard != None:
			output += "if (" + getinstruction(s.guard,o,D) + ") {\n" + indentspace
			output += "\ttarget = " + str(state_id[s.parent.target]) + ";\n" + indentspace
			output += "\t" + cudastore_new_vector(s,indent+1,o,D, sender_o=sender_o, sender_sm=sender_sm) + indentspace
			output += "}\n"
		else:
			output += "target = " + str(state_id[s.parent.target]) + ";\n" + indentspace
			output += cudastore_new_vector(s,indent,o,D, sender_o=sender_o, sender_sm=sender_sm)
	elif s.__class__.__name__ == "Expression":
		output += "target = " + str(state_id[s.parent.target]) + ";\n" + indentspace
		output += cudastore_new_vector(s,indent,o,D)
	return output

def cudafetchdata(s, indent, o, D, unguarded, resetfetched):
	"""Produce CUDA code to fetch the necessary data from the shared memory cache into thread register variables for the processing
	of statement s in Object o. M is a mapping from SLCO variables to register buffer variables. unguarded indicates whether or not
	only values for unguarded variables must be fetched. If not, then all guarded variables are fetched, assuming that the unguarded
	ones have already been fetched. If resetfetched, set the fetched variables to initial values."""
	global connected_channel, signalsize, fetched, vectorsize
	indentspace = ""
	for i in range(0,indent):
		indentspace += "\t"
	output = ""
	t = s.parent
	sm = t.parent
	# The vectorparts for values of unguarded variables
	L = transition_read_varrefs(t,o,True)
	# The 'old' VP list, containing vectorparts for fetching of values of unguarded variables (only relevant when fetching values of guarded variables)
	VPold = []
	# a special case: if s is a ReceiveSignal statement connected to an asynchronous channel, the vectorpart(s) containing the associated buffer size have already been retrieved.
	if not resetfetched and s.__class__.__name__ == "ReceiveSignal":
		c = connected_channel[(o,s.target)]
		if c.synctype == 'async':
			VPtmp = get_vectorparts([(c,"_size")],o)
			fetched[0] = VPtmp[0]
			if len(VPtmp) > 1:
				fetched[1] = VPtmp[1]
	Ln = []
	for v in L:
		if no_dynamic_indexing(v,o):
			Ln.append(v)
	VP = get_vectorparts(Ln,o)
	if not unguarded:
		# The remaining vectorparts for guarded variables
		L = transition_read_varrefs(t,o,False)
		VPrem = []
		Ln = []
		for v in L:
			if no_dynamic_indexing(v,o):
				Ln.append(v)
		VPold = VP
		VP = get_remaining_vectorparts(Ln,o,VP)
	for st in t.statements:
		# buffer allocations for dynamic array accessing
		allocs = get_buffer_arrayindex_allocs(t, o)
		if unguarded:
			first = True
			for v in get_vars(statement_varrefs(st,o,sm,0)):
				if has_dynamic_indexing(v, v.name, t, o):
					if first:
						output += "// Reset storage of array indices.\n" + indentspace
						first = False
					for i in range(0,allocs[(o,v)]):
						output += "idx_" + str(i+D[v][2]) + " = EMPTY_INDEX;\n" + indentspace
		if len(VP) > 0:
			if unguarded:
				output += "// Fetch values of unguarded variables.\n" + indentspace
			else:
				output += "// Fetch values of guarded variables.\n" + indentspace
		# are we considering the first vector part accessed by this statement?
		vpfirst = True
		# are we accessing the first variable from the current vector part for this statement?
		first = True
		if resetfetched:
			fetched = {0: -1, 1: -1}
		for i in range(0,len(VP)):
			R = transition_read_varrefs(t,o,False)
			for (v,j) in R:
				success = (v.__class__.__name__ == "Channel")
				if not success:
					success = no_dynamic_indexing((v,j),o)
				if success and not unguarded:
					success = vectorparts_not_covered((v,j),o,VPold)
				if success:
					name = scopename(v,j,o)
					p = vectorelem_in_structure_map[name]
					if p[1][0] == VP[i]:
						success = (len(p) == 2)
						if not success and i < len(VP)-1:
							success = (p[len(p)-1][0] == VP[i+1])
						if success:
							if vpfirst and first:
								if VP[i] != fetched[0] and VP[i] != fetched[1]:
									fetched[0] = VP[i]
									if vectorsize > 62:
										output += "part1 = get_vectorpart(node_index, " + str(VP[i]) + ");\n" + indentspace
								elif VP[i] == fetched[1]:
									fetched[0] = fetched[1]
									output += "part1 = part2\n" + indentspace
							if (not vpfirst) and VP[i] != fetched[0]:
								# we have already fetched part VP[i] as the second part (part2) in the previous iteration.
								fetched[0] = fetched[1]
								output += "part1 = part2;\n" + indentspace
							if i+1 < len(VP):
								if VP[i+1] != fetched[0] and VP[i+1] != fetched[1]:
									fetched[1] = VP[i+1]
									output += "part2 = get_vectorpart(node_index, " + str(VP[i+1]) + ");\n" + indentspace
								elif VP[i+1] == fetched[0]:
									fetched[1] = fetched[0]
									output += "part2 = part1;\n" + indentspace
							elif fetched[1] != fetched[0]:
								output += "part2 = part1;\n" + indentspace
							if not (v.__class__.__name__ == "Channel" and RepresentsInt(j)):
								if v.__class__.__name__ != "Channel" and v.__class__.__name__ != "StateMachine":
									output += "get_" + scopename(v,None,o).replace("'","_")
									if j != None:
										output += "_" + getinstruction(j,o,{})
									output += "(&" + D[v][0] + "_"
									if j == None:
										output += str(D[v][1])
									else:
										output += str(int(D[v][1]) + int(getinstruction(j,o,{})))
									output += ", part1, part2);\n" + indentspace
								else:
									if v.__class__.__name__ == "Channel" and "'" in str(j):
										# i is an (Object,Statemachine) pair. this refers to the current state variable of a state machine.
										output += "get_" + scopename(v,j,o).replace("[","_").replace("]","").replace("'","_") + "(&" + D[(v,"state")][0] + "_" + str(D[(v,"state")][1]) + ", part1, part2);\n" + indentspace
									else:
										output += "get_" + scopename(v,j,o).replace("[","_").replace("]","").replace("'","_") + "((statetype *) &" + D[(v,j)][0] + "_" + str(D[(v,j)][1]) + ", part1, part2);\n" + indentspace
							else:
								if no_dynamic_indexing((s.params[j-1].var, s.params[j-1].index), o):
									output += "get_" + scopename(c, j, o).replace("[","_").replace("]","").replace("'","_")
									output += "(&" + D[s.params[j-1].var][0] + "_"
									success = (j == None)
									if not success:
										success = (s.params[j-1].index == None)
									if success:
										output += str(D[s.params[j-1].var][1])
									else:
										coffset += int(D[s.params[j-1].var][1])
										if s.params[j-1].index != None:
											coffset += int(getinstruction(s.params[j-1].index,o,{}))
										output += str(coffset)
									output += ", part1, part2);\n" + indentspace
								else:
									size = allocs[(o,s.params[j-1].var)]
									output += "\t// Fetch and store value.\n" + indentspace
									output += "\tget_" + scopename(v,None,o).replace("'","_") + "(node_index, &buf" + str(gettypesize(s.params[j-1].var.type)) + "_" + str(D[s.params[j-1].var][1]+size-1) + ", " + getinstruction(s.params[j-1].index,o,D) + ");\n" + indentspace
									output += "\tA_STR_" + str(size) + "("
									for k in range(0,size):
										output += "&idx_" + str(k+D[s.params[j-1].var][2]) + ", "
									for k in range(0,size):
										output += "&buf" + str(gettypesize(s.params[j-1].var.type)) + "_" + str(D[s.params[j-1].var][1]+k) + ", "
									output += "(array_indextype) " + getinstruction(s.params[j-1].index,o,D) + ", buf" + str(gettypesize(s.params[j-1].var.type)) + "_" + str(D[s.params[j-1].var][1]+size-1) + ");\n" + indentspace
									output += "}\n" + indentspace
							first = False
			vpfirst = False
		first = True
		# DYNAMIC ARRAY ACCESSING HAS BEEN REMOVED. SHOULD BE DONE WHEN NEEDED IN THE EXECUTION OF STATEMENT BLOCK ITSELF.
		# for (v,j) in transition_sorted_dynamic_read_varrefs(t,o,unguarded):
		# 	if first:
		# 		output += "// Fetch values of variables involving dynamic array indexing.\n" + indentspace
		# 	print(D)
		# 	output += "// Check for presence of index in buffer indices.\n" + indentspace
		# 	size = allocs[(o,v)]
		# 	output += "if (!A_IEX_" + str(size) + "("
		# 	for i in range(0,size):
		# 		output += "idx_" + str(i+D[v][2]) + ", "
		# 	output += getinstruction(j,o,D) + ")) {\n" + indentspace
		# 	output += "\t// Fetch and store value.\n" + indentspace
		# 	output += "\tget_" + scopename(v,None,o).replace("'","_") + "(node_index, &buf" + str(gettypesize(v.type)) + "_" + str(D[v][1]+size-1) + ", " + getinstruction(j,o,D) + ");\n" + indentspace
		# 	output += "\tA_STR_" + str(size) + "("
		# 	for i in range(0,size):
		# 		output += "&idx_" + str(i+D[v][2]) + ", "
		# 	for i in range(0,size):
		# 		output += "&buf" + str(gettypesize(v.type)) + "_" + str(D[v][1]+i) + ", "
		# 	output += "(array_indextype) " + getinstruction(j,o,D) + ", buf" + str(gettypesize(v.type)) + "_" + str(D[v][1]+size-1) + ");\n" + indentspace
		# 	output += "}\n" + indentspace
		# 	first = False
	return output

def getinstruction(s, o, D):
	"""Get the CUDA instruction for the given statement s. o is Object owning s, D is a dictionary mapping variable refs to variable names."""
	global model, connected_channel

	result = ''
	if s.__class__.__name__ == "Assignment":
		e = D.get(s.left.var, (s.left.var.name, None))
		vname = e[0]
		offset = e[1]
		idx_offset = 0
		if len(e) > 2:
			idx_offset = e[2]
		rightexp = getinstruction(s.right, o, D)
		offsetcnt = -1
		if offset != None:
			offsetcnt = offset
		if s.left.index != None:
			indexresult = getinstruction(s.left.index, o, D)
			# find a suitable object (object is irrelevant for outcome)
			t = s.parent
			while t.__class__.__name__ != "Transition":
				t = t.parent
			for o in model.objects:
				if o.type == s.parent.parent:
					break
			if has_dynamic_indexing(s.left.var, s.left.var.name, t, o):
				allocs = get_buffer_arrayindex_allocs(t, o)
				size = allocs[(o,s.left.var)]
				tpsize = str(gettypesize(s.left.var.type))
				result += "A_STR_" + str(size) + "("
				for j in range(0,size):
					result += "&idx_" + str(j+idx_offset) + ", "
				for j in range(0,size):
					result += "&buf" + tpsize + "_" + str(j+offset) + ", "
				result += "(array_indextype) " + indexresult + ", (" + cudatype(s.left.var.type, True) + ") " + rightexp + ")"
			else:
				indexdict = get_constant_indices(s.left.var, s.left.var.name, t, o)
				offsetcnt += indexdict[indexresult]
		if result == '':
			result += vname
			if offsetcnt != -1:
				result += "_" + str(offsetcnt)
			result += " = "
			if s.left.var.type.base == 'Byte':
				result += "(elem_chartype) ("
			result += rightexp
			if s.left.var.type.base == 'Byte':
				result += ")"
	elif s.__class__.__name__ == "Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2" or s.__class__.__name__ == "ExprPrec1":
		if s.op != '':
			result += getinstruction(s.left, o, D) + " " + operator(s.op) + " " + getinstruction(s.right, o, D)
		else:
			result += getinstruction(s.left, o, D)
	elif s.__class__.__name__ == "Primary":
		result += sign(s.sign)
		result2 = ''
		if s.sign == 'not':
			result += "("
		if s.value != None:
			newvalue = s.value
			result += str(newvalue).lower()
		elif s.ref != None:
			# look for a match on s.ref.ref in D
			vname = ''
			offset = 0
			idx_offset = 0
			for r in D.keys():
				if not isinstance(r, tuple):
					if s.ref.ref == r.name:
						vname = D[r][0]
						offset = D[r][1]
						if len(D[r]) > 2:
							idx_offset = D[r][2]
						break
			offsetcnt = -1
			if offset != None:
				offsetcnt = offset
			if s.ref.index != None:
				indexresult = getinstruction(s.ref.index, o, D)
				# find the transition executing s by moving up the parse tree
				t = s.parent
				while t.__class__.__name__ != "Transition":
					t = t.parent
				if has_dynamic_indexing(s.ref, s.ref.ref, t, o):
					allocs = get_buffer_arrayindex_allocs(t, o)
					v = ''
					for (o1,v1) in allocs.keys():
						if o == o1 and v1.name == s.ref.ref:
							v = v1
							break
					size = allocs[(o,v)]
					tpsize = str(gettypesize(v.type))
					result2 += scopename(v,None,o).replace("[","_").replace("]","").replace("'","_") + "(node_index, "
					for j in range(0,size):
						result2 += "&idx_" + str(j+idx_offset) + ", "
					for j in range(0,size):
						result2 += "&buf" + tpsize + "_" + str(j+offset) + ", "
					result2 += indexresult + ")"
				else:
					indexdict = get_constant_indices(s.ref, s.ref.ref, t, o)
					offsetcnt += indexdict[indexresult]
			if result2 == '':
				result2 += vname
				if offsetcnt != -1:
					result2 += "_" + str(offsetcnt)
			# vname = s.ref.ref
			# offset = None
			# # look for an exact (name, index) match in Drec
			# if s.ref.index != None:
			# 	index_str = getinstruction(s.ref.index, o, {}, {})
			# else:
			# 	index_str = s.ref.index
			# (vname,offset) = Drec.get((s.ref.ref, index_str), (s.ref.ref, None))
			# if vname == s.ref.ref:
			# 	# look for a match on s.ref.ref in D
			# 	for r in D.keys():
			# 		if not isinstance(r, tuple):
			# 			if s.ref.ref == r.name:
			# 				vname = D[r][0]
			# 				offset = D[r][1]
			# 				break
			# result += vname
			# if offset != None or s.ref.index != None:
			# 	result += "_"
			# if offset != None:
			# 	result += str(offset)
			# 	if s.ref.index != None:
			# 		result += " + "
			# if s.ref.index != None:
			# 	indexresult = getinstruction(s.ref.index, o, D, Drec)
			# 	# find parent transition object
			# 	t = s.parent
			# 	while t.__class__.__name__ != "Transition":
			# 		t = t.parent
			# 	if has_dynamic_indexing(r, t, o):
			# 		result += "idx(idx_" + s.ref.ref + ", " + indexresult + ")"
			# 	else:
			# 		indexdict = get_constant_indices(r, t, o)
			# 		result += str(indexdict[indexresult])
		else:
			result2 += '(' + getinstruction(s.body, o, D) + ')'
		if s.sign == 'not':
			result2 += ")"
		result += result2
	# elif s.__class__.__name__ == "VariableRef":
	# 	(vname,offset) = D.get(s.var, (s.var.name, None))
	# 	result += vname
	# 	if offset != None or s.index != None:
	# 		result += "_"
	# 	if offset != None:
	# 		result += str(offset)
	# 		if s.index != None:
	# 			result += " + "
	# 	if s.index != None:
	# 		indexresult = getinstruction(s.index, o, D, Drec)
	# 		if (RepresentsInt(indexresult)):
	# 			result += indexresult
	# 		else:
	# 			result += "idx(idx_" + o.name + "_" + s.var.name + ", " + indexresult + ")"
	return result

def get_smart_fetching_vectorparts_bitmask(s, o):
	"""Construct a 32-bit bitmask indicating which vectorparts are relevant for processing the outgoing transitions of state s. o is Object owning s.
	Each bit in the bitmask indicates whether or not the corresponding vectorpart is needed for processing.
	Precondition: a vectortree has at most 32 parts."""
	parts = get_all_relevant_vectorparts_for_state(s, o)
	bitmask = 0
	for pid in parts:
		bitmask += (1 << (31 - pid))
	return hex(bitmask)

def transition_read_varrefs(t, o, only_unguarded):
	"""Return a set of variable refs appearing in block of transition t"""
	R = set([])
	sm = t.parent
	for st in t.statements:
		R |= statement_read_varrefs(st, o, sm, only_unguarded)
	filtered_R = set([])
	Vseen = set([])
	for (v,i) in R:
		i_str = getinstruction(i, o, {})
		if (v,i_str) not in Vseen:
			Vseen.add((v,i_str))
			filtered_R.add((v,i))
	return filtered_R

def transition_sorted_dynamic_read_varrefs(t, o, only_unguarded):
	"""Return a sorted list of variable refs involving dynamic indexing, in block of transition t. They are sorted on dependency."""
	R = set([])
	sm = t.parent
	for st in t.statements:
		Rtmp = statement_read_varrefs(st, o, sm, True)
		if not only_unguarded:
			Rtmp_all = statement_read_varrefs(st, o, sm, False)
			Rtmp = Rtmp_all - Rtmp
		R |= Rtmp
	# only keep refs with dynamic indexing
	R2 = set([])
	for (v,i) in R:
		if v.__class__.__name__ != "Channel" and v.__class__.__name__ != "StateMachine":
			i_str = getinstruction(i, o, {})
			if i != None and not RepresentsInt(i_str):
				R2.add((v,i))
	L = []
	R = set([])
	while R2 != set([]):
		for (v,i) in R2:
			R3 = statement_read_varrefs(i, o, sm, only_unguarded)
			add = True
			for (v2,i2) in R3:
				if (v2,i2) in R2:
					add = False
					break
			if add:
				L.append((v,i))
			else:
				R.add((v,i))
		R2 = R
		R = set([])
	return L

def is_dynamic_address(i, o):
	"""Return whether the given expression represents dynamic addressing, if used as an index for an array access."""
	if i == None:
		return False
	i_str = getinstruction(i, o, {})
	return not RepresentsInt(i_str)

def statement_varrefs(s, o, sm, subid):
	"""Return a set of variable refs appearing in statement s. o is the object owning s, and sm is the state machine owning s."""
	"""A variable ref is a triple (v, i, subid), with v an Object, i additional info (such as an array index), and subid a number.
	subid is in particular needed to distinguish the same dynamic array accesses in separate substatements of a Composite statement."""
	global connected_channel, signalsize, actions, syncactions, alphabet, smnames, smname_to_object, scopename
	R = set([])
	if s.__class__.__name__ == "Assignment":
		if is_dynamic_address(s.left.index, o):
			# we store this access with the subid identifier, since syntactically equivalent dynamic array accesses in different
			# substatements of a Composite should be treated as different accesses.
			R.add((s.left.var, s.left.index, subid))
		else:
			R.add((s.left.var, s.left.index, 0))
		if s.left.index != None:
			R |= statement_varrefs(s.left.index, o, sm, subid)
		R |= statement_varrefs(s.right, o, sm, subid)
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			R |= statement_varrefs(s.guard, o, sm, subid)
		for i in range(0,len(s.assignments)):
			R |= statement_varrefs(s.assignments[i], o, sm, subid+i)
	elif s.__class__.__name__ == "SendSignal":
		c = connected_channel[(o, s.target)]
		if c.synctype == 'async':
			# add channel with index '_size' to represent 'size' variable
			R.add((c, "_size", 0))
			# add channel with integer indices to represent items of a Message
			if signalsize[c] > 0:
				R.add((c, 0, 0))
			for i in range(1,len(c.type)+1):
				R.add((c, i, 0))
		#else:
			# for i in range(0,len(c.type)):
			# 	R.add((c, i))
			# add variable for temporary state storage
			# R.add((c, "state"))
			# add (Object,StateMachine) pairs for state machines that can potentially receive messages
			# for (o2,sm2) in get_syncrec_sms(o, c, s.signal):
			# 	R.add((c, scopename(sm2,None,o2)))
		for p in s.params:
			R |= statement_varrefs(p, o, sm, 0)
	elif s.__class__.__name__ == "ReceiveSignal":
		c = connected_channel[(o, s.target)]
		if c.synctype == 'async':
			# add channel with index "_size" to represent 'size' variable
			R.add((c, "_size", 0))
			# # add channel with integer indices to represent items of a Message
			if signalsize[c] > 0:
				R.add((c, 0, 0))
			# for i in range(1,len(c.type)+1):
			# 	R.add((c, "[" + str(i) + "][0]"))
			# Messagerefs = set([])
			# for p in s.params:
			# 	Messagerefs.add((p.var.name, getinstruction(p.index, o, {}, {})))
			# 	if p.index != None:
			# 		R |= statement_varrefs(p.index, o, sm)
			# Rguard = statement_varrefs(s.guard, o, sm)
			# for (v,j) in Rguard:
			# 	if (v.name, getinstruction(j, o, {}, {})) not in Messagerefs:
			# 		R.add((v,j))
		else:
			# add (Channel, (Object,Statemachine)) pair to represent current state variable
			R.add((c,(o,sm), 0))
		for p in s.params:
			R.add((p.var, p.index, 0))
			if p.index != None:
				R |= statement_varrefs(p.index, o, sm, subid)
		R |= statement_varrefs(s.guard, o, sm, subid)
		# else:
		# 	for p in s.params:
		# 		R |= statement_varrefs(p, sm)
		# 	R |= statement_varrefs(s.guard, sm)
	elif s.__class__.__name__ == "Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2" or s.__class__.__name__ == "ExprPrec1":
		R |= statement_varrefs(s.left, o, sm, subid)
		if s.op != '':
			R |= statement_varrefs(s.right, o, sm, subid)
	elif s.__class__.__name__ == "Primary":
		if s.ref != None:
			if s.ref.ref in actions:
				# we have a user-defined action
				if s.ref.ref in syncactions.get(o.type,set([])):
					# add references to other state machines in Object o that have the action in their alphabet
					# (i.e., we add a reference to their current state variable)
					for sname in smnames:
						(o2,sm2) = smname_to_object[sname]
						if o == o2 and sm != sm2:
								if s.ref.ref in alphabet[sm2]:
									# add two variables: src to store the current state of sm2, tgt to store target state of a transition
									R.add((sm2,"src", 0))
									R.add((sm2,"tgt", 0))
			if s.ref.index != None:
				R |= statement_varrefs(s.ref.index, o, sm, subid)
			# obtain suitable object matching name s.ref.ref
			for v1 in sm_variables(sm):
				if v1.name == s.ref.ref:
					if is_dynamic_address(s.ref.index, o):
						R.add((v1, s.ref.index, subid))
					else:
						R.add((v1, s.ref.index, 0))						
					break
		if s.body != None:
			R |= statement_varrefs(s.body, o, sm, subid)
	elif s.__class__.__name__ == "VariableRef":
		if s.index != None:
			R |= statement_varrefs(s.index, o, sm, subid)
		if is_dynamic_address(s.index, o):
			R.add((s.var, s.index, subid))
		else:
			R.add((s.var, s.index, 0))			
	return R

def statement_write_varrefs(s, o):
	"""Return a list of variable refs to which the statement is writing. o is Object owning s."""
	global connected_channel

	W = set([])
	if s.__class__.__name__ == "Assignment":
		W.add((s.left.var, s.left.index))
	elif s.__class__.__name__ == "Composite":
		for i in range(0,len(s.assignments)):
			W.add((s.assignments[i].left.var, s.assignments[i].left.index))
	elif s.__class__.__name__ == "ReceiveSignal":
		for i in range(0,len(s.params)):
			W.add((s.params[i].var, s.params[i].index))
	# elif s.__class__.__name__ == "SendSignal":
	# 	c = connected_channel[(o, s.target)]
	# 	if c.synctype == 'async':
	# 		if signalsize[c] > 0:
	# 			W.add((c, "[0][0]"))
	# 		for i in range(1,len(c.type)+1):
	# 			W.add((c, "[" + str(i) + "][0]"))
	return W

def statement_read_varrefs(s, o, sm, only_unguarded):
	"""Return a set of variable refs from which the statement is reading. o is Object owning s. sm is the state machine owning s. only_unguarded indicates whether only the unguarded objects (true) or all objects (false) should be returned."""
	global connected_channel, signalsize, smname_to_object, smnames, alphabet
	R = set([])
	if s.__class__.__name__ == "Assignment":
		if s.left.index != None:
			R |= statement_read_varrefs(s.left.index, o, sm, only_unguarded)
		R |= statement_read_varrefs(s.right, o, sm, only_unguarded)
	elif s.__class__.__name__ == "Composite":
		if s.guard != None:
			R |= statement_read_varrefs(s.guard, o, sm, only_unguarded)
		if s.guard == None or not only_unguarded:
			for i in range(0,len(s.assignments)):
				R |= statement_read_varrefs(s.assignments[i], o, sm, only_unguarded)
	elif s.__class__.__name__ == "SendSignal":
		c = connected_channel[(o, s.target)]
		if c.synctype == 'async':
			# add channel with index "_size" to represent 'size' variable
			R.add((c, "_size"))
		# if c.synctype == 'sync':
			# add (Object,StateMachine) pairs for state machines that can potentially receive messages
			# for (o2,sm2) in get_syncrec_sms(o, c, s.signal):
			# 	R.add((c, scopename(sm2,None,o2)))
		if not only_unguarded:
			for p in s.params:
				R |= statement_read_varrefs(p, o, sm, only_unguarded)
	elif s.__class__.__name__ == "ReceiveSignal":
		c = connected_channel[(o, s.target)]
		if c.synctype == 'async':
			# Note: checking size of buffer is NOT part of unguarded behaviour! (instead, this is checked in an earlier stage)
			# add channel with index to represent the signal of the buffer head element
			if signalsize[c] > 0:
				R.add((c, 0))
			if not only_unguarded:
				# add other elements of buffer head element
				for i in range(1,len(c.type)+1):
				 	R.add((c, i))
			# below: also holds for sync
			# Messagerefs = set([])
			# for p in s.params:
			# 	Messagerefs.add((p.var.name, getinstruction(p.index, o, {}, {})))
			# 	if p.index != None:
			# 		R |= statement_read_varrefs(p.index, o, sm, only_unguarded)
			# R1 = statement_read_varrefs(s.guard, o, sm, only_unguarded)
			# for (v,j) in R1:
			# 	if (v.name, getinstruction(j, o, {}, {})) not in Messagerefs:
			# 		R.add((v,j))
		else:
			# add (Object,Statemachine) pair to represent the current state variable
			R.add((c, scopename(sm,None,o)))
		if not only_unguarded:
			Messagerefs = []
			R1 = set([])
			for p in s.params:
				Messagerefs.append((p.var.name, getinstruction(p.index, o, {})))
				if p.index != None:
					R1 |= statement_read_varrefs(p.index, o, sm, only_unguarded)
			R1 |= statement_read_varrefs(s.guard, o, sm, only_unguarded)
			for (v,j) in R1:
				j_str = getinstruction(j, o, {})
				found = False
				for (v2name,j2_str) in Messagerefs:
					if v.name == v2name and j_str == j2_str:
						found = True
						break
				if not found:
					R.add((v,j))
	elif s.__class__.__name__ == "Expression" or s.__class__.__name__ == "ExprPrec4" or s.__class__.__name__ == "ExprPrec3" or s.__class__.__name__ == "ExprPrec2" or s.__class__.__name__ == "ExprPrec1":
		R |= statement_read_varrefs(s.left, o, sm, only_unguarded)
		if s.op != '':
			R |= statement_read_varrefs(s.right, o, sm, only_unguarded)
	elif s.__class__.__name__ == "Primary":
		if s.ref != None:
			if s.ref.ref in actions:
				# we have a user-defined action
				if s.ref.ref in syncactions.get(o.type,set([])):
					# add references to other state machines in Object o that have the action in their alphabet
					# (i.e., we add a reference to their current state variable)
					for sname in smnames:
						(o2,sm2) = smname_to_object[sname]
						if o == o2 and sm != sm2:
							if s.ref.ref in alphabet[sm2]:
								R.add((sm2,"src"))
			if s.ref.index != None:
				R |= statement_read_varrefs(s.ref.index, o, sm, only_unguarded)
			# obtain suitable object matching name s.ref.ref
			for v1 in sm_variables(sm):
				if v1.name == s.ref.ref:
					R.add((v1, s.ref.index))
					break
		if s.body != None:
			R |= statement_read_varrefs(s.body, o, sm, only_unguarded)
	elif s.__class__.__name__ == "VariableRef":
		if s.index != None:
			R |= statement_read_varrefs(s.index, o, sm, only_unguarded)
		R.add((s.var, s.index))
	return R

def get_vars(L):
	"""Given a list of variable refs, return the corresponding set of vars"""
	S = set([])
	for (v,j,subid) in L:
		S.add(v)
	return S

def sm_variables(sm):
	"""Provide list of variables accessible from current state machine sm"""
	Ldict = {}
	for v in sm.parent.variables:
		Ldict[v.name] = v
	for v in sm.variables:
		Ldict[v.name] = v
	return Ldict.values()

def state_varobjects(s):
	"""Return a list of objects the state is accessing (associated to its outgoing transitions)"""
	V = set([])
	for t in outgoingtrans(s,s.parent.transitions):
		for st in t.statements:
			V |= statement_varobjects_set(st, s.parent, False)
	return list(V)

def state_unguarded_varobjects(s):
	"""Return a list of objects the state is accessing. do not add objects that are guarded."""
	V = set([])
	for t in outgoingtrans(s,s.parent.transitions):
		for st in t.statements:
			V |= statement_varobjects_set(st, s.parent, True)
	return list(V)

def statement_guarded_varobjects(s):
	"""Return a list of guarded objects the statement is accessing."""
	return list(set(statement_varobjects(s)) - set(state_unguarded_varobjects(s.parent.source)))

def get_buffer_allocs(T):
	"""Return info on number of variable allocations to do for evaluation of transition blocks of (Object,transition) pairs in set T. o is Object owning s."""
	global vectorelem_in_structure_map, signalsize, max_statesize, vectortree, vectortree_T, vectorstructure, connected_channel, state_order
	max_32 = 0
	max_16 = 0
	max_8 = 0
	max_bool = 0
	max_idx = 0

	for (o2,t) in T:
		sm2 = t.parent
		Vseen = set([])
		O = set([])
		for st in t.statements:
			# O is set of variable refs occurring in block of t
			O |= statement_varrefs(st, o2, sm2, 0)
		nr_32 = 0
		nr_8 = 0
		nr_bool = 0
		dict_arrays_8 = {}
		dict_arrays_bool = {}
		for (v,i,subid) in O:
			if v.__class__.__name__ != "Channel" and v.__class__.__name__ != "StateMachine":
				i_str = getinstruction(i, o2, {})
			else:
				i_str = i
			if (v,i_str,subid) not in Vseen:
				Vseen.add((v,i_str,subid))
				if v.__class__.__name__ == "Channel":
					if v.synctype == 'async':
						# access channel buffer item or the buffer's size variable
						if i == "_size":
							size = vectorelem_in_structure_map[v.name + i][0]
						else:
							size = vectorelem_in_structure_map[v.name + "[" + str(i) + "][0]"][0]							
					else: # synchronous channel
						# if this entry refers to a ReceiveSignal statement, consider storing the source state of its transition
						# i is an (Object, Statemachine) pair
						if st.__class__.__name__ == "ReceiveSignal":
							size = max_statesize
						else:
							size = 0
						# else:
						# 	size = vectorelem_in_structure_map[i][0]
					if size > 0:
						if size <= 1:
							nr_bool += 1
						elif size <= 8:
							nr_8 += 1
						else:
							nr_32 += 1
				elif v.__class__.__name__ == "StateMachine":
					if max_statesize <= 1:
						nr_bool += 1
					elif max_statesize <= 8:
						nr_8 += 1
					else:
						nr_32 += 1
				elif v.type.base == 'Integer':
					nr_32 += 1
				elif v.type.base == 'Byte':
					nr_8 += 1
					if v.type.size > 0:
						count = dict_arrays_8.get(v, 0)
						count += 1
						dict_arrays_8[v] = count
				else:
					nr_bool += 1
					if v.type.size > 0:
						count = dict_arrays_bool.get(v, 0)
						count += 1
						dict_arrays_bool[v] = count
		# identify the largest number of array index variables needed
		dyn_allocs = get_buffer_arrayindex_allocs(t, o2)
		nr_idx = 0
		for v in dyn_allocs.keys():
			if dyn_allocs[v] > nr_idx:
				nr_idx = dyn_allocs[v]
		# if t has a SendSignal statement connected to a synchronous channel, also consider matching ReceiveSignal statements
		if st.__class__.__name__ == "SendSignal":
			c = connected_channel[(o2,st.target)]
			if c.synctype == 'sync':
				T2 = get_syncrec_sm_trans(o2, c, st.signal)
				(rec_32, rec_16, rec_8, rec_bool, rec_idx) = get_buffer_allocs(T2)
				nr_32 += rec_32
				nr_8 += rec_8
				nr_bool += rec_bool
				# consider the need for array index variables
				nr_idx2 = 0
				for (o3,t3) in T2:
					dyn_allocs = get_buffer_arrayindex_allocs(t3,o3)
					for v in dyn_allocs.keys():
						if dyn_allocs[v] > nr_idx2:
							nr_idx2 = dyn_allocs[v]
				nr_idx += nr_idx2
		# record maximum of idx
		if nr_idx > max_idx:
			max_idx = nr_idx
		# add additional 16-bit integers for pointers to store new vector trees
		bound = 1
		syncrecs = []
		if st.__class__.__name__ == "SendSignal":
			c = connected_channel[(o2,st.target)]
			if c.synctype == 'sync':
				bound = len(syncrecs)
				syncrecs = get_syncrec_sm_trans(o2, c, st.signal)
		for k in range(0,bound):
			O = set([])
			ST = []
			for st2 in t.statements:
				ST.append((o2,st2))
			if len(syncrecs) > k:
				(o3,t3) = syncrecs[k]
				for st3 in t3.statements:
					ST.append((o3,st3))
			for (o3,st3) in ST:
				O = get_write_vectorparts_info(st3,o3)
				O = sorted(list(O.keys()))
				n = len(vectorstructure)-1
				Onew = []
				for v in O:
					Onew.append(vectornode_id(v))
				O = Onew
				# explore vectortree to find maximum number of required pointers to store (the delta of) a new state vector tree
				navcounters = {}
				waiting = set([])
				L = deepcopy(O)
				seen = set([])
				while L != []:
					v = L.pop(0)
					if v != 0:
						nextnode = vectortree_T[v]
						C = navcounters.get(nextnode, 0)
						navcounters[nextnode] = C+1
						if nextnode not in seen:
							L.append(nextnode)
						seen.add(nextnode)
				maximum = 0
				nr_of_pointers = 0
				if len(O) > 0:
					current = O.pop(0)
					nr_of_pointers = 1
					maximum = 1
					while current != 0:
						parent = vectortree_T[current]
						navcounters[parent] -= 1
						if navcounters[parent] == 0:
							current = parent
							if parent in waiting:
								nr_of_pointers -= 1
								waiting.remove(parent)
						else:
							waiting.add(parent)
							current = O.pop(0)
							nr_of_pointers += 1
							if nr_of_pointers > maximum:
								maximum = nr_of_pointers
			if maximum > max_16:
				max_16 = maximum

		# record maxima
		if nr_32 > max_32:
			max_32 = nr_32
			diff = 0
		else:
			diff = max_32 - nr_32
		# if nr_16 > max_16 + diff:
		# 	max_16 = nr_16 - diff
		# 	diff = 0
		# else:
		# 	if nr_16 > max_16:
		# 		diff = diff - (nr_16 - max_16)
		# 	else:
		# 		diff = (max_16 + diff) - nr_16
		# iterate over byte arrays first
		tmp = 0
		for (v,nr) in dict_arrays_8.items():
			tmp += nr
		if tmp > max_8:
			max_8 = tmp
		nr_8 = nr_8 - tmp
		left = max_8 - tmp
		if nr_8 > left + diff:
			max_8 += nr_8 - left - diff
			diff = 0
		else:
			if nr_8 > left:
				diff -= (nr_8 - left)
			else:
				diff += (left - nr_8)
		# iterate over bool arrays
		tmp = 0
		for (v,nr) in dict_arrays_bool.items():
			tmp += nr
		if tmp > max_bool:
			max_bool = tmp
		nr_bool = nr_bool - tmp
		left = max_bool - tmp
		if nr_bool > left + diff:
			max_bool += nr_bool - left - diff
	return (max_32, max_16, max_8, max_bool, max_idx)

def get_buffer_arrayindex_allocs(t, o):
	"""Return info on variable allocations needed to do bookkeeping on dynamically accessing SLCO array elements, for the transition t. o is Object owning the transitions."""
	allocs = {}
	sm = t.parent
	Vseen = set([])
	access_counters = {}
	dynamic_access_seen = set([])
	O = set([])
	for st in t.statements:
		O |= statement_varrefs(st, o, sm, 0)
	for (v,i,subid) in O:
		# is v an array?
		if v.__class__.__name__ != "Channel" and v.__class__.__name__ != "StateMachine": 
			if v.type.size > 0:
				i_str = getinstruction(i, o, {})
				if (v,i_str,subid) not in Vseen:
					Vseen.add((v,i_str,subid))
					count = access_counters.get(v, 0)
					count += 1
					access_counters[v] = count
					if not RepresentsInt(i_str):
						dynamic_access_seen.add(v)
	for v in dynamic_access_seen:
		count1 = allocs.get((o,v), 0)
		count2 = access_counters.get(v, 0)
		if count2 > count1:
			allocs[(o,v)] = count2
	return allocs

def map_variables_on_buffer(t, o, buffer_allocs, prevM={}):
	"""Construct for block of given transition t a mapping from variables to buffer variables. o is Object owning t.
	prevM is a previous variable mapping, which the new one should extend. prevallocs are precomputed buffer allocations."""
	global vectorelem_in_structure_map, max_statesize
	sm = t.parent
	s = t.statements[0]

	allocs = get_buffer_arrayindex_allocs(t, o)

	# counters
	current_32 = 0
	current_16 = 0
	current_8 = 0
	current_bool = 0
	current_idx = 0
	# if a previous mapping has been given, derive suitable offsets to extend it with the new mapping.
	prevM_offset_32 = 0
	prevM_offset_8 = 0
	prevM_offset_bool = 0
	prevM_offset_idx = 0
	v_32 = ''
	v_8 = ''
	v_bool = ''
	v_idx = ''
	if prevM != {}:
		prevM_offset_32 = -1
		prevM_offset_8 = -1
		prevM_offset_bool = -1
		prevM_offset_idx = -1
		for (v,i) in prevM.items():
			if i[0] == "buf32" and i[1] > prevM_offset_32:
				prevM_offset_32 = i[1]
				v_32 = v
			elif i[0] == "buf8" and i[1] > prevM_offset_8:
				prevM_offset_8 = i[1]
				v_8 = v
			elif i[0] == "buf1" and i[1] > prevM_offset_bool:
				prevM_offset_bool = i[1]
				v_bool = v
			# check for use of indices
			if len(i) > 2:
				if i[2] > prevM_offset_idx:
					prevM_offset_idx = i[2]
					v_idx = v
		if v_32 != '':
			size = 1
			if v_32.__class__.__name__ != "Channel" and v_32.__class__.__name__ != "Statemachine":
				if v_32.type.size > 1:
					if allocs.get(v_32) != None:
						size = allocs[v_32]
					else:
						size = len(get_constant_indices(v_32, v_32.name, t, o))
			prevM_offset_32 += size
		else:
			prevM_offset_32 = 0
		if v_8 != '':
			size = 1
			if v_8.__class__.__name__ != "Channel" and v_8.__class__.__name__ != "Statemachine":
				if v_8.type.size > 1:
					if allocs.get(v_8) != None:
						size = allocs[v_8]
					else:
						size = len(get_constant_indices(v_8, v_8.name, t, o))
			prevM_offset_8 += size
		else:
			prevM_offset_8 = 0
		if v_bool != '':
			size = 1
			if v_bool.__class__.__name__ != "Channel" and v_bool.__class__.__name__ != "Statemachine":
				if v_bool.type.size > 1:
					if allocs.get(v_bool) != None:
						size = allocs[v_bool]
					else:
						size = len(get_constant_indices(v_bool, v_bool.name, t, o))
			prevM_offset_bool += size
		else:
			prevM_offset_bool = 0
		if v_idx != '':
			prevM_offset_idx += allocs[v_idx]
		else:
			prevM_offset_idx = 0
	O = set([])
	for st in t.statements:
		O |= statement_varrefs(st, o, sm, 0)
	M = {}
	access_counters = {}
	Vseen = set([])
	for (v,i,subid) in O:
		if v.__class__.__name__ != "Channel" and v.__class__.__name__ != "StateMachine":
			i_str = getinstruction(i, o, {})
			if (v,i_str,subid) not in Vseen:
				Vseen.add((v,i_str,subid))
				count = access_counters.get(v, 0)
				count += 1
				access_counters[v] = count
		elif v.__class__.__name__ == "StateMachine":
			access_counters[v] = 2
		else:
			access_counters[v] = 1
	
	# iterate over elements in descending order w.r.t. nr of items (arrays have multiple items)
	for (v, ac) in sorted(access_counters.items(), key=(lambda x: x[1]), reverse=True):
		if v.__class__.__name__ == "Channel":
			if v.synctype == 'async':
				size = vectorelem_in_structure_map[v.name + "_size"][0]
				if size <= 1 and current_bool < buffer_allocs[3]:
					M[(v, "_size")] = ("buf1", current_bool+prevM_offset_bool)
					current_bool += 1
				elif size <= 8 and current_8 < buffer_allocs[2]:
					M[(v, "_size")] = ("buf8", current_8+prevM_offset_8)
					current_8 += 1
				else:
					M[(v, "_size")] = ("buf32", current_32+prevM_offset_32)
					current_32 += 1
				bound = len(v.type)+1
				if s.__class__.__name__ == "ReceiveSignal":
					bound = 1
				for i in range(0,bound):
					if i == 0:
						size = signalsize[v]
					else:
						size = gettypesize(v.type[i-1])
					vindex = i
					if size > 0:
						if size == 1 and current_bool < buffer_allocs[3]:
							M[(v, vindex)] = ("buf1", current_bool+prevM_offset_bool)
							current_bool += 1
						elif size <= 8 and current_8 < buffer_allocs[2]:
							M[(v, vindex)] = ("buf8", current_8+prevM_offset_8)
							current_8 += 1
						else:
							M[(v, vindex)] = ("buf32", current_32+prevM_offset_32)
							current_32 += 1
			else: # synchronous channel
				# we know that we are dealing with a ReceiveSignal statement, since SendSignal statements over a synchronous channel require no additional data (besides parameter variables).
				size = max_statesize
				if size <= 1 and current_bool < buffer_allocs[3]:
					M[(v,"state")] = ("buf1", current_bool+prevM_offset_bool)
					current_bool += 1
				elif size <= 8 and current_8 < buffer_allocs[2]:
					M[(v,"state")] = ("buf8", current_8+prevM_offset_8)
					current_8 += 1
				else:
					M[(v,"state")] = ("buf32", current_32+prevM_offset_32)
					current_32 += 1
				# handle (Object,StateMachine) pairs for state machines that can potentially receive messages
				# for (o2,sm2) in get_syncrec_sms(o, v, st.signal):
				# 	size = vectorelem_in_structure_map[o2.name + "'" + sm2.name][0]
				# 	if size <= 1 and current_bool < buffer_allocs[3]:
				# 		M[(v,o2.name + "'" + sm2.name)] = ("buf1", current_bool)
				# 		current_bool += 1
				# 	elif size <= 8 and current_8 < buffer_allocs[2]:
				# 		M[(v,o2.name + "'" + sm2.name)] = ("buf8", current_8)
				# 		current_8 += 1
				# 	elif size <= 16 and current_16 < buffer_allocs[1]:
				# 		M[(v,o2.name + "'" + sm2.name)] = ("bufaddr", current_16)
				# 		current_16 += 1
				# 	else:
				# 		M[(v,o2.name + "'" + sm2.name)] = ("buf32", current_32)
				# 		current_32 += 1
		else:
			if v.__class__.__name__ == "StateMachine":
				size = max_statesize
				for x in ["src","tgt"]:
					if size <= 1 and current_bool < buffer_allocs[3]:
						M[(v,x)] = ("buf1", current_bool+prevM_offset_bool)
						current_bool += 1
					elif size <= 8 and current_8 < buffer_allocs[2]:
						M[(v,x)] = ("buf8", current_8+prevM_offset_8)
						current_8 += 1
					else:
						M[(v,x)] = ("buf32", current_32+prevM_offset_32)
						current_32 += 1
			else:
				size = gettypesize(v.type)
				if size <= 1 and current_bool < buffer_allocs[3]:
					if v.type.size == 0:
						M[v] = ("buf1", current_bool+prevM_offset_bool)
						current_bool += access_counters[v]
					else:
						M[v] = ("buf1", current_bool+prevM_offset_bool, current_idx+prevM_offset_idx)
						current_bool += access_counters[v]
						current_idx += access_counters[v]
				elif size <= 8 and current_8 < buffer_allocs[2]:
					if v.type.size == 0:
						M[v] = ("buf8", current_8+prevM_offset_8)
						current_8 += access_counters[v]
					else:
						M[v] = ("buf8", current_8+prevM_offset_8, current_idx+prevM_offset_idx)
						current_8 += access_counters[v]
						current_idx += access_counters[v]
				else:
					if v.type.size == 0:
						M[v] = ("buf32", current_32+prevM_offset_32)
						current_32 += access_counters[v]
					else:
						M[v] = ("buf32", current_32+prevM_offset_32, current_idx+prevM_offset_idx)
						current_32 += access_counters[v]
						current_idx += access_counters[v]
	M.update(prevM)
	return M

def get_all_relevant_vectorparts_for_state(s, o):
	"""For the given Statemachine state s, return a sorted list of vector parts that contain relevant info to process its outgoing transitions."""
	global vectorelem_in_structure_map, connected_channel, signalsize, state_order

	sm = s.parent
	P = set([])
	smid = 0
	# look up id of Statemachine
	for j in range(0, len(state_order)):
		if state_order[j] == o.name + "'" + sm.name:
			smid = j
			break
	for t in outgoingtrans(s,s.parent.transitions):
		if must_be_processed_by(t, smid, o):
			for st in t.statements:
				O = statement_varrefs(st, o, sm, 0)
				for (v,i,subid) in O:
					if v.__class__.__name__ == "Channel":
						if v.synctype == 'async':
							# add all elements of the channel's buffer
							start = 0
							if signalsize[v] == 0:
								start = 1
							for j in range(start, len(v.type)):
								for k in range(0, v.size):
									PIDs = vectorelem_in_structure_map[scopename(v,None,o) + "[" + str(j) + "][" + str(k) + "]"]
									P.add(PIDs[1][0])
									if len(PIDs) > 2:
										P.add(PIDs[2][0])
							# add the vectorpart containing the size variable of the buffer
							PIDs = vectorelem_in_structure_map[scopename(v,None,o) + "_size"]
							P.add(PIDs[1][0])
							if len(PIDs) > 2:
								P.add(PIDs[2][0])
					elif v.__class__.__name__ != "StateMachine":
						if v.type.size > 0:
							if has_dynamic_indexing(v, v.name, t, o):
								# add all elements of array v
								for j in range(0, v.type.size):
									PIDs = vectorelem_in_structure_map[scopename(v,None,o) + "[" + str(j) + "]"]
									P.add(PIDs[1][0])
									if len(PIDs) > 2:
										P.add(PIDs[2][0])
							else:
								PIDs = vectorelem_in_structure_map[scopename(v,i,o)]
								P.add(PIDs[1][0])
								if len(PIDs) > 2:
									P.add(PIDs[2][0])
						else:
							PIDs = vectorelem_in_structure_map[scopename(v,None,o)]
							P.add(PIDs[1][0])
							if len(PIDs) > 2:
								P.add(PIDs[2][0])
	return sorted(list(P))

def get_vectorparts(L, o):
	"""For the given set of variable refs (of given Object o), return a sorted list of vector parts that contain that info"""
	global vectorelem_in_structure_map

	P = set([])
	for (v,i) in L:
		# if v.__class__.__name__ != "Channel":
		# 	i_str = getinstruction(i, {}, {})
		# else:
		# 	i_str = i
		name = scopename(v,i,o)
		PIDs = vectorelem_in_structure_map[name]
		P.add(PIDs[1][0])
		if len(PIDs) > 2:
			P.add(PIDs[2][0])
	return sorted(list(P))

def get_remaining_vectorparts(L, o, VPs):
	"""Return a sorted list of vector parts that are needed to obtain values of given variable refs in L, that are not covered by list of vector parts VPs"""
	global vectorelem_in_structure_map

	VPset = set(VPs)
	P = set([])
	for (v,i) in L:
		if v.__class__.__name__ != "Channel":
			i_str = getinstruction(i, o, {})
			name = scopename(v,None,o)
			if RepresentsInt(i_str):
				name += "[" + i_str + "]"
		else:
			if RepresentsInt(i):
				name = v.name + "[" + str(i) + "][0]"
			elif "'" in i:
				name = i
			else:
				name = v.name + i
		PIDs = vectorelem_in_structure_map[name]
		S = set([PIDs[1][0]])
		if len(PIDs) > 2:
			S.add(PIDs[2][0])
		if not S.issubset(VPset):
			P |= S
	return sorted(list(P))

def vectorparts_not_covered(vi, o, VPs):
	"""Return whether or not the vectorparts needed for variable ref (v,i) are in list of vector parts VPs"""
	global vectorelem_in_structure_map

	v = vi[0]
	i = vi[1]
	VPset = set(VPs)
	if v.__class__.__name__ == "Channel":
		if RepresentsInt(i):
			name = v.name + "[" + str(i) + "][0]"
		elif "'" in i:
			name = i
		else:
			name = v.name + i
	else:
		i_str = getinstruction(i, o, {})
		name = scopename(v,None,o)
		if RepresentsInt(i_str):
			name += "[" + i_str + "]"
	PIDs = vectorelem_in_structure_map[name]
	S = set([PIDs[1][0]])
	if len(PIDs) > 2:
		S.add(PIDs[2][0])
	return not S.issubset(VPset)

def get_write_vectorparts_info(s, o, sender_o='', sender_sm='', lossy=False):
	"""Return a dictionary of (vectorpart,varrefs) tuples indicating how the new values produced when executing statement s have to be stored in vector parts"""
	"""sender_o and sender_sm are optional arguments indicating a (synchronous) sender SM that needs to change state."""
	global vectorelem_in_structure_map, connected_channel, signalsize

	Refs = statement_write_varrefs(s,o)
	D = {}
	for (v,i) in Refs:
		i_str = getinstruction(i, o, {})
		if has_dynamic_indexing(v, v.name, s.parent, o):
			name = scopename(v,None,o)
		else:
			name = scopename(v,i,o)
		if i == None or (RepresentsInt(i_str) and not has_dynamic_indexing(v, v.name, s.parent, o)):
			PIDs = vectorelem_in_structure_map[name]
			p = PIDs[1][0]
			Dv = D.get(p,set([]))
			# Boolean flag indicates that this is the first part (of possibly two) in which the variable is stored
			Dv.add((v,i,False))
			D[p] = Dv
			if len(PIDs) > 2:
				p = PIDs[2][0]
				Dv = D.get(p,set([]))
				Dv.add((v,i,True))
				D[p] = Dv
		else:
			# obtain all parts that contain a part of array v
			size = v.type.size
			S = set([])
			for j in range(0,size):
				i_str = "[" + str(j) + "]"
				PIDs = vectorelem_in_structure_map[name + i_str]
				S.add(PIDs[1][0])
				if len(PIDs) > 2:
					S.add(PIDs[2][0])
			for p in S:
				Dv = D.get(p,set([]))
				# We write '*' as index to indicate dynamic indexing into array
				Dv.add((v,'*',False))
				D[p] = Dv
	# add references to target state of statement
	state = o.name + "'" + s.parent.parent.name
	PIDs = vectorelem_in_structure_map[state]
	p = PIDs[1][0]
	Dv = D.get(p,set([]))
	# Boolean flag indicates that this is the first part (of possibly two) in which the variable is stored
	Dv.add((o,s.parent.parent,False))
	D[p] = Dv
	if len(PIDs) > 2:
		p = PIDs[2][0]
		Dv = D.get(p,set([]))
		Dv.add((o,s.parent.parent,True))
		D[p] = Dv
	if s.__class__.__name__ == "SendSignal":
		c = connected_channel[(o,s.target)]
		if c.synctype == 'async':
			# in case of asynchronous communication, add all vectorparts comtaining the buffer of the channel, unless we are handling a 'lossy' case
			if not lossy:
				i = 0
				if signalsize[c] == 0:
					i = 1
				j = 0
				PIDs = vectorelem_in_structure_map[c.name + "[" + str(i) + "][" + str(j) + "]"]
				first = PIDs[1][0]
				PIDs = vectorelem_in_structure_map[c.name + "[" + str(len(c.type)) + "][" + str(c.size-1) + "]"]
				last = PIDs[1][0]
				if len(PIDs) > 2:
					last = PIDs[2][0]
				for p in range(first,last+1):
					Dv = D.get(p,set([]))
					# We write '*' as index to indicate dynamic indexing into buffer
					Dv.add((c,'*',False))
					D[p] = Dv
				# add the buffer size variable
				PIDs = vectorelem_in_structure_map[c.name + "_size"]
				p = PIDs[1][0]
				Dv = D.get(p,set([]))
				# Boolean flag indicates that this is the first part (of possibly two) in which the variable is stored
				Dv.add((c,"_size",False))
				D[p] = Dv
				if len(PIDs) > 2:
					p = PIDs[2][0]
					Dv = D.get(p,set([]))
					Dv.add((c,"_size",True))
					D[p] = Dv
	if s.__class__.__name__ == "ReceiveSignal":
		c = connected_channel[(o,s.target)]
		if c.synctype == 'async':
			# add all vectorparts containing the buffer of the channel
			i = 0
			if signalsize[c] == 0:
				i = 1
			j = 0
			PIDs = vectorelem_in_structure_map[c.name + "[" + str(i) + "][" + str(j) + "]"]
			first = PIDs[1][0]
			PIDs = vectorelem_in_structure_map[c.name + "[" + str(len(c.type)) + "][" + str(c.size-1) + "]"]
			last = PIDs[1][0]
			if len(PIDs) > 2:
				last = PIDs[2][0]
			for p in range(first,last+1):
				Dv = D.get(p,set([]))
				# We write '*' as index to indicate dynamic indexing into buffer
				Dv.add((c,'*',False))
				D[p] = Dv
			# add the buffer size variable
			PIDs = vectorelem_in_structure_map[c.name + "_size"]
			p = PIDs[1][0]
			Dv = D.get(p,set([]))
			# Boolean flag indicates that this is the first part (of possibly two) in which the variable is stored
			Dv.add((c,"_size",False))
			D[p] = Dv
			if len(PIDs) > 2:
				p = PIDs[2][0]
				Dv = D.get(p,set([]))
				Dv.add((c,"_size",True))
				D[p] = Dv
		else:
			# add reference to state of sending SM in case of synchronous communication
			if sender_o != '':
				PIDs = vectorelem_in_structure_map[sender_o.name + "'" + sender_sm.name]
				p = PIDs[1][0]
				Dv = D.get(p,set([]))
				# Boolean flag indicates that this is the first part (of possibly two) in which the variable is stored
				Dv.add((sender_o,sender_sm,False))
				D[p] = Dv
				if len(PIDs) > 2:
					p = PIDs[2][0]
					Dv = D.get(p,set([]))
					Dv.add((sender_o,sender_sm,True))
					D[p] = Dv

	# handle action synchronisation
	if statement_is_actionref(s):
		a = getlabel(s)
		sm = s.parent.parent
		if a in syncactions.get(o.type,set([])):
			# construct list of statemachines with which synchronisation is required
			SMs = []
			for m in smnames:
				(o2,sm2) = smname_to_object[m]
				if o == o2 and sm != sm2:
					if a in alphabet[sm2]:
						SMs.append(sm2)
			# obtain part ids
			for sm2 in SMs:
				PIDs = vectorelem_in_structure_map[o.name + "'" + sm2.name]
				p = PIDs[1][0]
				Dv = D.get(p,set([]))
				# Boolean flag indicates that this is the first part (of possibly two) in which the variable is stored
				Dv.add((o,sm2,False))
				D[p] = Dv
				if len(PIDs) > 2:
					p = PIDs[2][0]
					Dv = D.get(p,set([]))
					Dv.add((o,sm2,True))
					D[p] = Dv				
	return D

# filter to produce difference between two lists
def difference(L1, L2):
	L = []
	L2set = set(L2)
	for l in L1:
		if l not in L2set:
			L.append(l)
	return L

# filter to obtain string description of given state vector element
def vector_element_string_desc(s):
	global elements_strings
	return elements_strings[s]

def is_async_channel(name):
	"""Return whether given name refers to an asynchronous channel in model"""
	global model

	for c in model.channels:
		if name == c.name:
			if c.synctype == 'async':
				return True
			else:
				return False
	return False

def is_state(name):
	"""Returns whether given name refers to a state in model"""
	global model

	for o in model.objects:
		c = o.type
		for sm in c.statemachines:
			if name == o.name + "'" + sm.name:
				return True
	return False

# Test to check if given channel is asynchronous
def is_async(c):
	return c.synctype == 'async'

# Test to check if given variable ref has dynamic indexing. o is Object owning v.
def no_dynamic_indexing(v, o):
	if v[0].__class__.__name__ == "Channel":
		return True
	if v[0].__class__.__name__ == "StateMachine":
		return True
	if v[1] == None:
		return True
	i_str = getinstruction(v[1], o, {})
	return RepresentsInt(i_str)

def has_dynamic_indexing(v, vname, t, o):
	"""Returns whether for array v, dynamic indexing is done somewhere in the block of transition t. o is Object owning s."""
	if v.__class__.__name__ != "Channel" and v.__class__.__name__ != "StateMachine":
		sm = t.parent
		O = set([])
		for st in t.statements:
			O |= statement_varrefs(st, o, sm, 0)
		for (v1,i,subid) in O:
			if v1.name == vname:
				if i != None:
					i_str = getinstruction(i, o, {})
					if not RepresentsInt(i_str):
						return True
	return False

def get_constant_indices(v, vname, t, o):
	"""Returns for array v a dictionary for constant indices used in the block of transition t, to map these to thread buffer variables in CUDA code. o is Object owning s."""
	L = set([])
	if v.__class__.__name__ != "Channel" and v.__class__.__name__ != "StateMachine":
		sm = t.parent
		O = set([])
		for st in t.statements:
			O |= statement_varrefs(st, o, sm, 0)
		for (v1,i,subid) in O:
			if v1.name == vname:
				if i != None:
					i_str = getinstruction(i, o, {})
					if RepresentsInt(i_str):
						L.add(i_str)
	L = sorted(list(L))
	D = {}
	counter = 0
	for i in L:
		D[i] = counter
		counter += 1
	return D

def is_model_write_constant_indexed(vname):
	"""Returns whether or not to given array element (identified by string name) is at least once written to in the model by referring to the index statically (i.e., with a constant)."""
	global model

	# if vname does not refer to an array element, return False
	if len((vname.split("["))) == 1:
		return False
	for o in model.objects:
		for sm in o.type.statemachines:
			for t in sm.transitions:
				for st in t.statements:
					O = statement_write_varrefs(st, o)
					for (v1,i) in O:
						if scopename(v1,i,o) == vname:
							return True
	return False

def is_model_write_dynamic_indexed(vname):
	"""Returns whether or not to given array (identified by string name) is at least once written to in the model by referring to the index dynamically (i.e., with an expression)."""
	global model, arraynames

	# if vname does not refer to an array, return False
	if vname not in arraynames.keys():
		return False
	for o in model.objects:
		for sm in o.type.statemachines:
			for t in sm.transitions:
				for st in t.statements:
					O = statement_write_varrefs(st, o)
					for (v1,i) in O:
						if scopename(v1,None,o) == vname:
							if i != None:
								i_str = getinstruction(i, o, {})
								if not RepresentsInt(i_str):
									return True
	return False

def get_array_range_in_vectorpart(v, vname, pid):
	"""Return, for the given array with the given name and vectorpart pid, the range of the former's elements that are (partially) stored in part pid"""
	global vectorstructure

	size = v.type.size
	I = set([])
	S = vectorstructure[pid]
	for (name,size) in S:
		if vname in name:
			# extract index from name
			index = name.split("[")
			index = index[1]
			index = index[:-1]
			I.add(index)
	# pick maximum and minimum values
	lower = min(I)
	upper = max(I)
	return (lower,upper)

def must_be_processed_by(t, i, o):
	"""Return whether or not the given transition must be processed by the thread associated to the state machine with id i in the vector state order. o is Object owning t."""
	global state_order, smname_to_object, alphabet, syncactions, connected_channel

	st = t.statements[0]
	if st.__class__.__name__ == "ReceiveSignal":
		c = connected_channel[(o, st.target)]
		if c.synctype == 'sync':
			return False
	if not statement_is_actionref(st):
		return True
	else:
		a = getlabel(st)
		if a not in syncactions.get(o.type,set([])):
			return True
		else:
			for j in range(0,i):
				smname = state_order[j]
				(o2,sm2) = smname_to_object[smname]
				if o.name == o2.name:
					if a in alphabet[sm2]:
						return False
			return True

def syncaction_must_be_processed_by(a, sm):
	"""Return whether or not the given sync action must be processed by the thread associated to state machine sm."""
	global model, state_order, smname_to_object, alphabet, syncactions

	# Retrieve an object containing sm
	c = sm.parent
	smo = 0
	for o in model.objects:
		if o.type == c:
			smo = o
			break
		if smo == 0:
			return False
	# Retrieve state machine order number
	j = 0
	for i in range(0, len(state_order)):
		if state_order[i] == smo.name + "'" + sm.name:
			j = i
			break
	# Check for ownership
	for i in range(0,j):
		smname = state_order[i]
		(o2,sm2) = smname_to_object[smname]
		if smo.name == o2.name:
			if a in alphabet[sm2]:
				return False
	return True

def nr_of_transitions_to_be_processed_by(s,i,o):
	"""Return the number of outgoing transitions of state s to be processed by the thread associated to the state machine with id i"""
	nr = 0
	for t in outgoingtrans(s,s.parent.transitions):
		if must_be_processed_by(t,i,o):
			nr += 1
	return nr

def get_syncrec_sm_trans(o, c, signal):
	"""Return a list of (Object,StateMachine) pairs that can potentially receive messages from a StateMachine in the given Object o via the given synchronous channel c"""
	global syncreccomm, connected_channel
	S = syncreccomm[c]
	L = []
	for (o2,t2) in S:
		if o2 != o:
			for st2 in t2.statements:
				if st2.__class__.__name__ == "ReceiveSignal":
					if signal == st2.signal:
						L.append((o2,t2))
	return L

def get_all_syncrecs(c):
	"""Return a list of (Object,StateMachine,signal) triples of StateMachines that can potentially receive messages with signal 'signal' via the given synchronous channel c"""
	global connected_channel

	L = set([])
	for o in model.objects:
		for sm in o.type.statemachines:
			for t in sm.transitions:
				for st in t.statements:
					if st.__class__.__name__ == "ReceiveSignal":
						if c == connected_channel[(o,st.target)]:
							L.add((o,sm,st.signal))
	return sorted(list(L), key=lambda x: (x[0].name, x[1].name, x[2]))

# Function to iterate over asynchronous channel buffer elements
def next_buffer_element(c,i,j):
	global signalsize

	elsize = len(c.type) + 1
	size = c.size
	ni = i
	nj = j
	ni += 1
	if ni == elsize:
		ni = 0
		if signalsize[c] == 0:
			ni = 1
		nj += 1
		if nj == size:
			ni = -1
			nj = -1
	return (ni,nj)

# Filter for debugging
def debug(text):
	print(text)
	return ''

def preprocess():
	"""Preprocessing of model"""
	global model, vectorsize, vectorstructure, vectortree, vectortree_T, vectortree_group_size, vectortree_level_ids, vectortree_nr_reachable_state_parts, vectortree_node_thread, vectorstructure_string, smnames, vectorelem_in_structure_map, max_statesize, state_order, smname_to_object, state_id, arraynames, max_arrayindexsize, max_buffer_allocs, connected_channel, signalsize, signalnr, alphabet, syncactions, actiontargets, actions, syncreccomm, no_state_constant, no_prio_constant, dynamic_access_arrays, async_channel_vectorpart_buffer_range, vectortree_size, vectortree_depth, vectortree_level_nr_of_leaves, vectortree_level_nr_of_nodes_with_two_children, tilesize, gpuexplore2_succdist, regsort_nr_el_per_thread, all_arrayindex_allocs_sizes, smart_vectortree_fetching_bitmask, nr_warps_per_tile, compact_hash_table, elements_strings, nrblocks, nrthreadsperblock, array_in_structure_map, vectorpart_id_dict, vectornode_id_dict, no_smart_fetching, nr_bits_shared_mem_element, nr_cache_elements

	# construct set of statemachine names in the system
	# also construct a map from names to objects
	smnames = set([])
	smname_to_object = {}
	for o in model.objects:
		for sm in o.type.statemachines:
			smnames.add(o.name + "'" + sm.name)
			smname_to_object[o.name + "'" + sm.name] = (o,sm)
	stateelements = set([])
	dataelements = set([])
	elements_strings = {}

	# construct dictionary providing connected channel of given port
	connected_channel = {}
	for c in model.channels:
		connected_channel[(c.source, c.port0)] = c
		connected_channel[(c.target, c.port1)] = c

	# construct dictionary to obtain (Object,Transition) pairs that are potentially receiving messages from a given synchronous channel
	syncreccomm = {}
	for ch in model.channels:
		if ch.synctype == 'sync':
			S = set([])
			for o in [ch.source,ch.target]:
				for sm in o.type.statemachines:
					found = False
					for t in sm.transitions:
						for st in t.statements:
							if st.__class__.__name__ == "ReceiveSignal":
								if (o == ch.source and st.target == ch.port0) or (o == ch.target and st.target == ch.port1):
									S.add((o,t))
									found = True
									break
						if found:
							break
			syncreccomm[ch] = S

	# determine number of bits needed to represent channel signals for each channel, and encode for each channel the possible signals
	signalsize = {}
	signalnr = {}
	signals = {}
	for o in model.objects:
		c = o.type
		for sm in c.statemachines:
			for t in sm.transitions:
				for st in t.statements:
					if st.__class__.__name__ == "ReceiveSignal" or st.__class__.__name__ == "SendSignal":
						c = connected_channel[(o,st.target)]
						S = signals.get(c, set([]))
						S.add(st.signal)
						signals[c] = S
	# determine number of bits needed
	for c, S in signals.items():
		bitcount = int(math.ceil(math.log(len(S), 2)))
		signalsize[c] = bitcount
		# encode signals
		nr = 0
		for s in S:
			signalnr[(c,s)] = nr
			nr += 1

	# for each state machine, order its states, and store the ids
	state_id = {}
	for c in model.classes:
		for sm in c.statemachines:
			state_nr = 0
			for s in sm.states:
				state_id[s] = state_nr
				state_nr += 1
	# store maximum number of bits needed to encode an automaton state
	max_statesize = 0
	# calculate state vector size
	for o in model.objects:
		# object global variables
		for v in o.type.variables:
			size = gettypesize(v.type)
			dimension = 0
			if v.type.size > 1:
				dimension = v.type.size
			vectorsize += (size*max(1,dimension))
			dataelements.add((o.name + "'" + v.name, tuple([size]), dimension))
		for sm in o.type.statemachines:
			# number of bits needed to encode states of this state machine.
			size = int(max(1,math.ceil(math.log(len(sm.states), 2))))
			vectorsize += size
			stateelements.add((o.name + "'" + sm.name, size))
			# Update max_statesize. We do this with len(sm.states)+1 (i.e. we add an extra state,
			# to incorporate the possibility of encoding the NO_STATE constant, which is the value
			# for state variables not containing a state).
			tmpsize = int(max(1,math.ceil(math.log(len(sm.states)+1, 2))))
			if max_statesize < tmpsize:
				max_statesize = tmpsize
			# statemachine local variables
			for v in sm.variables:
				size = gettypesize(v.type)
				dimension = 0
				if v.type.size > 1:
					dimension = v.type.size
				vectorsize += (size*max(1,dimension))
				dataelements.add((o.name + "'" + sm.name + "'" + v.name, tuple([size]), dimension))
	# add buffers for channels
	for ch in model.channels:
		if ch.synctype == 'async':
			typelist = [signalsize[ch]]
			typesize = 0
			for t in ch.type:
				typelist.append(gettypesize(t))
				typesize += gettypesize(t)
			dimension = 1
			if ch.size > 1:
				dimension = ch.size
			vectorsize += typesize*dimension
			# take buffer counter into account
			vectorsize += int(max(1,math.ceil(math.log(dimension, 2))))
			dataelements.add((ch.name, tuple(typelist), dimension))
	# if the vectorsize is sufficiently small, compact hash table storage and smart fetching are not needed.
	if vectorsize < 63:
		compact_hash_table = False
		no_smart_fetching = True
	for (s,i) in stateelements:
		if i > max_statesize:
			max_statesize = i
	# set the number of bits required per shared memory element
	if vectorsize > 30 and vectorsize <= 62:
		nr_bits_shared_mem_element = 64
	else:
		nr_bits_shared_mem_element = 32
	# store maximum number of bits needed for indices of arrays and channel buffers
	max_arrayindexsize = 1
	maxsize = 1
	for c in model.classes:
		for v in c.variables:
			if v.type.size > maxsize:
				maxsize = v.type.size
		for sm in c.statemachines:
			for v in sm.variables:
				if v.type.size > maxsize:
					maxsize = v.type.size
	for c in model.channels:
		if c.size > maxsize:
			maxsize = c.size
	max_arrayindexsize = int(max(1,math.ceil(math.log(maxsize, 2))))
	# construct the state vector structure
	vectorstructure = []
	# construct the mapping from vector element to vector part (integer index)
	vectorelem_in_structure_map = {}
	# construct the order of states and variables in the vectors
	state_order = {}
	tmp = []
	tmpsize = 0
	intsize = 30
	if vectorsize > 30:
		intsize = 62
	if compact_hash_table:
		intsize += 1
	vp_id = 0
	state_nr = 0
	while stateelements != set([]) or dataelements != set([]):
		selected = ""
		selected_dim = 0
		selected_sizelist = []
		if stateelements != set([]):
			for (s,i) in stateelements:
				if i <= intsize - tmpsize:
					selected = s
					selected_sizelist = [i]
					stateelements.remove((s,i))
					state_order[state_nr] = selected
					state_nr += 1
					elements_strings[selected] = "state"
					break
		elif dataelements != set([]):
			for (d,slist,dimension) in dataelements:
				if slist[0] <= intsize - tmpsize:
					selected = d
					selected_sizelist = slist
					selected_dim = dimension
					dataelements.remove((d,slist,dimension))
					break
		if selected == "":
			# nothing selected; add an element partially to current part
			if stateelements != set([]):
				for (s,i) in stateelements:
					selected = s
					selected_sizelist = [i]
					stateelements.remove((s,i))
					state_order[state_nr] = selected
					state_nr += 1
					elements_strings[selected] = "state"
					break					
			elif dataelements != set([]):
				for (d,slist,dimension) in dataelements:
					selected = d
					selected_sizelist = slist
					selected_dim = dimension
					dataelements.remove((d,slist,dimension))
					break
		if selected != "":
			# if we are adding a channel buffer, add counters if needed
			if is_async_channel(selected):
				sname = selected
				# create a new part if the current one is full
				if intsize - tmpsize == 0:
					vectorstructure.append(tmp)
					tmp = []
					tmpsize = 0
					vp_id += 1
				dimbits = int(max(1,math.ceil(math.log(selected_dim, 2))))
				added_size = min(dimbits, intsize - tmpsize)
				tmp.append((sname + "_size", added_size))
				PIDs = [dimbits, (vp_id, (intsize)-(tmpsize+added_size), added_size)]
				tmpsize += added_size
				if dimbits - added_size > 0:
					# add remainder of element to new vector part
					vectorstructure.append(tmp)
					tmp = [(sname + "_size", dimbits - added_size)]
					tmpsize = dimbits - added_size
					vp_id += 1
					PIDs.append((vp_id, intsize-(dimbits - added_size), dimbits - added_size))
				vectorelem_in_structure_map[sname + "_size"] = PIDs
				elements_strings[sname + "_size"] = "channel size counter"
			# add selected element to current (and possibly subsequent) vector part(s)
			for i in range(0,max(1,selected_dim)):
				for j in range(0,len(selected_sizelist)):
					sname = selected
					sindex = ""
					if is_async_channel(selected):
						sindex = "[" + str(j) + "][" + str(i) + "]"
					elif selected_dim > 0:
						sindex = "[" + str(i) + "]"
					l = selected_sizelist[j]
					if l > 0:
						# create a new part if the current one is full
						if intsize - tmpsize == 0:
							vectorstructure.append(tmp)
							tmp = []
							tmpsize = 0
							vp_id += 1
						added_size = min(l, intsize - tmpsize)
						tmp.append((sname + sindex, added_size))
						PIDs = [l, (vp_id, intsize-(tmpsize+added_size), added_size)]
						tmpsize += added_size
						if l - added_size > 0:
							# add remainder of element to new vector part
							vectorstructure.append(tmp)
							tmp = [(sname + sindex, l - added_size)]
							added_size = l - added_size
							tmpsize = 0
							vp_id += 1
							PIDs.append((vp_id, intsize-(tmpsize+added_size), added_size))
							tmpsize = added_size
						vectorelem_in_structure_map[sname + sindex] = PIDs
						if elements_strings.get(sname + sindex) == None:
							if is_async_channel(sname):
								elements_strings[sname + sindex] = "channel buffer part"
							else:
								elements_strings[sname + sindex] = "variable"
	# add final vector part
	if tmp != []:
		vectorstructure.append(tmp)
	vectorstructure_string = vectorstructure_to_string(elements_strings)
	# create vectortree structures, indicating how to navigate between the leaves and the root.
	# number of nodes
	nrnodes = 2*len(vectorstructure) - 1
	# compensate for a final vector part integrated into a non-leaf node
	if vectorpart_is_combined_with_nonleaf_node(len(vectorstructure)-1):
		# update the positioning of the data in the final vector part to account for the left node pointer inside it
		vlist = vectorstructure[len(vectorstructure)-1]
		for (vname,vsize) in vlist:
			PIDs = vectorelem_in_structure_map.get(vname)
			newPIDslist = [PIDs[0]]
			for i in range(1, len(PIDs)):
				if PIDs[i][0] != len(vectorstructure)-1:
					newPIDslist.append(PIDs[i])
				else:
					newpos = 0
					if compact_hash_table:
						newpos = nr_bits_address_internal()-1-(63-PIDs[i][1])
					else:
						newpos = nr_bits_address_root()-1-(62-PIDs[i][1])
					newPIDslist.append((PIDs[i][0], newpos, PIDs[i][2]))
			vectorelem_in_structure_map[vname] = newPIDslist
	# number of vector parts with state machine states
	nrstatenodes = 0
	for t in vectorstructure:
		if t[0][0] in smnames:
			nrstatenodes += 1
	nrstatenodes += nrstatenodes-1
	# build the tree in a downwards direction first
	vectortree = {}
	vectortree_T = {}
	openlist = []
	statenodes = set([])
	datanodes = set([])
	statecount = 0
	datacount = 0
	children = []
	# distinguish different cases regarding the presence of states and data
	if nrnodes > 1:
		# add elements to openlist to check for left (False) and right (True) leaves
		children = [1]
		statenodes.add(1)
		nextnode = 2
		statecount += 1
	if nrnodes > 2:
		children.append(2)
		# we create an unbalanced tree intentionally: we want state nodes to be in the left subtree, for smart fetching.
		# unless all nodes are state nodes, the right node from the root should be a data node.
		if nrstatenodes == nrnodes:
			statenodes.add(2)
			statecount += 1
		else:
			datanodes.add(2)
			datacount += 1
		nextnode = 3
	vectortree[0] = children
	openlist = deepcopy(children)
	while openlist != []:
		current = openlist.pop(0)
		children = []
		# child one
		if current in statenodes and statecount < nrstatenodes:
			children.append(nextnode)
			statenodes.add(nextnode)
			statecount += 1
			nextnode += 1
		# the number of datanodes equals nrnodes - 1 - nrstatenodes.
		# a node that cannot reach a leaf with states is a datanode.
		# the root node is neither counted as a statenode nor as a datanode (therefore ... - 1 ...).
		elif current in datanodes and datacount < (nrnodes - 1 - nrstatenodes):
			children.append(nextnode)
			datanodes.add(nextnode)
			datacount += 1
			nextnode += 1
		# child two
		if current in statenodes and statecount < nrstatenodes:
			children.append(nextnode)
			statenodes.add(nextnode)
			statecount += 1
			nextnode += 1		
		elif current in datanodes and datacount < (nrnodes - 1 - nrstatenodes):
			children.append(nextnode)
			datanodes.add(nextnode)
			datacount += 1
			nextnode += 1		
		for c in reversed(children):
			openlist.insert(0, c)
		vectortree[current] = children
	# if the final vectorpart is to be integrated into a non-leaf node, do this
	if vectorpart_is_combined_with_nonleaf_node(len(vectorstructure)-1):
		# remove the right-most leaf in the tree
		current = 0
		children = vectortree[current]
		prev = 0
		while len(children) == 2:
			prev = current
			current = children[1]
			children = vectortree[current]
		if children == []:
			# remove the current node
			del vectortree[current]
			vectortree[prev].remove(current)
			# relabel subsequent nodes
			vectortree_new = {}
			for c in vectortree.keys():
				children = vectortree[c]
				children_new = []
				for c2 in children:
					if c2 < current:
						children_new.append(c2)
					else:
						children_new.append(c2-1)
				if c < current:
					vectortree_new[c] = children_new
				else:
					vectortree_new[c-1] = children_new
			vectortree = vectortree_new
	#print(vectortree)
	# now transpose this tree
	for v in vectortree.keys():
		C1 = vectortree[v]
		for w in C1:
			vectortree_T[w] = v
	# construct a dictionary providing the ids of the nodes in each level (from the left).
	# vectortree_level_nr_of_leaves indicates for each level how many leaves it contains.
	# vectortree_level_nr_of_nodes_with_two_children indicates for each level how many nodes
	# have two children.
	# vectortree_nr_reachable_state_parts provides per tree level how many parts containing state info
	# are reachable
	vectortree_level_ids = {}
	vectortree_level_nr_of_leaves = {}
	vectortree_level_nr_of_nodes_with_two_children = {}
	vectortree_nr_reachable_state_parts = {}
	curlevel = [0]
	# map nodes to threads.
	vectortree_node_thread = {}
	nextlevel = []
	level = 0
	leaf_counter = 0
	nonleaf_counter = 0
	while len(curlevel) > 0:
		vectortree_level_ids[level] = curlevel
		vectortree_level_nr_of_leaves[level] = 0
		vectortree_level_nr_of_nodes_with_two_children[level] = 0
		for i in range(0, len(curlevel)):
			children = vectortree.get(curlevel[i],[])
			# is the node a leaf?
			if len(children) == 0:
				vectortree_level_nr_of_leaves[level] += 1
				# assign the node to a GPU thread
				vectortree_node_thread[curlevel[i]] = leaf_counter
				leaf_counter += 1
			else:
				# assign the node to a GPU thread
				vectortree_node_thread[curlevel[i]] = nonleaf_counter
				nonleaf_counter += 1
			# does the node have two children?
			if len(children) == 2:
				vectortree_level_nr_of_nodes_with_two_children[level] += 1
			# add children to the next level
			nextlevel += children
		curlevel = nextlevel
		nextlevel = []
		level += 1
	# create dictionary mapping node ids to part ids
	vectorpart_id_dict = {}
	openstack = [0]
	next_partid = 0
	hybrid_node = -1
	while len(openstack) > 0:
		t = openstack.pop()
		if is_vectorpart(t):
			if is_non_leaf(t):
				hybrid_node = t
			else:
				vectorpart_id_dict[t] = next_partid
				next_partid += 1
		for w in reversed(vectortree[t]):
			openstack.append(w)
	# add info on extra, combined node
	if hybrid_node != -1:
		vectorpart_id_dict[hybrid_node] = next_partid
	# create dictionary in the other direction
	vectornode_id_dict = {}
	for i in range(0, len(vectortree)):
		pid = vectorpart_id_dict.get(i, -1)
		if pid != -1:
			vectornode_id_dict[pid] = i
	# construct vectortree_nr_reachable_state_parts by iterating over the leaves, and moving up the tree
	prevcount = 0
	for i in reversed(range(0, len(vectortree_level_ids))):
		count = 0
		for n in vectortree_level_ids[i]:
			if is_vectorpart(n):
				pid = vectorpart_id(n)
				if vectorstructure[pid][0][0] in smnames:
					# a state part
					count += 1
		# update count
		vectortree_nr_reachable_state_parts[i] = prevcount + count
		prevcount += count
	# construct smart_vectortree_fetching_bitmask dictionary, which provides bitmasks needed for smart fetching
	# of vectortrees.
	smart_vectortree_fetching_bitmask = {}
	for i in range(0,len(vectortree)):
		bitmask = 0
		if is_non_leaf(i):
			# iterate over the vectorparts, and determine which ones can be reached from node i
			for j in range(0,len(vectortree)):
				if is_vectorpart(j):
					pid = vectorpart_id(j)
					current = j
					while current != 0:
						parent = vectortree_T[current]
						if parent == i:
							bitmask += (1 << (31 - pid))
							break
						else:
							current = parent
		if is_vectorpart(i):
			pid = vectorpart_id(i)
			bitmask += (1 << (31 - pid))
		smart_vectortree_fetching_bitmask[i] = bitmask
	# in addition, add a bitmask for the fetching of the state machine states part(s) of a vector.
	bitmask = 0
	for i in range(0,len(vectorstructure)):
		if vectorstructure[i][0][0] in smnames:
			bitmask += (1 << (31 - i))
#			bitmask += smart_vectortree_fetching_bitmask[i]
	smart_vectortree_fetching_bitmask["smstates"] = bitmask
	# create dictionary of array names
	arraynames = {}
	for o in model.objects:
		for v in o.type.variables:
			if v.type.size > 0:
				arraynames[o.name + "'" + v.name] = (v.type, v.type.size)
		for sm in o.type.statemachines:
			for v in sm.variables:
				if v.type.size > 0:
					arraynames[o.name + "'" + sm.name + "'" + v.name] = (v.type, v.type.size)
	# create action alphabets and construct dictionary providing target states when performing a given action in a given state
	alphabet = {}
	Adict = {}
	actiontargets = {}
	actions = set([])
	for c in model.classes:
		for sm in c.statemachines:
			A = set([])
			smtrans = {}
			for t in sm.transitions:
				for st in t.statements:
					if statement_is_actionref(st):
						a = getlabel(st)
						A.add(a)
						atrans = smtrans.get(a, {})
						src = state_id[t.source]
						tgt = state_id[t.target]
						tgts = atrans.get(src, [])
						tgts.append(tgt)
						tgts = list(set(tgts))
						atrans[src] = tgts
						smtrans[a] = atrans
			actiontargets[sm] = smtrans
			alphabet[sm] = A
			for a in A:
				actions.add(a)
				C = Adict.get(a, {})
				count = C.get(c,0)
				count += 1
				C[c] = count
				Adict[a] = C
	# construct set of actions requiring synchronisation
	syncactions = {}
	for a in actions:
		C = Adict[a]
		for c in C.keys():
			if C[c] > 1:
				CC = syncactions.get(c,set([]))
				CC.add(a)
				syncactions[c] = CC
	# for c in model.channels:
	# 	if c.synctype == 'async':
	# 		if signalsize[c] > 0:
	# 			arraynames.append((c.name + "[0]", signalsize[c], c.size))
	# 		for i in range(0,len(c.type)):
	# 			arraynames.append((c.name + "[" + str(i+1) + "]", c.type[i], c.size))
	# compute maximum number of buffer variable allocs needed for transition block processing
	max_buffer_allocs = [0,0,0]
	Cseen = set([])
	for o in model.objects:
		c = o.type
		if c not in Cseen:
			Cseen.add(c)
			for sm in c.statemachines:
				for s in sm.states:
					allocs = get_buffer_allocs(object_trans_to_be_processed_by_sm_thread(s,o))
					if allocs[0] > max_buffer_allocs[0]:
						max_buffer_allocs[0] = allocs[0]
					if allocs[2] > max_buffer_allocs[1]:
						max_buffer_allocs[1] = allocs[2]
					if allocs[3] > max_buffer_allocs[2]:
						max_buffer_allocs[2] = allocs[3]
	max_buffer_allocs = max(max_buffer_allocs)
	# determine values for constants
	no_state_constant = 0
	for c in model.classes:
		for sm in c.statemachines:
			if no_state_constant < len(sm.states):
				no_state_constant = len(sm.states)
	no_prio_constant = 0
	for c in model.classes:
		for sm in c.statemachines:
			for t in sm.transitions:
				if no_prio_constant < t.priority:
					no_prio_constant = t.priority
	# construct set of array variables
	arrays = set([])
	for o in model.objects:
		for sm in o.type.statemachines:
			for v in sm.variables:
				if v.type.size > 0:
					arrays.add((v,o.name + "'" + sm.name + "'" + v.name))
		for v in o.type.variables:
			if v.type.size > 0:
				arrays.add((v,o.name + "'" + v.name))
	# construct dynamic write arrays dictionary
	dynamic_access_arrays = {}
	for o in model.objects:
		for sm in o.type.statemachines:
			for t in sm.transitions:
				for st in t.statements:
					W = statement_write_varrefs(st, o)
					W |= statement_read_varrefs(st, o, sm, False)
					# search for dynamic accesses
					for (v,i) in W:
						if i != None:
							vname = o.name + "'"
							if v.parent.__class__.__name__ == "StateMachine":
								vname += sm.name + "'"
							vname += v.name
							if not isinstance(i, str):
								i_str = getinstruction(i,o,{})
								if not RepresentsInt(i_str):
									size = v.type.size
									# lower bound
									PIDs = vectorelem_in_structure_map[vname + "[0]"]
									lower = PIDs[1][0]
									# upper bound
									PIDs = vectorelem_in_structure_map[vname + "[" + str(size-1) + "]"]
									upper = PIDs[1][0]
									if len(PIDs) > 2:
										upper = PIDs[2][0]
									dynamic_access_arrays[v] = (vname,lower,upper)
	# construct dictionary for array placement in vectorparts
	array_in_structure_map = {}
	for (v,vname) in arrays:
		# construct a list L containing the placement info
		L = []
		size = v.type.size
		# lower bound
		PIDs = vectorelem_in_structure_map[vname + "[0]"]
		current_vp = PIDs[1][0]
		# add starting part id to L
		L.append(current_vp)
		# create tuples, starting with the starting position of array element 0
		tmp = [0,0,PIDs[1][1],0,0]
		old_PIDs = PIDs
		for i in range(1,size):
			PIDs = vectorelem_in_structure_map[vname + "[" + str(i) + "]"]
			if current_vp != PIDs[1][0]:
				# tuple is ready for storage
				tmp[1] = i-1
				if len(old_PIDs) > 2:
					tmp[3] = old_PIDs[2][2]
					tmp[4] = old_PIDs[2][1]
				L.append(tuple(tmp))
				tmp = [i,0,PIDs[1][1],0,0]
				current_vp = PIDs[1][0]
			old_PIDs = PIDs
		# store final tuple
		tmp[1] = size-1
		if len(PIDs) > 2:
			tmp[3] = PIDs[2][2]
			tmp[4] = PIDs[2][1]
		L.append(tuple(tmp))
		array_in_structure_map[vname] = L
	# construct async_channel_vectorpart_buffer_range: for all (asynchronous channel, vectorpart) pairs, provide the range of buffer elements of that channel stored in that vectorpart of a vector.
	async_channel_vectorpart_buffer_range = {}
	for c in model.channels:
		if c.synctype == 'async':
			if signalsize[c] > 0:
				PIDs = vectorelem_in_structure_map[c.name + "[0][0]"]
				lower = (0,0)
				i_type = 0
				i_dim = 0
			else:
				PIDs = vectorelem_in_structure_map[c.name + "[1][0]"]
				lower = (1,0)
				i_type = 0
				i_dim = 0
			current = PIDs[1][0]
			PIDs = vectorelem_in_structure_map[c.name + "[" + str(len(c.type)) + "][" + str(c.size-1) + "]"]
			if len(PIDs) > 2:
				last = PIDs[2][0]
			else:
				last = PIDs[1][0]
			upper = lower
			while current <= last:
				i_type += 1
				if i_type == len(c.type)+1:
					if signalsize[c] > 0:
						i_type = 0
					else:
						i_type = 1
					i_dim += 1
				if i_dim == c.size:
					break
				PIDs = vectorelem_in_structure_map[c.name + "[" + str(i_type) + "][" + str(i_dim) + "]"]
				p = PIDs[1][0]
				if p == current:
					upper = (i_type,i_dim)
				else:
					R = async_channel_vectorpart_buffer_range.get(c,{})
					R[current] = (lower,upper)
					async_channel_vectorpart_buffer_range[c] = R
					current = p
					lower = (i_type, i_dim)
			# add final range
			R = async_channel_vectorpart_buffer_range.get(c,{})
			R[current] = (lower,upper)
			async_channel_vectorpart_buffer_range[c] = R
	all_arrayindex_allocs_sizes = set([])
	C = set([])
	for o in model.objects:
		if o.type not in C:
			C.add(o.type)
			for sm in o.type.statemachines:
				for t in sm.transitions:
					allocs = get_buffer_arrayindex_allocs(t, o)
					for i in allocs.values():
						if i not in all_arrayindex_allocs_sizes:
							all_arrayindex_allocs_sizes.add(i)
	all_arrayindex_allocs_sizes = list(all_arrayindex_allocs_sizes)
	all_arrayindex_allocs_sizes = sorted(all_arrayindex_allocs_sizes)
	# determine the size of a vectortree group (group of threads needed to fetch a vector tree from the global hash table)
	vectortree_size = len(vectorstructure)
	if vectorpart_is_combined_with_nonleaf_node(len(vectorstructure)-1):
		vectortree_size -= 1
	# determine power of two (< 32) nearest to vectortree_size for cooperative group definition
	if vectortree_size == 1:
		vectortree_group_size = 1
	elif vectortree_size == 2:
		vectortree_group_size = 2
	elif vectortree_size <= 4:
		vectortree_group_size = 4
	elif vectortree_size <= 8:
		vectortree_group_size = 8
	elif vectortree_size <= 16:
		vectortree_group_size = 16
	else:
		vectortree_group_size = 32
	# determine the depth of the vectortree
	vectortree_depth = 0
	children = [0]
	while children != []:
		vectortree_depth += 1
		nextchildren = []
		for c in children:
			nextchildren += vectortree.get(c, [])
		children = nextchildren
	# determine the tile size.
	if gpuexplore2_succdist:
		# determine the tile size based on GPUexplore 2.0 successor work distribution.
		# As devisor we take minimum value 2, to ensure that the worktile is smaller than the block size.
		tilesize = int(nrthreadsperblock / max(len(smnames), 2))
		# to handle models with much data, we divide the tilesize by a factor.
		datadiv = int(1 + (vectortree_size/4))
		tilesize = int(tilesize / datadiv)
	else:
		# divide number of warps per block by the number of statemachines (minimum 2) in the model (or 16, if the former is smaller than the latter).
		# multiply that number by warpsize, as each thread in a warp can work on a different state vector.
		tilesize = int(((nrthreadsperblock / warpsize) / min(max(len(smnames), 2), (nrthreadsperblock / warpsize))) * warpsize)
		# to handle models with much data, we divide the tilesize by a factor.
		datadiv = int(1 + (vectortree_size/4))
		tilesize = int(tilesize / datadiv)
		nr_warps_per_tile = int(math.ceil(float(tilesize) / float(warpsize)))
		if not no_regsort:
			# compute the number of elements per thread in intra-warp regsort of tile elements
			regsort_nr_el_per_thread = int(math.pow(2,math.ceil(math.log(tilesize, 2))) / warpsize)
	if gpu_querying:
		# determine the size of shared memory caches (nr of elements)
		if cuda.Device.count() == 0:
			print("No CUDA-capable GPUs detected! You can generate code with the '-l' option.")
			sys.exit(1)
		else:
			device = cuda.Device(0)
			attrs = device.get_attributes()
			shared_size = int(math.floor(attrs.get(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK) / (nr_bits_shared_mem_element/8)))
			if vectorsize > 62:
				offset = int(5+(2*tilesize)+(nrthreadsperblock/32))
			else:
				offset = int(5+tilesize+(nrthreadsperblock/32))
			if deadlock_check:
				offset += int(math.ceil(tilesize/(nr_bits_shared_mem_element/8)))
			nr_cache_elements = shared_size - offset
			if vectorsize > 62:
				nr_cache_elements = int(math.floor(nr_cache_elements / 3))
			print("Nr. of elements in cache hash table: " + str(nr_cache_elements))
	print("System state vector size: " + str(vectorsize))
	if vectorsize <= 30:
		print("Running in 32-bit state vector mode")
	elif vectorsize <= 62:
		print("Running in 64-bit state vector mode")
	elif compact_hash_table:
		print("Running in compact hash table mode")
	else:
		print("Running in non-compact hash table mode")
	if with_cuckoo:
		print("Cuckoo hashing enabled")

def translate():
	"""The translation function"""
	global modelname, model, vectorstructure, vectorstructure_string, vectortree, vectortree_T, vectortree_group_size, vectortree_level_ids, vectortree_level_nr_of_leaves, vectortree_level_nr_of_nodes_with_two_children, vectortree_nr_reachable_state_parts, vectorelem_in_structure_map, vectortree_node_thread, state_order, max_statesize, smnames, smname_to_object, state_id, arraynames, max_arrayindexsize, max_buffer_allocs, signalsize, connected_channel, alphabet, syncactions, actiontargets, no_state_constant, no_prio_constant, async_channel_vectorpart_buffer_range, vectortree_size, vectortree_depth, gpuexplore2_succdist, no_regsort, tilesize, regsort_nr_el_per_thread, warpsize, all_arrayindex_allocs_sizes, no_smart_fetching, compact_hash_table, nrblocks, nrthreadsperblock, array_in_structure_map, nr_bits_shared_mem_element, deadlock_check, with_cuckoo, max_evictions
	
	path, name = split(modelname)
	if name.endswith('.slco'):
		name = name[:-5]
	else:
		name = name[:-8]
	outFile = open(join(path,name + "_gpuexplore.cuh"), 'w')

	# Initialize the template engine.
	jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(join(this_folder,'../../jinja2_templates')), trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols','jinja2.ext.do',])

	# Register the filters
	jinja_env.filters['get_vector_tree_to_part_navigation'] = get_vector_tree_to_part_navigation
	jinja_env.filters['get_vector_tree_to_node_navigation'] = get_vector_tree_to_node_navigation
	jinja_env.filters['outgoingtrans'] = outgoingtrans
	jinja_env.filters['object_trans_to_be_processed_by_sm_thread'] = object_trans_to_be_processed_by_sm_thread
	jinja_env.filters['cuda_xor_lr'] = cuda_xor_lr
	jinja_env.filters['cuda_xor_lr_inv'] = cuda_xor_lr_inv
	jinja_env.filters['cuda_xor_r3'] = cuda_xor_r3
	jinja_env.filters['cuda_xor_r3_inv'] = cuda_xor_r3_inv
	jinja_env.filters['cuda_xor_r'] = cuda_xor_r
	jinja_env.filters['cuda_xor_r_inv'] = cuda_xor_r_inv
	jinja_env.filters['cudarecsizeguard'] = cudarecsizeguard
	jinja_env.filters['cudaguard'] = cudaguard
	jinja_env.filters['cudafetchdata'] = cudafetchdata
	jinja_env.filters['cudastatement'] = cudastatement
	jinja_env.filters['getinstruction'] = getinstruction
	jinja_env.filters['getlabel'] = getlabel
	jinja_env.filters['transition_read_varrefs'] = transition_read_varrefs
	jinja_env.filters['transition_sorted_dynamic_read_varrefs'] = transition_sorted_dynamic_read_varrefs
	jinja_env.filters['statement_varrefs'] = statement_varrefs
	jinja_env.filters['statement_write_varrefs'] = statement_write_varrefs
	jinja_env.filters['state_unguarded_varobjects'] = state_unguarded_varobjects
	jinja_env.filters['statement_guarded_varobjects'] = statement_guarded_varobjects
	jinja_env.filters['cudatype'] = cudatype
	jinja_env.filters['get_vectorparts'] = get_vectorparts
	jinja_env.filters['vectorpart_is_combined_with_nonleaf_node'] = vectorpart_is_combined_with_nonleaf_node
	jinja_env.filters['get_smart_fetching_vectorparts_bitmask'] = get_smart_fetching_vectorparts_bitmask
	jinja_env.filters['get_remaining_vectorparts'] = get_remaining_vectorparts
	jinja_env.filters['get_compact_thread_condition'] = get_compact_thread_condition
	jinja_env.filters['get_compact_leaf_thread_condition'] = get_compact_leaf_thread_condition
	jinja_env.filters['get_compact_nonleaf_thread_condition'] = get_compact_nonleaf_thread_condition
	jinja_env.filters['scopename'] = scopename
	jinja_env.filters['gettypesize'] = gettypesize
	jinja_env.filters['getlogarraysize'] = getlogarraysize
	jinja_env.filters['get_buffer_allocs'] = get_buffer_allocs
	jinja_env.filters['get_buffer_arrayindex_allocs'] = get_buffer_arrayindex_allocs
	jinja_env.filters['map_variables_on_buffer'] = map_variables_on_buffer
	jinja_env.filters['is_state'] = is_state
	jinja_env.filters['has_dynamic_indexing'] = has_dynamic_indexing
	jinja_env.filters['is_model_write_constant_indexed'] = is_model_write_constant_indexed
	jinja_env.filters['is_model_write_dynamic_indexed'] = is_model_write_dynamic_indexed
	jinja_env.filters['get_array_range_in_vectorpart'] = get_array_range_in_vectorpart
	jinja_env.filters['next_buffer_element'] = next_buffer_element
	jinja_env.filters['difference'] = difference
	jinja_env.filters['get_vars'] = get_vars
	jinja_env.filters['must_be_processed_by'] = must_be_processed_by
	jinja_env.filters['syncaction_must_be_processed_by'] = syncaction_must_be_processed_by
	jinja_env.filters['nr_of_transitions_to_be_processed_by'] = nr_of_transitions_to_be_processed_by
	jinja_env.filters['get_all_syncrecs'] = get_all_syncrecs
	jinja_env.filters['debug'] = debug
	jinja_env.filters['log2'] = log2
	jinja_env.filters['pow2'] = pow2
	jinja_env.filters['xor'] = xor
	jinja_env.filters['bitshift_one_right'] = bitshift_one_right
	jinja_env.filters['bitshift_left'] = bitshift_left
	jinja_env.filters['hexa'] = hexa
	jinja_env.filters['in_list'] = in_list
	jinja_env.filters['vector_element_string_desc'] = vector_element_string_desc

	# Register the tests
	jinja_env.tests['is_async'] = is_async
	jinja_env.tests['no_dynamic_indexing'] = no_dynamic_indexing
	jinja_env.tests['vectorparts_not_covered'] = vectorparts_not_covered

	# load the GPUexplore template
	template = jinja_env.get_template('gpuexplore.jinja2template')
	out = template.render(model=model, vectorsize=vectorsize, vectortree_group_size=vectortree_group_size, vectorstructure=vectorstructure, vectorstructure_string=vectorstructure_string, vectortree=vectortree, vectortree_T=vectortree_T, max_statesize=max_statesize, vectorelem_in_structure_map=vectorelem_in_structure_map, array_in_structure_map=array_in_structure_map, state_order=state_order, smnames=smnames, smname_to_object=smname_to_object, state_id=state_id, arraynames=arraynames, max_arrayindexsize=max_arrayindexsize, max_buffer_allocs=max_buffer_allocs, connected_channel=connected_channel, alphabet=alphabet, syncactions=syncactions, actiontargets=actiontargets, syncreccomm=syncreccomm, no_state_constant=no_state_constant, no_prio_constant=no_prio_constant, dynamic_access_arrays=dynamic_access_arrays, signalsize=signalsize, async_channel_vectorpart_buffer_range=async_channel_vectorpart_buffer_range, vectortree_depth=vectortree_depth, vectortree_level_ids=vectortree_level_ids, vectortree_level_nr_of_leaves=vectortree_level_nr_of_leaves, vectortree_level_nr_of_nodes_with_two_children=vectortree_level_nr_of_nodes_with_two_children, vectortree_nr_reachable_state_parts=vectortree_nr_reachable_state_parts, vectortree_node_thread=vectortree_node_thread, gpuexplore2_succdist=gpuexplore2_succdist, no_regsort=no_regsort, tilesize=tilesize, regsort_nr_el_per_thread=regsort_nr_el_per_thread, nr_warps_per_tile=nr_warps_per_tile, warpsize=warpsize, all_arrayindex_allocs_sizes=all_arrayindex_allocs_sizes, smart_vectortree_fetching_bitmask=smart_vectortree_fetching_bitmask, no_smart_fetching=no_smart_fetching, compact_hash_table=compact_hash_table, nr_bits_address_root=nr_bits_address_root(), nr_bits_address_internal=nr_bits_address_internal(), cuda_initial_vector=cudastore_initial_vector(), nrblocks=nrblocks, nrthreadsperblock=nrthreadsperblock, nr_bits_shared_mem_element=nr_bits_shared_mem_element, deadlock_check=deadlock_check, nr_cache_elements=nr_cache_elements, with_cuckoo=with_cuckoo, max_evictions=max_evictions)
	# write new SLCO model
	outFile.write(out)
	outFile.close()
	# create the main file for GPUexplore
	outFile = open(join(path,"gpuexplore.cu"), 'w')
	template = jinja_env.get_template('gpuexplore_main.jinja2template')
	out = template.render(name=name, vectorsize=vectorsize, vectortree_size=vectortree_size, vectortree_group_size=vectortree_group_size, compact_hash_table=compact_hash_table, global_memsize=global_memsize, nrthreadsperblock=nrthreadsperblock, tilesize=tilesize, deadlock_check=deadlock_check, nr_cache_elements=nr_cache_elements)
	outFile.write(out)
	outFile.close()
	# create a Makefile
	outFile = open(join(path,"Makefile"), 'w')
	outFile.write('all:\n')
	outFile.write('\tnvcc -arch=sm_75 -lcudart -o gpuexplore gpuexplore.cu\n')
	outFile.write('debug:\n')
	outFile.write('\tnvcc -arch=sm_75 -g -G -Xcompiler -rdynamic -lcudart -o gpuexplore gpuexplore.cu\n')
	outFile.close()

def main(args):
	"""The main function"""
	global modelname, model, property_file, deadlock_check, gpuexplore2_succdist, no_regsort, no_smart_fetching, compact_hash_table, with_cuckoo, max_evictions, global_memsize, nrblocks, nrthreadsperblock, vectorsize, gpu_querying, cuda
	if len(args) == 0:
		print("Missing argument: SLCO model")
		sys.exit(1)
	else:
		if args[0] == '-h' or args[0] == '-help':
			print("Usage: pypy/python3 slco2gpuexplore.py [-vrsc]")
			print("")
			print("Transform an SLCO 2.0 model to CUDA source code for GPUexplore 3.0.")
			print("")
			print(" -s                    size of the GPU global memory (in GB) (default 24)")
			print(" -d                    check for deadlocks")
			print(" -p                    verify given LTL property")
			print(" -b                    number of CUDA blocks to run (default 3120)")
			print(" -t                    number of threads per CUDA block (default 512)")
			print(" -l                    do not query GPU properties during code generation (default False)")
			print(" -g2                   apply GPUexplore 2.0 successor generation work distribution (default False)")
			print(" -noregsort            do not apply regsort for successor generation work distribution (default False)")
			print(" -nosmartfetching      disable smart fetching of vectortrees from global memory (default False)")
			print(" -nocompacthashtable   disable compact storage in global memory hash table (default False)")
			print(" -nocuckoo             disable Cuckoo hashing (default False)")
			sys.exit(0)
		else:
			for i in range(0,len(args)):
				if args[i] == '-s':
					global_memsize = int(args[i+1])
					i += 1
				elif args[i] == '-d' and property_file == "":
					deadlock_check = True
				elif args[i] == '-p':
					property_file = args[i+1]
					i += 1
				elif args[i] == '-g2':
					gpuexplore2_succdist = True
				elif args[i] == '-l':
					gpu_querying = False
				elif args[i] == '-noregsort':
					no_regsort = True
				elif args[i] == '-nosmartfetch':
					no_smart_fetching = True
				elif args[i] == '-nocompacthashtable':
					compact_hash_table = False
				elif args[i] == '-nocuckoo':
					with_cuckoo = False
				elif args[i] == '-b':
					nrblocks = int(args[i+1])
					i += 1
				elif args[i] == '-t':
					nrthreadsperblock = int(args[i+1])
					i += 1
				else:
					modelname = args[i]

	# if GPUexplore 2.0 successor generation is applied, regsorting is not.
	if gpuexplore2_succdist:
		no_regsort = True

	# if GPU querying is enabled, import pyCUDA
	if gpu_querying:
		cudainit = importlib.import_module("pycuda.autoinit")
		cuda = importlib.import_module("pycuda.driver")

	# Cuckoo hashing is disabled for non-compact state storage.
	# NOTE: for vectors in size <= 62, Cuckoo hashing currently leads to a deadlock.
	if not compact_hash_table or vectorsize <= 62:
		with_cuckoo = False

	# if Cuckoo hashing is disabled, the number of allowed evictions is 0.
	if not with_cuckoo:
		max_evictions = 0

	if not modelname.endswith('.slco'):
		print("please provide an SLCO model to be verified.")
	else:
		model = read_SLCO_model(modelname)
		print("processing model %s" % basename(modelname))
		try:
			preprocess()
			# translate
			translate()
		except Exception:
			print("failed to process model %s" % basename(modelname))
			print(traceback.format_exc())

if __name__ == '__main__':
	args = []
	for i in range(1,len(sys.argv)):
		args.append(sys.argv[i])
	main(args)
