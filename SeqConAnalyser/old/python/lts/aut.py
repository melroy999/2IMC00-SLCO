import os
import re
import logging

scanner=re.Scanner([
	(r"des",							lambda scanner,token:("HEADER", token)),
	(r"[0-9]+",							lambda scanner,token:("NAT", token)),
	(r"[\s\(\),]+",                     None), # None == skip token.
	(r"[A-Za-z0-9!\_\-?.\*]+\([A-Za-z0-9!\_\-?,.\*\(\)\{\[\}\]]+\)|[A-Za-z0-9!\_\-?.\*]+|[\"\'][A-Za-z0-9!?\_\-\#\s.,\*\(\)\{\}\[\]]+[\"\']",
										lambda scanner,token:("ACTION", token)),
])
	
	
def read_from_file(folder, autfile):
	"""Read the .aut file and place the data in a dictionary"""
	aut_file_path = os.path.join(folder, autfile)
	inFile = None
	try:
		inFile = open(aut_file_path, 'r')
	except IOError:
		logging.error('Input LTS file does not exist: \%s', aut_file_path)
		return None
	ret = read(inFile)
	inFile.close()
	return ret
	
def read(input_stream):
	"""Read the .aut file and place the data in a dictionary"""
	# scan the first line
	line = input_stream.readline()
	results, remainder = scanner.scan(line)
	# read the header
	if results[0][0] != 'HEADER':
		logging.error("Unexpected start of .aut description!")
	transheader = []
	# an aut file has three values in the header
	for index in range(1, 4):
		transheader.append(results[index][1])
		
	trans = {}
	# count to remove double entries
	ntrans = 0
	actset = set([])

	#store transitions in memory
	currentsrc = ''
	newentry = {}
	first = True
	for transline in input_stream:
		if first:
			first = False
			continue
		
		tl, remainder = scanner.scan(transline)
		src = str(tl[0][1])
		act = str(tl[1][1])
		tgt = str(tl[2][1])
		# remove quotes
		act = act.replace('\'', '')
		act = act.replace('"', '')
		# remove spaces
		act = act.replace(' ', '')
		# put transition label in actset
		actset.add(act)
		# are we building a new dictionary entry?
		if src != currentsrc:
			# possibly add old entry if not empty
			if currentsrc != '':
				trans[currentsrc] = newentry
			currentsrc = src
			# obtain entry built so far
			newentry = trans.get(currentsrc,{})
		outgoingwithlabel = newentry.get(act)
		if outgoingwithlabel is None:
			newentry[act] = set([tgt])
			ntrans += 1
		else:
			if tgt not in outgoingwithlabel:
				# insert new element
				outgoingwithlabel.add(tgt)
				newentry[act] = outgoingwithlabel
				ntrans += 1
	# add final entry
	trans[currentsrc] = newentry
	transheader[1] = str(ntrans)
	input_stream.close()
	return [transheader, trans, actset]

def write_to_file(folder, header, trans, autfile):
	"""Write the trans LTS to file autfile"""
	outfile = open(os.path.join(folder, autfile + ".aut"), 'w')
	for line in to_string(header, trans):
		outfile.write(outfile, header, trans)
	outfile.flush()
	outfile.close()

def to_string(header, trans):
	"""Write the trans LTS to file aut stream"""
	#print header of .aut file
	if header[0] == '':
		header[0] = '0'
	yield 'des (' + header[0] + ', ' + header[1] + ', ' + header[2] + ')\n'
	for s in sorted(trans):
		outgoing = trans.get(s,{})
		for k, v in outgoing.items():
			label = k
			if label[0] != '"':
				label = '"' + str(k) + '"'
			for t in v:
				yield '(' + s + ', ' + label + ', ' + t + ')\n'