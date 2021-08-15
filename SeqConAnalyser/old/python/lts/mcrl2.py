import subprocess
import logging
from lts import LTS

_equivalence_dict = {   LTS.Equivalence.WEAK_BISIM      : 'weak-bisim',
						LTS.Equivalence.BRANCHING_BISIM : 'branching-bisim'
					}

# minimisation of LTSs using mcrl2
ltsconvert = 'ltsconvert -e%s %s %s'
	
def minimize(input_str, equivalence, informat, outformat):
	equivalence = _equivalence_dict[equivalence]
	proc = subprocess.Popen(['ltsconvert','-e' + equivalence, '-i'+informat, '-o'+outformat],
							stdin=subprocess.PIPE, stdout=subprocess.PIPE,
							encoding = 'ascii')
	(std_out, std_err) = proc.communicate(input=input_str)
	#proc = subprocess.run(['ltsconvert','-e' + equivalence,input,output], stderr=subprocess.PIPE, universal_newlines=True)
	if proc.returncode != 0:
		logging.error("ltsconvert reports: "+ str(proc.stderr))
		raise subprocess.CalledProcessError(proc.returncode, 'ltsconvert', stderr=proc.stderr)
	
	return std_out