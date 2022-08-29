import ase
from ase.io import read,write
import ctypes
import numpy as np
from lammps import lammps, PyLammps

def py_extract_ace(atoms_file_prefix,L,ndescs):
	#atoms = ASE atoms object
	#atoms = read('ats.0.cfg')
	atoms = read('atsin.cfg')
	#atoms[0].position = atoms[0].position + np.array([1.,0,0])
        # always written to lammps data intermediate
	write('%s.data' % atoms_file_prefix,atoms,format='lammps-data')
	# atom info
	L.command('units  metal')
	L.command('atom_style  atomic')
	L.command('atom_modify  map array')
	L.command('neighbor  2.3 bin')
	# boundary
	L.command('boundary  p p p')
	# read atoms
	L.command('read_data  %s.data' % atoms_file_prefix)
	L.command('displace_atoms all random 0.01 0.01 0.01 42353')
	L.command('mass  1 183.84')#mass doesnt matter for run 0 and energy calculations rn...
        #  - mass per lammps type will need to be added for accurate force & virial terms
	# pairstyle 0 cutoff must be slightly larger than ACE cutoff
	L.command('pair_style      zero 7.502')
	L.command('pair_coeff  * *')
	L.command('compute         desc all pace coupling_coefficients.ace 0 0 ')
	L.command('timestep        0.0005')
	L.command('thermo          1')

	thermo_cmd = 'thermo_style    custom step temp c_desc[1][1] '
	for i in range(1,ndescs):
		thermo_cmd = thermo_cmd + ' c_desc[1][%d] ' % (i+1)
	L.command(thermo_cmd)

	L.run(0)
	#result type is 0 for global data
	result_type = 0
	result_style = 2
	#array_rows = (1 + (6*len(atoms)) + 6) 
	array_rows = (1 + (3*len(atoms)) + 6) 
	array_cols = ndescs +1
	array_shape = (array_rows,array_cols)
	total_size= np.prod(array_shape)

	#ptr = lmp.extract_compute('desc', 'pace', result_type)
	# array style is 2
	ptr = lmp.extract_compute('desc', result_type, result_style)
	buffer_ptr = ctypes.cast(ptr.contents, ctypes.POINTER(ctypes.c_double * total_size))
	array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
	array_np.shape = array_shape
	return array_np

lmp = lammps()
L = PyLammps(ptr=lmp)
ndescs = 232
atoms_file_prefix = 'ats.0'
arr = py_extract_ace(atoms_file_prefix=atoms_file_prefix,L=L,ndescs=ndescs)
print (arr)
np.save('A.npy',arr)
print (np.shape(arr))
for atind in range(4):
	for find in range(3):
		tmpind= 1+ (atind*3) + find 
		print (np.sum(arr[tmpind ][:-1]), np.sum([arr[tmpind][-1]]),  np.sum(arr[tmpind ][:-1])-np.sum([arr[tmpind][-1]]) )
		#print ('atom %d f_%d' %(atind,find+1), '%2.15f' % float(np.sum(arr[tmpind ][:-1])) - float(np.sum([arr[tmpind][-1]])))
#print ('atom 0 Fx', np.sum(arr[1]))
#print ('atom 1 Fx', np.sum(arr[1+3]))
#print ('atom 2 Fx', np.sum(arr[1+6]))
#print ('atom 3 Fx', np.sum(arr[1+9]))
