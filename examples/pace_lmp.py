import ase
from ase.io import read,write
import ctypes
import numpy as np
from lammps import lammps, PyLammps

def py_extract_ace(atoms_file_prefix,L,ndescs):
	#atoms = ASE atoms object
	atoms = read('ats.0.cfg')
        # always written to lammps data intermediate
	write('%s.data' % atoms_file_prefix,atoms,format='lammps-data')
	# atom info
	L.command('units  metal')
	L.command('atom_style  atomic')
	L.command('atom_modify  map array')
	# boundary
	L.command('boundary  p p p')
	# read atoms
	L.command('read_data  %s.data' % atoms_file_prefix)
	L.command('mass  1 183.84')#mass doesnt matter for run 0 and energy calculations rn...
        #  - mass per lammps type will need to be added for accurate force & virial terms
	# pairstyle 0 cutoff must be slightly larger than ACE cutoff
	L.command('pair_style      zero 7.502')
	L.command('pair_coeff  * *')
	L.command('compute         desc all pace coupling_coefficients.ace')
	L.command('timestep        0.001')
	L.command('thermo          1')

	thermo_cmd = 'thermo_style    custom step temp c_desc[1][1] '
	for i in range(1,ndescs):
		thermo_cmd = thermo_cmd + ' c_desc[1][%d] ' % (i+1)
	L.command(thermo_cmd)

	L.run(0)
	#result type is 0 for global data
	result_type = 0
	result_style = 2
	array_rows = (1 + (6*len(atoms)) + 6) 
	array_cols = ndescs 
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
print (np.shape(arr))
print (arr[0])
