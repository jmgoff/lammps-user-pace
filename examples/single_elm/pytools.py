import ase
from ase.io import read,write
from ase.build import bulk
import ctypes
import numpy as np
from lammps import lammps, PyLammps

def py_extract_force(atoms,L):
	pot_file = 'UO2_potential'
	#ASE atom info
	fname ='lmp_struct_tmp'
	write('%s.data' %fname, atoms, format='lammps-data')

	L.command('units  metal')
	L.command('atom_style  atomic')
	L.command('atom_modify  map array')
	L.command('neighbor  2.3 bin')
	# boundary
	L.command('boundary  p p p')
	# read atoms
	L.command('read_data  %s.data' % fname )
	L.command('mass  1 15.999')
	L.command('mass  2 238.0289')
	L.command('pair_style      pace product')
	L.command('pair_coeff  * * %s.yace O U' % pot_file)
	L.command('timestep	0.0005')
	L.command('thermo	  1')

	L.run(0)

	forces = []
	positions = []
	for atid, at in enumerate(atoms):
		# access LAMMPS id
		lmp_id = L.atoms[atid].id
		# access LAMMPS forces,positions
		pos_i = L.atoms[atid].position
		force_i = L.atoms[atid].force
		positions.append(pos_i)
		forces.append(force_i)

	positions = np.array(positions)
	forces = np.array(forces)
	return positions,forces

def py_extract_ei(atoms,L,atind):
	pot_file = 'UO2_potential'
	#ASE atom info
	fname ='lmp_struct_tmp'
	write('%s.data' %fname, atoms, format='lammps-data')

	L.command('units  metal')
	L.command('atom_style  atomic')
	L.command('atom_modify  map array')
	L.command('neighbor  2.3 bin')
	# boundary
	L.command('boundary  p p p')
	# read atoms
	L.command('read_data  %s.data' % fname )
	#for ind,atom in enumerate(atoms):
	L.command('group sub id %s' % ' '.join([str(b+1) for b in range(len(atoms)) ]))
	L.command('mass  1 15.999')
	L.command('mass  2 238.0289')
	L.command('pair_style      pace product')
	L.command('pair_coeff  * * %s.yace O U' % pot_file)
	#for ind,atom in enumerate(atoms):
	L.command('compute	peratom%d sub pe/atom' % (atind))#,atind))
	#L.command('compute      eat all property/atom e')
	L.command('timestep	0.0005')
	L.command('thermo	  1')

	L.run(0)
	#result type is 0 for global data
	array_shape = (len(atoms),)
	total_size= np.prod(array_shape)
	#total_size = len(atoms)

	result_type = 1
	result_style = 1
	enlst = []
	#for ind,atom in enumerate(atoms):
	# array style is 1 for vec?
	ptr = lmp.extract_compute('peratom%d' %atind, result_type, result_style)
	print (ptr.contents)
	#buffer_ptr = ctypes.cast(ptr.contents, ctypes.POINTER(ctypes.c_double))
	buffer_ptr = ctypes.cast(ptr.contents, ctypes.c_double)
	array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
	enlst.append(array_np[0])
	array_np = np.array(en_lst)

	return array_np


def py_compute_ei(atoms,L,ndescs,coeff_arr):
	#ASE atom info
	fname ='lmp_struct_tmp'
	write('%s.data' % fname, atoms, format='lammps-data')

	L.command('units  metal')
	L.command('atom_style  atomic')
	L.command('atom_modify  map array')
	L.command('neighbor  2.3 bin')
	# boundary
	L.command('boundary  p p p')
	# read atoms
	L.command('read_data  %s.data' % fname )
	L.command('mass  1 15.999')
	L.command('mass  2 238.0289')
	L.command('pair_style      zero 7.502')
	L.command('pair_coeff  * *')
	L.command('compute	 desc all pace coupling_coefficients.yace 0 0 ')
	L.command('timestep	0.0005')
	L.command('thermo	  1')

	L.run(0)

	#result type is 0 for global data
	result_type = 0
	result_style = 2
	array_rows = (1 + (3*len(atoms)) + 6)
	array_cols = ndescs +1
	array_shape = (array_rows,array_cols)
	total_size= np.prod(array_shape)

	# array style is 2
	ptr = lmp.extract_compute('desc', result_type, result_style)
	buffer_ptr = ctypes.cast(ptr.contents, ctypes.POINTER(ctypes.c_double * total_size))
	array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
	array_np.shape = array_shape

	en_desc_gradients = array_np[0][:-1]
	en_per_at = en_desc_gradients * coeff_arr
	
	return en_per_at


from descriptor_calc_local.convert_configs import convert_fitsnap
atoms = read('UO2_ex.cif')
#both functions should use the same ASE atoms object
#atoms = bulk('Mo','bcc',a=3.3,cubic=True)
#atoms = atoms *(2,2,2)

# example usage for getting forces (must be calculated using potential file)
lmp = lammps()
L = PyLammps(ptr=lmp)
positions,forces = py_extract_force(atoms,L=L)
# returns np array of positions followed by np array of forces. Takes ASE atoms object and PyLAMMPS instance as input.


# example usage for ACE energy correction per atom (must be calculated using coupling file)
lmp = lammps()
L = PyLammps(ptr=lmp)
expansion_coefficients = np.load('coefficient_array.npy')
print (np.shape(expansion_coefficients))
ndescs = len(expansion_coefficients)
e_per_atom = py_compute_ei(atoms,L=L,ndescs=ndescs,coeff_arr=expansion_coefficients)
#e_per_atom = py_extract_ei(atoms,L=L,atind=2)
e0s = [-0.001114, -0.001114]
energies = []
for ind,atom in enumerate(atoms):
	if atoms[ind].symbol == 'O':
		energies.append(e_per_atom[ind] + e0s[0])
	elif atoms[ind].symbol == 'U':
		energies.append(e_per_atom[ind] +  e0s[1])
print (energies)
np.save('force_per_at.npy',forces)
np.save('en_per_at.npy',np.array(energies))
