"""
compute_snap_dgrad.py
Purpose: Demonstrate extraction of descriptor gradient (dB/dR) array from compute snap.
         Show that dBi/dRj components summed over neighbors i yields same output as regular compute snap with dgradflag = 0.
         This shows that the dBi/dRj components extracted with dgradflag = 1 are correct.
Serial syntax:
    python compute_snap_dgrad.py
Parallel syntax:
    mpirun -np 4 python compute_snap_dgrad.py
"""

from __future__ import print_function
import sys
import ctypes
import numpy as np
from lammps import lammps, LMP_TYPE_ARRAY, LMP_STYLE_GLOBAL

# get MPI settings from LAMMPS

lmp = lammps()
me = lmp.extract_setting("world_rank")
nprocs = lmp.extract_setting("world_size")

cmds = ["-screen", "none", "-log", "none"]
lmp = lammps(cmdargs = cmds)

def run_lammps(dgradflag):

    # simulation settings
    fname = 'Cu_ex'
    lmp.command("clear")
    lmp.command("info all out log")
    lmp.command('units  metal')
    lmp.command('atom_style  atomic')
    lmp.command("boundary	p p p")
    lmp.command("atom_modify	map hash")
    lmp.command('neighbor  2.3 bin')
        # boundary
    lmp.command('boundary  p p p')
        # read atoms
    lmp.command('read_data  %s.data' % fname )
    lmp.command('mass  1 63.546')

    lmp.command("displace_atoms 	all random 0.001 0.001 0.001 123456")

    # potential settings

    lmp.command(f"pair_style 	zero 7.5")
    lmp.command(f"pair_coeff 	* *")

    lmp.command(f"pair_style 	zbl {zblcutinner} {zblcutouter}")
    lmp.command(f"pair_coeff 	* * {zblz} {zblz}")

    # define compute snap

    if dgradflag:
	    lmp.command(f"compute 	pace all pace coupling_coefficients.yace 1 1")
    else:
	    lmp.command(f"compute 	pace all pace coupling_coefficients.yace 0 0")

    # run

    lmp.command(f"thermo 		100")
    lmp.command(f"run {nsteps}")

# declare simulation/structure variables

nsteps = 0
ntypes = 1

# declare compute pace variables


bikflag = 1

# NUMBER of descriptors
nd = 81

# define reference potential (from W - needs edit)

zblcutinner = 4.0
zblcutouter = 4.8
zblz = 73

# run lammps with dgradflag on

if me == 0:
    print("Running with dgradflag on")

dgradflag = 1
run_lammps(dgradflag)

# get global snap array
print ('array_type_ind',LMP_TYPE_ARRAY)
print ('global_LMP_ind',LMP_STYLE_GLOBAL)

# get global snap array
lmp_pace = lmp.numpy.extract_compute("pace", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY)
print ('global shape',np.shape(lmp_pace))
# print snap array to observe
#if (me==0):
#    np.savetxt("test_pace.dat", lmp_pace, fmt="%d %d %d %f %f %f %f %f")

# take out rows with zero column

# extract dBj/dRi (includes dBi/dRi)

natoms = lmp.get_natoms()
fref1 = lmp_pace[0:natoms,0:3].flatten()
eref1 = lmp_pace[-1,0]
dbdr_length = np.shape(lmp_pace)[0]-(natoms) - 1
#dbdr_length = np.shape(lmp_pace)[0]-(natoms) - 7
dBdR = lmp_pace[natoms:(natoms+dbdr_length),3:(nd+3)]
force_indices = lmp_pace[natoms:(natoms+dbdr_length),0:3].astype(np.int32)

# strip rows with all zero descriptor gradients to demonstrate how to save memory

nonzero_rows = lmp_pace[natoms:(natoms+dbdr_length),3:(nd+3)] != 0.0
nonzero_rows = np.any(nonzero_rows, axis=1)
dBdR = dBdR[nonzero_rows, :]
force_indices = force_indices[nonzero_rows,:]
dbdr_length = np.shape(dBdR)[0]

# sum over atoms i that j is a neighbor of, like dgradflag = 0 does.

array1 = np.zeros((3*natoms,nd))
for k in range(0,nd):
    for l in range(0,dbdr_length):
        i = force_indices[l,0]
        j = force_indices[l,1]
        a = force_indices[l,2]
        #print(f"{i} {j} {a}")
        array1[3 * j + a, k] += dBdR[l,k]
print ('array1 shape',np.shape(array1))
# run lammps with dgradflag off

if me == 0:
    print("Running with dgradflag off")

dgradflag = 0
run_lammps(dgradflag)

# get global snap array

lmp_pace = lmp.numpy.extract_compute("pace", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY)
print ('sum shape',np.shape(lmp_pace))
natoms = lmp.get_natoms()
#fref2 = lmp_pace[natoms:(natoms+3*natoms),-1]
#fref2 = lmp_pace[1:(natoms+3*natoms),-1]
#fref2 = lmp_pace[0:(3*natoms),-1]
fref2 = lmp_pace[1:-6,:]
fref2 = fref2[:,-1]
print ('fref2 shape',np.shape(fref2))
#fref2 = lmp_pace[natoms:(natoms+3*natoms),-7]
eref2 = lmp_pace[0,-1]
#array2 = lmp_pace[1:natoms+(3*natoms) -6, nd:-1]
array2 = lmp_pace[1:-6,:-1]
print ('array2 shape', np.shape(array2))
#array2 = lmp_pace[natoms:natoms+(3*natoms), nd:-1]#drews

# take difference of arrays obtained from dgradflag on and off.

diffm = array1 - array2
difff = fref1 - fref2
diffe = eref1 - eref2

if me == 0:
    print(f"Max/min difference in dSum(Bi)/dRj: {np.max(diffm)} {np.min(diffm)}")
    print(f"Max/min difference in reference forces: {np.max(difff)} {np.min(difff)}")
    print(f"Difference in reference energy: {diffe}")
