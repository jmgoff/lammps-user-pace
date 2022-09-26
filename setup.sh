#this will overwrite the existing lammps/src/ML-PACE contents
cp ./additional_ML-PACE/* ../../src/ML-PACE
#this will overwrite the existing lammps/cmake/Modules/Packages/ML-PACE.cmake file (only needed for older versions of LAMMPS)
cp ./cmake/* ../../cmake/Modules/Packages/
