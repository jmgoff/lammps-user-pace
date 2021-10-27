#Build note for lammps with compute_pace

First download lammps from git

git clone -b unstable https://github.com/lammps/lammps.git 
mylammps_pace

or with SSH

git clone -b unstable git@github.com:lammps/lammps.git mylammps_pace

Next, set up initial lammps build by downloading the compute_pace enabled lammps-user-pace package move this package to your build directory 

cd lammps-user-pace
chmod +x setup.sh
./setup.sh

Now use cmake to install the default version of ML-PACE (for fitsnap compatibility, use the following flags) – Make sure you are in the build folder first, then execute:

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/mylammps_pace/build

cmake -D LAMMPS_EXCEPTIONS=on -D PKG_PYTHON=on -D BUILD_SHARED_LIBS=on -D CMAKE_BUILD_TYPE=Release -D PKG_ML-IAP=on -D PKG_ML-PACE=on -D PKG_ML-SNAP=on -D BUILD_MPI=on -D BUILD_OMP=off ../cmake/

(Note: this may result in a hash mismatch error for the pace library. If this results in an MD5 hash mismatch error, then look in the error message for the actual hash, copy it, and replace the corresponding line in the ML-PACE cmake file (found in  ../cmake/Modules/Packages/ML-PACE.cmake) 
If you are installing on a SRN cluster, there may be problems with downloading the libpace tarball from http. If you get an error “no rule to make pace”, then change the address in the ML-PACE.cmake file:
https://github.com/ICAMS/... ->git@github.com:ICAMS/...

Modules and settings used for tested build: (blake)

module load intel/compilers/20.2.254
module load openmpi/4.0.5/intel/20.2.254
module load cmake/3.12.3

anaconda (python 3.8.8)

