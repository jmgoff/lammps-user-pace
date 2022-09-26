# lammps-user-pace

## Installation:

git clone -b develop git@github.com:lammps/lammps.git

For known compatible versions of lammps, use:
git checkout  e745c8aac4be7301a15f4f6ed458fd4f41aedf64
or 
git checkout 10068a0fdeafba786ad0a03d6aa282404e8c2423


### Build with `cmake`

1. Create build directory and go there with 

```
cd lammps
mkdir build
cd build
```

2. Copy additional ML-PACE files into the source folder

```
cp ../lammps-user-pace/additional_ML-PACE/* ../src/ML-PACE
```

3. Configure the lammps build, adding:

ML-PACE = ON

to your dflags or 


4. After running cmake, copy the the modified ML-PACE code into the ace source folder:
(this copy and paste will be avoided after cmake files are updated or pull request to
official lammps-user-pace code has been made)


```
cp ../lammps-user-pace/ML-PACE/ace_evaluator* ./lammps-user-pace_<version>/ML-PACE
```

   
5. Build LAMMPS using `make` 

