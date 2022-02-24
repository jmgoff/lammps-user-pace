# lammps-user-pace

## Installation:

Before the multispecies PACE will be merged into main branch of the official LAMMPS repository, you could get the unofficial version of LAMMPS from [here](https://github.com/yury-lysogorskiy/lammps)
Clone this repo into your lammps directory.

### Build with `make`

Follow LAMMPS installation instructions

1. Go to `lammps/src` folder
2. Compile the ML-PACE library by running `make lib-pace args="-b"`
3. Include `ML-PACE` in the compilation by running `make yes-ml-pace`
4. Compile lammps as usual, i.e. `make serial` or `make mpi`.

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

```
cmake -DCMAKE_BUILD_TYPE=Release -DPKG_USER-PACE=ON ../cmake 
```

to your dflags or 

```
cmake -DCMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -DPKG_USER-PACE=ON ../cmake
```

4. After running cmake, copy the the modified ML-PACE code into the ace source folder:
(this copy and paste will be avoided after cmake files are updated or pull request to
official lammps-user-pace code has been made)


```
cp ../lammps-user-pace/ML-PACE/ace_evaluator* ./lammps-user-pace_<version>/ML-PACE
```

   
5. Build LAMMPS using `make` 

