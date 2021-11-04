#include <stdio.h>
#include <stdlib.h>

#ifndef cg_H
#define cg_H

using namespace LAMMPS_NS;

double clebsch_gordan(int l1, int m1, int l2, int m2, int l3, int m3);
double wigner_3j(int l1, int m1, int l2, int m2, int l3, int m3);
long long binomialCoefficients(int n, int k);
#endif
