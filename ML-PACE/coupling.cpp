#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "math_const.h"
#include "math_special.h"
#include "coupling.h"

using namespace std;
using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

long double sterling(int n){
    long double e= 2.718281828459045;
    long double p = pow(n,n);
    p*= pow(e,-n);
    p*= pow((2*MY_PI*n),(0.5));
    p*= (1 + (1/(12*n)) );
    return p;
}

long long binomialCoefficients(int n, int k){
    //VERIFIED
    long long numbc = factorial(n);
    long long denombc,kfac,diff_fac;

    kfac = factorial(k);
    diff_fac = factorial(n-k);

    denombc = kfac*diff_fac;
    if (n ==0 and k>0){
        return 0; // return 0 for 0 choose k!=0
    }
    if (n ==0 and k==0){
        return 1; // return 1 for n!=0 choose 0
    }
    else{
        return numbc/denombc;
    }
}

double clebsch_gordan(int l1, int m1, int l2, int m2, int l3, int m3){
    //VERIFIED
    //#rules:
    //bool rule1 = abs(l1-l2) <= l3;
    //bool rule2 = l3 <= l1+l2;
    //bool rule3 = m3 == m1 + m2;
    //bool rule4 = abs(m3) <= l3;

    long long G1,G2,G3,G4;
    //long double N=0.;
    double N=0.;

    long long N1,N2,N3,N4,N5,N6,N7;
    long long N1d, N2d, N3d, N4d;
    long double denom,num;
    //rules assumed by input
    //abs(m1) <= l1, 'm1 must bein {-l1,l1}'
    //abs(m2) <= l2, 'm2 must bein {-l2,l2}'

    // binomial representation
    N1 = (2*l3) + 1;
    N2 = factorial(l1 + m1);
    N3= factorial(l1 - m1);
    N4= factorial(l2 + m2);
    N5= factorial(l2 - m2);
    N6= factorial(l3 + m3);
    N7= factorial(l3 - m3);
    num = N1*N2*N3*N4*N5*N6*N7;

    N1d = factorial(l1 + l2 - l3);
    N2d = factorial(l1 - l2 + l3);
    N3d = factorial(-l1 + l2 + l3);
    N4d = factorial(l1 + l2 + l3 + 1);
    denom = N1d*N2d*N3d*N4d;
    N = (double)num/(double)denom;
    long long G = 0;

    // k <= l1 - m1
    // k <= l2 + m2
    for (int k =0 ; k<= fmin(l1-m1, l2+m2) ; k++ ){
        G1 = pow(-1,k);
        G2 = binomialCoefficients(l1 + l2 - l3, k);
        G3 = binomialCoefficients(l1 - l2 + l3, l1 - m1 - k);
        G4 = binomialCoefficients(-l1 +l2 + l3, l2 + m2 - k);
        G = G+ ( G1*G2*G3*G4);
    }
    return pow(N,0.5)*(double)G;

}

double wigner_3j(int l1, int m1, int l2, int m2, int l3, int m3){
    //uses relation between Clebsch-Gordann coefficients and W-3j symbols to evaluate W-3j
    double cg = clebsch_gordan(l1,m1,l2,m2,l3,-m3);
    double numw3j = pow((-1),(l1-l2-m3));
    double denomw3j = pow(((2*l3) +1),0.5);
    return cg*(numw3j/denomw3j);
}

