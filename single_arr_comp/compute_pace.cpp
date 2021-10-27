// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_pace.h"
#include "ace_evaluator.h"
#include "ace_c_basis.h"
#include "ace_abstract_basis.h"
#include "ace_types.h"
#include <cstring>

#include "atom.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
//namespace LAMMPS_NS {
//  class ACECTildeBasisSet basis_set(nullptr);
//  class ACECTildeEvaluator ace(nullptr);
//}
  
using namespace LAMMPS_NS;

enum{SCALAR,VECTOR,ARRAY};

ComputePACE::ComputePACE(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), cutsq(nullptr), list(nullptr), pace(nullptr),
  paceall(nullptr), pace_peratom(nullptr), radelem(nullptr), wjelem(nullptr),
  basis_set(nullptr),ace(nullptr)
{

  array_flag = 1;
  extarray = 0;

  double rfac0, rmin0;
  int twojmax, switchflag, bzeroflag, bnormflag, wselfallflag;

  int ntypes = atom->ntypes;
  int nargmin = 6+2*ntypes;

  if (narg < nargmin) error->all(FLERR,"Illegal compute pace command");

  // default values

  rmin0 = 0.0;
  switchflag = 1;
  bzeroflag = 1;
  chemflag = 0;
  bnormflag = 0;
  wselfallflag = 0;
  nelements = 1;

  // process required arguments

  memory->create(radelem,ntypes+1,"pace:radelem"); // offset by 1 to match up with types
  memory->create(wjelem,ntypes+1,"pace:wjelem");
  //memory->create(map,ntypes+1,"compute_pace:map");
  rcutfac = atof(arg[3]);
  rfac0 = atof(arg[4]);
  twojmax = atoi(arg[5]);
  for (int i = 0; i < ntypes; i++)
    radelem[i+1] = atof(arg[6+i]);
  for (int i = 0; i < ntypes; i++)
    wjelem[i+1] = atof(arg[6+ntypes+i]);

  // construct cutsq

  double cut;
  cutmax = 0.0;
  memory->create(cutsq,ntypes+1,ntypes+1,"pace:cutsq");
  for (int i = 1; i <= ntypes; i++) {
    cut = 2.0*radelem[i]*rcutfac;
    if (cut > cutmax) cutmax = cut;
    cutsq[i][i] = cut*cut;
    for (int j = i+1; j <= ntypes; j++) {
      cut = (radelem[i]+radelem[j])*rcutfac;
      cutsq[i][j] = cutsq[j][i] = cut*cut;
    }
  }

  // process optional args

  int iarg = nargmin;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"rmin0") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute pace command");
      rmin0 = atof(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"bzeroflag") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute pace command");
      bzeroflag = atoi(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"switchflag") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute pace command");
      switchflag = atoi(arg[iarg+1]);
      iarg += 2; 
    } else if (strcmp(arg[iarg],"bnormflag") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute pace command");
      bnormflag = atoi(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"wselfallflag") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute pace command");
      wselfallflag = atoi(arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal compute pace command");
  }
  //------------------------------------------------------------
  // this block added for ACE implementation

  basis_set = new ACECTildeBasisSet;
  //ace = new ACECTildeEvaluator;
  //read in dummy file with coefficients
  basis_set->load("coupling_coefficients.ace");
  //printf("after loading .ace : cutoff = %f\n",basis_set->radial_functions->cut(0, 0));
  //ace->set_basis(*basis_set);
  //manually set mu index to 0 (for single component system)
  //! try offsetting?
  SPECIES_TYPE mu = 0;
  //# of rank 1 functions
  const int total_basis_size_rank1 = basis_set->total_basis_size_rank1[mu];
  //# of rank 2+ functions
  const int total_basis_size = basis_set->total_basis_size[mu];
  
  int ncoeff = total_basis_size_rank1 + total_basis_size;
  printf("$! number of coefficients: %d\n",ncoeff);

  //! set up B_array
  //rank 2+
  //Array3D<DOUBLE_TYPE> Bs =ace->B_arr;
  //rank1 only
  //Array2D<DOUBLE_TYPE> B1s=ace->B1_arr;
  //-----------------------------------------------------------
  nperdim = ncoeff;
  ndims_force = 3;
  ndims_virial = 6;
  yoffset = nperdim;
  zoffset = 2*nperdim;
  natoms = atom->natoms;
  size_array_rows = 1+ndims_force*natoms+ndims_virial;
  size_array_cols = nperdim*atom->ntypes+1;
  lastcol = size_array_cols-1;

  ndims_peratom = ndims_force;
  size_peratom = ndims_peratom*nperdim*atom->ntypes;

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputePACE::~ComputePACE()
{
  memory->destroy(pace);
  memory->destroy(paceall);
  memory->destroy(pace_peratom);
  memory->destroy(radelem);
  memory->destroy(wjelem);
  memory->destroy(cutsq);
  //delete basis_set;
  //delete ace;
  //if (chemflag) memory->destroy(map);
}

/* ---------------------------------------------------------------------- */

void ComputePACE::init()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Compute pace requires a pair style be defined");

  if (cutmax > force->pair->cutforce)
    error->all(FLERR,"Compute pace cutoff is longer than pairwise cutoff");

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"pace") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute pace");
  //  snaptr->init();

  // allocate memory for global array

  memory->create(pace,size_array_rows,size_array_cols,
                 "pace:pace");
  memory->create(paceall,size_array_rows,size_array_cols,
                 "pace:paceall");
  array = paceall;

  // find compute for reference energy

  //std::string id_pe = std::string("thermo_pe");
  //int ipe = modify->find_compute(id_pe);
  //if (ipe == -1)
  //  error->all(FLERR,"compute thermo_pe does not exist.");
  //c_pe = modify->compute[ipe];

  // add compute for reference virial tensor

  //std::string id_virial = std::string("pace_press");
  //std::string pcmd = id_virial + " all pressure NULL virial";
  //modify->add_compute(pcmd);

  //int ivirial = modify->find_compute(id_virial);
  //if (ivirial == -1)
  //  error->all(FLERR,"compute pace_press does not exist.");
  //c_virial = modify->compute[ivirial];

}


/* ---------------------------------------------------------------------- */

void ComputePACE::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputePACE::compute_array()
{
  int ntotal = atom->nlocal + atom->nghost;

  invoked_array = update->ntimestep;

  // grow pace_peratom array if necessary
  if (atom->nmax > nmax) {
    printf("atnmax =%d , nmax = %d\n", atom->nmax,nmax);
    memory->destroy(pace_peratom);
    nmax = atom->nmax;
    memory->create(pace_peratom,nmax,size_peratom,
                   "pace:pace_peratom");
  }

  // clear global array

  for (int irow = 0; irow < size_array_rows; irow++)
    for (int icoeff = 0; icoeff < size_array_cols; icoeff++)
      pace[irow][icoeff] = 0.0;

  // clear local peratom array

  for (int i = 0; i < ntotal; i++)
    for (int icoeff = 0; icoeff < size_peratom; icoeff++) {
      pace_peratom[i][icoeff] = 0.0;
    }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);
  SPECIES_TYPE *mus;
  NS_TYPE *ns;
  LS_TYPE *ls;
  const int inum = list->inum;
  const int* const ilist = list->ilist;
  const int* const numneigh = list->numneigh;
  int** const firstneigh = list->firstneigh;
  int * const type = atom->type;
  //! added
  //determine the maximum number of neighbours
  int max_jnum = -1;
  int nei = 0;
  int jtmp =0;
  for (int iitmp = 0; iitmp < list->inum; iitmp++) {
    int itmp = ilist[iitmp];
    jtmp = numneigh[itmp];
    nei = nei + jtmp;
    if (jtmp > max_jnum)
      max_jnum = jtmp;
  }
  printf("max_jnum after loop over atoms: %d\n" , max_jnum);
  // compute pace derivatives for each atom in group
  // use full neighbor list to count atoms less than cutoff
  SPECIES_TYPE mu =0;
  const int total_basis_size_rank1 = basis_set->total_basis_size_rank1[mu];
  //# of rank 2+ functions
  const int total_basis_size = basis_set->total_basis_size[mu];
  printf("total basis size in compute_array: %d\n", total_basis_size + total_basis_size_rank1);
  double** const x = atom->x;
  const int* const mask = atom->mask;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (mask[i] & groupbit) {

      const double xtmp = x[i][0];
      const double ytmp = x[i][1];
      const double ztmp = x[i][2];
      const int itype = type[i];
      int ielem = 0;
      if (chemflag)
        ielem = map[itype];
      const double radi = radelem[itype];
      const int* const jlist = firstneigh[i];
      const int jnum = numneigh[i];
      const int typeoffset_local = ndims_peratom*nperdim*(itype-1);
      const int typeoffset_global = nperdim*(itype-1);

      // insure rij, inside, and typej  are of size jnum

      //      snaptr->grow_rij(jnum);

      // rij[][3] = displacements between atom I and those neighbors
      // inside = indices of neighbors of I within cutoff
      // typej = types of neighbors of I within cutoff
      // note Rij sign convention => dU/dRij = dU/dRj = -dU/dRi

      int ninside = 0;
      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj];
        j &= NEIGHMASK;

        const double delx = x[j][0] - xtmp;
        const double dely = x[j][1] - ytmp;
        const double delz = x[j][2] - ztmp;
        const double rsq = delx*delx + dely*dely + delz*delz;
        int jtype = type[j];
        int jelem = 0;
        if (chemflag)
          jelem = map[jtype];
        if (rsq < cutsq[itype][jtype]&&rsq>1e-20) {
          // snaptr->rij[ninside][0] = delx;
          // snaptr->rij[ninside][1] = dely;
          // snaptr->rij[ninside][2] = delz;
          // snaptr->inside[ninside] = j;
          // snaptr->wj[ninside] = wjelem[jtype];
          // snaptr->rcutij[ninside] = (radi+radelem[jtype])*rcutfac;
          // snaptr->element[ninside] = jelem; // element index for multi-element snap
          ninside++;
        }
      }

      // snaptr->compute_ui(ninside, ielem);
      // snaptr->compute_zi();
      // snaptr->compute_bi(ielem);

      //      for (int jj = 0; jj < ninside; jj++) {
	//        const int j = snaptr->inside[jj];
        // snaptr->compute_duidrj(snaptr->rij[jj], snaptr->wj[jj],
        //                             snaptr->rcutij[jj], jj, snaptr->element[jj]);
        // snaptr->compute_dbidrj();

        // Accumulate dBi/dRi, -dBi/dRj

        // double *snadi = pace_peratom[i]+typeoffset_local;
        // double *snadj = pace_peratom[j]+typeoffset_local;

        // for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
          // snadi[icoeff] += snaptr->dblist[icoeff][0];
          // snadi[icoeff+yoffset] += snaptr->dblist[icoeff][1];
          // snadi[icoeff+zoffset] += snaptr->dblist[icoeff][2];
          // snadj[icoeff] -= snaptr->dblist[icoeff][0];
          // snadj[icoeff+yoffset] -= snaptr->dblist[icoeff][1];
          // snadj[icoeff+zoffset] -= snaptr->dblist[icoeff][2];
      //        }

      // }

      // Accumulate Bi

      // linear contributions
      int k = typeoffset_global;
      //! not sure if I'm reassigning these correctly
      //basis_set = new ACECTildeBasisSet;
      ace = new ACECTildeEvaluator;
      ace->element_type_mapping.init(atom->ntypes+1);
      ace->set_basis(*basis_set);
      // resize the neighbor cache after setting the basis
      ace->resize_neighbours_cache(max_jnum);
      //! TODO add check to see if jnum == jnum(pace)
      // the rc in ace may be different than the lammps neighbor cutoff?
      //jnum =numneigh[i]
      //TODO turn into for loop over species types for multicomponent implementation
      SPECIES_TYPE * mu_init = 0;
      //ace->compute_atom(i, atom->x, mu_init, list->numneigh[i], list->firstneigh[i]);
      ace->compute_atom(i, atom->x, atom->type, list->numneigh[i], list->firstneigh[i]);
      //read in dummy file with coefficients
      //basis_set->load("coupling_coefficients.ace");
      //! set up per_atom B_array
      SPECIES_TYPE mu = 0;
      //try allocating a 2d array for only 1 species
      //Array3D<DOUBLE_TYPE> Bs =ace->B_arr;
      Array1D<DOUBLE_TYPE> Bs=ace->B_all;
      //Array2D<DOUBLE_TYPE> B1s=ace->B1_arr;
      //for (int icoeff_r1 = 0; icoeff_r1 < total_basis_size_rank1; icoeff_r1++){
      //  pace[0][k++] += B1s(mu,icoeff_r1);
        //printf("atom %d , B1 contribution %f , icoeff %d \n", i, B1s(mu,icoeff_r1),icoeff_r1 );
      //}
      //rank>1
      for (int icoeff = 0; icoeff < total_basis_size+total_basis_size_rank1; icoeff++){
        //! todo check that icoeff here maps to the correct func_ind
        //ACECTildeBasisFunction *func = &basis_set->basis[mu][icoeff];
        //ACECTildeBasisFunction func = basis_set->basis[mu][icoeff];
        //int rank =basis_set->basis[mu][icoeff].rank;
        //mus= func->mus;
        //ns = func->ns;
        //ls = func->ls;
	//pace[0][k++] += Bs(basis_set->basis[mu][icoeff].mus[mu],basis_set->basis[mu][icoeff].ns[rank],basis_set->basis[mu][icoeff].ls[rank]);
	pace[0][k++] += Bs(icoeff);
	//pace[0][k++] += Bs(func.mus[mu],func.ns[rank],func.ls[rank]);
	//pace[0][k++] += Bs(mus[mu],ns[rank],ls[rank]);
        //printf("atom %d , B contribution %f , icoeff %d \n", i, Bs(func->mus[mu],func->ns[rank],func->ls[rank]),icoeff );
      }
      //delete ace;
    }
  }

  // accumulate bispectrum force contributions to global array

  for (int itype = 0; itype < atom->ntypes; itype++) {
    const int typeoffset_local = ndims_peratom*nperdim*itype;
    const int typeoffset_global = nperdim*itype;
    for (int icoeff = 0; icoeff < nperdim; icoeff++) {
      for (int i = 0; i < ntotal; i++) {
        double *snadi = pace_peratom[i]+typeoffset_local;
        int iglobal = atom->tag[i];
        int irow = 3*(iglobal-1)+1;
        pace[irow][icoeff+typeoffset_global] += snadi[icoeff];
        pace[irow+1][icoeff+typeoffset_global] += snadi[icoeff+yoffset];
        pace[irow+2][icoeff+typeoffset_global] += snadi[icoeff+zoffset];
      }
    }
  }

 // accumulate forces to global array

  for (int i = 0; i < atom->nlocal; i++) {
    int iglobal = atom->tag[i];
    int irow = 3*(iglobal-1)+1;
    pace[irow][lastcol] = atom->f[i][0];
    pace[irow+1][lastcol] = atom->f[i][1];
    pace[irow+2][lastcol] = atom->f[i][2];
  }

  // accumulate bispectrum virial contributions to global array

  //dbdotr_compute();

  // sum up over all processes

  MPI_Allreduce(&pace[0][0],&paceall[0][0],size_array_rows*size_array_cols,MPI_DOUBLE,MPI_SUM,world);

  // assign energy to last column

  //TODO get reference energy working
  //int irow = 0;
  //double reference_energy = c_pe->compute_scalar();
  //paceall[irow++][lastcol] = reference_energy;

  // assign virial stress to last column
  // switch to Voigt notation

  /*c_virial->compute_vector();
  irow += 3*natoms;
  paceall[irow++][lastcol] = c_virial->vector[0];
  paceall[irow++][lastcol] = c_virial->vector[1];
  paceall[irow++][lastcol] = c_virial->vector[2];
  paceall[irow++][lastcol] = c_virial->vector[5];
  paceall[irow++][lastcol] = c_virial->vector[4];
  paceall[irow++][lastcol] = c_virial->vector[3];
  */
}

/* ----------------------------------------------------------------------
   compute global virial contributions via summing r_i.dB^j/dr_i over
   own & ghost atoms
------------------------------------------------------------------------- */

void ComputePACE::dbdotr_compute()
{
  double **x = atom->x;
  int irow0 = 1+ndims_force*natoms;

  // sum over bispectrum contributions to forces
  // on all particles including ghosts

  int nall = atom->nlocal + atom->nghost;
  for (int i = 0; i < nall; i++)
    for (int itype = 0; itype < atom->ntypes; itype++) {
      const int typeoffset_local = ndims_peratom*nperdim*itype;
      const int typeoffset_global = nperdim*itype;
      double *snadi = pace_peratom[i]+typeoffset_local;
      for (int icoeff = 0; icoeff < nperdim; icoeff++) {
        double dbdx = snadi[icoeff];
        double dbdy = snadi[icoeff+yoffset];
        double dbdz = snadi[icoeff+zoffset];
        int irow = irow0;
        pace[irow++][icoeff+typeoffset_global] += dbdx*x[i][0];
        pace[irow++][icoeff+typeoffset_global] += dbdy*x[i][1];
        pace[irow++][icoeff+typeoffset_global] += dbdz*x[i][2];
        pace[irow++][icoeff+typeoffset_global] += dbdz*x[i][1];
        pace[irow++][icoeff+typeoffset_global] += dbdz*x[i][0];
        pace[irow++][icoeff+typeoffset_global] += dbdy*x[i][0];
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double ComputePACE::memory_usage()
{
  printf("nmax %d\n", nmax);
  double bytes = (double)size_array_rows*size_array_cols *
    sizeof(double);                                     // pace
  bytes += (double)size_array_rows*size_array_cols *
    sizeof(double);                                     // paceall
  bytes += (double)nmax*size_peratom * sizeof(double);  // pace_peratom
  //bytes += snaptr->memory_usage();                      // no equivalent for ACEEvaluator object!
  int n = atom->ntypes+1;
  bytes += (double)n*sizeof(int);        // map

  return bytes;
}
