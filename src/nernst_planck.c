/*****************************************************************************
 *
 *  nernst_planck.c
 *
 *  A solution for the Nerst-Planck equation, which is the advection
 *  diffusion equation for charged species \rho_k in the presence of
 *  a potential \psi.
 *
 *  We have, in the most simple case:
 *
 *  d_t rho_k + div . (rho_k u) = div . D_k (grad rho_k + Z_k rho_k grad psi)
 *
 *  where u is the velocity field, D_k are the diffusion constants for
 *  each species k, and Z_k = (valancy_k e / k_bT) = beta valency_k e.
 *  e is the unit charge.
 *
 *  If the chemical potential is mu_k for species k, the diffusive
 *  flux may be written as
 *
 *    j_k = - D_k rho_k grad (beta mu_k)
 *
 *  with mu_k = mu_k^ideal + mu_k^ex = k_bT ln(rho_k) + valency_k e psi.
 *  (For more complex problems, there may be other terms in the chemical
 *  potential.)
 *
 *  As it is important to conserve charge, we solve in a flux form.
 *  Following Capuani, Pagonabarraga and Frenkel, J. Chem. Phys.
 *  \textbf{121} 973 (2004) we include factors to ensure that the
 *  charge densities follow a Boltzmann distribution in equilbrium.
 *
 *  This writes the flux as
 *    j_k = - D_k exp[beta mu_k^ex] grad (rho_k exp[beta mu_k^ex])
 *
 *  which we approximate at the cell faces by (e.g., for x only)
 *
 *    -D_k (1/2) { exp[-beta mu_k^ex(i)] + exp[-beta mu_k^ex(i+1)] }
 *    * { rho_k(i+1) exp[beta mu_k^ex(i+1)] - rho_k(i) exp[beta mu_k^ex(i)] }
 *
 *  We then compute the divergence of the fluxes to update via an
 *  Euler forward step. The advective fluxes (again computed at the
 *  cells faces) may be added to the diffusive fluxes to solve the
 *  whole thing. Appropraite advective fluxes may be computed via
 *  the advection.h interface.
 *
 *  Solid boundaries simply involve enforcing a no normal flux
 *  condition at the cell face.
 *
 *  The potential and charge species are available via the psi_s
 *  object.
 *
 *  A uniform external electric field may be applied; this is done
 *  by adding a contribution to the potential
 *     psi -> psi - eE.r
 *  which just appears as -eE in the calculation of grad psi.
 *
 *
 *  Edinbrugh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "advection.h"
#include "advection_bcs.h"
#include "nernst_planck.h"
#include "psi_gradients.h"
#include "psi_init.h"
#include "psi_rt.h"
#include "phi_cahn_hilliard.h"
#include "field.h"
#include "field_options.h"



int * en_filter = NULL;

/* This needs an input switch to make it active. */
int nernst_planck_fluxes_force_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
		map_t * map, colloids_info_t * cinfo, double ** flx);

static int nernst_planck_fluxes(psi_t * psi, fe_t * fel, double * fx,
				double * fy,
				double * fz);
static int nernst_planck_no_flux_condition(map_t * map, int nk, double * fx,
					   double * fy, double * fz);
static int nernst_planck_update(psi_t * psi, double * fe, double * fy,
				double * fz);
static int nernst_planck_fluxes_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
		map_t * map, colloids_info_t * cinfo, double ** flx);
static int nernst_planck_update_d3qx(psi_t * psi, 
				map_t * map, double ** flx);

static int charge_update_sanity_check(psi_t * psi, map_t * map, double ** flx, int timestep);

/* Statistiche cumulative del flux eraser (identiche su tutti i rank) */
static struct {
  long ncalls;          /* chiamate totali (piu' di una per ts se multistep) */
  long nts;             /* timestep distinti controllati */
  int  last_ts;
  long ncalls_active;   /* chiamate in cui almeno un link e' stato azzerato */
  long cells;           /* celle/specie marcate, sommate sui pass */
  long links;           /* link entries azzerate */
  long links_max;       /* massimo in una singola chiamata */
  int  ts_max;
  long nentries;        /* nk * nfluid * (npoints - 1) */
} np_eraser = {0, 0, -1, 0, 0, 0, 0, -1, 0};
static double max_acc; 

int flx_noise(psi_t * psi, double ** flx, noise_t * noise,  map_t * map);
int np_advective_fluxes(psi_t * psi, hydro_t * hydro, double ** flux);
int np_no_flux_boundary(psi_t * psi, map_t * map, double ** flux);
int ef_advective(psi_t * psi, double ** flx);
static int np_flux_io_write(psi_t * psi, double ** flx, int timestep);

/*****************************************************************************
 *
 *  nernst_planck_driver
 *
 *  No hydrodynamics (use nernst_plancj_driver_d3qx instead).
 *  Allows for a no-flux condition between solid and fluid sites.
 *
 *****************************************************************************/

int nernst_planck_driver(psi_t * psi, fe_t * fel, map_t * map) {

  int nk;              /* Number of electrolyte species */
  int nsites;          /* Number of lattice sites */

  double * fe = NULL;
  double * fy = NULL;
  double * fz = NULL;

  psi_nk(psi, &nk);
  cs_nsites(psi->cs, &nsites);

  /* Allocate fluxes and initialise to zero */

  fe = (double*) calloc((size_t) nsites*nk, sizeof(double));
  fy = (double*) calloc((size_t) nsites*nk, sizeof(double));
  fz = (double*) calloc((size_t) nsites*nk, sizeof(double));

  if (fe == NULL) pe_fatal(psi->pe, "calloc(fe) failed\n");
  if (fy == NULL) pe_fatal(psi->pe, "calloc(fy) failed\n");
  if (fz == NULL) pe_fatal(psi->pe, "calloc(fz) failed\n");

  /* Diffusive fluxes based on six-point stencil, followed
   * by no-flux condition for solid-fluid boundaries. */

  nernst_planck_fluxes(psi, fel, fe, fy, fz);
  nernst_planck_no_flux_condition(map, nk, fe, fy, fz);

  /* Update charge distribution */

  nernst_planck_update(psi, fe, fy, fz);

  free(fz);
  free(fy);
  free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_fluxes
 *
 *  Compute diffusive fluxes via a simple 7-point stencil in 3d.
 *
 *****************************************************************************/

static int nernst_planck_fluxes(psi_t * psi, fe_t * fel, double * fx,
				double * fy,
				double * fz) {
  int ic, jc, kc, index;
  int nlocal[3];
  int nsites;
  int zs, ys, xs;
  int n, nk;

  double eunit, reunit;
  double b0, b1;
  double mu0, mu1;
  double rho0, rho1;
  double mu_s0, mu_s1;   /* Solvation chemical potential, from free energy */

  assert(psi);
  assert(fx);
  assert(fy);
  assert(fz);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);
  cs_strides(psi->cs, &xs, &ys, &zs);

  double kt = 0.0;
  physics_t * phys = NULL;
  physics_ref(&phys);
  physics_kt(phys, &kt);

  psi_nk(psi, &nk);
  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);

	for (n = 0; n < nk; n++) {

	  fel->func->mu_solv(fel, index, n, &mu_s0);
	  mu0 = reunit*mu_s0
	    + psi->valency[n]*psi->psi->data[addr_rank0(nsites, index)];
	  rho0 = psi->rho->data[addr_rank1(nsites, nk, index, n)];

	  /* x-direction (between ic and ic+1) */

	  fel->func->mu_solv(fel, index + xs, n, &mu_s1);
	  mu1 = reunit*mu_s1
	    + psi->valency[n]*psi->psi->data[addr_rank0(nsites, index + xs)];

	  b0 = exp(mu1 - mu0);
	  b1 = exp(mu1 - mu0);
	  rho1 = psi->rho->data[addr_rank1(nsites, nk, (index + xs), n)]*b1;

	  fx[addr_rank1(nsites, nk, index, n)]
	    = -psi->mobility_elec[n]*kt*0.5*(1.0 + b0)*(rho1 - rho0);

	  /* y-direction (between jc and jc+1) */

	  fel->func->mu_solv(fel, index + ys, n, &mu_s1);
	  mu1 = reunit*mu_s1
	    + psi->valency[n]*psi->psi->data[addr_rank0(nsites, index + ys)];

	  b0 = exp(mu1 - mu0);
	  b1 = exp(mu1 - mu0);
	  rho1 = psi->rho->data[addr_rank1(nsites, nk, (index + ys), n)]*b1;

	  fy[nk*index + n] = -psi->mobility_elec[n]*kt*0.5*(1.0 + b0)*(rho1 - rho0);

	  /* z-direction (between kc and kc+1) */

	  fel->func->mu_solv(fel, index + zs, n, &mu_s1);
	  mu1 = reunit*mu_s1
	    + psi->valency[n]*psi->psi->data[addr_rank0(nsites, index + zs)];

	  b0 = exp(mu1 - mu0);
	  b1 = exp(mu1 - mu0);
	  rho1 = psi->rho->data[addr_rank1(nsites, nk, (index + zs), n)]*b1;

	  fz[addr_rank1(nsites, nk, index, n)]
	    = -psi->mobility_elec[n]*kt*0.5*(1.0 + b0)*(rho1 - rho0);
	}

	/* Next face */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_no_flux_condition
 *
 *  Use map to determine solid/fluid boudaries and set no normal flux.
 *
 *****************************************************************************/

static int nernst_planck_no_flux_condition(map_t * map, int nk, double * fx,
					   double * fy, double * fz) {

  cs_t * cs = NULL;
  int nsites = 0;
  int nlocal[3] = {0};

  assert(map);
  assert(fx);
  assert(fy);
  assert(fz);

  cs = map->cs;
  cs_nsites(cs, &nsites);
  cs_nlocal(cs, nlocal);

  for (int ic = 0; ic <= nlocal[X]; ic++) {
    for (int jc = 0; jc <= nlocal[Y]; jc++) {
      for (int kc = 0; kc <= nlocal[Z]; kc++) {
	int index = cs_index(cs, ic, jc, kc);
	int status = MAP_FLUID;
	map_status(map, index, &status);
	if (status == MAP_BOUNDARY) {
	  for (int n = 0; n < nk; n++) {
	    fx[addr_rank1(nsites, nk, index, n)] = 0.0;
	    fy[addr_rank1(nsites, nk, index, n)] = 0.0;
	    fz[addr_rank1(nsites, nk, index, n)] = 0.0;
	  }
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_update
 *
 *  Update the rho_k from the fluxes. Euler forward step.
 *
 *****************************************************************************/

static int nernst_planck_update(psi_t * psi, double * fx, double * fy,
				double * fz) {
  int ic, jc, kc, index;
  int nlocal[3];
  int zs, ys, xs;
  int n, nk;
  
  double dt;

  assert(psi);
  assert(fx);
  assert(fy);
  assert(fz);

  cs_nlocal(psi->cs, nlocal);
  cs_strides(psi->cs, &xs, &ys, &zs);

  psi_nk(psi, &nk);
  psi_multistep_timestep(psi, &dt);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);

	for (n = 0; n < nk; n++) {
	  psi->rho->data[addr_rank1(psi->nsites, nk, index, n)]
	    -= (+ fx[addr_rank1(psi->nsites, nk, index, n)]
		- fx[addr_rank1(psi->nsites, nk, (index-xs), n)]
		+ fy[addr_rank1(psi->nsites, nk, index, n)]
		- fy[addr_rank1(psi->nsites, nk, (index-ys), n)]
		+ fz[addr_rank1(psi->nsites, nk, index, n)]
		- fz[addr_rank1(psi->nsites, nk, (index-zs), n)])*dt;
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_driver_d3qx
 *
 *  The hydro object is allowed to be NULL, in which case there is
 *  no advection.
 *
 *  The map object is allowed to be NULL, in which case no boundary
 *  condition corrections are attempted.
 *
 *****************************************************************************/

int nernst_planck_driver_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
			      map_t * map, colloids_info_t * cinfo, int print_current_flag, int timestep, noise_t * noise) {

  int n, nk;              /* Number of electrolyte species */
  int ia;
  
  double ** flx = NULL;

  psi_nk(psi, &nk);

  /* Allocate fluxes and initialise to zero */
  flx = (double **) calloc((size_t) psi->nsites*nk, sizeof(double *));
  assert(flx);
  if (flx == NULL) pe_fatal(psi->pe, "calloc(flx) failed\n");

  for (ia = 0; ia < psi->nsites*nk; ia++) {
    int nflux = psi->stencil->npoints;
    flx[ia] = (double *) calloc(nflux-1, sizeof(double));
    assert(flx[ia]);
    if (flx[ia] == NULL) pe_fatal(psi->pe, "calloc(flx[]) failed\n");
  }
  
  /* Add thermal fluxes */
  if (noise) flx_noise(psi, flx, noise, map);
  

  

  /* Add fluxes for advective field*/
  //ef_advective(psi, flx);
  /*

  if (print_current_flag == 1) {
    
	
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char filename[64];
    char filename_p[64];
    char filename_m[64];

    char ts[10]; // 9 cifre + terminatore null
    sprintf(ts, "%09d", timestep);
    //sprintf(filename, "fluxes_%s.txt", ts);
    sprintf(filename, "fluxes_EFadv_%s-rank%04d.txt", ts, rank);
    sprintf(filename_p, "fluxes_EFadv_p_%s-rank%04d.txt", ts, rank);
    sprintf(filename_m, "fluxes_EFadv_m_%s-rank%04d.txt", ts, rank);

    FILE *fp = fopen(filename, "w");
    FILE *fp_m = fopen(filename_m, "w");
    FILE *fp_p = fopen(filename_p, "w");

    if (fp == NULL) {
      perror("Errore nell'apertura del file flussi_correnti.dat");
      exit(EXIT_FAILURE);
    }
    
    if (fp_m == NULL) {
      perror("Errore nell'apertura del file flussi_correnti.dat");
      exit(EXIT_FAILURE);
    }

    if (fp_p == NULL) {
      perror("Errore nell'apertura del file flussi_correnti.dat");
      exit(EXIT_FAILURE);
    }
    
    stencil_t * s = psi->stencil;
    int nflux = s->npoints;
    int nlocal[3], noffset[3];
    cs_nlocal(psi->cs, nlocal);
    cs_nlocal_offset(psi->cs, noffset);

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
        for (int kc = 1; kc <= nlocal[Z]; kc++) {
          int index = cs_index(psi->cs, ic, jc, kc);
          double jx = 0.0, jy = 0.0, jz = 0.0;
          double jx_p = 0.0, jy_p = 0.0, jz_p = 0.0;
          double jx_m = 0.0, jy_m = 0.0, jz_m = 0.0;


          for (int n = 0; n < nk; n++) {
            int valency = psi->valency[n];
            int ia = addr_rank1(psi->nsites, nk, index, n);
            for (int c = 1; c < nflux; c++) {
              int8_t cx = s->cv[c][X];
              int8_t cy = s->cv[c][Y];
              int8_t cz = s->cv[c][Z];
              double f = flx[ia][c - 1];
              
              jx += f * cx *  valency / 2;
              jy += f * cy *  valency / 2;
              jz += f * cz *  valency / 2;

              

              if (valency > 0) {
                jx_p += f * cx *  valency / 2;
                jy_p += f * cy *  valency / 2;
                jz_p += f * cz *  valency / 2;

                //if (ic == 20 && jc == 10 && kc == 1){
                //printf("f value: %.15e x=20;y=10;z=1 direction %d \n", f, cx);
                //}
              }
              else if (valency < 0) {
                jx_m += f * cx *  valency / 2;
                jy_m += f * cy *  valency / 2;
                jz_m += f * cz *  valency / 2;
              }
            }
          }

          int gx = ic + noffset[X];
          int gy = jc + noffset[Y];
          int gz = kc + noffset[Z];

          fprintf(fp, "%d %d %d %.15e %.15e %.15e\n", gx, gy, gz, jx, jy, jz);
          fprintf(fp_p, "%d %d %d %.15e %.15e %.15e\n", gx, gy, gz, jx_p, jy_p, jz_p);
          fprintf(fp_m, "%d %d %d %.15e %.15e %.15e\n", gx, gy, gz, jx_m, jy_m, jz_m);
          //fprintf(fp, " % .15e\n", jx);
              
        }
      }
    }


  fclose(fp);
  fclose(fp_m);
  fclose(fp_p);
  }
  */
  /* Add diffusive fluxes */
  nernst_planck_fluxes_d3qx(psi, fe, hydro, map, cinfo, flx);
  /*
  if (print_current_flag == 1) {
    
	///*
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char filename[64];
    char filename_p[64];
    char filename_m[64];

    char ts[10]; // 9 cifre + terminatore null
    sprintf(ts, "%09d", timestep);
    //sprintf(filename, "fluxes_%s.txt", ts);
    sprintf(filename, "fluxes_NP_EFadv_%s-rank%04d.txt", ts, rank);
    sprintf(filename_p, "fluxes_NP_EFadv_p_%s-rank%04d.txt", ts, rank);
    sprintf(filename_m, "fluxes_NP_EFadv_m_%s-rank%04d.txt", ts, rank);

    FILE *fp = fopen(filename, "w");
    FILE *fp_m = fopen(filename_m, "w");
    FILE *fp_p = fopen(filename_p, "w");

    if (fp == NULL) {
      perror("Errore nell'apertura del file flussi_correnti.dat");
      exit(EXIT_FAILURE);
    }
    ///*
    if (fp_m == NULL) {
      perror("Errore nell'apertura del file flussi_correnti.dat");
      exit(EXIT_FAILURE);
    }

    if (fp_p == NULL) {
      perror("Errore nell'apertura del file flussi_correnti.dat");
      exit(EXIT_FAILURE);
    }
      
    
    stencil_t * s = psi->stencil;
    int nflux = s->npoints;
    int nlocal[3], noffset[3];
    cs_nlocal(psi->cs, nlocal);
    cs_nlocal_offset(psi->cs, noffset);

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
        for (int kc = 1; kc <= nlocal[Z]; kc++) {
          int index = cs_index(psi->cs, ic, jc, kc);
          double jx = 0.0, jy = 0.0, jz = 0.0;
          double jx_p = 0.0, jy_p = 0.0, jz_p = 0.0;
          double jx_m = 0.0, jy_m = 0.0, jz_m = 0.0;


          for (int n = 0; n < nk; n++) {
            int valency = psi->valency[n];
            int ia = addr_rank1(psi->nsites, nk, index, n);
            for (int c = 1; c < nflux; c++) {
              int8_t cx = s->cv[c][X];
              int8_t cy = s->cv[c][Y];
              int8_t cz = s->cv[c][Z];
              double f = flx[ia][c - 1];
              
              jx += f * cx *  valency / 2;
              jy += f * cy *  valency / 2;
              jz += f * cz *  valency / 2;

              

              if (valency > 0) {
                jx_p += f * cx *  valency / 2;
                jy_p += f * cy *  valency / 2;
                jz_p += f * cz *  valency / 2;

                //if (ic == 20 && jc == 10 && kc == 1){
                //printf("f value: %.15e x=20;y=10;z=1 direction %d \n", f, cx);
                //}
              }
              else if (valency < 0) {
                jx_m += f * cx *  valency / 2;
                jy_m += f * cy *  valency / 2;
                jz_m += f * cz *  valency / 2;
              }
            }
          }

          int gx = ic + noffset[X];
          int gy = jc + noffset[Y];
          int gz = kc + noffset[Z];

          fprintf(fp, "%d %d %d %.15e %.15e %.15e\n", gx, gy, gz, jx, jy, jz);
          fprintf(fp_p, "%d %d %d %.15e %.15e %.15e\n", gx, gy, gz, jx_p, jy_p, jz_p);
          fprintf(fp_m, "%d %d %d %.15e %.15e %.15e\n", gx, gy, gz, jx_m, jy_m, jz_m);
          //fprintf(fp, " % .15e\n", jx);
              
        }
      }
    }


  fclose(fp);
  fclose(fp_m);
  fclose(fp_p);
  }
  */
  /* Add advective fluxes */
  if (hydro) np_advective_fluxes(psi, hydro, flx);


  /* Apply no-flux BC */
  if (map) np_no_flux_boundary(psi, map, flx); 
  
  /* Charge update Sanity Check */
  charge_update_sanity_check(psi, map, flx, timestep);


  /* Update charges */
  nernst_planck_update_d3qx(psi, map, flx);

  if (print_current_flag == 1) np_flux_io_write(psi, flx, timestep);

  for (ia = 0; ia < psi->nsites*nk; ia++) {
    free(flx[ia]);
  }
  free(flx);

  return 0;

}

/*****************************************************************************
 *
 *  np_flux_io_write
 *
 *  Compute the currents from the link fluxes and write them via MPI-IO
 *  (one file for all ranks, binary, with .meta description).
 *
 *  Three 3-component fields (one file each), (jx, jy, jz) per site:
 *    psi->jflux   "flux"    total
 *    psi->jflux_p "flux_p"  positive species
 *    psi->jflux_m "flux_m"  negative species
 *
 *****************************************************************************/

static int np_flux_io_write(psi_t * psi, double ** flx, int timestep) {

  int nk = 0;
  int nlocal[3] = {0};
  stencil_t * s = psi->stencil;
  field_t * jf[3] = {psi->jflux, psi->jflux_p, psi->jflux_m};

  for (int f = 0; f < 3; f++) {
    assert(jf[f]);
    assert(jf[f]->nf == 3);
  }

  psi_nk(psi, &nk);
  cs_nlocal(psi->cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

        int index = cs_index(psi->cs, ic, jc, kc);
        double j[9] = {0};

        for (int n = 0; n < nk; n++) {
          int valency = psi->valency[n];
          int ia = addr_rank1(psi->nsites, nk, index, n);
          int off = (valency > 0) ? 3 : 6;

          for (int c = 1; c < s->npoints; c++) {
            for (int d = 0; d < 3; d++) {
              double jd = 0.5*valency*flx[ia][c - 1]*s->cv[c][d];
              j[d] += jd;
              if (valency != 0) j[off + d] += jd;
            }
          }
        }

        /* j[0-2] total, j[3-5] positive, j[6-8] negative */
        for (int f = 0; f < 3; f++) {
          for (int d = 0; d < 3; d++) {
            jf[f]->data[addr_rank1(jf[f]->nsites, 3, index, d)] = j[3*f + d];
          }
        }
      }
    }
  }

  for (int f = 0; f < 1; f++) {   /* 1: solo flux totale; 3: anche flux_p, flux_m */
    /* field_io_write() copies device -> host; keep device copy current */
    io_event_t event = {0};
    field_memcpy(jf[f], tdpMemcpyHostToDevice);
    field_io_write(jf[f], timestep, &event);
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_fluxes_d3qx
 *
 *  Compute diffusive fluxes.
 *
 *  We assume we can accumulate the diffusive and advective fluxes separately.
 *
 *  As we compute rho(n+1) = rho(n) - div.flux in the update routine,
 *  there is an extra minus sign in the fluxes here. This conincides
 *  with the sign of the advective fluxes, if present.
 *
 *****************************************************************************/

static int nernst_planck_fluxes_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
				     map_t * map, colloids_info_t * cinfo,
				     double ** flx) {

  int ic, jc, kc; 
  int index0, index1;
  int nlocal[3];
  int n, nk; /* Number of charged species */
  int c;
  int status1;

  double b0, b1;
  double mu0, mu1;
  double rho0, rho1;
  double mu_s0, mu_s1;   /* Solvation chemical potential, from free energy */
  
  double eunit, reunit;
  double dt;

  double kt = 0.0;
  physics_t * phys = NULL;
  physics_ref(&phys);
  physics_kt(phys, &kt);

  colloid_t * pc = NULL;

  double * __restrict__ psidata = psi->psi->data;
  double * __restrict__ rhodata = psi->rho->data;

  LB_RCS_TABLE(rcs);

  assert(psi);
  assert(fe);
  assert(fe->func->mu_solv);
  assert(flx);

  cs_nlocal(psi->cs, nlocal);

  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;

  psi_nk(psi, &nk);
  psi_multistep_timestep(psi, &dt);


  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

		index0 = cs_index(psi->cs, ic, jc, kc);
        colloids_info_map(cinfo, index0, &pc);

        if (pc) {
          continue;
        }
        else {
          stencil_t * s = psi->stencil;
          assert(s);
          
          for (c = 1; c < s->npoints; c++) {

            //printf("npoints: %d \n", s->npoints);
            //exit(0);

            int8_t cx  = s->cv[c][X];
            int8_t cy  = s->cv[c][Y];
            int8_t cz  = s->cv[c][Z];
            int8_t pcv = cx*cx + cy*cy + cz*cz;

		    index1 = cs_index(psi->cs, ic + cx, jc + cy, kc + cz);
		    map_status(map, index1, &status1);

		    if (status1 == MAP_FLUID) {

		      for (n = 0; n < nk; n++) {

				fe->func->mu_solv(fe, index0, n, &mu_s0);
				mu0 = reunit*mu_s0
				  + psi->valency[n]*psidata[addr_rank0(psi->nsites, index0)];
				rho0 = rhodata[addr_rank1(psi->nsites, nk, index0, n)];

                fe->func->mu_solv(fe, index1, n, &mu_s1);
                mu1 = reunit*mu_s1
                  + psi->valency[n]* psidata[addr_rank0(psi->nsites, index1)];
                b0 = exp(mu0 - mu1);
                b1 = exp(mu1 - mu0);
                rho1 = rhodata[addr_rank1(psi->nsites, nk, index1, n)]*b1;

                //flx[addr_rank1(psi->nsites, nk, index0, n)][c - 1]
                //  -= psi->diffusivity[n]*0.5*(1.0 + b0)*(rho1 - rho0)*rcs[pcv];

                flx[addr_rank1(psi->nsites, nk, index0, n)][c - 1]
                  -= psi->mobility_elec[n]*kt*0.5*(1.0 + b0)*(rho1 - rho0)*rcs[pcv];
                /*
                  flux = flx[addr_rank1(psi->nsites, nk, index0, n)][c - 1];
                jx_tot += flux * cx * psi->valency[n];
                jy_tot += flux * cy * psi->valency[n];
                jz_tot += flux * cz * psi->valency[n];
                jx = flux * cx  * psi->valency[n];
                jy = flux * cy * psi->valency[n];
                jz = flux * cz * psi->valency[n];
                double fattore = rho1 - rho0;
                if (ic == 1 && jc == 10 && kc == 1) {
                  printf("Processing cell (%d, %d, %d)\n", ic, jc, kc);
                  printf("rho0: %.6e rho1: %.6e\n", rho0, rho1);
                  printf("status0: %d status1: %d\n", status0, status1);
                  printf("Valency: %d exp: %.6e fattore: %.6e dir (%+d %+d %+d): → jx=%.6e jy=%.6e \n",
                  psi->valency[n], b1, fattore, cx, cy, cz, jx, jy);
                }
                
                  //printf("rho0: %.6e cell (%d, %d, %d) \n", rho0, ic, jc, kc);
                

                
                
                */
              }
            }
	  	  }
          

        }
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_fluxes_force_d3qx
 *
 *  Compute diffusive fluxes and link-flux force on fluid.
 *
 *  We assume we can accumulate the diffusive and advective fluxes separately.
 *
 *  As we compute rho(n+1) = rho(n) - div.flux in the update routine,
 *  there is an extra minus sign in the fluxes here. This conincides
 *  with the sign of the advective fluxes, if present.
 *
 *****************************************************************************/

int nernst_planck_fluxes_force_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
				    map_t * map, colloids_info_t * cinfo,
				    double ** flx) {

  int ic, jc, kc; 
  int index0, index1;
  int nlocal[3];
  int n, nk; // Number of charged species 
  int nsites;
  int c;
  int status1;

  //exit(0);

  double eunit;
  double beta, rbeta;
  double b0, b1;
  double mu0, mu1;
  double rho0, rho1;
  double mu_s0, mu_s1;   // Solvation chemical potential, from free energy /
  
  double rho_elec;
  double e[3];
  double flocal[4] = {0.0, 0.0, 0.0, 0.0}, fsum[4], f[3]; 
  double flxtmp[2];
  double dt;

  double kt = 0.0;
  physics_t * phys = NULL;
  physics_ref(&phys);
  physics_kt(phys, &kt);

  MPI_Comm comm;
  colloid_t * pc = NULL;

  double * __restrict__ psidata = psi->psi->data;
  double * __restrict__ rhodata = psi->rho->data;

  LB_RCS_TABLE(rcs);

  assert(psi);
  assert(flx);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);
  cs_cart_comm(psi->cs, &comm);

  psi_nk(psi, &nk);
  psi_unit_charge(psi, &eunit);
  psi_beta(psi, &beta);
  psi_multistep_timestep(psi, &dt);

  rbeta = 1.0/beta;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index0 = cs_index(psi->cs, ic, jc, kc);
        colloids_info_map(cinfo, index0, &pc);

        f[X] = 0.0;
        f[Y] = 0.0;
        f[Z] = 0.0;

        psi_rho_elec(psi, index0, &rho_elec);

        // Total electrostatic force on colloid /
        if (pc) {

          psi_electric_field(psi, index0, e);

          f[X] = rho_elec * e[X] * dt;
          f[Y] = rho_elec * e[Y] * dt;
          f[Z] = rho_elec * e[Z] * dt;

          pc->force[X] += f[X];
          pc->force[Y] += f[Y];
          pc->force[Z] += f[Z];

	      }
        else {

          stencil_t * s = psi->stencil;
          // Internal electrostatic force on fluid 
          for (c = 1; c < s->npoints; c++) {
            int8_t cx  = s->cv[c][X];
            int8_t cy  = s->cv[c][Y];
            int8_t cz  = s->cv[c][Z];
            int8_t pcv = cx*cx + cy*cy + cz*cz;

		    index1 = cs_index(psi->cs, ic + cx, jc + cy, kc + cz);
		    map_status(map, index1, &status1);

		    if (status1 == MAP_FLUID) {

              for (n = 0; n < nk; n++) {
                fe->func->mu_solv(fe, index0, n, &mu_s0);
                mu0 = mu_s0
                  + psi->valency[n]*eunit*psidata[addr_rank0(nsites, index0)];
                rho0 = rhodata[addr_rank1(nsites, nk, index0, n)];

                fe->func->mu_solv(fe, index1, n, &mu_s1);
                mu1 = mu_s1
                  + psi->valency[n]*eunit*psidata[addr_rank0(nsites, index1)];
                b0 = exp(-beta*(mu1 - mu0));
                b1 = exp(+beta*(mu1 - mu0));
                rho1 = rhodata[addr_rank1(nsites, nk, index1, n)]*b1;

                flxtmp[0] = - 0.5*(1.0 + b0)*(rho1 - rho0)*rcs[pcv];

                // Diffusive flux accumulated 
                //flx[addr_rank1(nsites, nk, index0, n)][c - 1] += psi->diffusivity[n]*flxtmp[0];
                flx[addr_rank1(nsites, nk, index0, n)][c - 1] += psi->mobility_elec[n]*kt*flxtmp[0];


                // Force, including ideal gas part in chemical potential 
                f[X] -= s->wgradients[c]*cx*flxtmp[0]*rbeta;
                f[Y] -= s->wgradients[c]*cy*flxtmp[0]*rbeta;
                f[Z] -= s->wgradients[c]*cz*flxtmp[0]*rbeta;
              }
            }
          }

          //

          // Electrostatic force in external field /

          f[X] *= dt;
          f[Y] *= dt;
          f[Z] *= dt;

          // Count number of fluid sites 
          flocal[3] += 1.0;

          if (hydro) hydro_f_local_add(hydro, index0, f);

        }   

        // Accumulate contribution to total force on system 
        flocal[X] += f[X];
        flocal[Y] += f[Y];
        flocal[Z] += f[Z];

        }
      }
    }

    // On fluid sites apply correction for momentum conservation 

    MPI_Allreduce(flocal, fsum, 4, MPI_DOUBLE, MPI_SUM, comm);
  
    fsum[X] /= fsum[3];
    fsum[Y] /= fsum[3];
    fsum[Z] /= fsum[3];

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {

          index0 = cs_index(psi->cs, ic, jc, kc);
                colloids_info_map(cinfo, index0, &pc);

          if (pc) continue;

          f[X] = -fsum[X];
          f[Y] = -fsum[Y];
          f[Z] = -fsum[Z];

          if (hydro) hydro_f_local_add(hydro, index0, f);
        }
      }
    }

  return 0;
}

/*
int nernst_planck_fluxes_force_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
				    map_t * map, colloids_info_t * cinfo,
				    double ** flx) {

  int ic, jc, kc; 
  int index0, index1;
  int nlocal[3];
  int n, nk;
  int nsites;
  int c;
  int status0, status1;

  double eunit;
  double beta, rbeta;
  double b0, b1;
  double mu0, mu1;
  double rho0, rho1;
  double mu_s0, mu_s1;

  double rho_elec;
  double e[3];
  double flocal[4] = {0.0, 0.0, 0.0, 0.0}, fsum[4], f[3]; 
  double flxtmp;
  double dt;

  MPI_Comm comm;
  colloid_t * pc = NULL;

  double * __restrict__ psidata = psi->psi->data;
  double * __restrict__ rhodata = psi->rho->data;

  LB_RCS_TABLE(rcs);

  assert(psi);
  assert(fe);
  assert(flx);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);
  cs_cart_comm(psi->cs, &comm);

  int nx = nlocal[X];  // Per gestire le PBC in X

  psi_nk(psi, &nk);
  psi_unit_charge(psi, &eunit);
  psi_beta(psi, &beta);
  psi_multistep_timestep(psi, &dt);

  rbeta = 1.0 / beta;

  for (ic = 1; ic <= nx; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index0 = cs_index(psi->cs, ic, jc, kc);
        colloids_info_map(cinfo, index0, &pc);
        map_status(map, index0, &status0);

        f[X] = f[Y] = f[Z] = 0.0;

        psi_rho_elec(psi, index0, &rho_elec);

        if (pc) {
          // Forza su colloide
          psi_electric_field(psi, index0, e);
          f[X] = rho_elec * e[X] * dt;
          f[Y] = rho_elec * e[Y] * dt;
          f[Z] = rho_elec * e[Z] * dt;

          pc->force[X] += f[X];
          pc->force[Y] += f[Y];
          pc->force[Z] += f[Z];
        }
        else {
          // Forza interna su fluido
          stencil_t * s = psi->stencil;
          for (c = 1; c < s->npoints; c++) {

            int8_t cx  = s->cv[c][X];
            int8_t cy  = s->cv[c][Y];
            int8_t cz  = s->cv[c][Z];
            int8_t pcv = cx*cx + cy*cy + cz*cz;

            int ic1 = ic + cx;
            int jc1 = jc + cy;
            int kc1 = kc + cz;

            index1 = cs_index(psi->cs, ic1, jc1, kc1);
            map_status(map, index1, &status1);

            // BLOCCO PBC X PER SOLIDO-SOLIDO
            int block_force_pbc_x = 0;
            if (status0 != MAP_FLUID && status1 != MAP_FLUID) {
              if ((ic == 1 && ic1 == nx + 1) || (ic == nx && ic1 == 0)) {
                block_force_pbc_x = 1;
              }
            }

            if (block_force_pbc_x) continue;

            for (n = 0; n < nk; n++) {
              fe->func->mu_solv(fe, index0, n, &mu_s0);
              mu0 = mu_s0 + psi->valency[n] * eunit * psidata[addr_rank0(nsites, index0)];
              rho0 = rhodata[addr_rank1(nsites, nk, index0, n)];

              fe->func->mu_solv(fe, index1, n, &mu_s1);
              mu1 = mu_s1 + psi->valency[n] * eunit * psidata[addr_rank0(nsites, index1)];
              b0 = exp(-beta * (mu1 - mu0));
              b1 = exp(+beta * (mu1 - mu0));
              rho1 = rhodata[addr_rank1(nsites, nk, index1, n)] * b1;

              flxtmp = -0.5 * (1.0 + b0) * (rho1 - rho0) * rcs[pcv];

              // Accumula flusso diffuso
              flx[addr_rank1(nsites, nk, index0, n)][c - 1] += psi->diffusivity[n] * flxtmp;

              // Contributo alla forza interna
              f[X] -= s->wgradients[c] * cx * flxtmp * rbeta;
              f[Y] -= s->wgradients[c] * cy * flxtmp * rbeta;
              f[Z] -= s->wgradients[c] * cz * flxtmp * rbeta;
            }
          }

          // Applica timestep e accumula forza
          f[X] *= dt;
          f[Y] *= dt;
          f[Z] *= dt;

          flocal[3] += 1.0;

          if (hydro) hydro_f_local_add(hydro, index0, f);
        }

        // Somma forza totale per la correzione
        flocal[X] += f[X];
        flocal[Y] += f[Y];
        flocal[Z] += f[Z];
      }
    }
  }

  // Correzione per conservazione della quantità di moto
  MPI_Allreduce(flocal, fsum, 4, MPI_DOUBLE, MPI_SUM, comm);

  fsum[X] /= fsum[3];
  fsum[Y] /= fsum[3];
  fsum[Z] /= fsum[3];

  for (ic = 1; ic <= nx; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index0 = cs_index(psi->cs, ic, jc, kc);
        colloids_info_map(cinfo, index0, &pc);

        if (pc) continue;

        f[X] = -fsum[X];
        f[Y] = -fsum[Y];
        f[Z] = -fsum[Z];

        if (hydro) hydro_f_local_add(hydro, index0, f);
      }
    }
  }

  return 0;
}
*/

/*****************************************************************************
 *
 *  nernst_planck_update_d3qx
 *
 *  Update the rho_k from the fluxes (D3QX stencil). Euler forward step.
 *
 *****************************************************************************/
/*
static int nernst_planck_update_d3qx(psi_t * psi, map_t * map, double ** flx) {

  int ic, jc, kc, index;
  int nsites;
  int nlocal[3];
  int n, nk;
  int c;
  int status;
  double acc, maxacc = 0.0;
  double dt;

  const double rho_min = 1.0e-30; 

  assert(psi);
  assert(flx);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);

  psi_nk(psi, &nk);
  psi_multistep_timestep(psi, &dt);

  stencil_t * s = psi->stencil;
  assert(s);

  
  double * corr = (double *) calloc((size_t) nsites*nk, sizeof(double));
  if (corr == NULL) {
    pe_fatal(psi->pe, "calloc(corr) failed in nernst_planck_update_d3qx\n");
  }

  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(psi->cs, ic, jc, kc);
        map_status(map, index, &status);

        if (status != MAP_FLUID) continue;

        for (n = 0; n < nk; n++) {

          double rho_old = psi->rho->data[addr_rank1(nsites, nk, index, n)];
          double delta = 0.0;

          for (c = 1; c < s->npoints; c++) {
            delta += flx[addr_rank1(nsites, nk, index, n)][c - 1]*dt;
          }

          double rho_new = rho_old - delta;

          if (rho_new >= rho_min || delta <= 0.0) {
            psi->rho->data[addr_rank1(nsites, nk, index, n)] = (rho_new >= 0.0)
                                                               ? rho_new
                                                               : rho_min;
          }
          else {

            double delta_target = rho_old - rho_min;
            double scale = 0.0;

            if (delta_target > 0.0 && delta > 0.0) {
              scale = delta_target / delta;
              if (scale < 0.0) scale = 0.0;
              if (scale > 1.0) scale = 1.0;
            }
            else {
              scale = 0.0;
            }

            for (c = 1; c < s->npoints; c++) {

              int8_t cx = s->cv[c][X];
              int8_t cy = s->cv[c][Y];
              int8_t cz = s->cv[c][Z];

              int index1 = cs_index(psi->cs, ic + cx, jc + cy, kc + cz);

              double f_old = flx[addr_rank1(nsites, nk, index, n)][c - 1];
              double f_new = f_old * scale;

              flx[addr_rank1(nsites, nk, index, n)][c - 1] = f_new;

              double d_old = f_old*dt;
              double d_new = f_new*dt;
              double removed = d_old - d_new; 

              if (removed != 0.0) {
                int status1;
                map_status(map, index1, &status1);
                if (status1 == MAP_FLUID) {
                  /* Per conservare il bilancio sul link:
                     vicino deve vedere +d_new invece di +d_old,
                  corr[addr_rank1(nsites, nk, index1, n)] += (d_new - d_old);
                }
              }
            }

            double delta_scaled = 0.0;
            for (c = 1; c < s->npoints; c++) {
              delta_scaled += flx[addr_rank1(nsites, nk, index, n)][c - 1]*dt;
            }
            rho_new = rho_old - delta_scaled;

            if (rho_new < rho_min) rho_new = rho_min;
            psi->rho->data[addr_rank1(nsites, nk, index, n)] = rho_new;
          }

          acc = 0.0;
          for (c = 1; c < s->npoints; c++) {
            double d = flx[addr_rank1(nsites, nk, index, n)][c - 1]*dt;
            acc += fabs(d);
          }

          if (psi->rho->data[addr_rank1(nsites, nk, index, n)] > 0.0) {
            acc /= fabs(psi->rho->data[addr_rank1(nsites, nk, index, n)]);
            if (maxacc < acc) maxacc = acc;
          }
        }
      }
    }
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(psi->cs, ic, jc, kc);
        map_status(map, index, &status);
        if (status != MAP_FLUID) continue;

        for (n = 0; n < nk; n++) {
          double rho = psi->rho->data[addr_rank1(nsites, nk, index, n)];
          double dcorr = corr[addr_rank1(nsites, nk, index, n)];

          rho += dcorr;

          if (rho < rho_min) rho = rho_min;

          psi->rho->data[addr_rank1(nsites, nk, index, n)] = rho;
        }
      }
    }
  }

  free(corr);

  nernst_planck_maxacc_set(maxacc);

  return 0;
}
*/
static int nernst_planck_update_d3qx(psi_t * psi, map_t * map, double ** flx) {

  int ic, jc, kc, index;
  int nsites;
  int nlocal[3];
  int n, nk;
  int c;
  int status;
  double acc, maxacc=0.0;
  double dt;
  double kt = 0.0;
  physics_t * phys = NULL;
  physics_ref(&phys);
  physics_kt(phys, &kt);

  double flusso_tot = 0.0; // Per evitare warning 

  assert(psi);
  assert(flx);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);

  psi_nk(psi, &nk);
  psi_multistep_timestep(psi, &dt);
  
  
  int suppress_fluct_z = 0;

  

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	    index = cs_index(psi->cs, ic, jc, kc);
        map_status(map, index, &status);

        if (status == MAP_FLUID) {
	  		stencil_t * s = psi->stencil;
	  		for (n = 0; n < nk; n++) {

			    acc = 0.0;
			    for (c = 1; c < s->npoints; c++) {
					if (suppress_fluct_z){
						if (s->cv[c][Z] != 0){
							flx[addr_rank1(nsites, nk, index, n)][c - 1] = 0.0;
						} 
					}
			      psi->rho->data[addr_rank1(nsites, nk, index, n)]
				-= flx[addr_rank1(nsites, nk, index, n)][c - 1] * dt;
			      acc += fabs(flx[addr_rank1(nsites, nk, index, n)][c - 1] * dt);
			    }
          flusso_tot = acc; // Per evitare warning
			    acc /= fabs(psi->rho->data[addr_rank1(nsites, nk, index, n)]);
			    if (maxacc < acc) maxacc = acc;
          
			  }
		}

      }
    }
  }
  
  nernst_planck_maxacc_set(maxacc);

  return 0;
}

/*****************************************************************************
 *
 *  charge_update_sanity_check
 *
 *  Versione parallela (MPI).
 *
 *  Ogni pass e' diviso in due fasi:
 *
 *  1. Ogni rank marca le celle interne fluide per cui la previsione
 *     rho - dt*sum_c flx sarebbe negativa o non finita. La maschera
 *     viene poi scambiata negli halo (inclusa la periodicita').
 *
 *  2. Ogni rank modifica SOLO i propri flussi flx[i][c], i interno.
 *     Il link i -> j = i + c viene azzerato se:
 *       - i e' marcata e il flusso esce da i  (f > 0), oppure
 *       - j e' marcata e il flusso esce da j  (f < 0, poiche'
 *         flx[j][copp] = -flx[i][c]).
 *     Il rank che possiede j valuta la stessa condizione dal suo lato,
 *     quindi entrambe le rappresentazioni del link vengono azzerate
 *     senza scrivere sui siti di halo.
 *
 *  Il numero di pass e' deciso globalmente (MPI_Allreduce), perche'
 *  lo scambio di halo e' collettivo.
 *
 *****************************************************************************/

static int charge_update_sanity_check(psi_t * psi, map_t * map, double ** flx,
				      int timestep) {

  int nsites;
  int nlocal[3];
  int nk;
  double dt;
  MPI_Comm comm;

  static field_t * mask = NULL;   /* 1.0 se la cella/specie e' invalida */

  long counter = 0;
  long counter_links = 0;

  assert(psi);
  assert(flx);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);
  cs_cart_comm(psi->cs, &comm);
  psi_nk(psi, &nk);
  psi_multistep_timestep(psi, &dt);

  stencil_t * s = psi->stencil;
  assert(s);

  if (mask == NULL) {
    field_options_t opts = field_options_ndata_nhalo(nk, 1);
    field_create(psi->pe, psi->cs, psi->rho->le, "np-sanity-mask", &opts,
		 &mask);
  }

  /*
   * Iteriamo un paio di volte perche' cancellare un link puo' cambiare
   * la previsione di rho nelle celle vicine.
   */
  int max_pass = 4;

  for (int pass = 0; pass < max_pass; pass++) {

    int nbad_local = 0;

    /* Fase 1: maschera delle celle invalide (halo incluso a zero) */

    for (int ia = 0; ia < nsites*nk; ia++) mask->data[ia] = 0.0;

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
        for (int kc = 1; kc <= nlocal[Z]; kc++) {

          int index0 = cs_index(psi->cs, ic, jc, kc);

          int status0 = MAP_FLUID;
          if (map) map_status(map, index0, &status0);
          if (status0 != MAP_FLUID) continue;

          for (int n = 0; n < nk; n++) {

            int ia0 = addr_rank1(nsites, nk, index0, n);
            double rho_new = psi->rho->data[ia0];

            for (int c = 1; c < s->npoints; c++) {
              rho_new -= flx[ia0][c - 1]*dt;
            }

            if (isfinite(rho_new) && rho_new >= 0.0) continue;

            mask->data[ia0] = 1.0;
            nbad_local += 1;
          }
        }
      }
    }

    /* Decisione globale: se nessun rank ha celle invalide, abbiamo finito */
    {
      int nbad = 0;
      MPI_Allreduce(&nbad_local, &nbad, 1, MPI_INT, MPI_SUM, comm);
      if (nbad == 0) break;
    }

    counter += nbad_local;

    /* Stesso schema di psi_halo_rho() */
    field_memcpy(mask, tdpMemcpyHostToDevice);
    field_halo(mask);
    field_memcpy(mask, tdpMemcpyDeviceToHost);

    /* Fase 2: azzera i link uscenti dalle celle marcate */

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
        for (int kc = 1; kc <= nlocal[Z]; kc++) {

          int index0 = cs_index(psi->cs, ic, jc, kc);

          int status0 = MAP_FLUID;
          if (map) map_status(map, index0, &status0);
          if (status0 != MAP_FLUID) continue;

          for (int c = 1; c < s->npoints; c++) {

            int index1 = cs_index(psi->cs, ic + s->cv[c][X],
				  jc + s->cv[c][Y], kc + s->cv[c][Z]);

            for (int n = 0; n < nk; n++) {

              int ia0 = addr_rank1(nsites, nk, index0, n);
              int ia1 = addr_rank1(nsites, nk, index1, n);

              double f = flx[ia0][c - 1];
              int bad0 = (mask->data[ia0] > 0.5);
              int bad1 = (mask->data[ia1] > 0.5);

              /*
               * f > 0: la cella i perde massa,   rho_i <- rho_i - dt*f
               * f < 0: la cella j perde massa,   rho_j <- rho_j + dt*f
               */
              int out0 = (!isfinite(f) || f > 0.0);
              int out1 = (!isfinite(f) || f < 0.0);

              if ((bad0 && out0) || (bad1 && out1)) {
                flx[ia0][c - 1] = 0.0;
                counter_links += 1;
              }
            }
          }
        }
      }
    }
  }

    /* Diagnostica: accumula, il riepilogo viene stampato a fine run */
  {
    long local[2] = {counter, counter_links};
    long global[2] = {0, 0};
    long nfluid_local = (long) nlocal[X]*nlocal[Y]*nlocal[Z];
    long nfluid = 0;

    MPI_Allreduce(local, global, 2, MPI_LONG, MPI_SUM, comm);
    MPI_Allreduce(&nfluid_local, &nfluid, 1, MPI_LONG, MPI_SUM, comm);

    np_eraser.ncalls += 1;
    if (timestep != np_eraser.last_ts) {
      np_eraser.nts += 1;
      np_eraser.last_ts = timestep;
    }
    np_eraser.cells += global[0];
    np_eraser.links += global[1];
    if (global[1] > 0) np_eraser.ncalls_active += 1;
    if (global[1] > np_eraser.links_max) {
      np_eraser.links_max = global[1];
      np_eraser.ts_max = timestep;
    }
    np_eraser.nentries = (long) nk*nfluid*(s->npoints - 1);
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_maxacc_set
 *
 *  Setter function for the local maximal accuracy in the Nernst-Planck 
 *  equation. This is defined as the absolut value of the ratio of the change 
 *  during one fractional LB timestep (multistep dt) and the charge 
 *  density itself.
 *  
 *****************************************************************************/

int nernst_planck_maxacc_set(double acc) {
  max_acc = acc;
  return 0;
}

/*****************************************************************************
 *
 *  FILTRO ENERGETICO
 *
 *  
 *****************************************************************************/

int def_en_filter(psi_t * psi) {
  
  int nlocal[3];
  int ntotal[3];
  int nsites;
  int mpi_cartsz[3];
  int mpicoords[3];
  int nlocal_offset[3];

  cs_nlocal(psi->cs, nlocal);
  cs_ntotal(psi->cs, ntotal);
  cs_cartsz(psi->cs, mpi_cartsz);
  cs_cart_coords(psi->cs, mpicoords);
  cs_nsites(psi->cs, &nsites);
  cs_nlocal_offset(psi->cs, nlocal_offset);

  en_filter = calloc(psi->nsites, sizeof(int));

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	      int gx = ic + nlocal_offset[X];
        int index = cs_index(psi->cs, ic, jc, kc);

        if (gx < 49 || gx > 208){
          en_filter[index] = 0;
        }
        else {
          en_filter[index] = 1;
        }

      }
    }
  }
  
  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_maxacc
 *
 *  Getter function for the local maximal accuracy 
 *  in the Nernst-Planck equation.
 *
 *****************************************************************************/

int nernst_planck_maxacc(double * acc) {
  * acc = max_acc;
  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_adjust_multistep
 *
 *  
 *****************************************************************************/

int nernst_planck_adjust_multistep(psi_t * psi) {

  double maxacc_local[1], maxacc[1], diffacc; /* actual and preset value of diffusive accuracy */
  double diff, diffmax=0.0;                   /* diffusivity of species and maximal value      */ 
  int n, nk, multisteps;
  MPI_Comm comm;

  psi_diffacc(psi, &diffacc);

  /* Take local maximum and reduce for global maximum */
  nernst_planck_maxacc(&maxacc_local[0]);
  cs_cart_comm(psi->cs, &comm);
  MPI_Allreduce(maxacc_local, maxacc, 1, MPI_DOUBLE, MPI_MAX, comm);

  /* Compare maximal accuracy with preset value for */ 
  /*   diffusion and adjust number of multisteps    */

  /* Increase no. of multisteps */
  if (* maxacc > diffacc && diffacc > 0.0) {
    psi_multisteps(psi, &multisteps);
    multisteps *= 2;
    psi->multisteps = multisteps;
    pe_info(psi->pe, "\nMaxacc > diffacc: changing no. of multisteps to %d\n",
	    multisteps);
  }    

  /* Reduce no. of multisteps */
  /* The factor 0.1 prevents too frequent changes. */
  if (* maxacc < 0.1*diffacc && diffacc > 0.0) {
  
    psi_multisteps(psi, &multisteps);
    psi_nk(psi, &nk);

    for (n = 0; n < nk; n++) {
	psi_diffusivity(psi, n, &diff);
	if (diff > diffmax) diffmax = diff;
    }

    /* Only reduce if sanity criteria fulfilled */  
    if (multisteps > 1 && diffmax/multisteps < 0.05) { 
      multisteps *= 0.5; 
      psi->multisteps = multisteps;
      pe_info(psi->pe, "\nMaxacc << diffacc: changing no. of multisteps to %d\n", multisteps);
    }    

  }    

  return 0;
} 

/*****************************************************************************
 *
 *  np_advective_ef
 *
 *  'Centred difference' advective fluxes for ef.
 *
 *  Symmetric two-point stencil.
 *
 *****************************************************************************/
int ef_advective(psi_t * psi, double ** flx) {

  int nlocal[3] = {0};
  int ntotal[3] = {0};
  int noffset[3] = {0};
  cs_t * cs = NULL;
  stencil_t * s = NULL;

  double kt = 0.0;
  physics_t * phys = NULL;
  physics_ref(&phys);
  physics_kt(phys, &kt);

  double * __restrict__ rho = NULL;

  const double pi = 3.14159265358979323846;
  const double E_0 = 0.0; //0.0001;

  assert(psi);
  assert(flx);

  cs = psi->cs;
  s  = psi->stencil;
  rho = psi->rho->data;

  assert(cs);
  assert(s);
  assert(rho);

  cs_nlocal(cs, nlocal);
  cs_ntotal(cs, ntotal);
  cs_nlocal_offset(cs, noffset);

  /* stesso numero d'onda usato in electric_field_periodic() */
  int n = (int) POTENTIAL;
  double k = 2.0*pi*((double) n)/((double) ntotal[X]);
  //double k = POTENTIAL;

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

        int index0 = cs_index(cs, ic, jc, kc);

        for (int p = 1; p < s->npoints; p++) {

          int8_t cx = s->cv[p][X];
          int8_t cy = s->cv[p][Y];
          int8_t cz = s->cv[p][Z];

          int index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);

          /* coordinate globali periodiche dei due estremi del link */
          int gx0 = noffset[X] + (ic - 1);
          int gx1 = gx0 + cx;

          gx0 = (gx0 % ntotal[X] + ntotal[X]) % ntotal[X];
          gx1 = (gx1 % ntotal[X] + ntotal[X]) % ntotal[X];

          /* campo ai due estremi del link */
          double E0[3] = {0.0, 0.0, 0.0};
          double E1[3] = {0.0, 0.0, 0.0};

          E0[X] = ef_adv[X] + E_0*sin(k*(double) gx0);
          E0[Y] = ef_adv[Y];
          E0[Z] = ef_adv[Z];

          E1[X] = ef_adv[X] + E_0*sin(k*(double) gx1);
          E1[Y] = ef_adv[Y];
          E1[Z] = ef_adv[Z];

          /* campo medio sul link */
          double Eface[3];
          Eface[X] = 0.5*(E0[X] + E1[X]);
          Eface[Y] = 0.5*(E0[Y] + E1[Y]);
          Eface[Z] = 0.5*(E0[Z] + E1[Z]);

          for (int n = 0; n < psi->nk; n++) {

            double rho0 = rho[addr_rank1(psi->nsites, psi->nk, index0, n)];
            double rho1 = rho[addr_rank1(psi->nsites, psi->nk, index1, n)];

            /* velocità di drift proiettata sul link */
            //double uem = psi->valency[n]*psi->diffusivity[n] *
            //             (Eface[X]*cx + Eface[Y]*cy + Eface[Z]*cz);
            double uem = psi->valency[n]*psi->mobility_elec[n] * kt *
                         (Eface[X]*cx + Eface[Y]*cy + Eface[Z]*cz);

            /* scelta robusta: upwind */
            double rho_face = 0.5 * (rho0 + rho1); //(uem >= 0.0) ? rho0 : rho1;

            double flux = uem * rho_face;

            flx[addr_rank1(psi->nsites, psi->nk, index0, n)][p-1] += flux;
          }
        }
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  Thermal Noise
 *
 *****************************************************************************/
int flx_noise(psi_t * psi, double ** flx, noise_t * noise,  map_t * map) {

  int nlocal[3] = {0};
  int ntotal[3] = {0};
  int noffset[3] = {0};
  int index0, index1;
  int vaddr0_X, vaddr0_Y, vaddr0_Z, vaddr1_X, vaddr1_Y, vaddr1_Z;
  double rho0, rho1, rho_face;
  double flux;
  int n, nk; /* Number of charged species */
  double kt = 0.0;
  double mobility_elec = 0.0;
  int status0, status1;

  cs_t * cs = NULL;
  stencil_t * s = NULL;
  physics_t * phys = NULL;
  //field_t * var = NULL;
  static field_t * var = NULL;

  double * __restrict__ rho = NULL;

  cs = psi->cs;
  s  = psi->stencil;
  rho = psi->rho->data;

  assert(psi);
  assert(flx);
  assert(noise);
  assert(cs);
  assert(s);
  assert(rho);

  cs_nlocal(cs, nlocal);
  cs_ntotal(cs, ntotal);
  cs_nlocal_offset(cs, noffset);
  physics_ref(&phys);
  physics_kt(phys, &kt);
  psi_nk(psi, &nk);

  //field_options_t opts = field_options_ndata_nhalo(3, 3);
  //field_create(psi->pe, psi->cs, psi->rho->le, "noise-var", &opts, &var);

  if (var == NULL) {
    field_options_t opts = field_options_ndata_nhalo(3, 3);
    field_create(psi->pe, psi->cs, psi->rho->le, "noise-var", &opts, &var);
  }

  for (n = 0; n < nk; n++) {
    
    mobility_elec = psi->mobility_elec[n];
    //printf("mobility %lf \n", mobility);
    //printf("kt %lf \n", kt);
    //kt = 1.0;
    //mobility_elec = 5.0e-01;
    //kt = 5.0e-06;

    phi_ch_var_flux_driver(var, noise, mobility_elec, kt);

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
        for (int kc = 1; kc <= nlocal[Z]; kc++) {

          index0 = cs_index(cs, ic, jc, kc);
          vaddr0_X = addr_rank1(var->nsites, 3, index0, X);
          vaddr0_Y = addr_rank1(var->nsites, 3, index0, Y);
          vaddr0_Z = addr_rank1(var->nsites, 3, index0, Z);

          map_status(map, index0, &status0);

          for (int p = 1; p < s->npoints; p++) {

            int8_t cx = s->cv[p][X];
            int8_t cy = s->cv[p][Y];
            int8_t cz = s->cv[p][Z];

            index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);
            vaddr1_X = addr_rank1(var->nsites, 3, index1, X);
            vaddr1_Y = addr_rank1(var->nsites, 3, index1, Y);
            vaddr1_Z = addr_rank1(var->nsites, 3, index1, Z);
            map_status(map, index1, &status1);

            rho0 = rho[addr_rank1(psi->nsites, psi->nk, index0, n)];
            rho1 = rho[addr_rank1(psi->nsites, psi->nk, index1, n)];
            rho_face = 0.5 * (rho0 + rho1); //(uem >= 0.0) ? rho0 : rho1;
            flux = 0.0;
            if (rho_face <= 0.0) continue;
            //rho_face = 0.0000001;

            if (status1 == MAP_FLUID && status0 == MAP_FLUID) {
              // VALIDO PER STENCIL A 7 PUNTI
              if (cx == 1 || cy == 1 || cz == 1){
                flux = sqrt(rho_face) * (var->data[vaddr0_X] * cx + var->data[vaddr0_Y] * cy+ var->data[vaddr0_Z] * cz);
                //flux = var->data[vaddr0_X] * cx + var->data[vaddr0_Y] * cy+ var->data[vaddr0_Z] * cz;
              }

              else if (cx == -1 || cy == -1 || cz == -1){
                flux = sqrt(rho_face) * (var->data[vaddr1_X] * cx + var->data[vaddr1_Y] * cy+ var->data[vaddr1_Z] * cz);
                //flux = var->data[vaddr1_X] * cx + var->data[vaddr1_Y] * cy+ var->data[vaddr1_Z] * cz;
              }
            
              flx[addr_rank1(psi->nsites, psi->nk, index0, n)][p-1] += flux;

              if (cx != 0){
                //printf("LIQUID x: %d y: %d z: %d flux = %.25f\n", ic, jc, kc, flux);
              }
            }
            else {
              flux = 0.0;
              if (cx != 0){
                //printf("SOLID x: %d y: %d z: %d flux = %.25f\n", ic, jc, kc, flux);
              }
            }
          }
        }
      }
    }
  }
  //exit(0);
  //field_free(var);
  //exit(0);

  
  return 0;
}

int np_advective_fluxes(psi_t * psi, hydro_t * hydro, double ** flx) {

  int nlocal[3] = {0};
  cs_t * cs = NULL;
  stencil_t * s = NULL;

  double * __restrict__ rho = psi->rho->data;

  assert(psi);
  assert(hydro);
  assert(flx);

  cs = psi->cs;
  s = psi->stencil;
  assert(cs);
  assert(s);

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index0 = cs_index(cs, ic, jc, kc);
	double u0[3] = {0};
	hydro_u(hydro, index0, u0);

        for (int p = 1; p < s->npoints; p++) {

	  int8_t cx = s->cv[p][X];
	  int8_t cy = s->cv[p][Y];
	  int8_t cz = s->cv[p][Z];
	  int index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);
	  double u1[3] = {0};
	  double u = 0.0;
	  hydro_u(hydro, index1, u1);

	  u = 0.5*((u0[X]+u1[X])*cx + (u0[Y]+u1[Y])*cy + (u0[Z]+u1[Z])*cz);

	  for (int n = 0; n < psi->nk; n++) {
	    double rho0 = rho[addr_rank1(psi->nsites, psi->nk, index0, n)];
	    double rho1 = rho[addr_rank1(psi->nsites, psi->nk, index1, n)];
      double rho_face;
      if (u > 0)
        rho_face = rho0;   // upwind = nodo di partenza
      else
        rho_face = rho1;   // upwind = nodo di arrivo (se flusso inverso)
      //double rho_ARM = 2 * rho0 * rho1 / (rho0 + rho1 + 1e-55); // Avoid division by zero 
	    double flux = u*0.5*(rho0 + rho1);
      //double flux = u*rho_ARM;
      //double flux = u*rho_face; // Upwind scheme
	    flx[addr_rank1(psi->nsites, psi->nk, index0, n)][p-1] += flux;
	  }
	}
	/* Next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  advective_bcs_no_flux_d3qx
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/
/*
int np_no_flux_boundary(psi_t * psi, map_t * map, double ** flx) {

  int nlocal[3] = {0};
  cs_t * cs = NULL;
  stencil_t * s = NULL;

  assert(psi);
  assert(map);
  assert(flx);

  cs = psi->cs;
  s  = psi->stencil;
  assert(cs);
  assert(s);

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

        int index0 = cs_index(cs, ic, jc, kc);
        int mask0  = 1;
        int status0 = MAP_BOUNDARY;
        map_status(map, index0, &status0);
        mask0 = (status0 == MAP_FLUID);

        for (int p = 1; p < s->npoints; p++) {

          int8_t cx = s->cv[p][X];
          int8_t cy = s->cv[p][Y];
          int8_t cz = s->cv[p][Z];
          int index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);
          int mask = 1;

          /*
          map_status(map, index1, &status);
          mask = (status == MAP_FLUID);
          mask = mask*mask0;

          for (int n = 0;  n < psi->nk; n++) {
            flx[addr_rank1(psi->nsites, psi->nk, index0, n)][p-1] *= mask;
          }
          //*
          int status1 = MAP_FLUID;
          map_status(map, index1, &status1);

          // ❌ Blocca flusso SOLO se è tra liquido e solido (o viceversa)
          int is_cross = ((status0 == MAP_FLUID && status1 != MAP_FLUID) ||
                          (status1 == MAP_FLUID && status0 != MAP_FLUID));

          if (is_cross) {
            for (int n = 0; n < psi->nk; n++) {
              flx[addr_rank1(psi->nsites, psi->nk, index0, n)][p-1] = 0.0;
            }
          }
        }
      }
    }
  }

  return 0;
}
*/

int np_no_flux_boundary(psi_t * psi, map_t * map, double ** flx) {

  int nlocal[3] = {0};
  cs_t * cs = NULL;
  stencil_t * s = NULL;

  assert(psi);
  assert(map);
  assert(flx);

  cs = psi->cs;
  s  = psi->stencil;
  assert(cs);
  assert(s);

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

		int index0 = cs_index(cs, ic, jc, kc);
		int mask0  = 1;
		int status = MAP_BOUNDARY;
		map_status(map, index0, &status);
		mask0 = (status == MAP_FLUID);

        for (int p = 1; p < s->npoints; p++) {

		  int8_t cx = s->cv[p][X];
		  int8_t cy = s->cv[p][Y];
		  int8_t cz = s->cv[p][Z];
		  int index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);
		  int mask = 1;

		  map_status(map, index1, &status);
		  mask = (status == MAP_FLUID);
		  mask = mask*mask0;

		  for (int n = 0;  n < psi->nk; n++) {
		    flx[addr_rank1(psi->nsites, psi->nk, index0, n)][p-1] *= mask;
		  }
		}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_eraser_report
 *
 *  Riepilogo del flux eraser su tutta la simulazione.
 *
 *****************************************************************************/

int nernst_planck_eraser_report(pe_t * pe) {

  assert(pe);
  if (np_eraser.ncalls == 0) return 0;

  double nts = (double) np_eraser.nts;
  double mean_links = (double) np_eraser.links / nts;
  double mean_frac = (np_eraser.nentries > 0)
    ? mean_links / (double) np_eraser.nentries : 0.0;

  pe_info(pe, "\n");
  pe_info(pe, "Flux eraser summary\n");
  pe_info(pe, "[eraser] timesteps checked          %ld\n", np_eraser.nts);
  pe_info(pe, "[eraser] calls (incl. multistep)    %ld\n", np_eraser.ncalls);
  pe_info(pe, "[eraser] calls with erased links    %ld\n", np_eraser.ncalls_active);
  pe_info(pe, "[eraser] total erased link entries  %ld\n", np_eraser.links);
  pe_info(pe, "[eraser] mean erased links per ts   %14.7e\n", mean_links);
  pe_info(pe, "[eraser] mean erased fraction per ts %14.7e\n", mean_frac);
  pe_info(pe, "[eraser] total flagged cells        %ld\n", np_eraser.cells);
  pe_info(pe, "[eraser] max erased in one call     %ld at ts %d\n",
	  np_eraser.links_max, np_eraser.ts_max);

  return 0;
}
