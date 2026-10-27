/*****************************************************************************
 *
 *  moment_propagation.c
 *
 *  Velocity autocorrelation function Z_g(t) = <v_g(0) v_g(t)> of a
 *  tracer of charge q = z e advected by the LB fluid, via moment
 *  propagation following
 *
 *    B. Rotenberg, I. Pagonabarraga, D. Frenkel, EPL 83, 34004 (2008).
 *
 *  Transition probability for a jump r -> r + c_i (eq. 4, 5):
 *
 *    p_i(r) = f_i(r)/rho_f(r) - w_i
 *           + lambda w_i { (1/4) beta q E.c_i
 *                          + 1/(1 + exp(-beta[V(r) - V(r+c_i)])) }
 *
 *  and p_i = 0 if r + c_i is not fluid; p_0 = 1 - sum_i p_i.
 *  f_i are post-collision distributions, V = q psi, and
 *  lambda = 4 D / (c_s^2 dt) fixes the tracer diffusivity D.
 *
 *  The uniform external field e0 enters psi as a linear ramp
 *  (psi -> psi - e0.r, see psi_halo_psijump()). Here the ramp is
 *  removed from V, and the field is included via the (1/4) beta q E.c
 *  term as in the paper, so that it is not counted twice.
 *
 *  The probabilities are computed once, from the state at time step
 *  mp_tracer_tstart (frozen), and P(r,t;g) is then propagated for
 *  mp_tracer_nsteps steps (eq. 3):
 *
 *    P(r,1;g)   = sum_i W(r - c_i) p_i(r - c_i) c_ig
 *    P(r,t+1;g) = p_0(r) P(r,t;g) + sum_i P(r - c_i,t;g) p_i(r - c_i)
 *    Z_g(t)     = sum_r P(r,t;g) u*_g(r),   u*_g = sum_i p_i c_ig
 *    Z_g(0)     = sum_r W(r) sum_i p_i(r) c_ig c_ig
 *
 *  with W(r) = exp(-beta V(r))/Q the equilibrium tracer distribution.
 *
 *  Units: dt = 1 (one LB step), dx = 1.
 *
 *  Input keys:
 *    mp_tracer               on
 *    mp_tracer_valency       z      (integer, default 0)
 *    mp_tracer_diffusivity   D      (lattice units)
 *    mp_tracer_tstart        t0     (LB step at which f, psi are frozen)
 *    mp_tracer_nsteps        N      (length of Z(t))
 *
 *  Output (rank 0): mp_vacf_q<z>.dat with columns
 *    t  Zx Zy Zz  Dx Dy Dz  norm
 *  where D_g(t) = (1/2) C_g(0) + sum_{s=1}^{t} C_g(s), C_g = Z_g - vbar_g^2,
 *  and norm = sum_r P(r,t) (should stay 1).
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "field.h"
#include "field_options.h"
#include "moment_propagation.h"

struct mp_tracer_s {
  pe_t * pe;
  cs_t * cs;
  int valency;           /* Tracer valency z */
  double diffusivity;    /* Tracer molecular diffusivity D */
  int tstart;            /* Time step at which probabilities are frozen */
  int nsteps;            /* Number of moment propagation steps */
  int done;
};

#define MP_F(fld, index, n) \
  (fld)->data[addr_rank1((fld)->nsites, (fld)->nf, (index), (n))]

/*****************************************************************************
 *
 *  mp_field_halo
 *
 *  Host data -> halo swap -> host data.
 *
 *****************************************************************************/

static int mp_field_halo(field_t * f) {

  field_memcpy(f, tdpMemcpyHostToDevice);
  field_halo(f);
  field_memcpy(f, tdpMemcpyDeviceToHost);

  return 0;
}

/*****************************************************************************
 *
 *  mp_tracer_create_rt
 *
 *  Returns *mp = NULL if "mp_tracer" is not switched on.
 *
 *****************************************************************************/

int mp_tracer_create_rt(pe_t * pe, cs_t * cs, rt_t * rt, mp_tracer_t ** mp) {

  mp_tracer_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(rt);
  assert(mp);

  *mp = NULL;
  if (rt_switch(rt, "mp_tracer") == 0) return 0;

  obj = (mp_tracer_t *) calloc(1, sizeof(mp_tracer_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(mp_tracer_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  rt_int_parameter(rt, "mp_tracer_valency", &obj->valency);

  if (rt_double_parameter(rt, "mp_tracer_diffusivity", &obj->diffusivity) == 0)
    pe_fatal(pe, "mp_tracer: please set mp_tracer_diffusivity\n");
  if (rt_int_parameter(rt, "mp_tracer_tstart", &obj->tstart) == 0)
    pe_fatal(pe, "mp_tracer: please set mp_tracer_tstart\n");
  if (rt_int_parameter(rt, "mp_tracer_nsteps", &obj->nsteps) == 0)
    pe_fatal(pe, "mp_tracer: please set mp_tracer_nsteps\n");

  if (obj->diffusivity <= 0.0) pe_fatal(pe, "mp_tracer: diffusivity <= 0\n");
  if (obj->tstart < 1)         pe_fatal(pe, "mp_tracer: tstart < 1\n");
  if (obj->nsteps < 1)         pe_fatal(pe, "mp_tracer: nsteps < 1\n");

  pe_info(pe, "\n");
  pe_info(pe, "Moment propagation (charged tracer VACF)\n");
  pe_info(pe, "----------------------------------------\n");
  pe_info(pe, "Tracer valency:         %d\n", obj->valency);
  pe_info(pe, "Tracer diffusivity:     %14.7e\n", obj->diffusivity);
  pe_info(pe, "Frozen at time step:    %d\n", obj->tstart);
  pe_info(pe, "Propagation steps:      %d\n", obj->nsteps);

  *mp = obj;

  return 0;
}

/*****************************************************************************
 *
 *  mp_tracer_free
 *
 *****************************************************************************/

int mp_tracer_free(mp_tracer_t ** mp) {

  assert(mp);

  free(*mp);
  *mp = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  mp_tracer_run
 *
 *  To be called between lb_collide() and the bounce-back/propagation,
 *  so that lb holds post-collision distributions. Does nothing unless
 *  timestep == tstart. psi may be NULL (neutral tracer only).
 *
 *****************************************************************************/

int mp_tracer_run(mp_tracer_t * mp, int timestep, lb_t * lb, psi_t * psi,
		  map_t * map) {

  int nlocal[3] = {0};
  int noffset[3] = {0};
  int periodic[3] = {0};
  int nvel;
  int status0, status1;

  double cs2;
  double lambda;
  double beq = 0.0;              /* beta z e */
  double eext[3] = {0.0};        /* External field (periodic directions) */

  double * zt = NULL;            /* Z_g(t), 3 per time */
  double * normt = NULL;         /* sum_r P(r,t) */
  double vbar[3] = {0.0};

  field_t * wf = NULL;           /* beta V, then W = exp(-beta V)/Q */
  field_t * prob = NULL;         /* p_i, i = 0 .. nvel-1 */
  field_t * ust = NULL;          /* u*_g */
  field_t * pa = NULL;           /* P(r,t;g) g = 0,1,2 and P(r,t) */
  field_t * pb = NULL;

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  MPI_Comm comm;

  if (mp == NULL) return 0;
  if (mp->done || timestep != mp->tstart) return 0;

  assert(lb);
  assert(map);

  mp->done = 1;
  pe = mp->pe;
  cs = mp->cs;

  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);
  cs_periodic(cs, periodic);
  cs_cart_comm(cs, &comm);

  nvel = lb->model.nvel;
  cs2 = lb->model.cs2;
  lambda = 4.0*mp->diffusivity/cs2;

  if (psi) {
    /* psi is the reduced potential (the Poisson solve carries the
     * factor e/kT), so beta z e psi is just z psi, as in the
     * exponents of nernst_planck_fluxes_d3qx(). */
    beq = mp->valency;
    for (int ia = 0; ia < 3; ia++) {
      if (periodic[ia]) eext[ia] = psi->e0[ia];
    }
  }
  else if (mp->valency != 0) {
    pe_fatal(pe, "mp_tracer: charged tracer requires electrokinetics\n");
  }

  pe_info(pe, "\nmp_tracer: freezing transition probabilities at step %d\n",
	  timestep);
  pe_info(pe, "mp_tracer: lambda = %14.7e  z = %14.7e\n", lambda, beq);

  if (lb->param->noise) {
    pe_info(pe, "mp_tracer: WARNING: LB noise is on; p_i are computed from "
	    "a single fluctuating snapshot\n");
  }

  lb_memcpy(lb, tdpMemcpyDeviceToHost);

  {
    field_options_t o1 = field_options_ndata_nhalo(1, 1);
    field_options_t on = field_options_ndata_nhalo(nvel, 1);
    field_options_t o3 = field_options_ndata_nhalo(3, 1);
    field_options_t o4 = field_options_ndata_nhalo(4, 1);
    field_create(pe, cs, NULL, "mp-w",    &o1, &wf);
    field_create(pe, cs, NULL, "mp-prob", &on, &prob);
    field_create(pe, cs, NULL, "mp-ust",  &o3, &ust);
    field_create(pe, cs, NULL, "mp-pa",   &o4, &pa);
    field_create(pe, cs, NULL, "mp-pb",   &o4, &pb);
  }

  /* 1. beta V(r) = beta z e (psi + e0.r): external ramp removed */

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(cs, ic, jc, kc);
	double bv = 0.0;
	if (psi) {
	  double phi = psi->psi->data[addr_rank0(psi->nsites, index)];
	  phi += eext[X]*(noffset[X] + ic)
	       + eext[Y]*(noffset[Y] + jc)
	       + eext[Z]*(noffset[Z] + kc);
	  bv = beq*phi;
	}
	MP_F(wf, index, 0) = bv;
      }
    }
  }

  mp_field_halo(wf);

  /* 2. Transition probabilities p_i(r) and u*(r) */

  {
    long nneg = 0;
    long nover = 0;
    long nloc[2] = {0};
    long nglb[2] = {0};
    double pmin = 1.0;
    double pmin_glb = 1.0;

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {

	  int index0 = cs_index(cs, ic, jc, kc);
	  double rhof = 0.0;
	  double sum = 0.0;

	  for (int p = 0; p < nvel; p++) MP_F(prob, index0, p) = 0.0;
	  for (int ia = 0; ia < 3; ia++) MP_F(ust, index0, ia) = 0.0;

	  map_status(map, index0, &status0);
	  if (status0 != MAP_FLUID) continue;

	  lb_0th_moment(lb, index0, LB_RHO, &rhof);

	  for (int p = 1; p < nvel; p++) {

	    int8_t cx = lb->model.cv[p][X];
	    int8_t cy = lb->model.cv[p][Y];
	    int8_t cz = lb->model.cv[p][Z];
	    int index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);
	    double f = 0.0;
	    double padv, dv, fermi, edrift, pp;

	    map_status(map, index1, &status1);
	    if (status1 != MAP_FLUID) continue;

	    lb_f(lb, index0, p, LB_RHO, &f);

	    padv   = f/rhof - lb->model.wv[p];
	    dv     = MP_F(wf, index1, 0) - MP_F(wf, index0, 0);
	    fermi  = 1.0/(1.0 + exp(dv));
	    edrift = 0.25*beq*(eext[X]*cx + eext[Y]*cy + eext[Z]*cz);
	    pp     = padv + lambda*lb->model.wv[p]*(edrift + fermi);

	    if (pp < 0.0) nneg += 1;
	    if (pp < pmin) pmin = pp;

	    MP_F(prob, index0, p) = pp;
	    MP_F(ust, index0, X) += pp*cx;
	    MP_F(ust, index0, Y) += pp*cy;
	    MP_F(ust, index0, Z) += pp*cz;
	    sum += pp;
	  }

	  MP_F(prob, index0, 0) = 1.0 - sum;
	  if (1.0 - sum < 0.0) nover += 1;
	}
      }
    }

    nloc[0] = nneg;
    nloc[1] = nover;
    MPI_Allreduce(nloc, nglb, 2, MPI_LONG, MPI_SUM, comm);
    MPI_Allreduce(&pmin, &pmin_glb, 1, MPI_DOUBLE, MPI_MIN, comm);

    pe_info(pe, "mp_tracer: min p_i = %14.7e\n", pmin_glb);

    if (nglb[0] > 0 || nglb[1] > 0) {
      pe_info(pe, "mp_tracer: %ld negative p_i, %ld sites with p_0 < 0\n",
	      nglb[0], nglb[1]);
      pe_info(pe, "mp_tracer: need roughly |u|/c_s^2 < lambda/2 and "
	      "lambda*sum_i w_i < 1 (adjust diffusivity or flow)\n");
      pe_fatal(pe, "mp_tracer: invalid transition probabilities\n");
    }
  }

  mp_field_halo(prob);

  /* 3. Equilibrium weights W(r) = exp(-beta V)/Q on fluid sites */

  {
    double bvmin = DBL_MAX;
    double bvmin_glb = DBL_MAX;
    double qloc = 0.0;
    double qglb = 0.0;

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  map_status(map, index, &status0);
	  if (status0 == MAP_FLUID && MP_F(wf, index, 0) < bvmin) {
	    bvmin = MP_F(wf, index, 0);
	  }
	}
      }
    }
    MPI_Allreduce(&bvmin, &bvmin_glb, 1, MPI_DOUBLE, MPI_MIN, comm);

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  double w = 0.0;
	  map_status(map, index, &status0);
	  if (status0 == MAP_FLUID) w = exp(-(MP_F(wf, index, 0) - bvmin_glb));
	  MP_F(wf, index, 0) = w;
	  qloc += w;
	}
      }
    }
    MPI_Allreduce(&qloc, &qglb, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (qglb <= 0.0) pe_fatal(pe, "mp_tracer: no fluid sites\n");

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  MP_F(wf, index, 0) /= qglb;
	}
      }
    }
  }

  mp_field_halo(wf);

  /* 4. Mean velocity vbar_g = sum_r W u*_g and Z_g(0) */

  zt = (double *) calloc(3*(mp->nsteps + 1), sizeof(double));
  normt = (double *) calloc(mp->nsteps + 1, sizeof(double));
  assert(zt);
  assert(normt);
  if (zt == NULL || normt == NULL) pe_fatal(pe, "calloc(zt) failed\n");

  {
    double sloc[6] = {0};
    double sglb[6] = {0};

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  double w = MP_F(wf, index, 0);
	  for (int ia = 0; ia < 3; ia++) {
	    sloc[ia] += w*MP_F(ust, index, ia);
	  }
	  for (int p = 1; p < nvel; p++) {
	    double pp = MP_F(prob, index, p);
	    for (int ia = 0; ia < 3; ia++) {
	      double c = lb->model.cv[p][ia];
	      sloc[3 + ia] += w*pp*c*c;
	    }
	  }
	}
      }
    }
    MPI_Allreduce(sloc, sglb, 6, MPI_DOUBLE, MPI_SUM, comm);

    for (int ia = 0; ia < 3; ia++) {
      vbar[ia] = sglb[ia];
      zt[ia] = sglb[3 + ia];
    }
    normt[0] = 1.0;
  }

  /* 5. P(r,1;g) = sum_i W(r - c_i) p_i(r - c_i) c_ig; P(r,1) likewise */

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(cs, ic, jc, kc);
	double sp[4] = {0};

	map_status(map, index, &status0);

	if (status0 == MAP_FLUID) {
	  sp[3] = MP_F(wf, index, 0)*MP_F(prob, index, 0);
	  for (int p = 1; p < nvel; p++) {
	    int8_t cx = lb->model.cv[p][X];
	    int8_t cy = lb->model.cv[p][Y];
	    int8_t cz = lb->model.cv[p][Z];
	    int index0 = cs_index(cs, ic - cx, jc - cy, kc - cz);
	    double wp = 0.0;
	    map_status(map, index0, &status1);
	    if (status1 != MAP_FLUID) continue;
	    wp = MP_F(wf, index0, 0)*MP_F(prob, index0, p);
	    sp[X] += wp*cx;
	    sp[Y] += wp*cy;
	    sp[Z] += wp*cz;
	    sp[3] += wp;
	  }
	}

	for (int n = 0; n < 4; n++) MP_F(pa, index, n) = sp[n];
      }
    }
  }

  /* 6. Propagate and accumulate Z_g(t) = sum_r P(r,t;g) u*_g(r) */

  for (int t = 1; t <= mp->nsteps; t++) {

    double sloc[4] = {0};
    double sglb[4] = {0};

    if (t > 1) {
      field_t * tmp = NULL;

      mp_field_halo(pa);

      for (int ic = 1; ic <= nlocal[X]; ic++) {
	for (int jc = 1; jc <= nlocal[Y]; jc++) {
	  for (int kc = 1; kc <= nlocal[Z]; kc++) {

	    int index = cs_index(cs, ic, jc, kc);
	    double sp[4] = {0};

	    map_status(map, index, &status0);

	    if (status0 == MAP_FLUID) {
	      double p0 = MP_F(prob, index, 0);
	      for (int n = 0; n < 4; n++) sp[n] = p0*MP_F(pa, index, n);
	      for (int p = 1; p < nvel; p++) {
		int8_t cx = lb->model.cv[p][X];
		int8_t cy = lb->model.cv[p][Y];
		int8_t cz = lb->model.cv[p][Z];
		int index0 = cs_index(cs, ic - cx, jc - cy, kc - cz);
		double pp = 0.0;
		map_status(map, index0, &status1);
		if (status1 != MAP_FLUID) continue;
		pp = MP_F(prob, index0, p);
		for (int n = 0; n < 4; n++) sp[n] += pp*MP_F(pa, index0, n);
	      }
	    }

	    for (int n = 0; n < 4; n++) MP_F(pb, index, n) = sp[n];
	  }
	}
      }

      tmp = pa;
      pa = pb;
      pb = tmp;
    }

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  for (int ia = 0; ia < 3; ia++) {
	    sloc[ia] += MP_F(pa, index, ia)*MP_F(ust, index, ia);
	  }
	  sloc[3] += MP_F(pa, index, 3);
	}
      }
    }
    MPI_Allreduce(sloc, sglb, 4, MPI_DOUBLE, MPI_SUM, comm);

    for (int ia = 0; ia < 3; ia++) zt[3*t + ia] = sglb[ia];
    normt[t] = sglb[3];
  }

  /* 7. Output (rank 0) */

  {
    double dg[3] = {0};
    char filename[FILENAME_MAX] = {0};
    FILE * fp = NULL;

    for (int ia = 0; ia < 3; ia++) {
      dg[ia] = 0.5*(zt[ia] - vbar[ia]*vbar[ia]);
    }

    if (pe_mpi_rank(pe) == 0) {
      snprintf(filename, FILENAME_MAX, "mp_vacf_q%+d.dat", mp->valency);
      fp = fopen(filename, "w");
      if (fp == NULL) pe_info(pe, "mp_tracer: cannot open %s\n", filename);
    }

    if (fp) {
      fprintf(fp, "# Moment propagation VACF (Rotenberg et al. EPL 2008)\n");
      fprintf(fp, "# valency %d  D %14.7e  lambda %14.7e  tstart %d\n",
	      mp->valency, mp->diffusivity, lambda, mp->tstart);
      fprintf(fp, "# vbar %14.7e %14.7e %14.7e\n", vbar[X], vbar[Y], vbar[Z]);
      fprintf(fp, "# t Zx Zy Zz Dx Dy Dz norm\n");
      fprintf(fp, "%d %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e\n",
	      0, zt[X], zt[Y], zt[Z], dg[X], dg[Y], dg[Z], normt[0]);
    }

    for (int t = 1; t <= mp->nsteps; t++) {
      for (int ia = 0; ia < 3; ia++) {
	dg[ia] += zt[3*t + ia] - vbar[ia]*vbar[ia];
      }
      if (fp) {
	fprintf(fp, "%d %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e\n",
		t, zt[3*t + X], zt[3*t + Y], zt[3*t + Z],
		dg[X], dg[Y], dg[Z], normt[t]);
      }
    }

    if (fp) fclose(fp);

    pe_info(pe, "mp_tracer: vbar  = %14.7e %14.7e %14.7e\n",
	    vbar[X], vbar[Y], vbar[Z]);
    pe_info(pe, "mp_tracer: Z(0)  = %14.7e %14.7e %14.7e\n",
	    zt[X], zt[Y], zt[Z]);
    pe_info(pe, "mp_tracer: D(%d) = %14.7e %14.7e %14.7e\n",
	    mp->nsteps, dg[X], dg[Y], dg[Z]);
    pe_info(pe, "mp_tracer: norm(%d) - 1 = %14.7e\n",
	    mp->nsteps, normt[mp->nsteps] - 1.0);
  }

  free(normt);
  free(zt);
  field_free(pb);
  field_free(pa);
  field_free(ust);
  field_free(prob);
  field_free(wf);

  return 0;
}
