/*****************************************************************************
 *
 *  psi_init.c
 *
 *  Various initial states for electrokinetics.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
 
#include "pe.h"
#include "coords.h"
#include "psi_init.h"
#include "psi_rt.h"

//GENERALE
int *fixed_potential = NULL;  
int line_count = 0;
//double POTENTIAL = 15.0; //0.00001665;

//POTENZIALE CONTINUO
static double *mask0 = NULL;

//ADVECTIVE EF
double ef_adv[3] = {0.0, 0.0, 0.0};
double *ef_per = NULL;


//ASSONE
int *axon_patches = NULL;  //patchi dell'assone da filoe
int *indices_patches_local = NULL; //indici dei patch nel dominio locale
int *state_patches_local = NULL; //stato dei patch nel dominio locale
int *timer_refractory_local = NULL; //cronometro nel dominio locale
int *timer_active_local = NULL; //cronometro nel dominio locale
int REFRACTORY_time = 700; //tempo di refrattarietà
int ACTIVE_time = 72; //tempo di attività

int num_patches_local = 0;  //numero di patch nel dominio locale
int line_count_axon = 0;
double rho = 0.00125;
double THRESHOLD_percentage = 0.1; //percentuale per attivare il patch

//EVOLUZIONE POTENZIALE
char **potential_evolution = NULL;
int line_count_evolution = 0;

/*****************************************************************************
 *
 *  psi_init_uniform
 *
 *  Set the charge density for all species to be rho_el everywhere.
 *  The potential is initialised to zero.
 *
 *****************************************************************************/

int psi_init_uniform(psi_t * obj, double rho_el) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n, nk;
  
  assert(obj);
  assert(rho_el >= 0.0);

  cs_nlocal(obj->cs, nlocal);
  psi_nk(obj, &nk);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);

	psi_psi_set(obj, index, 0.0);

	for (n = 0; n < nk; n++) {
	  psi_rho_set(obj, index, n, rho_el);
	}

      }
    }
  }
  
  return 0;
}

/*****************************************************************************
 *
 * psi_init_gouy_chapman
 *
 *  Set rho(x = 1)  = + (1/2NyNz)
 *      rho(x = Lx) = + (1/2NyNz)
 *      rho         = - 1/(NyNz*(Nx-2)) + electrolyte
 *
 *  This sets up the system for Gouy-Chapman.
 *
 *  rho_el is the electrolyte (background) charge density.
 *  sigma is the sufrace charge density at the wall.
 *
 *****************************************************************************/

int psi_init_gouy_chapman(psi_t * obj, map_t * map, double rho_el,
			      double sigma) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpi_cartsz[3];
  int mpi_cartcoords[3];
  double rho_w, rho_i;
  double ltot[3];

  assert(obj);
  assert(map);

  cs_nlocal(obj->cs, nlocal);
  cs_ltot(obj->cs, ltot);
  cs_cartsz(obj->cs, mpi_cartsz);
  cs_cart_coords(obj->cs, mpi_cartcoords);

  /* wall surface charge density */
  rho_w = sigma;

  /* counter charge density */
  rho_i = rho_w * 2.0 *ltot[Y]*ltot[Z] / (ltot[Y]*ltot[Z]*(ltot[X] - 2.0));

  /* apply counter charges & electrolyte */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);

	psi_psi_set(obj, index, 0.0);
	psi_rho_set(obj, index, 0, rho_el);
	psi_rho_set(obj, index, 1, rho_el + rho_i);

      }
    }
  }

  /* apply wall charges */
  if (mpi_cartcoords[X] == 0) {
    ic = 1;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY);

	psi_rho_set(obj, index, 0, rho_w);
	psi_rho_set(obj, index, 1, 0.0);

      }
    }
  }

  if (mpi_cartcoords[X] == mpi_cartsz[X] - 1) {
    ic = nlocal[X];
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY);

	psi_rho_set(obj, index, 0, rho_w);
	psi_rho_set(obj, index, 1, 0.0);

      }
    }
  }

  map_halo(map);
  map_pm_set(map, 1);

  return 0;
}

/*****************************************************************************
 *
 *  psi_init_liquid_junction
 *
 *  This follows Mafe et al., so we set for two species:
 *
 *    rho_left  = rho_el + delta_el / 2
 *    rho_right = rho_el - delta_el / 2
 *
 *  where left and right are separated by the half way point in the
 *  x-direction.
 *
 *  We should have delta_el << rho_el.
 *
 *****************************************************************************/

int psi_init_liquid_junction(psi_t * obj, double rho_el, double delta_el) {

  int ic, jc, kc, index;
  int ntotal[3];
  int nlocal[3], noff[3];

  assert(obj);

  cs_nlocal(obj->cs, nlocal);
  cs_ntotal(obj->cs, ntotal);
  cs_nlocal_offset(obj->cs, noff);

  /* Set electrolyte densities */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);

	psi_psi_set(obj, index, 0.0);

	if ((1 <= noff[X] + ic) && (noff[X] + ic <= ntotal[X]/2)) {
	  psi_rho_set(obj, index, 0, rho_el + 0.5*delta_el);
	  psi_rho_set(obj, index, 1, rho_el + 0.5*delta_el);
	}
	else {
	  psi_rho_set(obj, index, 0, rho_el - 0.5*delta_el);
	  psi_rho_set(obj, index, 1, rho_el - 0.5*delta_el);
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * psi_init_sigma
 *
 *  This sets up surface charges as specified in the porous media file.
 *
 *****************************************************************************/

int psi_init_sigma(psi_t * psi, map_t * map) {

  int ic, jc, kc, index;
  int nlocal[3];
  int ntotal[3];
  double sigma; /* point charge or surface charge density */

  
  assert(psi);
  assert(map);

  cs_nlocal(psi->cs, nlocal);
  cs_ntotal(psi->cs, ntotal);

  fixed_potential = (int *) calloc(ntotal[X] * ntotal[Y] * ntotal[Z], sizeof(int));


  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);
	map_data(map, index, &sigma);

	psi_psi_set(psi, index, 0.0);

	if (sigma) {
	  if (sigma > 0) {
	    psi_rho_set(psi, index, 0, sigma);
	    psi_rho_set(psi, index, 1, 0);
	  }
	  if (sigma < 0) {
	    psi_rho_set(psi, index, 0, 0);
	    psi_rho_set(psi, index, 1, -sigma);
	  }
	}

      }
    }
  }

  map_halo(map);

  return 0;
}

int psi_init_sigma_fixed_potential(psi_t * psi, map_t * map) {

  int ic, jc, kc, index;
  int mpi_cartsz[3];
  int mpicoords[3];
  int ntotal[3];

  cs_cartsz(psi->cs, mpi_cartsz);
  cs_cart_coords(psi->cs, mpicoords);
  cs_ntotal(psi->cs, ntotal);
  //int *fixed_potential;
  //int line_count;
  int counter = mpicoords[X] * (ntotal[Y] * ntotal[Z]) * ntotal[X] / mpi_cartsz[X] + mpicoords[Y] * ntotal[Z] * ntotal[Y] / mpi_cartsz[Y] + mpicoords[Z] * ntotal[Z] / mpi_cartsz[Z];
  int nlocal[3];
  int shift_z = ntotal[Z] * (mpi_cartsz[Z] - 1) / mpi_cartsz[Z];
  int shift_y = ntotal[Y] * ntotal[Z] * (mpi_cartsz[Y] - 1) / mpi_cartsz[Y];

  double sigma; /* point charge or surface charge density */
  fixed_potential = read_map_file("potential_map.txt", &line_count);
  //double * __restrict__ psidata = psi->psi->data;
  
  assert(psi);
  assert(map);

  cs_nlocal(psi->cs, nlocal);
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	      index = cs_index(psi->cs, ic, jc, kc);
	      map_data(map, index, &sigma);

        if (counter >= line_count) {
          printf("Errore: counter %d supera line_count %d\n", counter, line_count);
          
          }
        
        if (fixed_potential[counter] == 1){
          psi->psi->data[addr_rank0(psi->nsites, index)] = POTENTIAL;
          counter++;  
        } 
        else if (fixed_potential[counter] == -1){
          psi->psi->data[addr_rank0(psi->nsites, index)] = -POTENTIAL;
          counter++; 
        }
	      else if (fixed_potential[counter] == 2){
          psi->psi->data[addr_rank0(psi->nsites, index)] = 0.0;
          counter++; 
        }
        else if (fixed_potential[counter] == 0){
	      psi->psi->data[addr_rank0(psi->nsites, index)] = 0.0; //commenta se vuoi farlo partire da 0
          counter++;
        }
        else {
          printf("ERROR READING potential_map \n");
	        exit(0);
          //psi->psi->data[addr_rank0(psi->nsites, index)] = 0.0; //commenta se vuoi farlo partire da 0
          //counter++;
        }
        //printf("row: %d psi: %f \n", ic + jc + kc, psi->psi->data[addr_rank0(psi->nsites, index)]);
	      //psi_psi_set(psi, index, 1.0);

	      if (sigma) {
	        if (sigma > 0) {
	          psi_rho_set(psi, index, 0, sigma);
	          psi_rho_set(psi, index, 1, 0);
	       }
	       if (sigma < 0) {
	          psi_rho_set(psi, index, 0, 0);
	          psi_rho_set(psi, index, 1, -sigma);
	        }
	      }

      }
      counter = counter + shift_z;
    }
    counter = counter + shift_y;
  }

  map_halo(map);
  

  return 0;
}


//////////////////////////////////////////
//
//
//  funzione per evolvere il potenziale
//
//
//////////////////////////////////////////

int psi_evolve_potential(psi_t * psi, map_t * map, int contatore_update_potenziale) {

  int ic, jc, kc, index;

  int nlocal[3];
  int ntotal[3];
  int mpi_cartsz[3];
  int mpicoords[3];

  cs_nlocal(psi->cs, nlocal);
  cs_ntotal(psi->cs, ntotal);
  cs_cartsz(psi->cs, mpi_cartsz);
  cs_cart_coords(psi->cs, mpicoords);

  int shift_z = ntotal[Z] * (mpi_cartsz[Z] - 1) / mpi_cartsz[Z];
  int shift_y = ntotal[Y] * ntotal[Z] * (mpi_cartsz[Y] - 1) / mpi_cartsz[Y];

  //int *fixed_potential;
  //int line_count;
  
  int counter = mpicoords[X] * (ntotal[Y] * ntotal[Z]) * ntotal[X] / mpi_cartsz[X] + mpicoords[Y] * ntotal[Z] * ntotal[Y] / mpi_cartsz[Y] + mpicoords[Z] * ntotal[Z] / mpi_cartsz[Z];
  double sigma; // point charge or surface charge density 
  potential_evolution = read_evolution_file("potential_evolution.txt", &line_count_evolution);
  //double * __restrict__ psidata = psi->psi->data;
  double potential_zero = 0.0;
  assert(psi);
  assert(map);

  cs_nlocal(psi->cs, nlocal);
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	      index = cs_index(psi->cs, ic, jc, kc);
        
	      map_data(map, index, &sigma);

        if (counter >= line_count_evolution) {
          printf("Errore: counter %d supera line_count %d\n", counter, line_count_evolution); 
        }
        else if (strcmp(potential_evolution[counter], "0") == 0){
          if (counter == 0) {
            potential_zero = psi->psi->data[addr_rank0(psi->nsites, index)];
          }
          counter++; 
        }
        else if (strcmp(potential_evolution[counter], "0") != 0){
          if (contatore_update_potenziale == 1){
            if (strcmp(potential_evolution[counter], "t_0+") == 0){
              //psi->psi->data[addr_rank0(psi->nsites, index)] = potential_zero;
              psi->psi->data[addr_rank0(psi->nsites, index)] -= POTENTIAL;
              counter++;  
            } 
            else if (strcmp(potential_evolution[counter], "t_1+") == 0){
              //psi->psi->data[addr_rank0(psi->nsites, index)] = potential_zero + POTENTIAL;
              psi->psi->data[addr_rank0(psi->nsites, index)] += POTENTIAL;
              counter++; 
            }
            else if (strcmp(potential_evolution[counter], "t_0-") == 0){
              //psi->psi->data[addr_rank0(psi->nsites, index)] = potential_zero;
              psi->psi->data[addr_rank0(psi->nsites, index)] -= -POTENTIAL;
              counter++;  
            } 
            else if (strcmp(potential_evolution[counter], "t_1-") == 0){
              //psi->psi->data[addr_rank0(psi->nsites, index)] = potential_zero + POTENTIAL;
              psi->psi->data[addr_rank0(psi->nsites, index)] += -POTENTIAL;
              counter++; 
            }
            else {
              counter++;
            }
          }
          else if (contatore_update_potenziale == 2){
            if (strcmp(potential_evolution[counter], "t_1+") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] -= POTENTIAL;
              counter++;  
            } 
            else if (strcmp(potential_evolution[counter], "t_2+") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] += POTENTIAL;
              counter++; 
            }
            else if (strcmp(potential_evolution[counter], "t_1-") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] -= -POTENTIAL;
              counter++;  
            } 
            else if (strcmp(potential_evolution[counter], "t_2-") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] += -POTENTIAL;
              counter++; 
            }
            else {
              counter++;
            }
           }
          else if (contatore_update_potenziale == 3){
            if (strcmp(potential_evolution[counter], "t_2+") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] -= POTENTIAL;
              counter++;  
            } 
            else if (strcmp(potential_evolution[counter], "t_3+") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] += POTENTIAL;
              counter++; 
            }
            else if (strcmp(potential_evolution[counter], "t_2-") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] -= -POTENTIAL;
              counter++;  
            } 
            else if (strcmp(potential_evolution[counter], "t_3-") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] += -POTENTIAL;
              counter++; 
            }
            else {
              counter++;
            }
          }
          else if (contatore_update_potenziale == 4){
            if (strcmp(potential_evolution[counter], "t_3+") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] -= POTENTIAL;
              counter++;  
            } 
            else if (strcmp(potential_evolution[counter], "t_3-") == 0){
              psi->psi->data[addr_rank0(psi->nsites, index)] -= -POTENTIAL;
              counter++;  
            } 
            else {
              counter++;
            }
          }
        }
        
        
        else {
          printf("ERROR READING potential_evolution.txt \n");
	        exit(0);
        }

        //printf("row: %d psi: %f \n", counter, psi->psi->data[addr_rank0(psi->nsites, index)]);
        // LO COMMENTO PERCHE' NON SO SE SERVA
        /*
	      if (sigma) {
	        if (sigma > 0) {
	          psi_rho_set(psi, index, 0, sigma);
	          psi_rho_set(psi, index, 1, 0);
	       }
	       if (sigma < 0) {
	          psi_rho_set(psi, index, 0, 0);
	          psi_rho_set(psi, index, 1, -sigma);
	        }
	      }
        */ 
      }
      counter = counter + shift_z;
    }
    counter = counter + shift_y;
  }

  map_halo(map);
  

  return 0;
}

//////////////////////////////////////////
//
//
//  funzione per evolvere il potenziale 2
//
//
//////////////////////////////////////////

int psi_evolve_potential_shift(psi_t * psi, map_t * map) {

  int ic, jc, kc, index, index_sx;
  

  int nlocal[3];
  int ntotal[3];
  int mpi_cartsz[3];
  int mpicoords[3];

  cs_nlocal(psi->cs, nlocal);
  cs_ntotal(psi->cs, ntotal);
  cs_cartsz(psi->cs, mpi_cartsz);
  cs_cart_coords(psi->cs, mpicoords);

  int shift_z = ntotal[Z] * (mpi_cartsz[Z] - 1) / mpi_cartsz[Z];
  int shift_y = ntotal[Y] * ntotal[Z] * (mpi_cartsz[Y] - 1) / mpi_cartsz[Y];

  //int *fixed_potential;
  //int line_count;
  
  int counter = mpicoords[X] * (ntotal[Y] * ntotal[Z]) * ntotal[X] / mpi_cartsz[X] + mpicoords[Y] * ntotal[Z] * ntotal[Y] / mpi_cartsz[Y] + mpicoords[Z] * ntotal[Z] / mpi_cartsz[Z];
  double potential_zero = 0.0;
  assert(psi);
  assert(map);

  double *psi_old_sx = calloc(nlocal[X] * nlocal[Y] * nlocal[Z], sizeof(double));
  if (psi_old_sx == NULL) {
      perror("Errore nell'allocare la memoria per psi_old_sx");
      return -1;
  }
  cs_nlocal(psi->cs, nlocal);
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
        index_sx = cs_index(psi->cs, ic-1, jc, kc);
        int idx = (ic - 1) * (nlocal[Y] * nlocal[Z]) + (jc - 1) * nlocal[Z] + (kc - 1);
        if (fixed_potential[counter] != 0) {
          psi_old_sx[idx] = psi->psi->data[addr_rank0(psi->nsites, index_sx)];
        }
      }
      counter = counter + shift_z;	      
    }
    counter = counter + shift_y;
  }

  counter = mpicoords[X] * (ntotal[Y] * ntotal[Z]) * ntotal[X] / mpi_cartsz[X] + mpicoords[Y] * ntotal[Z] * ntotal[Y] / mpi_cartsz[Y] + mpicoords[Z] * ntotal[Z] / mpi_cartsz[Z];


  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
        index = cs_index(psi->cs, ic, jc, kc);
        int idx = (ic - 1) * (nlocal[Y] * nlocal[Z]) + (jc - 1) * nlocal[Z] + (kc - 1);
        if (fixed_potential[counter] != 0) {
          psi->psi->data[addr_rank0(psi->nsites, index)] = psi_old_sx[idx];
        }
      }	     
      counter = counter + shift_z; 
    }
    counter = counter + shift_y;
  }

  map_halo(map);
  

  return 0;
}

//////////////////////////////////////////
//
//
//  funzione per evolvere il potenziale 2
//
//
//////////////////////////////////////////

int psi_evolve_potential_cont(psi_t * psi, map_t * map, int ts) {

  int ic, jc, kc, index, idx;
  int nlocal[3];
  int ntotal[3];
  int mpi_cartsz[3];
  int mpicoords[3];
  int status;

  cs_nlocal(psi->cs, nlocal);
  cs_ntotal(psi->cs, ntotal);
  cs_cartsz(psi->cs, mpi_cartsz);
  cs_cart_coords(psi->cs, mpicoords);

  int shift_z = ntotal[Z] * (mpi_cartsz[Z] - 1) / mpi_cartsz[Z]; //
  int shift_y = ntotal[Y] * ntotal[Z] * (mpi_cartsz[Y] - 1) / mpi_cartsz[Y];
  int counter = mpicoords[X] * (ntotal[Y] * ntotal[Z]) * ntotal[X] / mpi_cartsz[X] + mpicoords[Y] * ntotal[Z] * ntotal[Y] / mpi_cartsz[Y] + mpicoords[Z] * ntotal[Z] / mpi_cartsz[Z];

  if (fixed_potential == NULL) {
    fixed_potential = read_map_file("potential_map.txt", &line_count);
  }
  assert(psi);
  assert(map);

  double *psi_X = calloc(ntotal[X], sizeof(double));
  if (psi_X == NULL) {
      perror("Errore nell'allocare la memoria per psi_old_sx");
      return -1;
  }

  if (mask0 == NULL) {
    mask0 = calloc(ntotal[X], sizeof(double));
    if (mask0 == NULL) {
      perror("Errore alloc mask0");
      return -1;
    }
  }



  
  double tau = 8.0 * 0.85 / 0.02; 
  double t = (double) ts / tau;

  for (int i = 0; i < ntotal[X]; i++) { //parte per potenziale rigido
    if (t < 1){
      psi_X[i] = POTENTIAL;
      //psi_X[i] = POTENTIAL * (7.00672*t-5.94127*t*t);
      //psi_X[i] = POTENTIAL * (70.6209*t-185.181*t*t+115.56*t*t*t); 
      //psi_X[i] = POTENTIAL * (315.12*t-1551.52*t*t+2297.38*t*t*t-1059.98*t*t*t*t);
      //psi_X[i] = POTENTIAL * (959.596*t-7254.27*t*t+18049.7*t*t*t-18367.9*t*t*t*t+6613.96*t*t*t*t*t);
      //psi_X[i] = POTENTIAL * (2635.7*t-28032.4*t*t+102472*t*t*t-170201*t*t*t*t+132200*t*t*t*t*t-39073.4*t*t*t*t*t*t);
    }
    else if (t >= 1){
      psi_X[i] = POTENTIAL;
    }
  }
    //Questa è la parte per traslazione continua di una patch
    /*
    int left_patch_border = 10; // ricordati che il vettore parte da 0
    int right_patch_border = 19; //cella destra della finestra iniziale

    for (int j0 = 0; j0 < ntotal[X]; j0++) { //mascher per potenziale continuo
      if (j0 >= left_patch_border && j0 <= right_patch_border) {
        mask0[j0] = 1.0;  // dentro la finestra iniziale 
      } else {
        mask0[j0] = 0.0;
      }
    }

    if (mpicoords[X] == 0 && mpicoords[Y] == 0 && mpicoords[Z] == 0) {
      printf("Patch initial Left Border: %d \n", left_patch_border);
      printf("Patch initial Right Border: %d \n", right_patch_border);
      printf("Velocity signal: %f \n", velocity_signal);
      printf("Patch initial Width: %d \n",
            right_patch_border - left_patch_border + 1);
    }
    */
  

  //RIMUOVI STA PARTE FINO A GIU SE VUOI USARE CONTINUOUS MOVING PATCH 
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
        index = cs_index(psi->cs, ic, jc, kc);
        map_status(map, index, &status);
        idx = (ic - 1) + ntotal[X] / mpi_cartsz[X] * mpicoords[X];
        if (fixed_potential[counter] != 0 && status != MAP_FLUID && jc >= nlocal[Y]-1) {
          psi->psi->data[addr_rank0(psi->nsites, index)] = psi_X[idx];
        }
        else if (fixed_potential[counter] != 0 && status != MAP_FLUID && jc <= 2) {
          psi->psi->data[addr_rank0(psi->nsites, index)] = -psi_X[idx];
        }
        counter++;
        //printf("row: %d psi: %f \n", counter, psi->psi->data[addr_rank0(psi->nsites, index)]);
      }	      
      counter = counter + shift_z; 
    }
    counter = counter + shift_y; 
  }
  //RIMUOVI STA PARTE FINO A SU SE VUOI USARE CONTINUOUS MOVING PATCH

  /*
  if (ts == 0){
    
   for (int i = 0; i < ntotal[X]; i++) { //parte per potenziale rigido
      if (i >= 10 && i < 20){
        psi_X[i] = POTENTIAL; //potenziale iniziale
      }
      else {
        psi_X[i] = 0.0;
      }
    }
  }
  //FUNZIONE POTENZIALE RIGIDO
  // spostamento in celle (approssimato a int) //
  int shift = (int) (velocity_signal * ts);

  // rendo shift periodico nel dominio [0, Lx) //
  shift = ((shift % ntotal[X]) + ntotal[X]) % ntotal[X];

  // per ogni punto j lungo X decido se è dentro la finestra spostata //
  for (int j = 0; j < ntotal[X]; j++) {
    // “coordinate originali” della finestra a t=0 //
    int j0 = (j - shift + ntotal[X]) % ntotal[X];

    if (j0 >= 10 && j0 < 20) {
      psi_X[j] = POTENTIAL;
    }
    else {
      psi_X[j] = 0.0; 
    }
  }
    // maschera "di base" a t = 0: 1 nella finestra, 0 fuori 

  // spostamento continuo in celle //
  double s = velocity_signal * (double) ts;   // può essere anche negativo //
  // rendo s periodico in [0, Lx) //
  double s_wrapped = fmod(s, (double) ntotal[X]);
  if (s_wrapped < 0.0) s_wrapped += (double) ntotal[X];

  // parte intera e frazionaria dello spostamento //
  int    k = (int) floor(s_wrapped);       // shift intero //
  double a = s_wrapped - (double) k;       // 0 <= a < 1, parte frazionaria

  // traslazione con interpolazione lineare 
  for (int j = 0; j < ntotal[X]; j++) {
    // indici originali che contribuiscono a j 
    int j0 = (j - k + ntotal[X]) % ntotal[X];          // “dietro” di k celle
    int j1 = (j0 - 1 + ntotal[X]) % ntotal[X];         // il vicino ancora più indietro 

    double phi = (1.0 - a)*mask0[j0] + a*mask0[j1];
    //printf("j: %d, j0: %d, j1: %d, a: %f, phi: %f \n", j, j0, j1, a, phi);

    psi_X[j] = POTENTIAL * phi;
  }

  //free(mask0);
  
  cs_nlocal(psi->cs, nlocal);
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
        index = cs_index(psi->cs, ic, jc, kc);
        idx = (ic - 1) + ntotal[X] / mpi_cartsz[X] * mpicoords[X];
        if (fixed_potential[counter] != 0) {
          psi->psi->data[addr_rank0(psi->nsites, index)] = psi_X[idx];
        }
        counter++;
        //printf("row: %d psi: %f \n", counter, psi->psi->data[addr_rank0(psi->nsites, index)]);
      }	      
      counter = counter + shift_z; 
    }
    counter = counter + shift_y; 
  }
  */
  //exit(0);

  map_halo(map);
  free(psi_X);
  
  return 0;
}

int electric_field_periodic(psi_t * psi){

  int ic;
  int nlocal[3];
  int ntotal[3];

  int noffset[3];
  const double pi = 3.14159265358979323846;
  

  cs_nlocal(psi->cs, nlocal);
  cs_ntotal(psi->cs, ntotal);
  cs_nlocal_offset(psi->cs, noffset);
	
  
  if (ef_per) free(ef_per);	
  ef_per = calloc(nlocal[X], sizeof(double));
  
  
  if (ef_per == NULL) {
	pe_fatal(psi->pe, "ef_per non inizializzato\n");
  }
  

  int n = (int) POTENTIAL; // WAVENUMBER
  double k = 2.0*pi*((double) n)/((double) ntotal[X]); 
  //double k = POTENTIAL; 
  double E_0 = 0.0;
  printf("WAVENUMBER = %f \n", k);
  //printf("OFFSET = %d \n", noffset[X]);


  for (ic = 1; ic <= nlocal[X]; ic++) {
	  //continue;
    ef_per[ic-1] = E_0 * sin(k * (ic-1 + noffset[X]));
    //if (ic < nlocal[X]) ef_per[ic-1] = 0.0;
    //if (ic >= nlocal[X]) ef_per[ic-1] = 1.0;
    //printf("x: %d ef_per: %f \n", ic-1, ef_per[ic-1]);
  }
  
  

  return 0;
}

//////////////////////////////////////////
//
//
//  (C)ASSONE
//
//
//////////////////////////////////////////

int psi_init_axon(psi_t * psi, map_t * map) {

  int ic, jc, kc, index;
  int mpi_cartsz[3];
  int mpicoords[3];
  int ntotal[3];
  int nlocal[3];
  int status;
  int nk;
  int nsites;
  

  cs_cartsz(psi->cs, mpi_cartsz);
  cs_cart_coords(psi->cs, mpicoords);
  cs_ntotal(psi->cs, ntotal);
  cs_nlocal(psi->cs, nlocal);
  psi_nk(psi, &nk);
  cs_nsites(psi->cs, &nsites);

  if (mpi_cartsz[Y] != 1 || mpi_cartsz[Z] != 1) {
    pe_fatal(psi->pe, "This function requires a single MPI process in Y and Z directions.\n");
    exit(0);
  }

  int nlocal_domain = nlocal[X] * nlocal[Y] * nlocal[Z];
  int ntotal_domain = ntotal[X] * ntotal[Y] * ntotal[Z];
  double Q_threshold = rho * ntotal_domain * THRESHOLD_percentage;
  printf("Global charge: %f, Threshold to activate axon patches: %f \n",rho * ntotal_domain, Q_threshold);
  int base = mpicoords[X] * (ntotal[Y] * ntotal[Z]) * ntotal[X] / mpi_cartsz[X] + mpicoords[Y] * ntotal[Z] * ntotal[Y] / mpi_cartsz[Y] + mpicoords[Z] * ntotal[Z] / mpi_cartsz[Z];
  int counter = base;
  int counter_axon = base;
  int counter_local = 0;
  int counter_indices = 0;
  int shift_z = ntotal[Z] * (mpi_cartsz[Z] - 1) / mpi_cartsz[Z];
  int shift_y = ntotal[Y] * ntotal[Z] * (mpi_cartsz[Y] - 1) / mpi_cartsz[Y];

  printf("base %d \n", base);


  fixed_potential = read_map_file("potential_map.txt", &line_count);
  axon_patches = read_map_file("axon_patches.txt", &line_count_axon);
  int num_patches_total = count_unique_values(axon_patches, line_count_axon);
  printf("Numero di patch assone totali nel dominio globale: %d\n", num_patches_total);
  indices_patches_local = (int *) calloc(num_patches_total / mpi_cartsz[X], sizeof(int));

  pe_info(psi->pe, "nsites = %d, line_count_axon = %d\n", ntotal_domain, line_count_axon);

  if (line_count_axon != ntotal_domain) {
    pe_fatal(psi->pe,
            "axon_patches.txt ha %d righe ma il dominio ha %d siti.\n"
            "Devi avere una riga per ogni cella del dominio.\n",
            line_count_axon, ntotal_domain);
    exit(0);
  }
    
  int * indices_local_gross = (int *) calloc(nlocal_domain, sizeof(int));

  assert(psi);
  assert(map);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	      index = cs_index(psi->cs, ic, jc, kc);
	      

        if (counter >= line_count) {
          printf("Errore: counter %d supera line_count %d\n", counter, line_count);
        }

        indices_local_gross[counter_local] = axon_patches[counter]; 

        if(ic==1 && jc==1 && kc==1){
          indices_patches_local[0] = indices_local_gross[0];
        }

        if (indices_patches_local[counter_indices] != indices_local_gross[counter_local]) {
          counter_indices++;
          indices_patches_local[counter_indices] = indices_local_gross[counter_local];
          
        }
        counter_local++;
         
        if (fixed_potential[counter] != 0){
          if (fixed_potential[counter] == 1){
            psi->psi->data[addr_rank0(psi->nsites, index)] = POTENTIAL;
            counter++;
          }
          else if (fixed_potential[counter] == 2){
            psi->psi->data[addr_rank0(psi->nsites, index)] = 0.0;
            counter++;
          }
          else if (fixed_potential[counter] == -1){
            psi->psi->data[addr_rank0(psi->nsites, index)] = -POTENTIAL;
            counter++;
          }
          else{
            printf("ERROR READING potential_map \n");
	          exit(0);
          }
            
        } 
        else if (fixed_potential[counter] == 0){
          counter++;
        }
        else {
          printf("ERROR READING potential_map \n");
	        exit(0);
        }

      }
      counter = counter + shift_z;
    }
    counter = counter + shift_y;
  }

  num_patches_local = count_unique_values(indices_local_gross, nlocal_domain);
  free(indices_local_gross);
  state_patches_local = (int *) calloc(num_patches_local, sizeof(int));
  timer_refractory_local = (int *) calloc(num_patches_local, sizeof(int));
  timer_active_local = (int *) calloc(num_patches_local, sizeof(int));


  for (int i = 0; i < num_patches_local; i++) {
    state_patches_local[i] = 0; // inizializza tutti gli stati a 0
    timer_refractory_local[i] = 0;
    timer_active_local[i] = 0;
  }

  // SANITY CHECKS

  if (num_patches_local != num_patches_total / mpi_cartsz[X]) {
    pe_fatal(psi->pe, "Mismatch in number of axon patches: local %d vs total %d / mpi_cartsz[X] %d\n",
             num_patches_local, num_patches_total, mpi_cartsz[X]);
    exit(0);
  } 

  if (nlocal_domain % num_patches_local != 0 || ntotal[X] * ntotal[Y] * ntotal[Z] % num_patches_local != 0 ) {
    pe_fatal(psi->pe, "Mismatch in number of axon patches: total domain or subdomain %d not divisible by local patches %d\n",
             ntotal[X] * ntotal[Y] * ntotal[Z], num_patches_local);
    exit(0);
  }

  printf("Processo %d di %d:\n", mpicoords[X], mpi_cartsz[X]);
  printf("Numero di patch assone locali: %d\n", num_patches_local);
  printf("Numero di patch assone totali: %d\n", num_patches_total);
  for (int i = 0; i < num_patches_local; i++) {
    printf("Patch locale %d: %d\n", i, indices_patches_local[i]);
  }

  for (int i = 0; i < num_patches_local; i++) {
    double patch_rho_local = 0.0;
    counter_axon = base;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {

          if (counter_axon >= line_count_axon) {
            printf("Errore: counter axon %d supera line_count %d\n", counter_axon, line_count_axon);
            exit(0);
          }

          index = cs_index(psi->cs, ic, jc, kc);
          map_status(map, index, &status);

          if (indices_patches_local[i] == axon_patches[counter_axon] && status == MAP_FLUID) {
            for (int n = 0; n < nk; n++) {
              if ( psi->valency[n] < 0) {
              patch_rho_local = patch_rho_local + psi->rho->data[addr_rank1(nsites, nk, index, n)];
              }
            }
          }
          counter_axon++;
        }
        counter_axon = counter_axon + shift_z;
      }
      counter_axon = counter_axon + shift_y;
    }
    if (patch_rho_local >= Q_threshold) {
      state_patches_local[i] = 1; // patch attivata
      timer_active_local[i]++; // inizio il timer di attivazione
      printf("Patch %d attivata con carica %f (soglia %f)\n", indices_patches_local[i], patch_rho_local, Q_threshold);
    } else {
      state_patches_local[i] = 0; // patch non attivata
      printf("Patch %d non attivata con carica %f (soglia %f)\n", indices_patches_local[i], patch_rho_local, Q_threshold);
    }
  }


  for (int i = 0; i < num_patches_local; i++) {
    counter_axon = base;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {

          index = cs_index(psi->cs, ic, jc, kc);
          map_status(map, index, &status);

          if (state_patches_local[i] == 1 && indices_patches_local[i] == axon_patches[counter_axon] && status != MAP_FLUID) {
            psi->psi->data[addr_rank0(psi->nsites, index)] = POTENTIAL;
          }
          else if (state_patches_local[i] == 0 && indices_patches_local[i] == axon_patches[counter_axon] && status != MAP_FLUID) {
            psi->psi->data[addr_rank0(psi->nsites, index)] = 0.0;
          }
          counter_axon++;
        }
        counter_axon = counter_axon + shift_z;
      }
      counter_axon = counter_axon + shift_y;
    }    
  } 

  /*
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        if (ic < 81){
          //printf("cella (%d,%d,%d), potenziale %f, axon patch %d\n",
            //  ic, jc, kc, psi->psi->data[addr_rank0(psi->nsites, cs_index(psi->cs, ic, jc, kc))], axon_patches[counter]);
              //printf("%f\n",psi->psi->data[addr_rank0(psi->nsites, cs_index(psi->cs, ic, jc, kc))]);
        }
        //printf("cella (%d,%d,%d), potenziale %f, axon patch %d\n",
         //     ic, jc, kc, psi->psi->data[addr_rank0(psi->nsites, cs_index(psi->cs, ic, jc, kc))], axon_patches[counter]);
        
      }
    }
  }
  */
  map_halo(map);
  

  return 0;
}

int psi_axon_update(psi_t * psi, map_t * map, int ts) {

  int ic, jc, kc, index;
  int mpi_cartsz[3];
  int mpicoords[3];
  int ntotal[3];
  int nlocal[3];
  int status;
  int nk;
  int nsites;
  

  cs_cartsz(psi->cs, mpi_cartsz);
  cs_cart_coords(psi->cs, mpicoords);
  cs_ntotal(psi->cs, ntotal);
  cs_nlocal(psi->cs, nlocal);
  psi_nk(psi, &nk);
  cs_nsites(psi->cs, &nsites);

  if (mpi_cartsz[Y] != 1 || mpi_cartsz[Z] != 1) {
    pe_fatal(psi->pe, "This function requires a single MPI process in Y and Z directions.\n");
    exit(0);
  }

  int ntotal_domain = ntotal[X] * ntotal[Y] * ntotal[Z];
  double Q_threshold = rho * ntotal_domain * THRESHOLD_percentage;
  printf("Global charge: %f, Threshold to activate axon patches: %f \n",rho * ntotal_domain, Q_threshold);
  int base = mpicoords[X] * (ntotal[Y] * ntotal[Z]) * ntotal[X] / mpi_cartsz[X] + mpicoords[Y] * ntotal[Z] * ntotal[Y] / mpi_cartsz[Y] + mpicoords[Z] * ntotal[Z] / mpi_cartsz[Z];
  int counter_axon = base;
  int shift_z = ntotal[Z] * (mpi_cartsz[Z] - 1) / mpi_cartsz[Z];
  int shift_y = ntotal[Y] * ntotal[Z] * (mpi_cartsz[Y] - 1) / mpi_cartsz[Y];

  for (int i = 0; i < num_patches_local; i++) {
    double patch_rho_local = 0.0;
    counter_axon = base;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {

          if (counter_axon >= line_count_axon) {
            printf("Errore: counter axon %d supera line_count %d\n", counter_axon, line_count_axon);
            exit(0);
          }

          index = cs_index(psi->cs, ic, jc, kc);
          map_status(map, index, &status);

          if (indices_patches_local[i] == axon_patches[counter_axon] && status == MAP_FLUID) {
            for (int n = 0; n < nk; n++) {
              if ( psi->valency[n] < 0) {
              patch_rho_local = patch_rho_local + psi->rho->data[addr_rank1(nsites, nk, index, n)];
              }
            }
          }
          counter_axon++;
        }
        counter_axon = counter_axon + shift_z;
      }
      counter_axon = counter_axon + shift_y;
    }
    if (patch_rho_local >= Q_threshold && state_patches_local[i] != 2) {
      state_patches_local[i] = 1; // patch attivata
      timer_active_local[i]++; // misura quando inizia il periodo refrattario
      //printf("Patch %d attivata con carica %f (soglia %f)\n", indices_patches_local[i], patch_rho_local, Q_threshold);
    } else if (patch_rho_local < Q_threshold && state_patches_local[i] != 2) {
      state_patches_local[i] = 0; // patch non attivata
      timer_active_local[i] = 0; // reset timer refrattario
      //printf("Patch %d non attivata con carica %f (soglia %f)\n", indices_patches_local[i], patch_rho_local, Q_threshold);
    } else if (state_patches_local[i] == 2) {
      timer_refractory_local[i]++;
      if (timer_refractory_local[i] >= REFRACTORY_time) {
        state_patches_local[i] = 0; // torna a stato inattivo
        timer_refractory_local[i] = 0;
        timer_active_local[i] = 0;
        printf("Patch %d esce dal periodo refrattario\n", indices_patches_local[i]);
      }
    }
    if (state_patches_local[i] == 1 && timer_active_local[i] >= ACTIVE_time) {
      state_patches_local[i] = 2; // entra in stato refrattario
      timer_refractory_local[i]++;
      printf("Patch %d entra nel periodo refrattario\n", indices_patches_local[i]);
    }
  }

  for (int i = 0; i < num_patches_local; i++) {
    counter_axon = base;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {

          index = cs_index(psi->cs, ic, jc, kc);
          map_status(map, index, &status);

          if (indices_patches_local[i] == axon_patches[counter_axon] && status != MAP_FLUID) {
            if (state_patches_local[i] == 1) {
              psi->psi->data[addr_rank0(psi->nsites, index)] = POTENTIAL;
            } else {
              /* stati 0 e 2: potenziale nullo */
              psi->psi->data[addr_rank0(psi->nsites, index)] = 0.0;
            }
          }
          counter_axon++;
        }
        counter_axon = counter_axon + shift_z;
      }
      counter_axon = counter_axon + shift_y;
    }    
  } 

  return 0;
}


int electric_field_time_update(int ts){

  double tau = 0.85 * 8.5 / 0.0002;
  double t = ts / tau;
  if (t < 1){
    //ef_adv[X] = 0.001; //005;
    ef_adv[X] = 0.0;
    //ef_adv[X] = 0.00011765 *(258827.0*t-2.10818e9*t*t+6.39132e12*t*t*t-9.17317e15*t*t*t*t+6.318e18*t*t*t*t*t-1.68577e21*t*t*t*t*t*t);
	  //ef_adv[X] = 0.00000011765 *(7006.72*t-5.94127e6*t*t);
    //ef_adv[X] = 0.125 * (7.17729*t-6.17729*t*t); 
    //ef_adv[X] = 0.125 * (25.3915*t-57.568*t*t+33.1765*t*t*t); 
    //ef_adv[X] = 0.125 * (65.5339*t-264.558*t*t+346.885*t*t*t-146.862*t*t*t*t);
    //ef_adv[X] = 0.125 * (141.473*t-858.242*t*t+1850.5*t*t*t-1694.5*t*t*t*t+561.769*t*t*t*t*t);
    //ef_adv[X] = 0.125 * (272.382*t-2259.58*t*t+6965.41*t*t*t-10156.8*t*t*t*t+7103.66*t*t*t*t*t-1924.03*t*t*t*t*t*t);
  }
  if (t >= 1){
    ef_adv[X] = 0.0;
    //ef_adv[X] = 0.001; //005;
  }
  return 0;
}

//////////////////////////////////////////
//
//
//  FUNZIONI DI SERVIZIO
//
//
//////////////////////////////////////////


int count_unique_values(const int *array, int n) {
    if (n == 0) return 0;

    // Copia l'array, così non modifichi axon_patches originale
    int *tmp = malloc(n * sizeof(int));
    if (!tmp) {
        perror("malloc");
        return -1;  // errore
    }

    for (int i = 0; i < n; i++) {
        tmp[i] = array[i];
    }

    // Ordina
    qsort(tmp, n, sizeof(int), compare_int);

    // Conta quanti valori diversi
    int count = 1;  // almeno uno se n > 0
    for (int i = 1; i < n; i++) {
        if (tmp[i] != tmp[i - 1]) {
            count++;
        }
    }

    free(tmp);
    return count;
}

int compare_int(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

//////////////////////////////////////////
//
//
//  funzione per copiare i valori fissi del potenziale
//
//
//////////////////////////////////////////

int* read_map_file(const char *filename, int *line_count) {
  FILE *file;
  char line[1024];
  int *array;
  int value;

  // Apre il file per la lettura
  file = fopen(filename, "r");
  if (file == NULL) {
      perror("Errore nell'aprire il file");
      return NULL;
  }

  // Conta il numero di righe nel file
  *line_count = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
      (*line_count)++;
  }

  // Riavvolge il puntatore del file all'inizio
  rewind(file);

  // Alloca l'array per contenere i valori (0 o 1)
  array = (int *)malloc(*line_count * sizeof(int));
  if (array == NULL) {
      perror("Errore nell'allocare la memoria");
      fclose(file);
      return NULL;
  }

  // Legge il file linea per linea e salva i valori nell'array
  int i = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
      // Converte il valore della riga in un intero (0 o 1)
      if (sscanf(line, "%d", &value) == 1){ //&& (value == 0 || value == 1 || value == -1 || value == 2)) {
          array[i] = value;
          i++;
      } else {
          printf("Errore nel leggere il valore dalla riga: %s", line);
      }
  }

  // Chiude il file
  fclose(file);

  return array;
}

//////////////////////////////////////////
//
//
//  funzione per registrare i valori in cui evolve il potenziale
//
//
//////////////////////////////////////////

char **read_evolution_file(const char *filename, int *line_count_evolution) {
    FILE *file;
    char line[1024];
    char **array;

    file = fopen(filename, "r");
    if (file == NULL) {
        perror("Errore nell'aprire il file");
        return NULL;
    }

    *line_count_evolution = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        (*line_count_evolution)++;
    }

    rewind(file);

    array = (char **)malloc(*line_count_evolution * sizeof(char *));
    if (array == NULL) {
        perror("Errore nell'allocare la memoria");
        fclose(file);
        return NULL;
    }

    int i = 0;
    while (fgets(line, sizeof(line), file) != NULL && i < *line_count_evolution) {
        line[strcspn(line, "\r\n")] = '\0';  // rimuove newline e carriage return
        if (strcmp(line, "t_0+") == 0 || strcmp(line, "t_1+") == 0 ||
            strcmp(line, "t_2+") == 0 || strcmp(line, "t_3+") == 0 ||
            strcmp(line, "t_0-") == 0 || strcmp(line, "t_1-") == 0 ||
            strcmp(line, "t_2-") == 0 || strcmp(line, "t_3-") == 0 || strcmp(line, "0") == 0) {
            array[i] = strdup(line);  // copia la stringa
            i++;
        } else {
            printf("Errore nel leggere il valore dalla riga: %s\n", line);
        }
    }

    /*
      array[i] = strdup(line); // copia la stringa
      array[i][strcspn(array[i], "\n")] = 0; // rimuove newline
      i++;
    */

    fclose(file);
    *line_count_evolution = i;  // aggiornamento reale in caso di linee scartate
    return array;
}