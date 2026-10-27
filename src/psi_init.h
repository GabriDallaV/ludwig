/*****************************************************************************
*
*  psi_init.h
*
*  Various initial states for electrokinetics.
*
*  $Id$
*
*  Edinburgh Soft Matter and Statistical Physics Group and
*  Edinburgh Parallel Computing Centre
*
*  Oliver Henrich (o.henrich@ucl.ac.uk) wrote most of these.
*  (c) 2012 The University of Edinburgh
*
*****************************************************************************/

#ifndef PSI_INIT_H
#define PSI_INIT_H

#include "psi.h"
#include "map.h"

extern int *fixed_potential;
extern int line_count;

extern double ef_adv[3];
extern double *ef_per;

int psi_init_uniform(psi_t * obj, double rho_el);
int psi_init_gouy_chapman(psi_t * obj, map_t * map, double rho_el,
			      double sigma);
int psi_init_liquid_junction(psi_t * obj, double rho_el, double delta_el);
int psi_init_sigma(psi_t * obj, map_t * map);

//Aggiunte da GDV
int psi_init_sigma_fixed_potential(psi_t * obj, map_t * map);
int psi_evolve_potential(psi_t * psi, map_t * map, int contatore_update_potenziale);
int psi_evolve_potential_shift(psi_t * psi, map_t * map);
int psi_evolve_potential_cont(psi_t * psi, map_t * map, int ts);
int psi_init_axon(psi_t * psi, map_t * map);
int psi_axon_update(psi_t * psi, map_t * map, int ts);
int electric_field_time_update(int ts);
int electric_field_periodic(psi_t * psi);


//Funzioni di servizio
int count_unique_values(const int *array, int n);
int compare_int(const void *a, const void *b);
int* read_map_file(const char *filename, int *line_count);
char **read_evolution_file(const char *filename, int *line_count);


#endif 
