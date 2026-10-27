/*****************************************************************************
 *
 *  psi_sor.c
 *
 *  A solution of the Poisson equation for the potential and
 *  charge densities stored in the psi_t object.
 *
 *  The simple Poisson equation looks like
 *
 *    nabla^2 \psi = - rho_elec / epsilon
 *
 *  where psi is the potential, rho_elec is the free charge density, and
 *  epsilon is a permeability.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2023 The University of Edinburgh
 *
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga (ipagonabarraga@ub.edu)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <mpi.h>

#include "pe.h"
#include "coords.h"
#include "psi_sor.h"
#include "util.h"
#include "psi_init.h"
#include "psi_rt.h"

double *psi_ext = NULL;
//double *ef_per = NULL;

/* Function table */

static psi_solver_vt_t vt_ = {
  (psi_solver_free_ft)  psi_solver_sor_free,
  (psi_solver_solve_ft) psi_solver_sor_solve
};

static psi_solver_vt_t vart_ = {
  (psi_solver_free_ft)  psi_solver_sor_free,
  (psi_solver_solve_ft) psi_solver_sor_var_epsilon_solve
};

/*****************************************************************************
 *
 *  psi_solver_sor_create
 *
 *****************************************************************************/

int psi_solver_sor_create(psi_t * psi, psi_solver_sor_t ** sor) {

  int ifail = 0;
  psi_solver_sor_t * solver = NULL;
  printf("SOR CREATE \n");
  //exit(0);
  //printf("psi: %f \n", psi->psi->data[addr_rank0(psi->nsites, 85)]);

  assert(psi);
  assert(sor);

  solver = (psi_solver_sor_t *) calloc(1, sizeof(psi_solver_sor_t));
  if (solver == NULL) {
    ifail = -1;
  }
  else {
    /* Set the function table ... */
    solver->super.impl = &vt_;
    solver->psi = psi;
  }

  *sor = solver;

  return ifail;
}

/*****************************************************************************
 *
 *  psi_solver_sor_free
 *
 *****************************************************************************/

int psi_solver_sor_free(psi_solver_sor_t ** sor) {

  assert(sor);
  assert(*sor);

  free(*sor);
  *sor  = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  funzione per copiare i valori fissi del potenziale
 *
 *****************************************************************************/
/*
int* read_map_file2(const char *filename, int *line_count) {
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
        if (sscanf(line, "%d", &value) == 1 && (value == 0 || value == 1 || value == -1)) {
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
*/
/*****************************************************************************
 *
 *  psi_solver_sor_solve
 *
 *  Solve Poisson equation with uniform permittivity.
 *
 *  The differencing is a seven point stencil for \nabla^2 \psi. So
 *
 *  epsilon [ psi(i+1,j,k) - 2 psi(i,j,k) + psi(i-1,j,k)
 *          + psi(i,j+1,k) - 2 psi(i,j,k) + psi(i,j-1,k)
 *          + psi(i,j,k+1) - 2 psi(i,j,k) + psi(i,j,k-1) ] = -rho_elec(i,j,k)
 *
 *  We use the asymptotic estimate of the spectral radius for
 *  the Jacobi iteration
 *      radius ~= 1 - (pi^2 / 2N^2)
 *  where N is the linear dimension of the problem. It's important
 *  to get this right to keep the number of iterations as small as
 *  possible.
 *
 *  If this is an initial solve, the initial norm of the residual
 *  may be quite large (e.g., psi(t = 0)  = 0; rhs \neq 0); in this
 *  case a relative tolerance would be appropriate to decide
 *  termination. On subsequent calls, we might expect the initial
 *  residual to be relatively small (psi not charged much since
 *  previous solve), and an absolute tolerance might be appropriate.
 *
 *  The actual residual is checked against both at every 'ncheck'
 *  iterations, and either condition met will result in termination
 *  of the iteration. If neither criterion is met, the iteration will
 *  finish after 'niteration' iterations.
 *
 *  "its" is the time step for statistics purposes.
 *
 *  See, e.g., Press et al. Chapter 19.
 *
 *****************************************************************************/

int psi_solver_sor_solve(psi_solver_sor_t * sor, int its) {
  
  int niteration = 1000;       /* Maximum number of iterations */
  const int ncheck = 5;        /* Check global residual every n iterations */
  
  int nhalo;
  int nlocal[3];
  int ntotal[3];
  int nsites;
  int xs, ys, zs;              /* Memory strides */
  double rho_elec;             /* Right-hand side */
  double residual;             /* Residual at given point */
  double rnorm[2];             /* Initial and current norm of residual */
  double rnorm_local[2];       /* Local values */
  double epsilon;              /* Uniform permittivity */
  double dpsi;
  double omega;                /* Over-relaxation parameter 1 < omega < 2 */
  double radius;               /* Spectral radius of Jacobi iteration */
  double eunit, beta;
  double ltot[3];

  int mpi_cartsz[3];
  int mpicoords[3];

  //int *fixed_potential;
  //int line_count;

  int kk=0;

  MPI_Comm comm;               /* Cartesian communicator */

  psi_t * psi = sor->psi;
  double * __restrict__ psidata = psi->psi->data;
  //fixed_potential = read_map_file2("potential_map.txt", &line_count);


  cs_ltot(psi->cs, ltot);
  cs_nhalo(psi->cs, &nhalo);
  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);
  cs_ntotal(psi->cs, ntotal);
  cs_cart_comm(psi->cs, &comm);
  cs_strides(psi->cs, &xs, &ys, &zs);
  cs_cartsz(psi->cs, mpi_cartsz);
  cs_cart_coords(psi->cs, mpicoords);

  assert(nhalo >= 1);

  /* The red/black operation needs to be tested for odd numbers
   * of points in parallel. */

  assert(nlocal[X] % 2 == 0);
  assert(nlocal[Y] % 2 == 0);
  assert(nlocal[Z] % 2 == 0);

  assert(ntotal[X] % 2 == 0);
  assert(ntotal[Y] % 2 == 0);
  assert(ntotal[Z] % 2 == 0);

  //assert(psi_ext != NULL);
  /* Compute initial norm of the residual */

  radius = 1.0 - 0.5*pow(4.0*atan(1.0)/dmax(ltot[X],ltot[Z]), 2);

  psi_epsilon(psi, &epsilon);
  psi_maxits(psi, &niteration);
  //niteration=1000;
  psi_beta(psi, &beta);
  psi_unit_charge(psi, &eunit);
  
  int counter = 0;

  int shift_z = ntotal[Z] * (mpi_cartsz[Z] - 1) / mpi_cartsz[Z];
  int shift_y = ntotal[Y] * ntotal[Z] * (mpi_cartsz[Y] - 1) / mpi_cartsz[Y];
  //printf("shift_y = %d \n", shift_y);
  //exit(0);
  //shift_y=0;


  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  
  /*counter = 0;
  for (int ic = 1; ic <= nlocal[X]; ic++) {    
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
        int index = cs_index(psi->cs, ic, jc, kc);
        printf("row: %d psiactive: %d \n", counter, fixed_potential[counter]);
        counter++;
        }
      }
  }
  exit(0);
  
  printf("nhalo: %d\n",nhalo);
  printf("nlocalX: %d\n",nlocal[X]);
  printf("nlocalY: %d\n",nlocal[Y]);
  printf("nlocalZ: %d\n",nlocal[Z]);
  
  
 //////PRIMO CICLO !!!!!!!!!
  for (int ic = 1; ic <= nlocal[X]; ic++) {    
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
        int index = cs_index(psi->cs, ic, jc, kc);
        
        printf("index: %d psi: %f \n",index, psi->psi->data[addr_rank0(psi->nsites, index)]);
        
        if (counter >= line_count) {
          printf("Errore: counter %d supera line_count %d\n", counter, line_count);
          
          }
        
        if (fixed_potential[counter] == 1){
          psidata[addr_rank0(nsites, index)] = 5.0;
          counter++;  
        } 
        else if (fixed_potential[counter] == -1){
          psidata[addr_rank0(nsites, index)] = -5.0;
          counter++; 
        }
        else if (fixed_potential[counter] == 0){
          counter++;
        }
        else {
          printf("ERROR READING potential_map \n");
        }
        
      }
    }
  }
  //exit(1);
  counter = 0;
  
  for (int ic = 1; ic <= nlocal[X]; ic++) {    
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
        int index = cs_index(psi->cs, ic, jc, kc);
        printf("row: %d psi: %f \n", counter, psidata[addr_rank0(nsites, index)]);
        counter++;
        }
      }
  }
  */
  
  
  
  
  
  


  rnorm_local[0] = 0.0;

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	      int index = cs_index(psi->cs, ic, jc, kc);
      	psi_rho_elec(psi, index, &rho_elec);

	      /* Non-dimensional potential in Poisson eqn requires e/kT */
	      /* This is just the L2 norm of the right hand side. */

	      residual = eunit*beta*rho_elec;
	      rnorm_local[0] += residual*residual;
      }
    }
  }

  rnorm_local[0] = sqrt(rnorm_local[0]);

  /* Iterate to solution */

  omega = 1.0;
  counter = 0;
  /*
  for(int i = 0; i < 3119; i++) {
    printf("counter: %d numero riga file: %d \n", counter, fixed_potential[counter]);
    counter++;
  }
  */
  //exit(0);

  //printf("mpicoords[X]: %d mpicoords[Y]: %d mpicoords[Z]: %d \n", mpicoords[X], mpicoords[Y], mpicoords[Z]);
  
  for (int n = 0; n < niteration; n++) {
    counter = 0;
    /* Compute current normal of the residual */

    rnorm_local[1] = 0.0;

    for (int pass = 0; pass < 2; pass++) {

      if (pass == 0){
        counter = mpicoords[X] * (ntotal[Y] * ntotal[Z]) * ntotal[X] / mpi_cartsz[X] + mpicoords[Y] * ntotal[Z] * ntotal[Y] / mpi_cartsz[Y] + mpicoords[Z] * ntotal[Z] / mpi_cartsz[Z];
        //printf("pass 0 counter: %d \n", counter);
      }
      else if (pass == 1){
        counter = mpicoords[X] * (ntotal[Y] * ntotal[Z]) * ntotal[X] / mpi_cartsz[X] + mpicoords[Y] * ntotal[Z] * ntotal[Y] / mpi_cartsz[Y] + mpicoords[Z] * ntotal[Z] / mpi_cartsz[Z] + 1;
        //printf("pass 1 counter: %d \n", counter);
      }
      else {
        printf("ERROR \n");
      }
	
      //////SECONDO CICLO !!!!!!!!!

      for (int ic = 1; ic <= nlocal[X]; ic++) { 
	      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	        int kst = 1 + (ic + jc + pass) % 2;
	        for (int kc = kst; kc <= nlocal[Z]; kc += 2) {
            //printf("counter: %d \n", counter);

				//if (counter >= line_count) {
				//  printf("Errore: counter %d supera line_count %d\n", counter, line_count);
				//  exit(1);
				//}
				
				if (n == 0){
					if (pass == 0){
						//printf("pass 0 rank: %d counter: %d numero riga file: %d \n", rank, counter, fixed_potential[counter]);
						//printf("checkpoint 1 counter: %d kst: %d \n", counter, kst);
					}
					if (pass == 1){
						//printf("pass 1 rank: %d counter: %d numero riga file: %d \n", rank, counter, fixed_potential[counter]);
						//printf("checkpoint 1 counter: %d kst: %d \n", counter, kst);
					}
				}
				

				int index = cs_index(psi->cs, ic, jc, kc);
				
			
				if (fixed_potential[counter] == 1 || fixed_potential[counter] == -1 || fixed_potential[counter] == 2){
				  
				  counter = counter + 2;
				}
				

				else {
				  counter = counter + 2;
			
				  psi_rho_elec(psi, index, &rho_elec);

				  ///* 6-point stencil of Laplacian 

				  dpsi
					= psidata[addr_rank0(nsites, index + xs)] 
					+ psidata[addr_rank0(nsites, index - xs)] 
					+ psidata[addr_rank0(nsites, index + ys)]
					+ psidata[addr_rank0(nsites, index - ys)]
					+ psidata[addr_rank0(nsites, index + zs)]
					+ psidata[addr_rank0(nsites, index - zs)]
					- 6.0*psidata[addr_rank0(nsites, index)]; 
          
          /*
          double dpsi_ext
          = psi_ext[ic + 1]
          + psi_ext[ic - 1]
          - 2.0*psi_ext[ic];
          
          dpsi = dpsi + dpsi_ext;
          */

				  ///* Non-dimensional potential in Poisson eqn requires e/kT 

				  residual = epsilon*dpsi + eunit*beta*rho_elec;
				  psidata[addr_rank0(nsites, index)] -= omega*residual / (-6.0*epsilon);
				  rnorm_local[1] += residual*residual;
								
				}

				
				if (/*n == */0){
				  printf("%d rho_elec : %.10f rnorm_local : %.10f \n", kk, rho_elec, rnorm_local[1]);
				  kk++;
				}
				//printf("Numero iter: %d \n", n);
            }
			
			if (pass == 0){
				if (kst % 2 == 0) { //se � l'ultimo implica che: x disp
				counter = counter - 2 + 1 + shift_z; //ricordati il +2 che c'� sopra
				//printf("checkpoint 3 counter: %d numero riga file: %d \n", counter, fixed_potential[counter]);
				}
				else if (kst % 2 != 0) { //se � l'ultimo implica che: x pari
				  counter = counter - 2 + 3 + shift_z; //ricordati il +2 che c'� sopra
				  //printf("checkpoint 3 counter: %d numero riga file: %d \n", counter, fixed_potential[counter]);
				}
			}
			if (pass == 1){
				if (kst % 2 == 0) { //se � l'ultimo implica che: x pari
				counter = counter - 2 + 1 + shift_z; //ricordati il +2 che c'� sopra
				//printf("checkpoint 3 counter: %d numero riga file: %d \n", counter, fixed_potential[counter]);
				}
				else if (kst % 2 != 0) { //se � l'ultimo implica che: x disp 
				  counter = counter - 2 + 3 + shift_z; //ricordati il +2 che c'� sopra
				  //printf("checkpoint 3 counter: %d numero riga file: %d \n", counter, fixed_potential[counter]);
				}
			}
          
        }
        if (pass == 0){
          if (ic % 2 != 0) { //ritorna semplicemente counter + 2 come dentro il ciclo proncipale
            counter = counter - 1 + 2 + shift_y; // ntotal[Z] / mpi_cartsz[Z] * mpicoords[Z] +1; //ricordati il +2 che c'� sopra
          }
          else if (ic % 2 == 0) { 
            counter = counter + 2 - 3 + shift_y;
          }
        }

        if (pass == 1){
          if (ic % 2 == 0) {
            counter = counter - 1 + 2 + shift_y;// ntotal[Z] / mpi_cartsz[Z] * mpicoords[Z] +1; //ricordati il +2 che c'� sopra
          }
          else if (ic % 2 != 0) { 
            counter = counter + 2 - 3 + shift_y;
          }
        }
      }

      //exit(0);

      


      /* Recompute relaxation parameter and next pass */

      if (n == 0 && pass == 0) {
	      omega = 1.0 / (1.0 - 0.5*radius*radius);
      }
      else {
	      omega = 1.0 / (1.0 - 0.25*radius*radius*omega);
      }
      assert(1.0 < omega && omega < 2.0);

      psi_halo_psi(psi);
      psi_halo_psijump(psi);
      
    }
    //exit(0);
    if (0){
      printf("Potenziali a %d \n", n);
      counter = 0;
      for (int ic = 1; ic <= nlocal[X]; ic++) {    
        for (int jc = 1; jc <= nlocal[Y]; jc++) {
          for (int kc = 1; kc <= nlocal[Z]; kc++) {
            int index = cs_index(psi->cs, ic, jc, kc);
            printf("row: %d psi: %f \n", counter, psidata[addr_rank0(nsites, index)]);
            counter++;
          }
        }
      }
    
    }
    

    
    
    if ((n % ncheck) == 0) {

      // Compare residual and exit if small enough 
      pe_t * pe = psi->pe;

      rnorm_local[1] = sqrt(rnorm_local[1]);

      MPI_Allreduce(rnorm_local, rnorm, 2, MPI_DOUBLE, MPI_SUM, comm);
      

      if (rnorm[1] < psi->solver.abstol) {
        
        if (its % psi->solver.nfreq == 0) {
          pe_info(pe, "\n");
          pe_info(pe, "SOR solver converged to absolute tolerance\n");
          pe_info(pe, "SOR residual %14.7e at %d iterations\n", rnorm[1], n);
        }
	    break;
      }
      
      if (rnorm[1] < psi->solver.reltol*rnorm[0]) {
        

        if (its % psi->solver.nfreq == 0) {
          pe_info(pe, "\n");
          pe_info(pe, "SOR solver converged to relative tolerance\n");
          pe_info(pe, "SOR residual %14.7e at %d iterations\n", rnorm[1], n);
        }
	    break;
      }
    }
 
    if (n == niteration-1) {
      
      pe_info(psi->pe, "\n");
      
      pe_info(psi->pe, "SOR solver exceeded %d iterations\n", n+1);
      
      pe_info(psi->pe, "SOR residual %le (initial) %le (final)\n\n",
	      rnorm[0], rnorm[1]);
    }
   
  }
  /*
   counter = 0;
  for (int ic = 1; ic <= nlocal[X]; ic++) {    
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
        int index = cs_index(psi->cs, ic, jc, kc);
        printf("row: %d psi: %f \n", counter, psidata[addr_rank0(nsites, index)]);
        counter++;
        }
      }
  }
  */
  
  

  //exit(0);
  return 0;
  
}

/*****************************************************************************
 *
 *  psi_solver_sor_var_epsilon_create
 *
 *****************************************************************************/

int psi_solver_sor_var_epsilon_create(psi_t * psi, var_epsilon_t user,
				      psi_solver_sor_t ** sor) {
  int ifail = 0;
  psi_solver_sor_t * solver = NULL;

  assert(psi);
  assert(sor);

  solver = (psi_solver_sor_t *) calloc(1, sizeof(psi_solver_sor_t));
  if (solver == NULL) {
    ifail = -1;
  }
  else {
    /* Set the function table etc... */
    solver->super.impl = &vart_;
    solver->psi = psi;
    solver->fe = user.fe;
    solver->epsilon = user.epsilon;
  }

  *sor = solver;

  return ifail;
}

/*****************************************************************************
 *
 *  psi_solver_sor_var_epsilon_solve
 *
 *  This is essentially a copy of the above, but it allows a spatially
 *  varying permittivity epsilon:
 *
 *    div [epsilon(r) grad phi(r) ] = -rho(r)
 *
 *  Only the electro-symmetric free energy is relevant at the moment.
 *
 ****************************************************************************/

int psi_solver_sor_var_epsilon_solve(psi_solver_sor_t * sor, int its) {

  int niteration = 2000;       /* Maximum number of iterations */
  const int ncheck = 1;        /* Check global residual every n iterations */

  int nlocal[3];
  int nsites;
  int xs, ys, zs;              /* Memory strides */

  double rho_elec;             /* Right-hand side */
  double residual;             /* Residual at given point */
  double rnorm[2];             /* Initial and current norm of residual */
  double rnorm_local[2];       /* Local values */

  double depsi;                /* Differenced left-hand side */
  double eps0, eps1;           /* Permittivity values */

  double omega;                /* Over-relaxation parameter 1 < omega < 2 */
  double radius;               /* Spectral radius of Jacobi iteration */

  double ltot[3];
  double eunit, beta;

  MPI_Comm comm;               /* Cartesian communicator */

  psi_t * psi = sor->psi;
  double * __restrict__ psidata = psi->psi->data;

  cs_ltot(psi->cs, ltot);
  cs_nlocal(psi->cs, nlocal);
  cs_nsites(psi->cs, &nsites);
  cs_cart_comm(psi->cs, &comm);
  cs_strides(psi->cs, &xs, &ys, &zs);

  /* The red/black operation needs to be tested for odd numbers
   * of points in parallel. */

  assert(nlocal[X] % 2 == 0);
  assert(nlocal[Y] % 2 == 0);
  assert(nlocal[Z] % 2 == 0);

  /* Compute initial norm of the residual */

  radius = 1.0 - 0.5*pow(4.0*atan(1.0)/dmax(ltot[X],ltot[Z]), 2);

  psi_maxits(psi, &niteration);
  psi_beta(psi, &beta);
  psi_unit_charge(psi, &eunit);


  /* Compute the initial norm of the right hand side. */

  rnorm_local[0] = 0.0;

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(psi->cs, ic, jc, kc);
	psi_rho_elec(psi, index, &rho_elec);

	residual = eunit*beta*rho_elec;
	rnorm_local[0] += residual*residual;
      }
    }
  }

  rnorm_local[0] = sqrt(rnorm_local[0]);

  /* Iterate to solution */

  omega = 1.0;

  for (int n = 0; n < niteration; n++) {

    /* Compute current normal of the residual */

    rnorm_local[1] = 0.0;

    for (int pass = 0; pass < 2; pass++) {

      for (int ic = 1; ic <= nlocal[X]; ic++) {
	for (int jc = 1; jc <= nlocal[Y]; jc++) {
	  int kst = 1 + (ic + jc + pass) % 2;
	  for (int kc = kst; kc <= nlocal[Z]; kc += 2) {

	    int index = cs_index(psi->cs, ic, jc, kc);
	    depsi  = 0.0;

	    psi_rho_elec(psi, index, &rho_elec);
	    sor->epsilon(sor->fe, index, &eps0);

	    /* Laplacian part of operator */

	    depsi += eps0*(-6.0*psidata[addr_rank0(nsites, index)]
			   + psidata[addr_rank0(nsites, index + xs)] 
			   + psidata[addr_rank0(nsites, index - xs)] 
			   + psidata[addr_rank0(nsites, index + ys)]
			   + psidata[addr_rank0(nsites, index - ys)]
			   + psidata[addr_rank0(nsites, index + zs)]
			   + psidata[addr_rank0(nsites, index - zs)]);

	    /* Additional terms in generalised Poisson equation */

	    sor->epsilon(sor->fe, index + xs, &eps1);
	    depsi += 0.25*eps1*(psidata[addr_rank0(nsites, index + xs)]
			      - psidata[addr_rank0(nsites, index - xs)]);

	    sor->epsilon(sor->fe, index - xs, &eps1);
	    depsi -= 0.25*eps1*(psidata[addr_rank0(nsites, index + xs)]
			      - psidata[addr_rank0(nsites, index - xs)]);

	    sor->epsilon(sor->fe, index + ys, &eps1);
	    depsi += 0.25*eps1*(psidata[addr_rank0(nsites, index + ys)]
			      - psidata[addr_rank0(nsites, index - ys)]);

	    sor->epsilon(sor->fe, index - ys, &eps1);
	    depsi -= 0.25*eps1*(psidata[addr_rank0(nsites, index + ys)]
			      - psidata[addr_rank0(nsites, index - ys)]);

	    sor->epsilon(sor->fe, index + zs, &eps1);
	    depsi += 0.25*eps1*(psidata[addr_rank0(nsites, index + zs)]
			      - psidata[addr_rank0(nsites, index - zs)]);

	    sor->epsilon(sor->fe, index - zs, &eps1);
	    depsi -= 0.25*eps1*(psidata[addr_rank0(nsites, index + zs)]
			      - psidata[addr_rank0(nsites, index - zs)]);

	    /* Non-dimensional potential in Poisson eqn requires e/kT */
	    residual = depsi + eunit*beta*rho_elec;
	    psidata[addr_rank0(nsites,index)] -= omega*residual / (-6.0*eps0);
	    rnorm_local[1] += residual*residual;
	  }
	}
      }

      psi_halo_psi(psi);
      psi_halo_psijump(psi);

    }

    /* Recompute relation parameter */
    /* Note: The default Chebychev acceleration causes a convergence problem */ 
    omega = 1.0 / (1.0 - 0.25*radius*radius*omega);

    if ((n % ncheck) == 0) {

      /* Compare residual and exit if small enough */

      rnorm_local[1] = sqrt(rnorm_local[1]);
      MPI_Allreduce(rnorm_local, rnorm, 2, MPI_DOUBLE, MPI_SUM, comm);

      if (rnorm[1] < psi->solver.abstol) {

	if (its % psi->solver.nfreq == 0) {
	  pe_info(psi->pe, "\n");
	  pe_info(psi->pe, "SOR (heterogeneous) solver converged to "
		  "absolute tolerance\n");
	  pe_info(psi->pe, "SOR residual %14.7e at %d iterations\n",
		  rnorm[1], n);
	}
	break;
      }

      if (rnorm[1] < psi->solver.reltol*rnorm[0]) {

	if (its % psi->solver.nfreq == 0) {
	  pe_info(psi->pe, "\n");
	  pe_info(psi->pe, "SOR (heterogeneous) solver converged to "
		  "relative tolerance\n");
	  pe_info(psi->pe, "SOR residual %14.7e at %d iterations\n",
		  rnorm[1], n);
	}
	break;
      }

      if (n == niteration-1) {
	pe_info(psi->pe, "\n");
	pe_info(psi->pe, "SOR solver (heterogeneous) exceeded %d iterations\n",
		n+1);
	pe_info(psi->pe, "SOR residual %le (initial) %le (final)\n\n",
		rnorm[0], rnorm[1]);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  CAMPO ESTERNO
 *
 *  
 *****************************************************************************/

int PSI_ext(psi_t * psi) {

  int nlocal[3];
  int ntotal[3];
  int noffset[3];

  cs_nlocal(psi->cs, nlocal);
  cs_ntotal(psi->cs, ntotal);
  cs_nlocal_offset(psi->cs, noffset);

  free(psi_ext);
  psi_ext = NULL;

  psi_ext = calloc(nlocal[X] + 2, sizeof(double));
  //ef_per = calloc(nlocal[X], sizeof(double));
  assert(psi_ext);

  double * psi_ext_global = NULL;
  psi_ext_global = calloc(ntotal[X], sizeof(double));
  assert(psi_ext_global);

  const double pi = 3.14159265358979323846;
  const double E0 = 0.0001;
  const double K  = 2.0*pi*16/ntotal[X];

  /* Costruzione del vettore globale */
  for (int gx = 0; gx < ntotal[X]; gx++) {
    psi_ext_global[gx] = E0*cos(K*gx)/K;
  }

  /* Copia della parte locale */
  for (int ic = 1; ic <= nlocal[X]; ic++) {
    int gx = noffset[X] + (ic - 1);
    psi_ext[ic] = psi_ext_global[gx];
    //ef_per[ic-1] = E0 * sin(K*0 * (ic-1 + noffset[X]));
  }

  /* Halo sinistro */
  {
    int gx_left = noffset[X] - 1;
    if (gx_left < 0) gx_left += ntotal[X];
    psi_ext[0] = psi_ext_global[gx_left];
  }

  /* Halo destro */
  {
    int gx_right = noffset[X] + nlocal[X];
    if (gx_right >= ntotal[X]) gx_right -= ntotal[X];
    psi_ext[nlocal[X] + 1] = psi_ext_global[gx_right];
  }

  free(psi_ext_global);

  return 0;
}