/*****************************************************************************
 *
 *  moment_propagation.h
 *
 *  Velocity autocorrelation function of charged tracers via moment
 *  propagation (Rotenberg, Pagonabarraga, Frenkel, EPL 83 34004 2008).
 *
 *****************************************************************************/

#ifndef LUDWIG_MOMENT_PROPAGATION_H
#define LUDWIG_MOMENT_PROPAGATION_H

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "lb_data.h"
#include "map.h"
#include "psi.h"

typedef struct mp_tracer_s mp_tracer_t;

int mp_tracer_create_rt(pe_t * pe, cs_t * cs, rt_t * rt, mp_tracer_t ** mp);
int mp_tracer_free(mp_tracer_t ** mp);
int mp_tracer_run(mp_tracer_t * mp, int timestep, lb_t * lb, psi_t * psi,
		  map_t * map);

#endif
