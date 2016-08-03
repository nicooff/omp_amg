#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "c99.h"
#include "name.h"
#include "types.h"
#include "fail.h"
#include "mem.h"
#include "sort.h"
#include "sarray_sort.h"
#include "gs_defs.h"
#include "amg_tools.h"

/*
  The user ids   id[n]   are assigned to unique procs.
  
  Output
    uid --- ulong array; uid[0 ... uid.n-1]  are the ids owned by this proc
    rid_map --- id[i] is uid[ rid_map[i].i ] on proc rid_map[i].p
    map = assign_dofs(...) --- uid[i] is id[map[i]]

    map is returned only when rid_map is null, and vice versa
*/
////////////////////////////////////////////////////////////////////////

#define rid_equal(a,b) ((a).p==(b).p && (a).i==(b).i)


#define nz_pos_equal(a,b) \
  (rid_equal((a).i,(b).i) && rid_equal((a).j,(b).j))

void mat_sort(struct array *const mat,
                     const enum mat_order order, buffer *const buf)
{
  switch(order) {
  case col_major: sarray_sort_4(struct rnz,mat->ptr,mat->n,
                                j.p,0,j.i,0, i.p,0,i.i,0, buf); break;
  case row_major: sarray_sort_4(struct rnz,mat->ptr,mat->n,
                                i.p,0,i.i,0, j.p,0,j.i,0, buf); break;
  }
}

/* assumes matrix is already sorted */
void mat_condense_sorted(struct array *const mat)
{
  struct rnz *dst,*src, *const end=(struct rnz*)mat->ptr+mat->n;
  if(mat->n<=1) return;
  for(dst=mat->ptr;;++dst) {
    if(dst+1==end) return;
    if(nz_pos_equal(dst[0],dst[1])) break;
  }
  for(src=dst+1;src!=end;++src) {
    if(nz_pos_equal(*dst,*src))
      dst->v += src->v;
    else
      *(++dst) = *src;
  }
  mat->n = (dst+1)-(struct rnz*)mat->ptr;
}

void mat_condense(
  struct array *const mat, const enum mat_order order, buffer *const buf)
{
  mat_sort(mat,order,buf); mat_condense_sorted(mat);
}


double get_time(void)
{
  return omp_get_wtime();
}

// z = alpha y + beta M x
double apply_M(double *z, const double alpha, const double *y,
  const double beta, const struct csr_mat *const M, const double *x)
{
  uint i; const uint rn=M->rn;
  const uint *const row_off = M->row_off, *col = M->col;
  const double *a = M->a;
  const double t0 = get_time();
  for(i=0;i<rn;++i) {
    uint j; 
    const uint je=row_off[i+1]; 
    double t = 0;
    for(j=row_off[i]; j<je; ++j) t += (*a++) * x[*col++];
    *z++ = alpha*(*y++) + beta*t;
  }
  return get_time()-t0;
}
