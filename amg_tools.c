#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "c99.h"
#include "name.h"
#include "types.h"
#include "fail.h"
#include "mem.h"
#include "sort.h"
#include "sarray_sort.h"
#include "gs_defs.h"
#include "comm.h"
#include "crystal.h"
//#include "sarray_transfer.h"
#include "gs.h"
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

void mat_distribute(struct array *const mat, const enum distr d, 
    const enum mat_order o, struct crystal *const cr)
{
  switch(d) {
  case row_distr: mat_condense(mat,row_major,&cr->data); break;
//                  sarray_transfer(struct rnz,mat, i.p,0, cr); break;
  case col_distr: mat_condense(mat,col_major,&cr->data); break;
  //                sarray_transfer(struct rnz,mat, j.p,0, cr); break;
  }
  mat_condense(mat,o,&cr->data);
}

void mat_list_nonlocal_sorted(
  struct array *const nonlocal_id,
  const struct array *const mat, const enum distr d,
  const ulong *uid, struct crystal *const cr)
{
  const uint pid = cr->comm.id;
  struct rid last_rid;
  const struct rnz *p, *const e=(const struct rnz*)mat->ptr+mat->n;
  uint count; struct labelled_rid *out, *end;
  #define BUILD_LIST(k) do { \
    last_rid.p=-(uint)1,last_rid.i=-(uint)1; \
    for(count=0,p=mat->ptr;p!=e;++p) { \
      if(p->k.p==pid || rid_equal(last_rid,p->k)) continue; \
      last_rid=p->k; ++count; \
    } \
    array_init(struct labelled_rid, nonlocal_id, count); \
    nonlocal_id->n=count; out=nonlocal_id->ptr; \
    last_rid.p=-(uint)1,last_rid.i=-(uint)1; \
    for(p=mat->ptr;p!=e;++p) { \
      if(p->k.p==pid || rid_equal(last_rid,p->k)) continue; \
      (out++)->rid=last_rid=p->k; \
    } \
  } while(0)
  switch(d) {
    case row_distr: BUILD_LIST(j); break;
    case col_distr: BUILD_LIST(i); break;
  }
  #undef BUILD_LIST
//  sarray_transfer(struct labelled_rid,nonlocal_id,rid.p,1,cr);
  for(out=nonlocal_id->ptr,end=out+nonlocal_id->n;out!=end;++out)
    out->id=uid[out->rid.i];
//  sarray_transfer(struct labelled_rid,nonlocal_id,rid.p,1,cr);
  sarray_sort_2(struct labelled_rid,nonlocal_id->ptr,nonlocal_id->n,
                rid.p,0, rid.i,0, &cr->data);
}

void barrier(const struct comm *c)
{
#ifdef GS_BARRIER
  comm_barrier(c);
#endif
}

double get_time(void)
{
#ifdef GS_TIMING
  return comm_time();
#else
  return 0;
#endif
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
