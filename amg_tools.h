#ifndef AMG_TOOLS
#define AMG_TOOLS

/* sparse matrix, condensed sparse row */
struct csr_mat {
  uint rn, cn, *row_off, *col;
  double *a;
};

struct Q { uint nloc; struct gs_data *gsh; };

/* remote id - designates the i-th uknown owned by proc p */
struct rid { uint p,i; };

/* rnz is a mnemonic for remote non zero */
struct rnz {
  double v; struct rid i,j;
};

struct labelled_rid {
  struct rid rid; ulong id;
};

struct crs_data {
  struct comm comm;
  struct gs_data *gs_top;
  uint un, *umap; /* number of unique id's on this proc, map to user ids */
  double tni; /* 1 / (total number of unique ids)  ... for computing mean */
  int null_space;
  unsigned levels;
  unsigned *cheb_m; /* cheb_m  [levels-1] : smoothing steps */
  double *cheb_rho; /* cheb_rho[levels-1] : spectral radius of smoother */
  uint *lvl_offset;
  double *Dff;      /* diagonal smoother, F-relaxation */
  struct Q *Q_W, *Q_AfP, *Q_Aff;
  struct csr_mat *W, *AfP, *Aff;
  double *b, *x, *c, *c_old, *r, *buf;
  double *timing; uint timing_n;
};

uint *assign_dofs(struct array *const uid,
                         struct rid *const rid_map,
                         const ulong *const id, const uint n,
                         const uint p,
                         struct gs_data *gs_top, buffer *const buf);
////////////////////////////////////////////
enum mat_order { row_major, col_major };
enum distr { row_distr, col_distr };

#define rid_equal(a,b) ((a).p==(b).p && (a).i==(b).i)


#define nz_pos_equal(a,b) \
  (rid_equal((a).i,(b).i) && rid_equal((a).j,(b).j))

void mat_sort(struct array *const mat,
                     const enum mat_order order, buffer *const buf);
/* assumes matrix is already sorted */
void mat_condense_sorted(struct array *const mat);

void mat_condense(
  struct array *const mat, const enum mat_order order, buffer *const buf);

void mat_distribute(struct array *const mat, const enum distr d, 
    const enum mat_order o, struct crystal *const cr);

void mat_list_nonlocal_sorted(
  struct array *const nonlocal_id,
  const struct array *const mat, const enum distr d,
  const ulong *uid, struct crystal *const cr);

void barrier(const struct comm *c);
double get_time(void);
double apply_M(double *z, const double alpha, const double *y,
  const double beta, const struct csr_mat *const M, const double *x);

#endif
