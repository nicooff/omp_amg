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

struct amg_setup_data 
{
    // Same as Matlab
    double tolc;
    double gamma;
    double *n;
    double *nnz;
    double *nnzf;
    double *nnzfp;
    double *m;
    double *rho;
    struct csr_mat **A;
    uint **id;
    double **C;
    double **F;
    double **D;
    struct csr_mat **Af;
    struct csr_mat **Wt;
    struct csr_mat **AfPt;
    // Additional variables
    uint nlevels;
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

double get_time(void);
double apply_M(double *z, const double alpha, const double *y,
    const double beta, const struct csr_mat *const M, const double *x);
double apply_Mt(double *const z, const struct csr_mat *const M, 
    const double *x);

#endif
