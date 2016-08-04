#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>
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
#include "amg_setup.h"

/* 
    Serial version of the AMG setup for Nek5000 based on the Matlab version. 

    Algorithm is based on the the Ph.D. thesis of J. Lottes:
    "Towards Robust Algebraic Multigrid Methods for Nonsymmetric Problems"

    - Author of the original version (Matlab): James Lottes
    - Author of the serial version in C: Nicolas Offermans

    - Last update: 4 August, 2016

    - Status: finished solve_weights.

*/

/*
    REMARKS:
        - at some point in the Matlab code, conj(Af) is used. If I am right, 
          the matrices A or Af should never be complex so I ignored this
          operation in this code. This should be checked though!

        - in function interpolation, u is a vector of 1s (default value) but
          other choices are possible (vector needs to be "near null space"
          --> cf. thesis)
*/

/*
    TODO: 
        - properly check anyvc (Done but not tested)
        - write sym_sparsify
*/

void amg_setup(uint n, const uint *Ai, const uint* Aj, const double *Av,
    struct amg_setup_data *data)
/*    Build data, the "struct crs_data" defined in amg_setup.h that is required
      to execute the AMG solver. */
{   
/* Declare csr matrix A (assembled matrix Av under csr format) */  
    struct csr_mat *A = tmalloc(struct csr_mat, 1);

/* Build A the required data for the setup */
    build_csr(A, n, Ai, Aj, Av);

/* At this point, A is stored on proc 0 using csr format
*/

/**************************************/
/* Memory allocation for data struct. */
/**************************************/

    uint rn = A->rn;
    uint cn = A->cn;

//    data->tni = 1./rn;

    // Initial size for number of sublevels
    // If more levels than this, realloction is required!
    uint initsize = 10; 
    data->cheb_m = tmalloc(uint, initsize);
    data->cheb_rho = tmalloc(double, initsize); 
    data->lvl_offset = tmalloc(uint, initsize+1);

    data->Dff = tmalloc(double, rn);

//    data->Q_W = tmalloc(struct Q, initsize);
//    data->Q_AfP = tmalloc(struct Q, initsize);
//    data->Q_Aff = tmalloc(struct Q, initsize);

    data->W = tmalloc(struct csr_mat, initsize);
    data->AfP = tmalloc(struct csr_mat, initsize);
    data->Aff = tmalloc(struct csr_mat, initsize);

/**************************************/
/* AMG setup (previously Matlab code) */
/**************************************/

    // Sublevel number
    uint slevel = 0;
    
    uint offset = 0;
    data->lvl_offset[slevel] = offset;

/* Tolerances (hard-coded so far) */
    double tol = 0.5; 
    double ctol = 0.7; // Coarsening tolerance
    double itol = 1e-4; // Interpolation tolerance
    double gamma2 = 1. - sqrt(1. - tol);
    double gamma = sqrt(gamma2);
    double stol = 1e-4;

// BEGIN WHILE LOOP

/* Make sure that enough memory is allocated */
    if (slevel > 0 && slevel % initsize == 0)
    {
        int memsize = (slevel/initsize+1)*initsize;
        data->cheb_m = trealloc(uint, data->cheb_m, memsize);
        data->cheb_rho = trealloc(double, data->cheb_rho, memsize); 
        data->lvl_offset = trealloc(uint, data->lvl_offset, memsize);

//        data->Q_W = trealloc(struct Q, data->Q_W, memsize);
//        data->Q_AfP = trealloc(struct Q, data->Q_AfP, memsize);
//        data->Q_Aff = trealloc(struct Q, data->Q_Aff, memsize);

        data->W = trealloc(struct csr_mat, data->W, memsize);
        data->AfP = trealloc(struct csr_mat, data->AfP, memsize);
        data->Aff = trealloc(struct csr_mat, data->Aff, memsize);
    }

/* Coarsen */ 
    double *vc = tmalloc(double, cn);
    // compute vc for i = 1,...,rn
    coarsen(vc, A, ctol); 

    double *vf = tmalloc(double, cn);
    bin_op(vf, vc, cn, not_op); // vf  = ~vc for i = 1,...,cn

/* Smoother */
    // Af = A(F, F)
    struct csr_mat *Af = tmalloc(struct csr_mat, 1);
    sub_mat(Af, A, vf, vf);

    // Letter f denotes dimensions for Af
    uint rnf = Af->rn;
    uint cnf = Af->cn;
    uint ncolf = Af->row_off[rnf];

    // af2 = Af.*Af ( Af.*conj(Af) in Matlab --> make sure Af is never complex)  
    double *af = tmalloc(double, ncolf); 
    memcpy(af, Af->a, ncolf*sizeof(double));
    double *af2 = af;
    vv_op(af2, af2, ncolf, ewmult); // af2 = Af.*Af

    // s = 1./sum(Af.*Af)
    double *s = tmalloc(double, rnf);
    uint i;
    for (i=0; i<rnf; i++)
    {
        uint js = Af->row_off[i];
        uint je = Af->row_off[i+1]; 
        uint nsum = je-js;

        s[i] = array_op(af2, nsum, sum_op); // s = sum(af2)
        af2 += nsum; 
    }

    array_op(s, rnf, minv_op); // s = 1./s

    // D = diag(Af)' .* s
    double *D = tmalloc(double, rnf);
    diag(D, Af);

    vv_op(D, s, rnf, ewmult);

    double gap;

    if (rnf >= 2)
    {
        // Dh = sqrt(D)
        double *Dh = tmalloc(double, cnf);
        memcpy(Dh, D, rnf*sizeof(double));
        array_op(Dh, rnf, sqrt_op);

        struct csr_mat *DhAfDh = tmalloc(struct csr_mat, 1);
        copy_csr(DhAfDh, Af); // DhAfDh = Af
        diagcsr_op(DhAfDh, Dh, dmult); // DhAfDh = Dh*Af
        diagcsr_op(DhAfDh, Dh, multd); // DhAfDh = Dh*Af*Dh

        // Vector of eigenvalues
        double *lambda;
        // Number of eigenvalues
        uint k = lanczos(&lambda, DhAfDh);  

        // First and last eigenvalues
        double a = lambda[0];
        double b = lambda[k-1];

        ar_scal_op(D, 2./(a+b), rnf, mult_op);
        data->Dff += offset;
        data->Dff = D;        

        double rho = (b-a)/(b+a);
        data->cheb_rho[slevel] = rho;
        
        double m, c;
        chebsim(&m, &c, rho, gamma2);
        data->cheb_m[slevel] = m;

        gap = gamma2-c;

        /* Sparsification is skipped for the moment */
        //sym_sparsify(Sf, DhAfDh, (1.-rho)*(.5*gap)/(2.+.5*gap)); => not implemented

        free(Dh);    
        free(lambda);        
    }
    else
    {
        gap = 0;

        data->Dff += offset;
        data->Dff = D;  

        data->cheb_rho[slevel] = 0;
        data->cheb_m[slevel] = 1;
    }
        
    data->Aff = Af;
//    data->Q_Aff->nloc = Af->cn;

/* Interpolation */
    // Afc = A(F, C)
    struct csr_mat *Afc = tmalloc(struct csr_mat, 1);
    sub_mat(Afc, A, vf, vc);

    // Ac = A(C, C)
    struct csr_mat *Ac = tmalloc(struct csr_mat, 1);
    sub_mat(Ac, A, vc, vc);

    // Letter c denotes dimensions for Ac
    //uint rnc = Ac->rn; Unused
    //uint cnc = Ac->cn; Unused
    //uint ncolc = Ac->row_off[rnc]; Unused

    // W
    struct csr_mat *W = tmalloc(struct csr_mat, 1);

    interpolation(W, Af, Ac, Afc, gamma2, itol);
    
/* Update data structure */
    offset += rnf;
    slevel += 1;

    data->lvl_offset[slevel] = offset;

// END WHILE LOOP

    data->levels = slevel;

    // Compute dimensions for remaining arrays
/*    uint max_f = 0, max_e = 0;
    for (i=0; i<slevel; i++)
    {
        uint f = data->lvl_offset[i+1] - data->lvl_offset[i];
        if (f > max_f) max_f = f;

        uint e = data->W[i].cn;
        if (e > max_e) max_e = e;

        e = data->AfP[i].cn;
        if (e > max_e) max_e = e;

        e = data->Aff[i].cn;
        if (e > max_e) max_e = e;
    }

    // Initialize remaining arrays of data structure to 0
    data->b = tmalloc(double, rn);
    init_array(data->b, rn, 0.);
    data->x = tmalloc(double, rn);
    init_array(data->x, rn, 0.);

    data->c = tmalloc(double, max_f);
    init_array(data->c, max_f, 0.);
    data->c_old = tmalloc(double, max_f);
    init_array(data->c_old, max_f, 0.);
    data->r = tmalloc(double, max_f);
    init_array(data->r, max_f, 0.);

    data->buf = tmalloc(double, max_e);
    init_array(data->buf, max_e, 0.);

    data->timing_n = 0;
    data->timing = tmalloc(double, 6*(slevel-1));
    init_array(data->timing, 6*(slevel-1), 0.);*/

/* Free */
    // Free arrays
    free(vc);
    free(af);
    free(s);
    free(D);

    // Free csr matrices
    free_csr(&Af);
    free_csr(&Afc);
    free_csr(&Ac);
    //free_csr(&W);

    free_csr(&A);
}

/* Interpolation */
void interpolation(struct csr_mat *W, struct csr_mat *Af, struct csr_mat *Ac, 
    struct csr_mat *Ar, double gamma2, double tol)
{

    // Dimensions of the matrices
    uint rnf = Af->rn;//, cnf = Af->cn; Unused
    uint rnc = Ac->rn, cnc = Ac->cn;
    // uint rnr = Ar->rn, cnr = Ar->cn;
    // rnr = rnf and cnr = cnc

    // If nc==0
    if (rnc == 0)
    {
        W->rn = rnf;
        W->cn = 0;
    }

    // d = 1./diag(Af)
    double *Df = tmalloc(double, rnf);
    diag(Df, Af);

    double *Dfinv = tmalloc(double, rnf);
    diag(Dfinv, Af);
    array_op(Dfinv, rnf, minv_op);

    // uc = ones(nf, 1) /!\ in the Matlab code, uc could be diffent from ones
    //                      though this is the default value.
    double *uc = tmalloc(double, cnc);
    init_array(uc, cnc, 1.);

    // v = pcg(Af,full(-Ar*uc),d,1e-16);
    double *r = tmalloc(double, rnf); 
    apply_M(r, 0, uc, -1, Ar, uc);

    double *v = tmalloc(double, rnf); 

    double *b = tmalloc(double, rnf);
    init_array(b, rnf, 1.0);    

    pcg(v, Af, r, Df, 1e-16, b);   
    
    // dc = diag(Ac)
    double *Dc = tmalloc(double, cnc);
    diag(Dc, Ac);

    double *Dcinv = tmalloc(double, cnc);
    diag(Dcinv, Ac);
    array_op(Dcinv, rnc, minv_op);

    // W_skel = intp.min_skel( (Ar/Dc) .* (Df\Ar) );
    struct csr_mat *ArD = tmalloc(struct csr_mat, 1); //ArD = (Ar/Dc) .* (Df\Ar)
    copy_csr(ArD, Ar);
    
    array_op(ArD->a, ArD->row_off[ArD->rn], sqr_op);

    diagcsr_op(ArD, Dfinv, dmult);
    diagcsr_op(ArD, Dcinv, multd);

    // Minimum interpolation skeleton
    struct csr_mat *W_skel = tmalloc(struct csr_mat, 1);
    min_skel(W_skel, ArD);

    free_csr(&ArD);

    // Initialize eigenvalues
    double *lam = tmalloc(double, rnf);
    init_array(lam, rnf, 0.);

    //while(true){

        struct csr_mat *Wtmp = tmalloc(struct csr_mat, 1);            
        struct csr_mat *W0   = tmalloc(struct csr_mat, 1);

        solve_weights(Wtmp, W0, lam, W_skel, Af, Ar, rnc, Dc, uc, v, tol);
        
        struct csr_mat *Arhat0 = tmalloc(struct csr_mat, 1);
        struct csr_mat *Arhat  = tmalloc(struct csr_mat, 1);

        mxm(Arhat0, Af, W0  , 0.0);
        mxm(Arhat , Af, Wtmp, 0.0);
        

        print_csr(Arhat);

        free_csr(&Wtmp); 
        free_csr(&W0); 
        free_csr(&Arhat0); 
        free_csr(&Arhat); 

    //}

    free(Df);
    free(Dfinv);
    free(Dc);
    free(Dcinv);
    free(uc);
    free(v);
    free(r);

    // free matrices
    free_csr(&W_skel);

}

/* Solve interpolation weights */
void solve_weights(struct csr_mat *W, struct csr_mat *W0, double *lam, 
    struct csr_mat *W_skel, struct csr_mat *Af, struct csr_mat *Ar, uint rnc,
    double *alpha, double *u, double *v, double tol)
{
    // Dimensions of the matrices
    uint rnf = Af->rn; //, cnf = Af->cn; Unused
    //uint rnr = Ar->rn, cnr = Ar->cn; Unused

    // au = alpha.*u
    double *au = tmalloc(double, rnc);
    memcpy(au, alpha, rnc*sizeof(double));
    vv_op(au, u, rnc, ewmult);

    // W0t is initialised with W_skel'
    struct csr_mat *W0t = tmalloc(struct csr_mat, 1);
    transpose(W0t, W_skel); 

    // Arminust = -Ar'
    struct csr_mat *Arminust = tmalloc(struct csr_mat, 1);
    transpose(Arminust, Ar); 
    ar_scal_op(Arminust->a,-1.0,Arminust->row_off[Arminust->rn],mult_op);
    
    // au = alpha.*u
    double *zeros = tmalloc(double, rnf);
    init_array(zeros, rnf, 0.0);  

    // Matrices W0, Af and Ar have to be transposed
    // Af is assumed symmetric so not transposed!
    interp(W0t, Af, Arminust, au, zeros);

    transpose(W0, W0t); 

    free_csr(&W0t);

    // Matrices W_skel has to be transposed
    struct csr_mat *W_skelt = tmalloc(struct csr_mat, 1);
    transpose(W_skelt, W_skel); 

    solve_constraint(lam, W_skel, W_skelt, Af, W0, alpha, u, v, tol);

    // Wt is initialised by W_skel'
    struct csr_mat *Wt = W_skelt;

    interp(Wt, Af, Arminust, au, lam);

    free_csr(&Arminust);

    transpose(W, Wt);

    free_csr(&Wt); 
}

/* Solve constraint
   This function requires both W_skel and W_skel' so both are given as input
   as W_skel' is already available in solve_weights and we don't want to
   recompute it. */
void solve_constraint(double *lam, struct csr_mat *W_skel,
    struct csr_mat *W_skelt, struct csr_mat *Af, struct csr_mat *W0, 
    double *alpha, double *u, double *v, double tol)
{
    uint nf = W_skel->rn, nc = W_skel->cn;
    double *au2 = tmalloc(double, nc);

    memcpy(au2, u, nc*sizeof(double)); // au2 = u
    array_op(u, nc, sqr_op); // au2 = u.^2
    vv_op(au2, alpha, nc, ewmult); // au2 = alpha.*(u.^2)

    // Need to initialize S by (W_skel*W_skel') before calling interp_lmop
    struct csr_mat *S = tmalloc(struct csr_mat, 1);
    mxm(S, W_skel, W_skel, 1.0);  
    
    // S and Af are assumed to be symmetric -> not transposed
    interp_lmop(S, Af, au2, W_skelt);

    double *resid = tmalloc(double, nf);

    apply_M(resid, 1.0, v, -1.0, W0, u);
    
    double *d = tmalloc(double, nf);

    diag(d, S);

    double *dlogic = tmalloc(double, nf);
    mask_op(dlogic, d, nf, 0.0, ne);

    uint i, ifall = 0;
    for (i=0;i<nf;i++)
    {
        if (dlogic[i]==0.)
        {
            ifall = 1;
            lam[i] = 0.;
        }
    }

    if (ifall != 0)
    {
        struct csr_mat *subS = tmalloc(struct csr_mat, 1);
        sub_mat(subS, S, dlogic, dlogic);
        free_csr(&S);
        S = subS;
    }

    uint ncond = condense_array(resid, dlogic, 0.0, nf);
    condense_array(d, dlogic, 0.0, nf);
    double *lam_cond = tmalloc(double, nf);
    memcpy(lam_cond, lam, nf*sizeof(double));
    condense_array(lam_cond, dlogic, 0.0, nf);

    double *q = tmalloc(double, ncond);
    apply_M(q, 1., resid, -1., S, lam_cond);    

    array_op(d, ncond, minv_op);

    double *x = tmalloc(double, ncond);

    pcg(x, S, q, d, tol, resid);

    for (i=0;i<nf;i++)
    {
        if (dlogic[i] != 0.) lam[i] = *x++;
    }

// Outpost for checking
    printf("Eigenvalues when exiting solve_constraint:\n");
    uint ii;
    for (ii=0;ii<nf;ii++) printf("lam[%u] = %lf\n", ii, lam[ii]);
//

    free_csr(&S);
    free(au2);
    free(resid);
    free(d);
    free(dlogic);
    free(lam_cond);
}

/* Interp_lmop 
   St, At and W_skelt are assumed to be transposed
   St needs to be initialized to (W_skel*W_skel')'! */
int interp_lmop(struct csr_mat *St, struct csr_mat *At, double *u,
    struct csr_mat *W_skelt)
{
    uint nf = W_skelt->rn, nc = W_skelt->cn;
    uint max_nz=0, max_Q;
    uint i;
  
    double *sqv1, *sqv2;
    double *Q, *QQt;
  
    for(i=0;i<nf;++i) {
        uint nz=W_skelt->row_off[i+1]-W_skelt->row_off[i];
        if(nz>max_nz) max_nz=nz;
    }
    max_Q = (max_nz*(max_nz+1))/2;
  
    if(!(sqv1=tmalloc(double, 2*max_nz + max_Q + max_nz*max_nz)))
        return 0;
    sqv2 = sqv1+max_nz, Q = sqv2+max_nz, QQt = Q+max_Q;

    { uint nz=St->row_off[nc]; for(i=0;i<nz;++i) St->a[i]=0.0; }
    for(i=0;i<nf;++i)
    {
        const uint *Qj = &W_skelt->col[W_skelt->row_off[i]];
        uint nz = W_skelt->row_off[i+1]-W_skelt->row_off[i];
        uint m,k;
        double *qk = Q;
        double ui = u[i];
        for(k=0;k<nz*nz;++k) QQt[k]=0;
        for(k=0;k<nz;++k,qk+=k) 
        {
            double alpha;
            uint s = Qj[k];
            /* sqv1 := R_(k+1) A e_s */
            sp_restrict_sorted(sqv1, k+1,Qj, At->row_off[s+1]-At->row_off[s],
                &At->col[At->row_off[s]], &At->a[At->row_off[s]]);
            /* sqv2 := Q^t A e_s */
            mv_utt(sqv2, k,Q, sqv1);
            /* qk := Q Q^t A e_s */ 
            mv_ut(qk, k,Q, sqv2);
            /* alpha := ||(I-Q Q^t A)e_s||_A^2 = (A e_s)^t (I-Q Q^t A)e_s */
            alpha = sqv1[k];
            for(m=0;m<k;++m) alpha -= sqv1[m] * qk[m];
            /* qk := Q e_(k+1) = alpha^{-1/2} (I-Q Q^t A)e_s */
            alpha = -1.0 / sqrt(alpha);
            for(m=0;m<k;++m) qk[m] *= alpha;
            qk[k] = -alpha;
            /* QQt := QQt + qk qk^t */
            for(m=0;m<=k;++m) 
            {
                uint j, mnz = m*nz; double qkm = qk[m];
                for(j=0;j<=k;++j) QQt[mnz+j] += qkm * qk[j];
            }
        }   
        /* St := St + u_j QQt */
        qk=QQt;
        for(k=0;k<nz;++k,qk+=nz) 
        {
            uint j = Qj[k], tj = St->row_off[j];
            sp_add(St->row_off[j+1]-tj,&St->col[tj],&St->a[tj], ui, nz,Qj,qk);
        }
    }
    free(sqv1);
    return 1;
}

/*--------------------------------------------------------------------------
   sparse add
   
   y += alpha * x
   
   the sparse vector x is added to y
   it assumed that yi and xi are sorted,
   and that xi is a subset of yi
--------------------------------------------------------------------------*/
static void sp_add(uint yn, const uint *yi, double *y, double alpha,
                   uint xn, const uint *xi, const double *x)
{
  const uint *xe = xi+xn;
  uint iy;
  if(yn==0) return; iy = *yi;
  for(;xi!=xe;++xi,++x) {
    uint ix = *xi;
    while(iy<ix) ++y, iy=*(++yi);
    *y++ += alpha * (*x), iy=*(++yi);
  }
}

/* Matrix-matrix multiplication
   if (iftrsp == 0.0) X = A*B
   if (iftrsp != 0.0) X = A*(B^T)

   Matrix B needs to be stored using Condensed Sparse Column.
   This is done by transposing it. Therefore, B is transposed if (iftrsp == 0.0)
   and vice versa (which is counterintuitive).
*/
void mxm(struct csr_mat *X, struct csr_mat *A, struct csr_mat *B, 
    double iftrsp)
{
    uint rna = A->rn, cna = A->cn;
    // uint rnb = B->rn, cnb = B->cn; Unused

    // Transpose B if necessary
    struct csr_mat *Bt;
    if (iftrsp == 0.0) 
    {
        Bt = tmalloc(struct csr_mat, 1);
        transpose(Bt, B);
    }
    else Bt = B;

    uint rnbt = Bt->rn, cnbt = Bt->cn;

    if (cna != cnbt)
    {
        printf("Mismatch in matrix dimensions in mxm, cna != cnbt.");
        die(0);
    }

    // Compute number of non zeros in X and allocate memory
    uint ia, ib, ja, jb, jsa, jea, jsb, jeb;
    uint nnzx = 0; 
    for (ia=0;ia<rna;ia++)
    {
        jsa = A->row_off[ia];   
        jea = A->row_off[ia+1]; 
        for (ib=0;ib<rnbt;ib++)
        {
            jsb = Bt->row_off[ib];   
            jeb = Bt->row_off[ib+1];
            for (ja=jsa,jb=jsb;(ja<jea && jb<jeb);) 
            {
                if (A->col[ja] == Bt->col[jb]) 
                {
                    nnzx += 1;
                    break;
                }
                A->col[ja] < Bt->col[jb] ? ja++ : jb++; 
            }
        }
    }    

    malloc_csr(X, rna, rnbt, nnzx);

    X->row_off[0] = 0;

    // Multiply each line of A by B and build X
    double *x = tmalloc(double, cnbt);
    double *y = tmalloc(double, rnbt);
  
    uint counter = 0;
    for (ia=0;ia<rna;ia++)
    {   
        init_array(x, cnbt, 0.0);
        jsa = A->row_off[ia];   
        jea = A->row_off[ia+1];
        X->row_off[ia+1] = X->row_off[ia];  
        for (ja=jsa;ja<jea;ja++) x[A->col[ja]] = A->a[ja];
        apply_M(y, 0.0, x, 1.0, Bt, x);
    
        for (ib=0;ib<rnbt;ib++)
        {
            if (y[ib] != 0.0)
            {
                X->row_off[ia+1] += 1;
                X->a[counter] = y[ib];
                X->col[counter] = ib;
                counter++;
            }
        }
    }

    if (iftrsp == 0.0) 
    {
        free_csr(&Bt);
    }
}

/* Transpose csr matrix */
void transpose(struct csr_mat *At, const struct csr_mat *A)
{
    // Build matrix using coordinate list format
    uint rn = A->rn;
    uint cn = A->cn;
    uint nnz = A->row_off[rn];

    coo_mat *coo_A = tmalloc(coo_mat, nnz);

    uint i, j, je, js;
    for (i=0;i<rn;i++)
    {
        js = A->row_off[i];
        je = A->row_off[i+1];        
        for (j=js;j<je;j++)
        {
            coo_A[j].i = i;
            coo_A[j].j = A->col[j];
            coo_A[j].v = A->a[j];  
        }
    }
    
    // Sort matrix by columns then rows
    buffer buf = {0};
    sarray_sort_2(coo_mat, coo_A, nnz, j, 0, i, 0, &buf);
    buffer_free(&buf);

    // Build transpose matrix
    uint rnt = cn, cnt = rn;
    malloc_csr(At, rnt, cnt, nnz);
    uint row_cur, row_prev = coo_A[0].j, counter = 1;
    At->row_off[0] = 0;
    
    for (i=0;i<nnz;i++)
    {
        // row_off
        row_cur = coo_A[i].j;
        if (row_cur != row_prev)     
        {    
            uint k;
            for (k=row_prev;k<row_cur;k++)
            {
                At->row_off[counter++] = i;
            }
        }
        if (i == nnz-1) At->row_off[counter] = nnz;
    
        row_prev = row_cur;
        At->col[i] = coo_A[i].i; // col
        At->a[i] = coo_A[i].v;   // a
    }

    free(coo_A);
}

/* Build interpolation matrix.
   It is assumed that matrix W is initialized with minimum skeleton.
   Wt, At and Bt are transposed matrices ! */
void interp(struct csr_mat *Wt, struct csr_mat *At, struct csr_mat *Bt, 
    double *u, double *lambda)
{
    uint nf = Wt->rn; //, nc = Wt->cn;
    uint max_nz = 0, max_Q;
    uint i;

    double *sqv1, *sqv2;
    double *Q;

    for (i=0;i<nf;i++)
    {
        uint nz = Wt->row_off[i+1]-Wt->row_off[i];
        if (nz>max_nz) max_nz = nz;
    }
    
    max_Q = (max_nz*(max_nz+1))/2;

    sqv1 = tmalloc(double, 2*max_nz + max_Q);
    sqv2 = sqv1+max_nz, Q = sqv2+max_nz;

    for(i=0;i<nf;++i) 
    {
        uint wir = Wt->row_off[i];
        const uint *Qj = &Wt->col[wir];
        uint nz = Wt->row_off[i+1]-wir;
        uint m,k;
        double *qk = Q;
        for(k=0;k<nz;++k,qk+=k) 
        {
            double alpha;
            uint s = Qj[k];
            // sqv1 := R_(k+1) A e_s
            sp_restrict_sorted(sqv1, k+1,Qj, At->row_off[s+1]-At->row_off[s],
            &At->col[At->row_off[s]], &At->a[At->row_off[s]]);
            // sqv2 := Q^t A e_s 
            mv_utt(sqv2, k,Q, sqv1);
            // qk := Q Q^t A e_s 
            mv_ut(qk, k,Q, sqv2);
            // alpha := ||(I-Q Q^t A)e_s||_A^2 = (A e_s)^t (I-Q Q^t A)e_s
            alpha = sqv1[k];
            for(m=0;m<k;++m) alpha -= sqv1[m] * qk[m];
            // qk := Q e_(k+1) = alpha^{-1/2} (I-Q Q^t A)e_s
            alpha = -1.0 / sqrt(alpha);
            for(m=0;m<k;++m) qk[m] *= alpha;
            qk[k] = -alpha;
        }
        // sqv1 := R B e_i
        sp_restrict_sorted(sqv1, nz,Qj, Bt->row_off[i+1]-Bt->row_off[i],
                           &Bt->col[Bt->row_off[i]], &Bt->a[Bt->row_off[i]]);
        // sqv1 := R (B e_i + u_i lambda)
        for(k=0;k<nz;++k) sqv1[k] += u[i]*lambda[Qj[k]];
        // sqv2 := Q^t (B e_i + u_i lambda)
        mv_utt(sqv2, nz,Q, sqv1);
        // X e_i := Q Q^t (B e_i + u_i lambda)
        mv_ut(&Wt->a[wir], nz,Q, sqv2);

    }

    free(sqv1);
}

////////////////////////////////////////////////////////////////////////////////

/* Upper triangular transpose matrix vector product
   y[0] = U[0] * x[0]
   y[1] = U[1] * x[0] + U[2] * x[1]
   y[2] = U[3] * x[0] + U[4] * x[1] + U[5] * x[2]
   ... */
static void mv_utt(double *y, uint n, const double *U, const double *x)
{
  double *ye = y+n; uint i=1;
  for(;y!=ye;++y,++i) {
    double v=0;
    const double *xp=x, *xe=x+i;
    for(;xp!=xe;++xp) v += (*U++) * (*xp);
    *y=v;
  }
}

/* Upper triangular matrix vector product
   y[0] = U[0] * x[0] + U[1] * x[1] + U[3] * x[2] + ...
   y[1] =               U[2] * x[1] + U[4] * x[2] + ...
   y[2] =                             U[5] * x[2] + ...
   ... */
static void mv_ut(double *y, uint n, const double *U, const double *x)
{
  uint i,j;
  for(j=0;j<n;++j) {
    y[j]=0;
    for(i=0;i<=j;++i) y[i] += (*U++) * x[j];
  }
}

/*--------------------------------------------------------------------------
   sparse restriction
   
   y := R * x
   
   the sparse vector x is restricted to y
   R is indicated by map_to_y
   map_to_y[i] == k   <->    e_k^t R == e_i^t I
   map_to_y[i] == -1  <->    row i of I not present in R
--------------------------------------------------------------------------*/
static void sp_restrict_unsorted(double *y, uint yn, const uint *map_to_y,
    uint xn, const uint *xi, const double *x)
{
  const uint *xe = xi+xn; uint i;
  for(i=0;i<yn;++i) y[i]=0;
  for(;xi!=xe;++xi,++x) {
    uint i = map_to_y[*xi];
    y[i]=*x; // if(i>=0) y[i]=*x; comparison of unsigned expression >= 0 is always true
  }
}

/*--------------------------------------------------------------------------
   sparse restriction
   
   y := R * x
   
   the sparse vector x is restricted to y
   Ri[k] == i   <->   e_k^t R == e_i^t I
   Ri must be sorted
--------------------------------------------------------------------------*/
static void sp_restrict_sorted(double *y, uint Rn, const uint *Ri, uint xn, 
    const uint *xi, const double *x)
{
  const uint *xe = xi+xn;
  double *ye = y+Rn;
  uint iy;
  if(y==ye) return; iy = *Ri;
  for(;xi!=xe;++xi,++x) {
    uint ix = *xi;
    while(iy<ix) { *y++ = 0; if(y==ye) return; iy = *(++Ri); }
    if(iy==ix) { *y++ = *x; if(y==ye) return; iy = *(++Ri); }
  }
  while(y!=ye) *y++ = 0;
}


////////////////////////////////////////////////////////////////////////////////

/* Minimum skeleton */
void min_skel(struct csr_mat *W_skel, struct csr_mat *R)
{
    uint rn = R->rn;
    uint cn = R->cn;

    uint i, k;
    double *y_max = tmalloc(double, rn);
    init_array(y_max, rn, -DBL_MAX);

    // Create the array W_skel
    W_skel->rn = rn;
    W_skel->cn = cn;
    W_skel->row_off = tmalloc(uint,rn+1);
    W_skel->col = tmalloc(uint, rn);
    W_skel->a = tmalloc(double, rn);
    
    for (i = 0; i < rn; i++) 
    {
	    uint j = 0;
	    for(k = R->row_off[i]; k < R->row_off[i+1]; k++) 
        {
	        if (R->a[k] > y_max[i]) 
            {
		        y_max[i] = R->a[k];
		        j = R->col[k];
	        }
	    }
	    if (y_max[i] > 0.0) 
        {
	        W_skel->a[i] = 1.0; 
	    }
	    else 
        {
	        W_skel->a[i] = 0.0;
	    }
	    W_skel->col[i] = j;
	    W_skel->row_off[i] = i;
    }
    W_skel->row_off[rn] = rn;    
}

/* Preconditioned conjugate gradient */
uint pcg(double *x, struct csr_mat *A, double *r, double *M, double tol, 
    double *b)
{
    uint rn = A->rn;
    uint cn = A->cn;

    // x = zeros(size(r)); p=x;
    init_array(x, rn, 0.);

    double *p = tmalloc(double, cn);
    init_array(p, cn, 0.);

    // z = M(r); (M(r) = M.*r)
    double *z = tmalloc(double, rn);
    memcpy(z, M, rn*sizeof(double));
    vv_op(z, r, rn, ewmult);
    
    // rho_0 = b'*M(b);
    // rho_stop=tol*tol*rho_0
    double rho = vv_dot(r, z, rn);

    double *tmp = tmalloc(double, rn);
    double rho_0;
    memcpy(tmp, M, rn*sizeof(double));
    vv_op(tmp, b, rn, ewmult);
    rho_0 = vv_dot(tmp, b, rn);

    double rho_stop = tol*tol*rho_0;

    // n = min(length(r),100);
    uint n = rn;
    n = (n <= 100) ? n : 100;

    // if n==0; return; end
    if (n == 0)
    {
        return 0;
    } 

    // rho_old = 1;
    double rho_old = 1;
    
    uint k = 0;
    double alpha = 0, beta = 0;
    init_array(tmp, rn, 0.0);
    double *w = tmalloc(double, rn);

    // while rho > rho_stop && k<n
    while (rho > rho_stop && k < n)
    {
        k++;

        // beta = rho / rho_old;
        beta = rho / rho_old;

        // p = z + beta * p;
        ar_scal_op(p, beta, rn, mult_op);
        vv_op(p, z, rn, plus);

        // w = A(p); (A(p) = A*p)
        apply_M(w, 0, p, 1, A, p);

        // alpha = rho / (p'*w);
        alpha = vv_dot(p, w, rn);
        alpha = rho / alpha;

        // x = x + alpha*p;
        memcpy(tmp, p, rn*sizeof(double));
        ar_scal_op(tmp, alpha, rn, mult_op);
        vv_op(x, tmp, rn, plus);

        // r = r - alpha*w;
        ar_scal_op(w, alpha, rn, mult_op);
        vv_op(r, w, rn, minus);

        // z = M(r);
        memcpy(z, M, rn*sizeof(double));
        vv_op(z, r, rn, ewmult);

        // rho_old = rho;
        rho_old = rho;

        // rho = r'*z;
        rho = vv_dot(r, z, rn);
    }

    free(p);
    free(z);
    free(w);
    free(tmp);

    return k;
}

/*
  Sym_sparsify (for symmetric "sparsification")
*/
// TO BE IMPLEMENTED

/* 
  Sparsify (for non symmetric "sparsification")
  This function is slightly different from the Matlab version: S is an array 
  having the same number of elements as A.
   - S[i] = 0 if A->a[i] is "sparsified"
   - S[i] = 1 if A->a[i] is kept
  S needs to be allocated with A->row_off[rn] doubles before the call!
*/
void sparsify(double *S, struct csr_mat *A, double tol)
{
    uint rn = A->rn;
    // uint cn = A->cn; Unused
    uint *row_off = A->row_off;
    uint *col = A->col;
    double *a = A->a;
    uint ncol = A->row_off[rn];

    // S has the same number of elements as A -> initialized with 1s
    init_array(S, ncol, 1.);

    // E[i] is the sum of the smallest elements of line i such that the sum is
    // lower than the tolerance
    double *E = tmalloc(double, rn);    
    init_array(E, rn, 0.);

    // Build matrix A using coordinate list format for sorting
    // coo_A = abs(A)
    coo_mat *coo_A = tmalloc(coo_mat, ncol);  
    coo_mat *p = coo_A;  

    uint i;
    for(i=0;i<rn;++i) 
    {
        uint j; 
        uint je=row_off[i+1]; 
        for(j=row_off[i]; j<je; ++j) 
        {
            p->i = i;
            p->j = *col++;
            p->v = fabs(*a++);
            p++;
        }
    }

    buffer buf = {0};
    sarray_sort(coo_mat, coo_A, ncol, v, 0, &buf);;
    buffer_free(&buf);

    for (i=0; i<ncol; i++)
    {
        if (coo_A[i].v > tol) break;
        
        if (coo_A[i].i != coo_A[i].j)
        {
            E[coo_A[i].i] += fabs(coo_A[i].v);
            if (E[coo_A[i].i] < tol) S[i] = 0.;
        }
    }

    free(coo_A);
    free(E);
}

/*
  Chebsim
*/
void chebsim(double *m, double *c, double rho, double tol)
{
    double alpha = 0.25*rho*rho;
    *m = 1;
    double cp = 1;
    *c = rho;
    double gamma = 1;
    double d, cn;
    
    while (*c > tol)
    {
        *m += 1;
        d = alpha*(1+gamma);
        gamma = d/(1-d);
        cn = (1+gamma)*rho*(*c) - gamma*cp;
        cp = *c;
        *c = cn;
    }
}

/* 
  Eigenvalues evaluations by Lanczos algorithm
*/
uint lanczos(double **lambda, struct csr_mat *A)
{
    uint rn = A->rn;
    uint cn = A->cn;
    double *r = tmalloc(double, rn);

//    int seed = time(NULL);   // Better ?
//    srand(seed);

    uint i;
    for (i=0; i<rn; i++)
    {
        r[i] = (double)rand() / (double)RAND_MAX;
    }

    // Fixed length for max value of k
    uint kmax = 300;
    *lambda = tmalloc(double, kmax);
    double *l = *lambda;
    double *y = tmalloc(double, kmax);
    double *a = tmalloc(double, kmax);
    double *b = tmalloc(double, kmax);
    double *d = tmalloc(double, kmax+1);
    double *v = tmalloc(double, kmax);

    double beta = array_op(r, rn, norm2_op);
    double beta2 = beta*beta;
    beta = sqrt(beta2);

    uint k = 0; 
    double change = 0.0;

    // norm(A-speye(n),'fro') < 0.00000000001
    double *eye = tmalloc(double, rn);
    init_array(eye, rn, 1.);

    // Dummy matrix
    struct csr_mat *Acpy = tmalloc(struct csr_mat, 1);
    copy_csr(Acpy, A);

    // Frobenius norm
    diagcsr_op(Acpy, eye, dminus);

    // A assumed to be real ==> Frobenius norm is 2-norm of A->a
    double fronorm = array_op(Acpy->a, Acpy->row_off[rn], norm2_op);

    // Free matrix
    free_csr(&Acpy);

    double fronorm2 = fronorm * fronorm;
    fronorm = sqrt(fronorm2);

    if (fronorm < 1e-11)
    {
        l[0] = 1;
        l[1] = 1;
        y[0] = 0;
        y[1] = 0;
        k = 2;
        change = 0.0;
    }
    else
    {
        change = 1.0;
    }

    // If n == 1
    if (rn == 1)
    {
        double A00 = A->a[0];

        l[0] = A00;
        l[1] = A00;
        y[0] = 0;
        y[1] = 0;
        k = 2;
        change = 0.0; 
    }

    // While...
    double *qk = tmalloc(double, cn);
    init_array(qk, cn, 0.);
    double *qkm1 = tmalloc(double, rn);
    double *alphaqk = tmalloc(double, rn); // alpha * qk vector
    double *Aqk = tmalloc(double, rn); 
    uint na = 0, nb = 0;

    while (k < kmax && ( change > 1e-5 || y[0] > 1e-3 || y[k-1] > 1e-3))
    {
        k++;

        // qkm1 = qk
        memcpy(qkm1, qk, rn*sizeof(double));

        // qk = r/beta
        memcpy(qk, r, rn*sizeof(double));
        ar_scal_op(qk, 1./beta, rn, mult_op);

        // Aqk = A*qk
        apply_M(Aqk, 0, qk, 1, A, qk);

        // alpha = qk'*Aqk
        double alpha = vv_dot(qk, Aqk, rn); 

        //a = [a; alpha];
        a[na++] = alpha;

        // r = Aqk - alpha*qk - beta*qkm1
        memcpy(alphaqk, qk, rn*sizeof(double)); // alphaqk = qk
        ar_scal_op(alphaqk, alpha, rn, mult_op); // alphaqk = alpha*qk
        ar_scal_op(qkm1, beta, rn, mult_op); // qkm1 = beta*qkm1

        memcpy(r, Aqk, rn*sizeof(double)); // r = Aqk
        vv_op(r, alphaqk, rn, minus); // r = Aqk - alpha*qk
        vv_op(r, qkm1, rn, minus); // r = Aqk - alpha*qk - beta*qkm1
        
        if (k == 1)
        {
            l[0] = alpha;
            y[0] = 1;
        }
        else
        {
            double l0 = l[0];
            double lkm2 = l[k-2];
            // d
            //double *d = tmalloc(double, k+1);
            d[0] = 0;
            for (i=1; i<k; i++) d[i] = l[i-1];
            d[k] = 0;

            // v
            //double *v = tmalloc(double, k);
            v[0] = alpha;
            for (i=1; i<k; i++) v[i] = beta*y[i-1]; // y assumed to be real !!!

            tdeig(l, y, d, v, k-1);
  
            change = fabs(l0 - l[0]) + fabs(lkm2 - l[k-1]);
        }        

        beta = array_op(r, rn, norm2_op);
        beta2 = beta*beta;
        beta = sqrt(beta2);
        b[nb++] = beta;

        if (beta == 0) {break;}
    }

    uint n = 0;

    for (i=0; i<k; i++)
    {
        if (y[i] < 0.01)
        {
            (*lambda)[n++] = l[i];
        }
    }  

    free(r);
    free(eye);
    free(qk);
    free(qkm1);
    free(alphaqk);
    free(Aqk); 
    free(y);
    free(a);
    free(b);
    free(d);
    free(v);

    return n; 
}

/*
  TDEIG
*/

#define EPS (128*DBL_EPSILON)

/* minimizes cancellation error (but not round-off ...) */
static double sum_3(const double a, const double b, const double c)
{
  if     ( (a>=0 && b>=0) || (a<=0 && b<=0) ) return (a+b)+c;
  else if( (a>=0 && c>=0) || (a<=0 && c<=0) ) return (a+c)+b;
  else return a+(b+c);
}

/* solve     c
          - --- + b + a x == 0        with sign(x) = sign
             x
*/
static double rat_root(const double a, const double b, const double c,
                       const double sign)
{
  double bh = (fabs(b) + sqrt(b*b + 4*a*c))/2;
  return sign * (b*sign <= 0 ? bh/a : c/bh);
}

/*
  find d[ri] <= lambda <= d[ri+1]
  such that 0 = lambda - v[0] + \sum_i^n v[i]^2 / (d[i] - lambda)
*/
static double sec_root(double *y, const double *d, const double *v,
                       const int ri, const int n)
{
  double dl = d[ri], dr = d[ri+1], L = dr-dl;
  double x0l = L/2, x0r = -L/2;
  int i;
  double al, ar, bln, blp, brn, brp, cl, cr;
  double fn, fp, lambda0, lambda;
  double tol = L;
  if(fabs(dl)>tol) tol=fabs(dl);
  if(fabs(dr)>tol) tol=fabs(dr);
  tol *= EPS;
  for(;;) {
    if(fabs(x0l)==0 || x0l < 0) { *y=0; return dl; }
    if(fabs(x0r)==0 || x0r > 0) { *y=0; return dr; }
    lambda0 = fabs(x0l) < fabs(x0r) ? dl + x0l : dr + x0r;
    al = ar = cl = cr = bln = blp = brn = brp = 0;
    fn = fp = 0;
    for(i=1;i<=ri;++i) {
      double den = (d[i]-dl)-x0l;
      double fac = v[i]/den;
      double num = sum_3(d[i],-dr,-2*x0r);
      fn += v[i]*fac;
      fac *= fac;
      ar += fac;
      if(num > 0) brp += fac*num; else brn += fac*num;
      bln += fac*(d[i]-dl);
      cl  += fac*x0l*x0l;
    }
    for(i=ri+1;i<=n;++i) {
      double den = (d[i]-dr)-x0r;
      double fac = v[i]/den;
      double num = sum_3(d[i],-dl,-2*x0l);
      fp += v[i]*fac;
      fac *= fac;
      al += fac;
      if(num > 0) blp += fac*num; else bln += fac*num;
      brp += fac*(d[i]-dr);
      cr  += fac*x0r*x0r;
    }
    if(lambda0>0) fp+=lambda0; else fn+=lambda0;
    if(v[0]<0) fp-=v[0],blp-=v[0],brp-=v[0];
          else fn-=v[0],bln-=v[0],brn-=v[0];
    if(fp+fn > 0) { /* go left */
      x0l = rat_root(1+al,sum_3(dl,blp,bln),cl,1);
      lambda = dl + x0l;
      x0r = x0l - L;
    } else { /* go right */
      x0r = rat_root(1+ar,sum_3(dr,brp,brn),cr,-1);
      lambda = dr + x0r;
      x0l = x0r + L;
    }
    if( fabs(lambda-lambda0) < tol ) {
      double ty=0, fac;
      for(i=1;i<=ri;++i) fac = v[i]/((d[i]-dl)-x0l), ty += fac*fac;
      for(i=ri+1;i<=n;++i) fac = v[i]/((d[i]-dr)-x0r), ty += fac*fac;
      *y = 1/sqrt(1+ty);
      return lambda;
    }
  }
}

/*
  find the eigenvalues of
  
  d[1]           v[1]
       d[2]      v[2]
            d[n] v[n]
  v[1] v[2] v[n] v[0]
  
  sets d[0], d[n+1] to Gershgorin bounds
  
  also gives (n+1)th component of each orthonormal eigenvector in y
*/
static void tdeig(double *lambda, double *y, double *d, const double *v,
                  const int n)
{
  int i;
  double v1norm = 0, min=v[0], max=v[0];
  for(i=1;i<=n;++i) {
    double vi = fabs(v[i]), a=d[i]-vi, b=d[i]+vi;
    v1norm += vi;
    if(a<min) min=a;
    if(b>max) max=b;
  }
  d[0]   = v[0] - v1norm < min ? v[0] - v1norm : min;
  d[n+1] = v[0] + v1norm > max ? v[0] + v1norm : max;
  for(i=0;i<=n;++i) lambda[i] = sec_root(&y[i],d,v,i,n);
}

void coarsen(double *vc, struct csr_mat *A, double ctol)
{
    uint rn = A->rn, cn = A->cn;
    // array reduction for openmp in mat_max
    /*omp_set_num_threads(68);*/
    const int nthreads = omp_get_max_threads();
    const int ithread = omp_get_thread_num();
    printf("Number of threads: %d\n", nthreads);
    double *mp = malloc(sizeof(double)*cn*nthreads);

    // D = diag(A)
    double *D = tmalloc(double, cn);
    diag(D, A);

    // D = 1/sqrt(D)
    array_op(D, cn, sqrt_op);
    array_op(D, cn, minv_op);

    // S = abs(D*A*D)
    struct csr_mat *S = tmalloc(struct csr_mat, 1);
    copy_csr(S, A); // S = A
    diagcsr_op(S, D, dmult); // S = D*S
    diagcsr_op(S, D, multd); // S = S*D
    array_op(S->a, S->row_off[rn], abs_op); // S = abs(S)

    // S = S - diag(S)
    diag(D, S); // D = diag(S)
    diagcsr_op(S, D, dminus); // S = S - D

    // Free data not required anymore  
    free(D);

    /* Write out csr matrices for external tests (in matlab) */
    //write_mat(S, id_rows_owned, nnz, data);

    // vc = zeros(n, 1), vf = ones(n, 1)
    init_array(vc, cn, 0.); 
    int anyvc = 0; // = 0 if vc = all zeros / = 1 if at least one element is !=0
    double *vf = tmalloc(double, cn);
    init_array(vf, cn, 1.);
    
    // Other needed arrays
    double *g = tmalloc(double, cn);

    double *w1 = tmalloc(double, cn);
    double *w2 = tmalloc(double, cn);

    double *tmp = tmalloc(double, cn); // temporary array required for  
                                       // storing intermediate results
    double *w = tmalloc(double, cn);

    double *mask = tmalloc(double, cn);
    double *m = tmalloc(double, cn);
    printf("coarse A->rn: %u, A->cn: %u\n", A->rn, A->cn);

    while (1)
    {
        // w1 = vf.*(S*(vf.*(S*vf)))
        apply_M(g, 0, vf, 1., S, vf); // g = S*vf 
        vv_op(g, vf, cn, ewmult); // g = vf.*g
        apply_M(w1, 0, g, 1., S, g); // w1 = S*g
        vv_op(w1, vf, cn, ewmult); // w1 = vf.*w1

        // w2 = vf.*(S *(vf.*(S*w1)))
        apply_M(w2, 0, w1, 1., S, w1); // w2 = S*w1
        vv_op(w2, vf, cn, ewmult); // w2 = vf.*w2
        apply_M(tmp, 0, w2, 1., S, w2); // tmp = S*w2
        memcpy(w2, tmp, cn*sizeof(double)); // tmp = w2
        vv_op(w2, vf, cn, ewmult); // w2 = vf.*w2

        // w = w2./w1
        memcpy(w, w1, cn*sizeof(double)); // w = w1
        array_op(w, cn, minv_op); // w = 1./w1
        vv_op(w, w2, cn, ewmult); // w = w2./w1

        uint i;
        for (i=0;i<cn;i++) // w(w1==0) = 0;
        {
            if (w1[i] == 0) w[i] = 0.;
        }

        // b = sqrt(min(max(w1),max(w)));
        double b, w1m, wm; // w1m = max(w1), wm = max(w)
        uint mi, unused; // mi = index of w1m in w1
        extr_op(&w1m, &mi, w1, cn, max);
        extr_op(&wm, &unused, w, cn, max);

        b = (w1m < wm) ? sqrt(w1m) : sqrt(wm);

        // if b<=ctol; vc(mi)=true;
        if (b <= ctol)
        {
            if (anyvc == 0)
            {
                 vc[mi] = 1.;
            }
            break;
        }

        // mask = w > ctol^2;
        mask_op(mask, w, cn, ctol*ctol, gt);
 
        // m = mat_max(S,vf,mask.*g)
        double mat_max_tol = 0.1; // ==> hard-coded tolerance
        memcpy(tmp, g, cn*sizeof(double)); // tmp = g
            // g (needed later) copied into tmp
        vv_op(tmp, mask, cn, ewmult); // tmp = mask.*tmp (= mask.*g)        
        mat_max(m, mp, S, vf, tmp, mat_max_tol); // m = mat_max(S,vf,mask.*g)
 
        // mask = mask & (g-m>=0)
        vv_op(g, m, cn, minus); // g = g - m
        mask_op(tmp, g, cn, 0., ge); // tmp = (g-m>=0)
        bin_op(mask, tmp, cn, and_op); // mask = mask & tmp;

        // m = mat_max(S,vf,mask.*id)
        for (i=0;i<cn;i++)
        {
            g[i] = (double)i + 1.0; // g = (double) id
        }
        memcpy(tmp, mask, cn*sizeof(double)); // copy mask to tmp
        vv_op(tmp, g, cn, ewmult); // tmp = tmp.*g (= mask.*id)
        mat_max(m, mp, S, vf, tmp, mat_max_tol); // m = mat_max(S,vf,mask.*g)

        // mask = mask & (id-m>0)
        vv_op(g, m, cn, minus); // id = id - m
        mask_op(tmp, g, cn, 0., gt); // tmp = (id-m>0)
        bin_op(mask, tmp, cn, and_op); // mask = mask & tmp;

        // vc = vc | mask ; vf = xor(vf, mask)
        bin_op(vc, mask, cn, or_op); // vc = vc | mask;
        if (anyvc == 0)
        {
            for (i=0;i<cn;i++) 
            {
                if (vc[i] == 1.)
                {
                    anyvc = 1;
                    break;
                }
            }
        }
        bin_op(vf, mask, cn, xor_op); // vf = vf (xor) mask;
    }

    // Free S
    free_csr(&S);

    // Free arrays
    free(vf);
    free(w1);
    free(w2);
    free(w);
    free(g);
    free(mask);
    free(m);
    free(tmp);

    // Free openmp reduction array
    free(mp);
}

/* Exctract sub-matrix 
   subA = A(vr, vc) */
void sub_mat(struct csr_mat *subA, struct csr_mat *A, double* vr, double *vc)
{
    uint rn = A->rn, cn = A->cn;
    uint *row_off = A->row_off, *col = A->col;
    double *a = A->a;

    uint subrn = 0, subcn = 0, subncol = 0;

    uint i, j;

    // Compute number of rows and of non-zero elements for sub-matrix
    for (i=0; i<rn; i++)
    {
        if (vr[i] != 0.)
        {
            subrn += 1;
            uint je=row_off[i+1]; 
            for(j=row_off[i]; j<je; j++)
            {
                if (vc[col[j]] != 0.)
                {
                    subncol += 1;
                }
            }
        }
    }

    // Compute cn for sub-matrix
    uint c = 0; // just a counter
    uint *g2lcol = tmalloc(uint, cn); // correspondence between local and global
                                      // column ids
    for (i=0; i<cn; i++)
    {
        if (vc[i] != 0)
        {
            subcn += 1;
            g2lcol[i] = c;
            c++;
        }
        else
        {
            g2lcol[i] = -1; // dummy because g2lcol is of type uint
        }
    }

    // Initialize and build sub-matrix
    malloc_csr(subA, subrn, subcn, subncol);

    uint *subrow_off = subA->row_off;
    uint *subcol     = subA->col;
    double *suba     = subA->a;

    *subrow_off++ = 0;
    uint roffset = 0;

    for (i=0; i<rn; i++)
    {
        if (vr[i] != 0.)
        {
            uint je=row_off[i+1]; 
            for(j=row_off[i]; j<je; j++)
            {
                if (vc[col[j]] != 0.)
                {
                    roffset += 1;
                    *subcol++ = g2lcol[col[j]];
                    *suba++   = a[j];
                }
            }
            *subrow_off++ = roffset;
        }
    }

    free(g2lcol);
}

/* Exctract sub-vector of type double
   a = b(v) 
   n is length of b amd v */
void sub_vec(double *a, double *b, double* v, uint n)
{
    uint i;    
    for (i=0; i<n; i++)
    {
        if (v[i] != 0.)
        {
            *a++ = b[i];  
        }
    }
}

/* Exctract sub-vector of type slong
   a = b(v) 
   n is length of b and v */
void sub_slong(slong *a, slong *b, double* v, uint n)
{
    uint i;    
    for (i=0; i<n; i++)
    {
        if (v[i] != 0.)
        {
            *a++ = b[i];  
        }
    }
}

/* Vector-vector operations:
   a = a (op) b */
void vv_op(double *a, double *b, uint n, enum vv_ops op)
{
    uint i;
    switch(op)
    {
        case(plus):   for (i=0;i<n;i++) *a = *a + *b, a++, b++; break;
        case(minus):  for (i=0;i<n;i++) *a = *a - *b, a++, b++; break;
        case(ewmult): for (i=0;i<n;i++) *a = *a * (*b), a++, b++; break;
    }
}

// Dot product between two vectors
double vv_dot(double *a, double *b, uint n)
{
    double r = 0;
    uint i;
    for (i=0;i<n;i++)
    {
        r += (*a++)*(*b++);
    }

    return r;
}

/* Binary operations
   mask = mask (op) a */
void bin_op(double *mask, double *a, uint n, enum bin_ops op)
{
    uint i;
    switch(op)
    {
        case(and_op): for (i=0;i<n;i++) 
                      {
                          *mask = (*mask != 0. && *a != 0.) ? 1. : 0.; 
                          mask++, a++;
                      } 
                      break;
        case(or_op):  for (i=0;i<n;i++) 
                      {
                          *mask = (*mask == 0. && *a == 0.) ? 0. : 1.; 
                          mask++, a++; 
                      }
                      break;
        case(xor_op): for (i=0;i<n;i++) 
                      { 
                          *mask = ((*mask == 0. && *a != 0.) || 
                                   (*mask != 0. && *a == 0.)   ) ? 1. : 0.; 
                          mask++, a++; 
                      }
                      break;
        // mask = not(a)
        case(not_op): for (i=0;i<n;i++) 
                      { 
                          *mask = (*a == 0. ) ? 1. : 0.; 
                          mask++, a++; 
                      }
                      break;
    }
}

/* Mask operations */
void mask_op(double *mask, double *a, uint n, double trigger, enum mask_ops op)
{
    uint i;
    switch(op)
    {
        case(gt): for (i=0;i<n;i++) mask[i] = (a[i] > trigger)? 1. : 0.; break;
        case(lt): for (i=0;i<n;i++) mask[i] = (a[i] < trigger)? 1. : 0.; break;
        case(ge): for (i=0;i<n;i++) mask[i] = (a[i] >= trigger)? 1. : 0.; break;
        case(le): for (i=0;i<n;i++) mask[i] = (a[i] <= trigger)? 1. : 0.; break;
        case(eq): for (i=0;i<n;i++) mask[i] = (a[i] = trigger)? 1. : 0.; break;
        case(ne): for (i=0;i<n;i++) mask[i] = (a[i] != trigger)? 1. : 0.; break;
    }
}

/* Extremum operations */
void extr_op(double *extr, uint *idx, double *a, uint n, enum extr_ops op)
{
    *extr = a[0];
    *idx = 0;
    uint i;
    switch(op)
    {
        case(max): for (i=1;i<n;i++) if (a[i] > *extr) *extr = a[i], *idx = i;
                   break;
        case(min): for (i=1;i<n;i++) if (a[i] < *extr) *extr = a[i], *idx = i;
                   break;
    }
}

/* Array operations */
// a[i] = op(a[i])
// i = 0, n-1
double array_op(double *a, uint n, enum array_ops op)
{
    double r = 0;
    uint i;
    switch(op)
    {
        case(abs_op):  for (i=0;i<n;i++) *a = fabs(*a), a++; break;
        case(sqrt_op): for (i=0;i<n;i++) *a = sqrt(*a), a++; break;
        case(minv_op): for (i=0;i<n;i++) *a = 1./(*a), a++; break;
        case(sqr_op): for (i=0;i<n;i++) *a = (*a)*(*a), a++; break;
        case(sum_op)  :  for (i=0;i<n;i++) r += *a++; break;
        case(norm2_op):  for (i=0;i<n;i++)
                         { 
                             r += (*a)*(*a);
                             a++; 
                         }
                         r = sqrt(r); 
                         break;
    }

    return r;
}

// Array initialisation
// a = v*ones(n, 1)
void init_array(double *a, uint n, double v)
{
    uint i;
    for (i=0;i<n;i++)
    {
        *a++ = v;
    }
}

// Operations between an array and a scalar
// a[i] = a[i] op scal
void ar_scal_op(double *a, double scal, uint n, enum ar_scal_ops op)
{
    uint i;
    switch(op)
    {
        case(mult_op)  :  for (i=0;i<n;i++) *a = (*a)*scal, a++; break;
    }
}

// Condense array by deleting elment a[i] if b[i] == target
// Returns size of condense array
uint condense_array(double *a, double *b, const double target, const uint n)
{
    double *tmp = tmalloc(double, n);
    init_array(tmp, n, 0.0);
        
    uint i, counter = 0;
    for (i=0;i<n;i++) {if (b[i] != target) tmp[counter++] = a[i];}
    memcpy(a, tmp, counter*sizeof(double));

    free(tmp);
    return counter;
}

/*******************************************************************************
* Diagonal operations
*******************************************************************************/
/* Extract diagonal from square sparse matrix */
void diag(double *D, struct csr_mat *A)
{
    uint i; 
    uint rn=A->rn;
    uint *row_off = A->row_off, *col = A->col;
    double *a = A->a;
    
    for(i=0;i<rn;++i) 
    {
        uint j; 
        const uint jb=row_off[i], je=row_off[i+1];
        for(j=jb; j<je; ++j) 
        {  
            if (col[j] == i) 
            {
                *D++ = a[j]; 
                break;
            }
        }
    }
}

// Operations between csr and diagonal matrices
// Assumption: length(D) = A->cn
void diagcsr_op(struct csr_mat *A, double *D, enum diagcsr_ops op)
{
    uint i;
    uint rn = A->rn;
    uint *row_off = A->row_off, *col = A->col;
    double *a = A->a;

    switch(op)
    {
        case(dplus):   for (i=0;i<rn;i++) 
                      {
                          uint j; 
                          const uint jb=row_off[i], je=row_off[i+1];
                          for(j=jb; j<je; ++j) 
                          {
                              if (col[j] == i) 
                              {
                                  a[j] = a[j]+D[i]; 
                                  break;
                              }
                          }
                      }
                      break;
        case(dminus): for (i=0;i<rn;i++) 
                      {
                          uint j; 
                          const uint jb=row_off[i], je=row_off[i+1];
                          for(j=jb; j<je; ++j) 
                          {
                              if (col[j] == i) 
                              {
                                  a[j] = a[j]-D[i]; 
                                  break;
                              }
                          }
                      }
                      break;
        case(dmult):  for(i=0;i<rn;++i) 
                      {
                          uint j; 
                          const uint jb=row_off[i], je=row_off[i+1];
                          for(j=jb; j<je; ++j) 
                          {       
                              a[j] = a[j]*D[i];
                          }
                      }
                      break;
        case(multd):  for(i=0;i<rn;++i) 
                      {
                          uint j; 
                          const uint jb=row_off[i], je=row_off[i+1];
                          for(j=jb; j<je; ++j) 
                          {       
                              a[j] = a[j]*D[col[j]];
                          }
                      }
                      break;
    }
}

/*******************************************************************************
* Others
*******************************************************************************/
/* Create empty matrix */
void malloc_csr(struct csr_mat *A, uint rn, uint cn, uint nnz)
{
    A->rn = rn;
    A->cn = cn;
    A->row_off = tmalloc(uint, rn+1);
    A->col = tmalloc(uint, nnz);
    A->a = tmalloc(double, nnz);
}

/* Copy sparse matrix A into B */
void copy_csr(struct csr_mat *B, struct csr_mat *A)
{
    uint nnz = A->row_off[A->rn];
    malloc_csr(B, A->rn, A->cn, nnz);

    memcpy(B->row_off, A->row_off, (A->rn+1)*sizeof(uint));
    memcpy(B->col, A->col, (nnz)*sizeof(uint));
    memcpy(B->a, A->a, (nnz)*sizeof(double));
}

/* Free csr matrices */
void free_csr(struct csr_mat **A)
{
     if (A) 
    {
        free((*A)->row_off);
        free((*A)->col);
        free((*A)->a);
        free(*A);
        *A = NULL;
     }
}

// C-MEX FUNCTIONS TRANSFORMED TO C
/* /!\ Contrary to the original mex function, this one is valid for a square 
   symmetric matrix A only 
   - x and y are local
*/
static void mat_max(double *y, double *yp, struct csr_mat *A, double *f, double *x, 
    double tol)
{
    uint i, rn = A->rn, cn = A->cn;
    for(i=0;i<cn;++i) 
    {
        y[i] = -DBL_MAX;
    }
#pragma omp parallel 
    {
      const int nthreads = omp_get_num_threads();
      const int ithread = omp_get_thread_num();
#pragma omp for
      for(i=0;i<cn;++i) 
      {
        uint j;
        for(j=0;j<nthreads;++j) 
          yp[j*cn+i]=-DBL_MAX;
      }
#pragma omp for 
    for(i=0;i<rn;++i) 
    {
        double xj = x[i];
        uint j, jb = A->row_off[i], je = A->row_off[i+1];
        double Amax = 0;
        for(j=jb;j<je;++j)
            if(f[A->col[j]] != 0 && fabs(A->a[j])>Amax) Amax=fabs(A->a[j]);
        Amax *= tol;
        for(j=jb;j<je;++j) 
        {
            uint k = A->col[j];
            if(f[k] == 0 || fabs(A->a[j]) < Amax) continue;
            {
              if(xj>yp[ithread*cn+k]) yp[ithread*cn+k]=xj;
              /*if(xj>y[k]) y[k]=xj;*/
            }
        }
    }
#pragma omp for 
    for(i=0;i<cn;i++)
    {
      for(int j=0;j<nthreads;j++)
      {
        if(y[i]<yp[j*cn+i]) y[i]=yp[j*cn+i];
      }
    }
    }
}

/* Build csr matrix from arrays of indices and values */
void build_csr(struct csr_mat *A, uint n, const uint *Ai, const uint* Aj, 
    const double *Av)
{
    // Build matrix in coord. list format
    coo_mat *coo_A = tmalloc(coo_mat, n);
    coo_A = tmalloc(coo_mat,n);

    uint i;
    for (i=0;i<n;i++)
    {
        coo_A[i].i = Ai[i];
        coo_A[i].j = Aj[i];
        coo_A[i].v = Av[i];
    }
    
    // Build csr matrix
    coo2csr(A, coo_A, n);
    free(coo_A);
}

/* Build sparse matrix using the csr format
   It is assumed that  
   - the function is called in serial
   - coo_A is sorted by rows and columns
*/
void coo2csr(struct csr_mat *A, coo_mat *coo_A, uint nnz)
{
    // Sort matrix by rows then columns
    buffer buf = {0};
    sarray_sort_2(coo_mat, coo_A, nnz, i, 0, j, 0, &buf);
    buffer_free(&buf);

    /* Go to csr format */
    // Check for dimensions
    uint rn=0, cn=0;
    uint i;
    for (i=0;i<nnz;i++)
    {
        if (coo_A[i].i+1 > rn) rn = coo_A[i].i+1;
        if (coo_A[i].j+1 > cn) cn = coo_A[i].j+1;
    }

    malloc_csr(A, rn, cn, nnz);

    uint row_cur, row_prev = coo_A[0].i, counter = 1;
    A->row_off[0] = 0;
    
    for (i=0;i<nnz;i++)
    {
        // row_off
        row_cur = coo_A[i].i;
        if (row_cur != row_prev)     
        {    
            uint k;
            for (k=row_prev;k<row_cur;k++)
            {
                A->row_off[counter++] = i;
            }
        }
        if (i == nnz-1) A->row_off[counter++] = nnz;
    
        row_prev = row_cur;
        
        // col
        A->col[i] = coo_A[i].j;

        // a
        A->a[i] = coo_A[i].v;
    }
}

/* TO BE DELETED */
void print_csr(struct csr_mat *P)
{
    printf("P:\n"); 
    printf(" rn=%u, cn=%u, nnz=%u\n", P->rn, P->cn, P->row_off[P->rn]); 
    uint ip, jp, jpe, jps;
    for (ip=0;ip<P->rn;ip++)
    {
        jps = P->row_off[ip];
        jpe = P->row_off[ip+1];   
        printf("js = %u, je = %u\n", jps, jpe);      
        for (jp=jps;jp<jpe;jp++)
        {
            printf("P[%u,%u] = %lf\n", ip, P->col[jp], P->a[jp]); 
        }
    }
}
