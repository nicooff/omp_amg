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

    - Last update: 19 September, 2016

    - Status: 
        * Finished amg_export.
        * The part with memory reallocation has memory leakage -> use "initsize"
          (initial guess for number of levels) high enough!
        * Only tested for small 2D cavity.
        * No sparsification functions.
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
        - Properly check anyvc (Done but not tested).
        - Implement all the sparsification functions (sym_sparsify, 
          simple_sparsify...).
        - Check with large cases and debug.
*/

/**************************************/
/* AMG setup (previously Matlab code) */
/**************************************/
void amg_setup(uint n, const uint *Ai, const uint* Aj, const double *Av,
    struct amg_setup_data *data)
/*    Build data, the "struct crs_data" defined in amg_setup.h that is required
      to execute the AMG solver. */
{   
/* Declare csr matrix A (assembled matrix Av under csr format) */  
    struct csr_mat *A = tmalloc(struct csr_mat, 1);

/* Build A the required data for the setup */
    build_csr(A, n, Ai, Aj, Av);

/* Tolerances (hard-coded so far) */
    double tol = 0.5; 
    double ctol = 0.7; // Coarsening tolerance
    double itol = 1e-4; // Interpolation tolerance
    double gamma2 = 1. - sqrt(1. - tol);
    double gamma = sqrt(gamma2);
    double stol = 1e-4;

    data->tolc = ctol;
    data->gamma = gamma;

/**************************************/
/* Memory allocation for data struct. */
/**************************************/

    // Initial size for number of sublevels
    // If more levels than this, realloction is required!
    uint initsize = 100; // I'm not very confident with the rellocation part
                         // => value should be larger than expected number of levels

    data->n = tmalloc(double, initsize);
    data->nnz = tmalloc(double, initsize);
    data->nnzf = tmalloc(double, initsize-1);
    data->nnzfp = tmalloc(double, initsize-1);
    data->m = tmalloc(double, initsize-1);
    data->rho = tmalloc(double, initsize-1);
    data->id = tmalloc(uint, A->rn);
    data->idc = tmalloc(uint*, initsize);
    data->idf = tmalloc(uint*, initsize);
    data->C = tmalloc(double*, initsize-1);
    data->F = tmalloc(double*, initsize-1);
    data->D = tmalloc(double*, initsize-1);

    malloc_csr_arr(&(data->A), initsize);

    malloc_csr_arr(&(data->Af), initsize-1);

    malloc_csr_arr(&(data->W), initsize-1);

    malloc_csr_arr(&(data->AfP), initsize-1);

    // Sublevel number
    uint level = 0;

    // Init id array
    uint k;
    for (k=0;k<A->rn;k++) data->id[k] = k+1;

// BEGIN WHILE LOOP
    while (1)
    {
        if (level > 0 && level % initsize == 0)
        {
            uint newsize = (level/initsize+1)*initsize;
            data->n = trealloc(double, data->n, newsize);
            data->nnz = trealloc(double, data->nnz, newsize);
            data->nnzf = trealloc(double, data->nnzf, newsize-1);
            data->nnzfp = trealloc(double, data->nnzfp, newsize-1);
            data->m = trealloc(double, data->m, newsize-1);
            data->rho = trealloc(double, data->rho, newsize-1);
            data->idc = trealloc(uint*, data->idc, newsize-1);
            data->idf = trealloc(uint*, data->idf, newsize-1);
            data->C = trealloc(double*, data->C, newsize-1);
            data->F = trealloc(double*, data->F, newsize-1);
            data->D = trealloc(double*, data->D, newsize-1);

            realloc_csr_arr(&(data->A), newsize);

            realloc_csr_arr(&(data->Af), newsize-1);

            realloc_csr_arr(&(data->W), newsize-1);

            realloc_csr_arr(&(data->AfP), newsize-1);
        }

        // Dimensions of matrix A
        uint rn = A->rn;
        uint cn = A->cn;

        data->n[level] = cn;
        data->nnz[level] = A->row_off[rn];

        data->A[level] = tmalloc(struct csr_mat, 1);
        copy_csr(data->A[level], A);

        if (cn <= 1) 
        {
            data->nullspace = 0;
            if (A->a[0] < 1e-9) data->nullspace = 1;
            printf("Nullspace = %u\n", data->nullspace);
            break;
        }

    /* Coarsen */ 
        double *vc = tmalloc(double, rn);
        // compute vc for i = 1,...,rn
        coarsen(vc, A, ctol); 

        double *vf = tmalloc(double, rn);
        bin_op(vf, vc, cn, not_op); // vf  = ~vc for i = 1,...,cn

        data->C[level] = tmalloc(double, rn);
        memcpy(data->C[level], vc, rn*sizeof(double));

        data->F[level] = tmalloc(double, rn);
        memcpy(data->F[level], vf, rn*sizeof(double));

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
            
            data->D[level] = tmalloc(double, rnf); 
            memcpy(data->D[level], D, rnf*sizeof(double));

            double rho = (b-a)/(b+a);
            data->rho[level] = rho;
            
            double m, c;
            chebsim(&m, &c, rho, gamma2);
            data->m[level] = m;

            gap = gamma2-c;

            /* Sparsification is skipped for the moment */
            //sym_sparsify(Sf, DhAfDh, (1.-rho)*(.5*gap)/(2.+.5*gap)); => not implemented

            data->Af[level] = tmalloc(struct csr_mat, 1);
            copy_csr(data->Af[level], Af);

            free(Dh);    
            free(lambda);
            free_csr(&DhAfDh);
        }
        else
        {
            gap = 0;

            data->D[level] = tmalloc(double, rnf); 
            memcpy(data->D[level], D, rnf*sizeof(double));     

            data->rho[level] = 0;
            data->m[level] = 1;

            data->Af[level] = tmalloc(struct csr_mat, 1);
            copy_csr(data->Af[level], Af);
        }
            
        data->nnzf[level] = Af->row_off[Af->rn];

    /* Interpolation */
        // Afc = A(F, C)
        struct csr_mat *Afc = tmalloc(struct csr_mat, 1);
        sub_mat(Afc, A, vf, vc);

        // Ac = A(C, C)
        struct csr_mat *Ac = tmalloc(struct csr_mat, 1);
        sub_mat(Ac, A, vc, vc);

        // Update id arrays
        uint rnc = Ac->rn;

        data->idc[level] = tmalloc(uint, rnc);
        data->idf[level] = tmalloc(uint, rnf);
        uint *idl;

        if (level== 0) idl = data->id;
        else           idl = data->idc[level-1];

        uint cc, cf;
        for (i=0,cc=0,cf=0;i<rn;i++) 
        {
            if (vc[i] == 1.) (data->idc[level])[cc++] = idl[i];
            else             (data->idf[level])[cf++] = idl[i];
        }

        // W
        struct csr_mat *W = tmalloc(struct csr_mat, 1);

        interpolation(W, Af, Ac, Afc, gamma2, itol); // Maybe W should be transposed at output...

        data->W[level] = tmalloc(struct csr_mat, 1);
        copy_csr(data->W[level], W);

        // AfP = Af*W+A(F,C);
        struct csr_mat *AfW = tmalloc(struct csr_mat, 1);
        struct csr_mat *Wt = tmalloc(struct csr_mat, 1);
        transpose(Wt, W);
        mxm(AfW, Af, Wt, 1.);

        struct csr_mat *AfP = tmalloc(struct csr_mat, 1);
        mpm(AfP, 1., AfW, 1., Afc);
        free_csr(&AfW);

        //struct csr_mat *AfPt = tmalloc(struct csr_mat, 1);
        //transpose(AfPt, AfP);

        data->AfP[level] = tmalloc(struct csr_mat, 1);
        copy_csr(data->AfP[level], AfP);

        //free_csr(&AfPt);

        data->nnzfp[level] = AfP->row_off[AfP->rn];

        // A = W'*AfP + A(C,F)*W + A(C,C); u = u(C);
        free_csr(&A);
        A = tmalloc(struct csr_mat, 1);

        struct csr_mat *WtAfP = tmalloc(struct csr_mat, 1);
        mxm(WtAfP, Wt, AfP, 0.);

        struct csr_mat *Acf = tmalloc(struct csr_mat, 1);
        transpose(Acf, Afc);

        struct csr_mat *AcfW = tmalloc(struct csr_mat, 1);
        mxm(AcfW, Acf, Wt, 1.);

        struct csr_mat *Atmp = tmalloc(struct csr_mat, 1);
        mpm(Atmp, 1., WtAfP, 1., AcfW);

        mpm(A, 1., Atmp, 1, Ac);
        
        // Update level number
        level++;

        // Free arrays
        free(vf);
        free(vc);
        free(af);
        free(s);
        free(D);

        // Free csr matrices
        free_csr(&Af);
        free_csr(&Afc);
        free_csr(&Ac);
        free_csr(&W);
        free_csr(&Wt);
        free_csr(&Atmp);
        free_csr(&Acf);
        free_csr(&AcfW);
        free_csr(&AfP);
        free_csr(&WtAfP);
    }
// END WHILE LOOP

    data->nlevels = level+1;
    free_csr(&A);
}

void amg_export(struct amg_setup_data *data)
{
    uint nlevels = data->nlevels;
    uint n = data->n[0];
    
    // for i=1:nl; lvl(data.id{i})=i; end
    uint *lvl = tmalloc(uint, n);
    uint i, j;

    for (i=0;i<n;i++) lvl[i] = 1;
    for (i=0;i<nlevels-1;i++)
    {
        uint nl = data->n[i+1];
        uint *idl = data->idc[i];
        for (j=0;j<nl;j++)
        {
            lvl[idl[j]-1] += 1;
        }
    }

    //for (i=0;i<n;i++) printf("lvl[%u] = %u\n",i,lvl[i]);

    //for i=1:nl-1; dvec(xor(data.id{i},data.id{i+1}),1) = full(diag(data.D{i})); end
    double *dvec = tmalloc(double, n);
    for (i=0;i<nlevels-1;i++)
    {
        uint nl = data->n[i]-data->n[i+1];
        uint *idl = data->idf[i];
        double *Dl = data->D[i];
        for (j=0;j<nl;j++)
        {
            dvec[idl[j]-1] = Dl[j];
        }
    }

    // if nullspace==0; dvec(data.id{nl}) = 1/full(data.A{nl}); end;
    uint k = (data->idc[nlevels-2][0])-1;
    if (data->nullspace != 0) 
    {
        dvec[k] = 0.;
    }
    else 
    {
        double a = *(data->A[nlevels-1]->a);
        dvec[k] = 1./a;
    }

    // 
    uint *W_len = tmalloc(uint, n);
    savemats(W_len, n, nlevels-1, lvl, data->idc, data->W, "amg_W.dat");

    uint *AfP_len = tmalloc(uint, n);
    savemats(AfP_len, n, nlevels-1, lvl, data->idc, data->AfP, "amg_AfP.dat");

    uint *Aff_len = tmalloc(uint, n);
    savemats(Aff_len, n, nlevels-1, lvl, data->idf, data->Af, "amg_Aff.dat");

    savevec(nlevels, data, n, lvl, W_len, AfP_len, Aff_len, dvec, "amg.dat");

    free(W_len);
    free(AfP_len);
    free(Aff_len);
    free(dvec);
    free(lvl);
}

/*
    Function to save matrices
    OUTPUT:
    - len(n): length of each row
    INPUT:
    - nl: number of levels
    - n: number of points
    - lvl(n): last level at which each point appears
    - id(n): global id for each point
    - mat(nl): array pointing to matrices of the different levels
    - filename: name of the file
*/
static void savemats(uint *len, uint n, uint nl, uint *lvl, uint **id,
                     struct csr_mat **mat, const char *filename)
{
    const double magic = 3.14159;
    FILE *f = fopen(filename,"w");
    uint max=0;
    uint i;
    uint *row;
    double *buf;
    fwrite(&magic,sizeof(double),1,f);

    for(i=0;i<nl;++i) {uint l=max_row_nnz(mat[i]); if(l>max) max=l; }
    //printf("maximum row size = %d\n",(int)max);
    buf = tmalloc(double, 2*max);
    row = tmalloc(uint, nl);
    for(i=0;i<nl;++i) row[i]=0;
    for(i=0;i<n;++i) 
    {
        uint l = lvl[i]-1;
        struct csr_mat *M;
        uint j,k,kb,ke;
        double *p;
        if(l>nl) { printf("level out of bounds\n"); continue; }
        if(l==nl) { len[i]=0; continue; }
        M = mat[l];
        j = row[l]++;
        if(j>=M->rn) { printf("row out of bounds\n"); continue; }
        kb=M->row_off[j],ke=M->row_off[j+1];
        p = buf;
        for(k=kb;k!=ke;++k) *p++ = id[l][M->col[k]], *p++ = M->a[k];
        len[i] = ke-kb;
        fwrite(buf,sizeof(double),(double)(2*(ke-kb)),f);
    }
    for(i=0;i<nl;++i) 
    {
        if(row[i]!=mat[i]->rn) printf("matrices not exhausted\n");
    }
    free(row);
    free(buf);
    fclose(f);
}

/* Returns max number of non zero elements on a row */
static uint max_row_nnz(struct csr_mat *mat)
{
    uint n=mat->rn,max=0;
    uint i;
    for(i=0;i<n;++i) 
    {
        uint l=mat->row_off[i+1]-mat->row_off[i];
        if(l>max) max=l;
    }
    return max;
}

/*
    Function to save vectors
    INPUT:
    - nl: number of levels
    - data: data with all info about setup
    - n: number of points
    - lvl(n): last level at which each point appears
    - W_len(n), Aff_len(n), AfP_len(n): length of each row of corresponding matrix
    - dvec(n): diagonal smoother for each point
*/
static void savevec(uint nl, struct amg_setup_data *data, uint n, uint *lvl,
    uint *W_len, uint *AfP_len, uint *Aff_len, double *dvec, const char *name)
{
    const double magic = 3.14159;
    const double stamp = 2.01;
    FILE *f = fopen(name,"w");
    fwrite(&magic,sizeof(double),1,f);
    fwrite(&stamp,sizeof(double),1,f);

    double dbldum; // dummy double to convert uint to double
    dbldum = (double)nl;
    fwrite(&dbldum,sizeof(double),1,f);

    fwrite(data->m,sizeof(double),nl-1,f);

    fwrite(data->rho,sizeof(double),nl-1,f);

    dbldum = (double)n;
    fwrite(&dbldum,sizeof(double),1,f);

    uint i;
    for (i=0;i<n;i++)
    {
            dbldum = (double)data->id[i];
            fwrite(&dbldum,sizeof(double),1,f);

            dbldum = (double)lvl[i];
            fwrite(&dbldum,sizeof(double),1,f);

            dbldum = (double)W_len[i];
            fwrite(&dbldum,sizeof(double),1,f);

            dbldum = (double)AfP_len[i];
            fwrite(&dbldum,sizeof(double),1,f);

            dbldum = (double)Aff_len[i];
            fwrite(&dbldum,sizeof(double),1,f);

            fwrite(&(dvec[i]),sizeof(double),1,f);
    }

    fclose(f);
}

/* Interpolation */
void interpolation(struct csr_mat *W, struct csr_mat *Af, struct csr_mat *Ac, 
    struct csr_mat *Ar, double gamma2, double tol)
{

    // Dimensions of the matrices
    uint rnf = Af->rn;//, cnf = Af->cn; Unused
    uint rnc = Ac->rn, cnc = Ac->cn;
    //uint rnr = Ar->rn; Unused
    uint cnr = Ar->cn;
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
    double *uc = tmalloc(double, cnr);
    init_array(uc, cnr, 1.);

    // v = pcg(Af,full(-Ar*uc),d,1e-16);
    double *tmp = tmalloc(double, rnf); 
    apply_M(tmp, 0, NULL, -1, Ar, uc);

    double *v = tmalloc(double, rnf); 

    double *b = tmalloc(double, rnf);
    init_array(b, rnf, 1.0);    

    pcg(v, Af, tmp, Df, 1e-16, b);   
    
    // dc = diag(Ac)
    double *Dc = tmalloc(double, cnc);
    diag(Dc, Ac);

    // Dcinv = 1./dc
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

    struct csr_mat *Wtmp, *W0, *AfW, *Arhat0, *Arhat;

    // alpha = Dc
    double *alpha = tmalloc(double, cnc);
    memcpy(alpha, Dc, cnc*sizeof(double));
    
    // Declare everything that is required in while loop
    struct csr_mat *Arr;
    struct csr_mat *ArW;
    struct csr_mat *new_skel;

    double *Dcsqrti;
    double *w1;
    double *w2;
    double *ones;
    double *r;

    double *Dfsqrti = Dfinv;
    array_op(Dfsqrti, rnf, sqrt_op);

    while(1)
    {
        Wtmp = tmalloc(struct csr_mat, 1);            
        W0   = tmalloc(struct csr_mat, 1);

//      [W,W0,lam] = intp.solve_weights(W_skel,Af,Ar,alpha,uc,v,tol1,lam);
//      Arhat0 = Af*W0+Ar;
//      Arhat  = Af*W +Ar;
        solve_weights(Wtmp, W0, lam, W_skel, Af, Ar, rnc, alpha, uc, v, tol);

        AfW = tmalloc(struct csr_mat, 1);
        Arhat0 = tmalloc(struct csr_mat, 1);

        mxm(AfW, Af, W0  , 0.0);
        mpm(Arhat0, 1., AfW, 1., Ar);
        free_csr(&AfW);
        
        AfW = tmalloc(struct csr_mat, 1);
        Arhat = tmalloc(struct csr_mat, 1);

        mxm(AfW , Af, Wtmp, 0.0);
        mpm(Arhat , 1., AfW, 1., Ar);
        free_csr(&AfW);

        Arr = tmalloc(struct csr_mat, 1);

//      dchat  = full(sum(W .* Arhat + Ar .* W, 1).' + diag(Ac));
//      Dcsqrti = spdiag(1./sqrt(dchat));
        mpm(Arr, 1.0, Arhat, 1.0, Ar);
        ArW = tmalloc(struct csr_mat, 1);
        mxmpoint(ArW, Wtmp,Arr);
        free_csr(&Arr);

        Dcsqrti = tmalloc(double, cnc);
        sum(Dcsqrti, ArW, 1);
        free_csr(&ArW);

        vv_op(Dcsqrti, Dc, cnc, plus);        

        array_op(Dcsqrti, cnc, minv_op);
        array_op(Dcsqrti, cnc, sqrt_op);

//      R  = abs(Dfsqrti*Arhat )*Dcsqrti;
//      Dimensions of R and R0 are rnf x cnc
        struct csr_mat *R = tmalloc(struct csr_mat,1); 
        copy_csr(R, Arhat); // R = Arhat

        diagcsr_op(R, Dfsqrti, dmult);// R=Dfsqrti*R (Dfsqrti*Arhat)
        array_op(R->a, R->row_off[rnf], abs_op); // R = abs(R)
        diagcsr_op(R, Dcsqrti, multd);

//      R0 = abs(Dfsqrti*Arhat0)*Dcsqrti;
        struct csr_mat *R0 = tmalloc(struct csr_mat,1); 
        copy_csr(R0, Arhat0); // R = Arhat

        diagcsr_op(R0, Dfsqrti, dmult);// R=Dfsqrti*R (Dfsqrti*Arhat)
        array_op(R0->a, R0->row_off[rnf], abs_op); // R = abs(R)
        diagcsr_op(R0, Dcsqrti, multd);

//      w1 = full(((R*one)'*R)');
//      w2 = full(((R*w1 )'*R)');
//      r = w2./w1; r(w1==0) = 0;
        w1 = tmalloc(double, cnc);
        w2 = tmalloc(double, cnc);
        ones = tmalloc(double, cnc);
        init_array(ones, cnc, 1.0);
        
        apply_M(tmp, 0., NULL, 1., R, ones);
        apply_Mt(w1, R, tmp);

        apply_M(tmp, 0., NULL, 1., R, w1);
        apply_Mt(w2, R, tmp);

        r = tmalloc(double, cnc);
        vv_op3(r, w2, w1, cnc, ewdiv);

        uint i;
        for (i=0;i<cnc;i++) // r(w1==0) = 0;
        {
            if (w1[i] == 0) r[i] = 0.;
        }    

//      n = sum(r>gamma2);
        uint n = 0.;    
        for (i=0;i<cnc;i++)
        {
            if (r[i] > gamma2) n += 1;
        }    

//      if (n==0 || max(w1)<=gamma2)
        double w1m;
        uint w1mi;
        extr_op(&w1m, &w1mi, w1, cnc, max);

        if (n == 0 || w1m <= gamma2)
        {
            free_csr(&W0);
            W0 = tmalloc(struct csr_mat, 1);
            solve_weights(W, W0, lam, W_skel, Af, Ar, rnc, alpha, uc, v, 1e-16);

            double *wuc = tmalloc(double, rnf);
            apply_M(wuc, 0., NULL, 1., W, uc);

            for (i=0;i<rnf;i++)
            {
                if (wuc[i] != 0.) 
                {
                    uint j, js, je;
                    js = W->row_off[i];
                    je = W->row_off[i+1]; 
                    for (j=js; j<je; j++)
                    {
                        if (i == W->col[j])
                        {
                            double vwuc = v[i]/wuc[i];
                            W->a[j] = vwuc*W->a[j];
                        }                    
                    }                    
                }
            } 
            // free matrices before break
            free_csr(&Wtmp);
            free_csr(&W0); 
            free_csr(&Arhat0); 
            free_csr(&Arhat); 
            free_csr(&R0); 
            free_csr(&R);
            free_csr(&W_skel);

            // free arrays before break
            free(Dcsqrti);
            free(w1);
            free(w2);
            free(ones);
            free(r); 
            free(wuc); 

            break;
        }

        
 //     alpha = dc./max(w2,1e-6);
        for (i=0;i<cnc;i++)
        {
            double x = w2[i] > 1e-6 ? w2[i] : 1e-6;
            alpha[i] = Dc[i] / x;
            //printf("alpha[%u] = %lf\n", i, alpha[i]);
        }

//      W_skel = intp.expand_support(W_skel,R,R0,gamma2);
        new_skel = tmalloc(struct csr_mat, 1);
        expand_support(new_skel, W_skel, R, R0, gamma2);

        free_csr(&W_skel);    
        W_skel = tmalloc(struct csr_mat, 1);
        copy_csr(W_skel, new_skel);

        // free matrices
        free_csr(&Wtmp); 
        free_csr(&W0); 
        free_csr(&Arhat0); 
        free_csr(&Arhat); 
        free_csr(&R0); 
        free_csr(&R);
        free_csr(&new_skel);

        // free arrays
        free(Dcsqrti);
        free(w1);
        free(w2);
        free(ones);
        free(r); 

    // End of while loop
    }

    free(alpha);
    free(Df);
    free(Dfinv);
    free(Dc);
    free(Dcinv);
    free(uc);
    free(v);
    free(b);
    free(lam);
    free(tmp);
}

/* Expand support of interpolation matrix */
void expand_support(struct csr_mat *new_skel, struct csr_mat *W_skel, 
    struct csr_mat *R, struct csr_mat *R0, double gamma)
{
    struct csr_mat *M = tmalloc(struct csr_mat, 1);
    find_support(M, R, gamma);

    uint nf = W_skel->rn, nc = W_skel->cn;

    // Temporary new_skel
    struct csr_mat *ns_tmp = tmalloc(struct csr_mat, 1);
    mpm(ns_tmp, 1., M, 1., W_skel);
    free_csr(&M);

    // bad_row = logical(sum(M & W_skel, 2)); nbad = sum(bad_row);
    double *bad_row = tmalloc(double, nf);
    uint nbad = 0;

    uint i;
    for (i=0;i<nf;i++)
    {
        uint j, js, je;
        js = ns_tmp->row_off[i];
        je = ns_tmp->row_off[i+1]; 
        bad_row[i] = 0.;
        for (j=js; j<je; j++)
        {
            if (ns_tmp->a[j] == 2.) // i.e. if M & W_skel == true 
            {
                bad_row[i] = 1.;
                nbad += 1;
                break;
            }
        }
    }

    // if nbad==0; return; end
    // set new_skel = W_skel | M before returning
    if (nbad == 0)
    {
        uint nnz = ns_tmp->row_off[nf];
        for (i=0;i<nnz;i++)
        {
            if (ns_tmp->a[i] == 2.) ns_tmp->a[i] = 1.;
        }
        copy_csr(new_skel, ns_tmp);
        free_csr(&ns_tmp);
        free(bad_row);
        return;
    }

    // ibr = find(bad_row);
    uint *ibr = tmalloc(uint, nbad);
    uint *ibrp = ibr;
    for (i=0;i<nf;i++)
    {
        if (bad_row[i] != 0.) *ibrp++ = i;
    }

    // X = R0 - (R0 .* W_skel);
    struct csr_mat *R0W = tmalloc(struct csr_mat, 1);
    mxmpoint(R0W, R0, W_skel);

    struct csr_mat *Xfull = tmalloc(struct csr_mat, 1);
    mpm(Xfull, 1., R0, -1, R0W);
    
    free_csr(&R0W);

    struct csr_mat *X = tmalloc(struct csr_mat, 1);
    double *onec = tmalloc(double, nc);
    init_array(onec, nc, 1.);

    // X = abs(X(bad_row,:));
    sub_mat(X, Xfull, bad_row, onec);
    free_csr(&Xfull);
    uint nnzx = X->row_off[nbad];
    array_op(X->a, nnzx, abs_op);

    // rank elements of X by descending magnitude, in each row
    coo_mat *SX = tmalloc(coo_mat, nnzx);
    csr2coo(SX, X);
    
    // Rows are already sorted after csr2coo
    // sarray_sort cannot sort according to double -> use qsort instead
    for (i=0;i<nbad;i++)
    {
        uint js=X->row_off[i], je=X->row_off[i+1];
        qsort(&(SX[js]), je-js, sizeof(coo_mat), cmp_coo_v_revert);
    }

    // [i j v] = find(X); sx = [vec(i) vec(j) vec(v)];
    // [~,idx] = sort(-sx(:,3)); sx = sx(idx,:);
    // [~,idx] = sort( sx(:,1)); sx = sx(idx,:);
    uint *sx1 = tmalloc(uint, nnzx), *sy2 = tmalloc(uint, nnzx);
    double *sx2 = tmalloc(double, nnzx), *sx3 = tmalloc(double, nnzx);

    for (i=0;i<nnzx;i++)
    {
        sx1[i] = SX[i].i;
        sx2[i] = (double)(SX[i].j)+1.; // +1 because C convention for row numbering
        sx3[i] = SX[i].v;
    }
    free(SX);

    // [i j v] = find(sort(X,2,'descend')); sy = [vec(i) vec(j) vec(v)];
    // [~,idx] = sort(sy(:,1)); sy = sy(idx,:);
    uint w = 0; // max # of nonzeros in any row
    uint *sy2p = sy2;
    for (i=0;i<nbad;i++)
    {
        uint j, js=X->row_off[i], je=X->row_off[i+1];
        uint wi = 0; // max # of nonzeros in row i
        for (j=js;j<je;j++)
        {
            *sy2p++ = wi++;
            if (wi > w) w = wi;
        }
    }
    free_csr(&X);

    // X = sparse(sy(:,2),sx(:,1),sx(:,3),w,nbad);
    X = tmalloc(struct csr_mat, 1);
    build_csr_dim(X, nnzx, sy2, sx1, sx3, w, nbad);

    // J = sparse(sy(:,2),sx(:,1),sx(:,2),w,nbad);
    struct csr_mat *J = tmalloc(struct csr_mat, 1);
    build_csr_dim(J, nnzx, sy2, sx1, sx2, w, nbad);

    // S = cumsum(X);
    struct csr_mat *S = tmalloc(struct csr_mat, 1);
    cumsum(S, X);

    // V = ones(w,1) * sum(X)/2; ==> big waste of memory
    double *sumX = tmalloc(double, nbad);
    sum(sumX, X, 1); 
    free_csr(&X);
    ar_scal_op(sumX, 0.5, nbad, mult_op);

    double *onew = tmalloc(double, w);
    init_array(onew, w, 1.);

    struct csr_mat *V = tmalloc(struct csr_mat, 1);
    dyad(V, onew, w, sumX, nbad);

    // N = ones(w,1) * (1+sum((S-V)<0)); ==> big waste of memory
    struct csr_mat *SV = tmalloc(struct csr_mat, 1);
    mpm(SV, 1., S, -1., V);
    free_csr(&S);
    free_csr(&V);
    uint nnzsv = SV->row_off[w];
    double *maskSV = tmalloc(double, nnzsv);
    mask_op(maskSV, SV->a, nnzsv, 0., lt); //=> Lots of zeros
    free(SV->a);
    SV->a = maskSV;

    double *sumSV = tmalloc(double, nbad);
    sum(sumSV, SV, 1);
    free_csr(&SV);
    ar_scal_op(sumSV, 1., nbad, add_op);

    struct csr_mat *N = tmalloc(struct csr_mat, 1);
    dyad(N, onew, w, sumSV, nbad);

    // L = sparse( ((1:w)' * ones(1,nbad)) <= N); ==> big waste of memory
    double *linspace = tmalloc(double, w);
    for (i=0;i<w;i++) linspace[i] = i+1;

    double *onenbad = tmalloc(double, nbad);
    init_array(onenbad, nbad, 1.);

    struct csr_mat *L = tmalloc(struct csr_mat, 1);
    dyad(L, linspace, w, onenbad, nbad);

    uint nnzL = L->row_off[w], nnzN = N->row_off[w];
    if (nnzL != nnzN)
    {
        printf("L and N should have same number of non zero elements.\n");
        die(0);
    }
    double *maskLN = tmalloc(double, nnzL);
    mask_arrays(maskLN, L->a, N->a, nnzL, le);
    free_csr(&N);
    free(L->a);
    L->a = maskLN;

    // [~,i,j] = find(L .* J);
    struct csr_mat *LJ = tmalloc(struct csr_mat, 1);
    mxmpoint(LJ, L, J);
    free_csr(&L);
    free_csr(&J);

    uint *ibri = tmalloc(uint, nnzL); // upper bound for size, there are many zeros in L.*J
    uint *coln = tmalloc(uint, nnzL); // upper bound for size, there are many zeros in L.*J
    uint *ibrip = ibri, *colnp = coln;
    uint nnzn = 0;

    for (i=0;i<w;i++)
    {
        uint j, js=LJ->row_off[i], je=LJ->row_off[i+1];
        for (j=js;j<je;j++)
        {
            if (LJ->a[j] != 0.)
            {
                nnzn++;
                *ibrip++ = ibr[LJ->col[j]];
                *colnp++ = (uint)(LJ->a[j]-1.); // -1 because C convention for row numbering
            }
        }
    }
    free_csr(&LJ);

    // N = sparse(ibr(i),j,1,nf,nc);
    double *onen = tmalloc(double, nnzn);
    init_array(onen, nnzn, 1.);
    N = tmalloc(struct csr_mat, 1);
    build_csr_dim(N, nnzn, ibri, coln, onen, nf, nc); 

    // new_skel = new_skel | N;   
    mpm(new_skel, 1., ns_tmp, 1., N);

    free_csr(&N);    
    free_csr(&ns_tmp);

    uint nnzns = new_skel->row_off[nf];
    for (i=0;i<nnzns;i++)
    {
        if (new_skel->a[i] != 0.) new_skel->a[i] = 1.;
    }

    free(bad_row);
    free(ibr);
    free(onec);
    free(onen);
    free(sx1);
    free(sx2);
    free(sx3);
    free(sy2);
    free(sumX);
    free(onew);
    free(onenbad);
    free(sumSV);
    free(ibri);
    free(coln); 
    free(linspace);
}

/* Dyadic product 
   X[i,j] = a[i]*b[j] */
void dyad(struct csr_mat *X, double *a, uint na, double *b, uint nb)
{
    // Count number of nonzero elements first (better in case a and b are sparse)
    uint i, j;
    uint nnz=0;
    for (i=0;i<na;i++)
    {
        for (j=0;j<nb;j++)
        {
            if (a[i]*b[j] != 0.) nnz++;
        }
    }

    malloc_csr(X, na, nb, nnz);

    X->row_off[0] = 0;

    // Build matrix
    uint k=0;
    for (i=0;i<na;i++)
    {
        X->row_off[i+1] = X->row_off[i];
        for (j=0;j<nb;j++)
        {
            double ab = a[i]*b[j];
            if (ab != 0.) 
            {
                X->row_off[i+1] += 1;
                X->col[k] = j;
                X->a[k] = ab;
                k++;
            }
        }
    }
}

/* Sum over - columns if dim = 1
            - rows    if dim = 2 
   Assumes that memory for s has been allocated accordingly */
void sum(double *s, struct csr_mat *A, uint dim)
{
    uint rn = A->rn, cn = A->cn;
    if (dim == 1)
    {
        init_array(s, cn, 0.0);
        uint i;
        for (i=0; i<A->row_off[rn]; i++)
        {   
            uint col = A->col[i];
            s[col]  += A->a[i];                
        } 
    }
    else if (dim == 2)
    {
        init_array(s, rn, 0.0);
        uint i;
        for (i=0; i<rn; i++)
        {   
            uint j, js=A->row_off[i], je=A->row_off[i+1];
            for (j=js; j<je; j++) s[i] += A->a[j];           
        } 
    }
    else {printf("dim should be either 1 or 2 in sum.\n"); die(0);}
}

/* Cumulated sum */
void cumsum (struct csr_mat *S, struct csr_mat *A)
{
    uint rn = A->rn, cn = A->cn;
    uint nnza = A->row_off[rn];
    uint nnzs = rn*cn;

    malloc_csr(S, rn, cn, nnzs);
    S->row_off[0] = 0;

    double *cs = tmalloc(double, cn);
    init_array(cs, cn, 0.);    

    uint i, j=0, k;
    for (i=0;i<rn;i++)
    {
        S->row_off[i+1] = S->row_off[i]+cn;
        for (k=0;k<cn;k++)
        {
            if (j < nnza)
            {
                uint colA = A->col[j];
                if (k == colA) cs[k] += A->a[j++];
            }
            uint m = i*cn+k;
            S->col[m] = k;
            S->a[m] = cs[k];
        }
    }
    free(cs);
}

/* Comparison function for sorting coo_mat in reverse according to v field */
int cmp_coo_v_revert (const void *a, const void *b)
{
    double av = ((coo_mat*)a)->v, bv = ((coo_mat*)b)->v;
    if ( av > bv ) return -1;
    if ( av < bv ) return 1;
    return 0;
}

void find_support(struct csr_mat *Skel, struct csr_mat *R, double goal)
{
    uint nf = R->rn, nc = R->cn; // [nf,nc] = size(R);
    uint nnz = R->row_off[nf];
    double *onec = tmalloc(double, nc); // one = ones(nc,1);
    init_array(onec, nc, 1.);

    double *onef = tmalloc(double, nf); // onef = ones(nf,1);
    init_array(onef, nf, 1.);

    uint nskel = 0;
    uint *skeli = tmalloc(uint, nnz); // skel = zeros(nnz(R),2);  
    uint *skelj = tmalloc(uint, nnz); // => Split in two 1D arrays for ease
    uint i;
    for (i=0;i<nnz;i++) {skeli[i] = 0; skelj[i] = 0;}

    double theta = 0.5;

    double *rs = tmalloc(double, nf);
    double *w = tmalloc(double, nc); 
    double *w2 = tmalloc(double, nc);
    double *tmp = tmalloc(double, nf);
    double *v = tmalloc(double, nc); 
    
    double *r = tmalloc(double, nc);
    double *sumR = tmalloc(double, nc);

    struct csr_mat *Rloc;

    // Need a local copy of R
    Rloc = tmalloc(struct csr_mat, 1);
    copy_csr(Rloc, R);

    while (1)
    {
        // rs = R*one;
        apply_M(rs, 0., NULL, 1., Rloc, onec);
        
        // w = (rs'*R)';
        apply_Mt(w, Rloc, rs);
        
        // w2 = ((R*w)'*R)';
        apply_M(tmp, 0., NULL, 1., Rloc, w);
        apply_Mt(w2, Rloc, tmp);

        // v = w2./w; v(w==0) = 0;
        vv_op3(v, w2, w, nc, ewdiv);

        for (i=0;i<nc;i++)
        {
            if (w[i] == 0.) v[i] = 0.;
        }
        
        // if max( v ) < goal || max( w ) < goal; break; end    
        double mv, mw; // max(v), max(w) 
        uint mvi, mwi; // index of max
        extr_op(&mv, &mvi, v, nc, max);
        extr_op(&mw, &mwi, v, nc, max);
    
        if (mv < goal || mw < goal) break;

        while (mw <= (1+theta)*goal) theta = theta/2.;
        
        sum(sumR, Rloc, 1);

        uint nbad = 0;
        for (i=0;i<nc;i++)
        {
            if (w[i] > (1+theta)*goal && sumR[i] != 0.) 
            {
                r[i] = 1.;
                nbad += 1;
            }
            else r[i] = 0.;
        }

        double *maxx = tmalloc(double, nbad);
        init_array(maxx, nbad, -DBL_MAX);
        uint *maski = tmalloc(uint, nbad);

        // X = spdiag(rs) * R(:,r);  % X_ij = R_ij ( e_i' R 1 )
        // if nf>1; [~,i] = max(X); else i=ones(1,nbad); end
        // here, i = maski
        if (nf > 1)
        {
            struct csr_mat *X = tmalloc(struct csr_mat, 1);
            sub_mat(X, Rloc, onef, r);

            diagcsr_op(X, rs, dmult);

            for (i=0;i<nf;i++)
            {
                uint j, js, je;
                js = X->row_off[i];
                je = X->row_off[i+1]; 
                for (j=js; j<je; j++)
                {
                    uint col = X->col[j];
                    if (X->a[j] > maxx[col])
                    {
                        maxx[col] = X->a[j];
                        maski[col] = i;
                    }                    
                }                    
            }            
            free_csr(&X);
        }
        else
        {
            for (i=0;i<nbad;i++) maski[i] = 1;
        }

        // maskj = find(r)
        uint *maskj = tmalloc(uint, nbad);
        uint *maskjp = maskj;
        for (i=0;i<nc;i++) 
        {
            if (r[i] != 0.) *maskjp++ = i;
        }
        
        // M = logical(sparse(mask(:,1),mask(:,2),1,nf,nc));
        struct csr_mat *M = tmalloc(struct csr_mat, 1);
    
        double *onebad = tmalloc(double, nbad); // onef = ones(nf,1);
        init_array(onebad, nbad, 1.);
        build_csr_dim(M, nbad, maski, maskj, onebad, nf, nc);

        // R = R - (R.*M);
        struct csr_mat *RM = tmalloc(struct csr_mat, 1);
        mxmpoint(RM, Rloc, M);

        struct csr_mat *Rnew = tmalloc(struct csr_mat, 1);
        mpm(Rnew, 1., Rloc, -1., RM);

        free_csr(&M);
        free_csr(&RM);
        free_csr(&Rloc);

        Rloc = tmalloc(struct csr_mat, 1);
        copy_csr(Rloc, Rnew); //
   
        free_csr(&Rnew); 

        // skel(nskel+(1:nbad),:) = mask;
        // nskel=nskel+nbad;
        memcpy(skeli+nskel, maski, nbad*sizeof(uint));
        memcpy(skelj+nskel, maskj, nbad*sizeof(uint));
        nskel += nbad;

        free(maxx);
        free(maski);
        free(maskj);
        free(onebad);
    }


    free_csr(&Rloc);

    double *oneskel = tmalloc(double, nskel);
    init_array(oneskel, nskel, 1.);
    build_csr_dim(Skel, nskel, skeli, skelj, oneskel, nf, nc);

    free(onef);
    free(onec);
    free(skeli);
    free(skelj);
    free(rs);
    free(w);
    free(w2);
    free(tmp);
    free(v);
    free(r);
    free(sumR);
    free(oneskel);
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
    
    // Matrices W0, Af and Ar should be transposed to get the correct result !
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

    // Free arrays
    free(au);
    free(zeros);
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

    // S = cinterp_lmop(Af,alpha.*(u.*u),W_skel,logical( (+W_skel)*(+W_skel)' ));
    double *au2 = tmalloc(double, nc);
    memcpy(au2, u, nc*sizeof(double)); // au2 = u
    array_op(au2, nc, sqr_op); // au2 = u.^2
    vv_op(au2, alpha, nc, ewmult); // au2 = alpha.*(u.^2)

    // Need to initialize S by (W_skel*W_skel') before calling interp_lmop
    struct csr_mat *S = tmalloc(struct csr_mat, 1);
    mxm(S, W_skel, W_skel, 1.0);  
    
    // S and Af are assumed to be symmetric -> not transposed
    interp_lmop(S, Af, au2, W_skelt);

    // resid = v - W0*u;
    double *resid = tmalloc(double, nf);
    apply_M(resid, 1.0, v, -1.0, W0, u);
    
    // dlogic = logical(diag(S));
    double *d = tmalloc(double, nf);
    diag(d, S);

    double *dlogic = tmalloc(double, nf);
    mask_op(dlogic, d, nf, 0.0, ne);

    // if ~all(i); S = S(i,i); lam(~i) = 0; end
    struct csr_mat *subS = tmalloc(struct csr_mat, 1);

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
        sub_mat(subS, S, dlogic, dlogic);
        free_csr(&S);
        S = subS;
    }

    // [x,k] = pcg(S,resid(i)-S*lam(i),1./diag(S),tol,resid(i))
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

    // lam(i) = lam(i)+x;
    double *xp = x;
    for (i=0;i<nf;i++)
    {
        if (dlogic[i] != 0.) lam[i] += *xp++;
    }

    free_csr(&S);
    free(subS);
    free(au2);
    free(resid);
    free(d);
    free(q);
    free(x);
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
    iy=*(++yi-1);
    uint ix = *xi;
    while(iy<ix) ++y, iy=*(++yi-1);
    *y++ += alpha * (*x);
  }
}

/* Matrix-matrix addition
   X = alpha*A + beta*B

   Remark: if alpha*A(i,j) + beta*B(i,j) == 0., element is ignored. 
*/
void mpm(struct csr_mat *X, double alpha, struct csr_mat *A, double beta,
    struct csr_mat *B)
{
    uint rna = A->rn, cna = A->cn;
    uint rnb = B->rn, cnb = B->cn;

    if (rna != rnb || cna != cnb)
    {
        printf("Mismatch in matrix dimensions in mpm.");
        die(0);
    }

    // Compute max number of non zeros in X and allocate memory
    // Assume alpha != 0 & beta != 0
    uint i, ja, jb, jsa, jea, jsb, jeb;
    uint nnzx = 0;

    for (i=0;i<rna;i++)
    {
        jsa = A->row_off[i];   
        jea = A->row_off[i+1]; 
        jsb = B->row_off[i];   
        jeb = B->row_off[i+1];
        for (ja=jsa,jb=jsb;(ja<jea || jb<jeb);) 
        {
            if (ja<jea && jb<jeb)
            {
                if (A->col[ja] == B->col[jb])
                {
                    ja++, jb++;
                }
                else
                {
                    A->col[ja] < B->col[jb] ? ja++ : jb++;
                }
            }
            else if (ja == jea)
            {   
                jb++;
            }
            else if (jb == jeb)
            {   
                ja++;
            }
            else
            {
                printf("Error when computing size of X in mpm.\n");
                die(0);
            }
            nnzx++;
        }
    }
    malloc_csr(X, rna, cna, nnzx);

    X->row_off[0] = 0;
    uint counter = 0;

    // Compute X = alpha*A + beta*B
    for (i=0;i<rna;i++)
    {
        jsa = A->row_off[i];   
        jea = A->row_off[i+1]; 
        jsb = B->row_off[i];   
        jeb = B->row_off[i+1];
        X->row_off[i+1] = X->row_off[i];
        for (ja=jsa,jb=jsb;(ja<jea || jb<jeb);) 
        {
            if (ja<jea && jb<jeb)
            {
                if (A->col[ja] == B->col[jb])
                {
                    double s = alpha*(A->a[ja]) + beta*(B->a[jb]);
                    if (s != 0.)
                    {
                        X->col[counter] = A->col[ja];
                        X->a[counter] = s;
                        X->row_off[i+1] += 1;
                        counter++;
                    }
                    ja++ , jb++;
                }
                else if (A->col[ja] < B->col[jb])
                {
                    X->col[counter] = A->col[ja];
                    X->a[counter] = alpha*(A->a[ja]);
                    ja++;
                    X->row_off[i+1] += 1;
                    counter++;
                }
                else
                {
                    X->col[counter] = B->col[jb];
                    X->a[counter] = beta*(B->a[jb]);
                    jb++;
                    X->row_off[i+1] += 1;
                    counter++;
                }
            }
            else if (ja == jea)
            {   
                X->col[counter] = B->col[jb];
                X->a[counter] = beta*(B->a[jb]);
                jb++;
                X->row_off[i+1] += 1;
                counter++;
            }
            else if (jb == jeb)
            {   
                X->col[counter] = A->col[ja];
                X->a[counter] = alpha*(A->a[ja]);
                ja++;
                X->row_off[i+1] += 1;
                counter++;
            }
        }
    }
}

/* Matrix-matrix pointwise multiplication
   X = A.* B

   Remark: if X(i,j) == 0., the element is still stored in memory. 
*/
void mxmpoint(struct csr_mat *X, struct csr_mat *A, struct csr_mat *B)
{
    uint rna = A->rn, cna = A->cn;
    uint rnb = B->rn, cnb = B->cn;

    if (rna != rnb || cna != cnb)
    {
        printf("Mismatch in matrix dimensions in mxmpoint.");
        die(0);
    }

    // Compute max number of non zeros in X and allocate memory
    // Assume alpha != 0 & beta != 0
    uint i, ja, jb, jsa, jea, jsb, jeb;
    uint nnzx = 0;

    for (i=0;i<rna;i++)
    {
        jsa = A->row_off[i];   
        jea = A->row_off[i+1]; 
        jsb = B->row_off[i];   
        jeb = B->row_off[i+1];
        for (ja=jsa,jb=jsb;(ja<jea || jb<jeb);) 
        {
            if (ja<jea && jb<jeb)
            {
                if (A->col[ja] == B->col[jb])
                {
                    ja++, jb++;
                }
                else
                {
                    A->col[ja] < B->col[jb] ? ja++ : jb++;
                }
            }
            else if (ja == jea)
            {   
                jb++;
            }
            else if (jb == jeb)
            {   
                ja++;
            }
            else
            {
                printf("Error when computing size of X in mxmpoint.\n");
                die(0);
            }
            nnzx++;
        }
    }

    malloc_csr(X, rna, cna, nnzx);
    X->row_off[0] = 0;
    uint counter = 0;

    // Compute X = A.*B
    for (i=0;i<rna;i++)
    {
        jsa = A->row_off[i];   
        jea = A->row_off[i+1]; 
        jsb = B->row_off[i];   
        jeb = B->row_off[i+1];
        X->row_off[i+1] = X->row_off[i];
        for (ja=jsa,jb=jsb;(ja<jea && jb<jeb);) 
        {   
            if (A->col[ja] == B->col[jb])
            {
                X->col[counter] = A->col[ja];
                X->a[counter] = (A->a[ja]) * (B->a[jb]);
                counter++;
                X->row_off[i+1] += 1;
                ja++; jb++;
            }     
            else if (A->col[ja] < B->col[jb]) ja++;
            else if (A->col[ja] > B->col[jb]) jb++; 
        }
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
                    nnzx++;
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
        apply_M(y, 0.0, NULL, 1.0, Bt, x);
    
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

    // Free arrays
    free(x);
    free(y);
}

void csr2coo(coo_mat *coo_A, const struct csr_mat *A)
{
 // Build matrix using coordinate list format
    uint rn = A->rn;
    //uint cn = A->cn; Unused
    //uint nnz = A->row_off[rn]; Unused

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
}
/* Transpose csr matrix */
void transpose(struct csr_mat *At, const struct csr_mat *A)
{
    uint rn = A->rn;
    uint cn = A->cn;
    uint nnz = A->row_off[rn];

    coo_mat *coo_A = tmalloc(coo_mat, nnz);
    csr2coo(coo_A,A);

    // Sort matrix by columns then rows
    buffer buf = {0};
    sarray_sort_2(coo_mat, coo_A, nnz, j, 0, i, 0, &buf);
    buffer_free(&buf);

    // Build transpose matrix
    uint rnt = cn, cnt = rn;
    malloc_csr(At, rnt, cnt, nnz);
    uint row_cur, row_prev = coo_A[0].j, counter = 1;
    At->row_off[0] = 0;
    
    uint i;
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
static void sp_restrict_unsorted(double *y, uint yn, const int *map_to_y,
    uint xn, const uint *xi, const double *x)
{
  const uint *xe = xi+xn; uint i;
  for(i=0;i<yn;++i) y[i]=0;
  for(;xi!=xe;++xi,++x) {
    /*This and map_to_y were unsigned (uint). That does not seem to make sense
     * because how can map_to_y[i] be < 0*/
    /*uint i = map_to_y[*xi];*/
    int i = map_to_y[*xi];
    if(i>=0) y[i]=*x;
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
    
    free(y_max);  
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
    
    // rho_0 = rho;
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
        free(p);
        free(z);
        free(tmp);
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
        apply_M(w, 0, NULL, 1, A, p);

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

==> Needs to be reimplemented and return the sparsified matrix directly
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
        apply_M(Aqk, 0, NULL, 1, A, qk);

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
    /*printf("Number of threads: %d\n", nthreads);*/
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

    while (1)
    {
        // w1 = vf.*(S*(vf.*(S*vf)))
        apply_M(g, 0, NULL, 1., S, vf); // g = S*vf 
        vv_op(g, vf, cn, ewmult); // g = vf.*g
        apply_M(w1, 0, NULL, 1., S, g); // w1 = S*g
        vv_op(w1, vf, cn, ewmult); // w1 = vf.*w1

        // w2 = vf.*(S *(vf.*(S*w1)))
        apply_M(w2, 0, NULL, 1., S, w1); // w2 = S*w1
        vv_op(w2, vf, cn, ewmult); // w2 = vf.*w2
        apply_M(tmp, 0, NULL, 1., S, w2); // tmp = S*w2
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
        if (vr[i] != 0)
        {
            subrn += 1;
            uint je=row_off[i+1]; 
            for(j=row_off[i]; j<je; j++)
            {
                if (vc[col[j]] != 0)
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
        if (vr[i] != 0)
        {
            uint je=row_off[i+1]; 
            for(j=row_off[i]; j<je; j++)
            {
                if (vc[col[j]] != 0)
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
        case(ewdiv):  for (i=0;i<n;i++) *a = *a / (*b), a++, b++; break;
    }
}

/* Vector-vector operations. Result stored in 3rd array.
   c = a (op) b */
void vv_op3(double *c, double *a, double *b, uint n, enum vv_ops op)
{
    uint i;
    switch(op)
    {
        case(plus):   for (i=0;i<n;i++) *c++ = *a++ + *b++; break;
        case(minus):  for (i=0;i<n;i++) *c++ = *a++ - *b++; break;
        case(ewmult): for (i=0;i<n;i++) *c++ = *a++ * (*b++); break;
        case(ewdiv):  for (i=0;i<n;i++) *c++ = *a++ / (*b++); break;
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
        case(gt): for (i=0;i<n;i++) mask[i] = (a[i] >  trigger)? 1. : 0.; break;
        case(lt): for (i=0;i<n;i++) mask[i] = (a[i] <  trigger)? 1. : 0.; break;
        case(ge): for (i=0;i<n;i++) mask[i] = (a[i] >= trigger)? 1. : 0.; break;
        case(le): for (i=0;i<n;i++) mask[i] = (a[i] <= trigger)? 1. : 0.; break;
        case(eq): for (i=0;i<n;i++) mask[i] = (a[i] =  trigger)? 1. : 0.; break;
        case(ne): for (i=0;i<n;i++) mask[i] = (a[i] != trigger)? 1. : 0.; break;
    }
}

/* Apply mask */
void apply_mask(double *a, uint n, double trigger, enum mask_ops op)
{
    double *mask = tmalloc(double, n);
    mask_op(mask, a, n, trigger, op);
    vv_op(a, mask, n, ewmult);
    free(mask);
}

/* Mask between elements of two arrays */
void mask_arrays(double *mask, double *a, double *b, uint n, enum mask_ops op)
{
    uint i;
    switch(op)
    {
        case(gt): for (i=0;i<n;i++) mask[i] = (a[i] >  b[i])? 1. : 0.; break;
        case(lt): for (i=0;i<n;i++) mask[i] = (a[i] <  b[i])? 1. : 0.; break;
        case(ge): for (i=0;i<n;i++) mask[i] = (a[i] >= b[i])? 1. : 0.; break;
        case(le): for (i=0;i<n;i++) mask[i] = (a[i] <= b[i])? 1. : 0.; break;
        case(eq): for (i=0;i<n;i++) mask[i] = (a[i] =  b[i])? 1. : 0.; break;
        case(ne): for (i=0;i<n;i++) mask[i] = (a[i] != b[i])? 1. : 0.; break;
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
        case(add_op)   :  for (i=0;i<n;i++) *a = (*a)+scal, a++; break;
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
    memcpy(a, tmp, n*sizeof(double));

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
    if (*A) 
    {
        free((*A)->row_off);
        free((*A)->col);
        free((*A)->a);
        free(*A);
        *A = NULL;
     }
}

/* Free data struct */
void free_data(struct amg_setup_data **data)
{
    if(*data)
    {
        free((*data)->n);
        free((*data)->nnz);
        free((*data)->nnzf);
        free((*data)->nnzfp);
        free((*data)->m); 
        free((*data)->rho);        

        uint i;
        for (i=0;i<(*data)->nlevels;i++)
        {
            free_csr(&((*data)->A[i]));
        }

        for (i=0;i<(*data)->nlevels-1;i++)
        {
            free((*data)->C[i]);
            free((*data)->F[i]);
            free((*data)->D[i]);
            free((*data)->idc[i]);
            free((*data)->idf[i]);
            free_csr(&((*data)->Af[i]));
            free_csr(&((*data)->W[i]));
            free_csr(&((*data)->AfP[i]));
        }
        free((*data)->id);
        free((*data)->idc);
        free((*data)->idf);
        free((*data)->C);
        free((*data)->F);
        free((*data)->D);
        free((*data)->A);
        free((*data)->Af);
        free((*data)->W);
        free((*data)->AfP);
        free(*data);
        *data = NULL;
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
//#pragma omp parallel 
    {
      const int nthreads = omp_get_num_threads();
      const int ithread = omp_get_thread_num();
//#pragma omp for
      for(i=0;i<cn;++i) 
      {
        int j;
        for(j=0;j<nthreads;++j) 
          yp[j*cn+i]=-DBL_MAX;
      }
//#pragma omp for 
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
//#pragma omp for 
    for(i=0;i<cn;i++)
    {
      for(int j=0;j<nthreads;j++)
      {
        if(y[i]<yp[j*cn+i]) y[i]=yp[j*cn+i];
      }
    }
    }
}

/* Build csr matrix from arrays of indices and values 
   Matrix dimensions are deduced from input data */
void build_csr(struct csr_mat *A, uint n, const uint *Ai, const uint* Aj, 
    const double *Av)
{
    // Build matrix in coord. list format
    coo_mat *coo_A = tmalloc(coo_mat, n);

    uint i, k=0;
    uint rn=0, cn=0;
    uint nnz=n;
    for (i=0;i<n;i++)
    {
        if (Av[i] != 0.)
        {
            coo_A[k].i = Ai[i];
            coo_A[k].j = Aj[i];
            coo_A[k].v = Av[i];
            k++;
        }
        else nnz--;
        // Check for dimensions
        if (coo_A[i].i+1 > rn) rn = coo_A[i].i+1;
        if (coo_A[i].j+1 > cn) cn = coo_A[i].j+1;
    }
    
    // Build csr matrix
    coo2csr(A, coo_A, nnz, rn, cn);
    free(coo_A);
}

/* Build csr matrix from arrays of indices and values 
   Matrix dimensions are imposed */
void build_csr_dim(struct csr_mat *A, uint n, const uint *Ai, const uint* Aj, 
    const double *Av, uint rn, uint cn)
{
    // Build matrix in coord. list format
    coo_mat *coo_A = tmalloc(coo_mat, n);

    uint i, k=0;
    uint nnz=n;
    for (i=0;i<n;i++)
    {
        if (Av[i] != 0.)
        {
            coo_A[k].i = Ai[i];
            coo_A[k].j = Aj[i];
            coo_A[k].v = Av[i];
            k++;
        }
        else nnz--;
    }
    
    // Build csr matrix
    coo2csr(A, coo_A, nnz, rn, cn);
    free(coo_A);
}

/* Build sparse matrix using the csr format */
void coo2csr(struct csr_mat *A, coo_mat *coo_A, uint nnz, uint rn, uint cn)
{
    // Sort matrix by rows then columns
    buffer buf = {0};
    sarray_sort_2(coo_mat, coo_A, nnz, i, 0, j, 0, &buf);
    buffer_free(&buf);

    malloc_csr(A, rn, cn, nnz);

    uint row_cur, row_prev = 0, counter = 1;
    A->row_off[0] = 0;
    
    uint i;
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
        
        A->col[i] = coo_A[i].j; // col
        A->a[i] = coo_A[i].v; // a

        if (i == nnz-1) 
        {
            for (;counter<=rn;)
            {
               A->row_off[counter++] = nnz;
            }
        }
    
        row_prev = row_cur;
    }
}

/* Build sparse matrix using the csr format. Original version. */
void coo2csr_v1(struct csr_mat *A, coo_mat *coo_A, uint nnz, uint rn, uint cn)
{
    // Sort matrix by rows then columns
    buffer buf = {0};
    sarray_sort_2(coo_mat, coo_A, nnz, i, 0, j, 0, &buf);
    buffer_free(&buf);

    malloc_csr(A, rn, cn, nnz);

    uint row_cur, row_prev = coo_A[0].i, counter = 1;
    A->row_off[0] = 0;
    
    uint i;
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

// Malloc array of pointers to csr_mat
void malloc_csr_arr(struct csr_mat ***arr, uint n)
{
    (*arr) = (struct csr_mat **) malloc(sizeof(struct csr_mat *)*n);
}

// Realloc array of pointers to csr_mat
void realloc_csr_arr(struct csr_mat ***arr, uint n)
{
    (*arr) = (struct csr_mat **) realloc(*arr, sizeof(struct csr_mat *)*n);
}

/* TO BE DELETED */
void print_csr(struct csr_mat *P)
{
#ifdef PRINT_DEBUG
    printf("P:\n"); 
    printf(" rn=%u, cn=%u, nnz=%u\n", (unsigned int) P->rn, (unsigned int) P->cn, (unsigned int) P->row_off[P->rn]); 
#endif
    uint ip, jp, jpe, jps;
    for (ip=0;ip<P->rn;ip++)
    {

        jps = P->row_off[ip];
        jpe = P->row_off[ip+1];   
#ifdef PRINT_DEBUG
        printf("js = %u, je = %u\n", (unsigned int) jps, (unsigned int) jpe);      
        for (jp=jps;jp<jpe;jp++)
        {
            printf("P[%u,%u] = %lf\n", (unsigned int) ip, (unsigned int) P->col[jp], P->a[jp]); 
        }
#endif
    }
}

void print_coo(coo_mat *P, uint nnz)
{
    uint i;
    for (i=0;i<nnz;i++) 
    {
        printf("P[%u,%u] = %lf\n", P[i].i, P[i].j, P[i].v);
    }
}
