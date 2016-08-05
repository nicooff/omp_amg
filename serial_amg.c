#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>
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

#define N sizeof(double)
static double byteswap(double x)
{
  char buf[N]; char t;
  memcpy(buf,&x,N);
#define SWAP(i) if(N>2*(i)+1) t=buf[i],buf[i]=buf[N-1-(i)],buf[N-1-(i)]=t
#define SWAP2(i) SWAP(i); SWAP((i)+1)
#define SWAP4(i) SWAP2(i); SWAP2((i)+2)
#define SWAP8(i) SWAP4(i); SWAP4((i)+4)
#define SWAP16(i) SWAP8(i); SWAP8((i)+8)
  SWAP16(0);
#undef SWAP
#undef SWAP2
#undef SWAP4
#undef SWAP8
#undef SWAP16
  memcpy(&x,buf,N);
  return x;
}
#undef N

static long filesize(const char *name) {
  long n;
  FILE *f = fopen(name,"r");
  fseek(f,0,SEEK_END);
  n = ftell(f)/sizeof(double);
  fclose(f);
  return n;
}

static long readfile(double *data, long max, const char *name)
{
  const double magic = 3.14159;
  long n;
  FILE *f = fopen(name,"r");
  
  fseek(f,0,SEEK_END);
  n = ftell(f)/sizeof(double);
  if(n>max) printf("file longer than expected"),n=max;
  fseek(f,0,SEEK_SET);
  fread(data,sizeof(double),n,f);
  fclose(f);
  if(n>0 && fabs(data[0]-magic)>0.000001) {
    long i;
    printf("swapping byte order");
    if(fabs(byteswap(data[0])-magic)>0.000001) {
      printf("magic number for endian test not found");
    } else
      for(i=0;i<n;++i) data[i]=byteswap(data[i]);
  }
  return n;
}

int main(int argc, char *argv[])
{
    uint n = filesize("amgdmp_i.dat");
    double *v  = tmalloc(double, n);
    double *Aid = tmalloc(double, n);
    double *Ajd = tmalloc(double, n);
    double *Av = tmalloc(double, n);
    
    readfile(v,n,"amgdmp_i.dat");
    memcpy(Aid, v+1, (n-1)*sizeof(double));
    readfile(v,n,"amgdmp_j.dat");
    memcpy(Ajd, v+1, (n-1)*sizeof(double));
    readfile(v,n,"amgdmp_p.dat");
    memcpy(Av, v+1, (n-1)*sizeof(double));

    uint *Ai = tmalloc(uint, n-1);
    uint *Aj = tmalloc(uint, n-1);

    uint i;
    for (i=0;i<n-1;i++) 
    {
        Ai[i] = (uint)Aid[i]-1;
        Aj[i] = (uint)Ajd[i]-1;
    }

    struct amg_setup_data *data = tmalloc(struct amg_setup_data, 1);
    amg_setup(n-1, Ai, Aj, Av, data);
    //free_data(&data); NOT WORKING

    free(v);
    free(Aid);
    free(Ajd);
    free(Ai);
    free(Aj);
    free(Av);
}
