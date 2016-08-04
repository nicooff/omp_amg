
SRC_FILES=$(wildcard *.c)
BASENAMES=$(basename $(SRC_FILES))
OBJ_FILES=$(addsuffix .o,$(BASENAMES))
OBJECTS= "serial_amg.o amg_tools.o amg_setup.o fail.o"
#comm.o comm_test.o crs_test.o crystal.o crystal_test.o fail.o fcrs.o fcrystal.o findpts.o findpts_el_2.o findpts_el_2_test2.o findpts_el_2_test.o findpts_el_3.o findpts_el_3_test2.o findpts_el_3_test.o findpts_local.o findpts_local_test.o findpts_test.o gen_poly_imp.o gs.o gs_local.o gs_test.o gs_test_old.o gs_unique_test.o lob_bnd.o lob_bnd_test.o obbox.o obbox_test.o poly.o poly_test2.o poly_test.o rand_elt_test.o sarray_sort.o sarray_sort_test.o sarray_transfer.o sarray_transfer_test.o sort.o sort_test2.o sort_test.o sparse_cholesky.o spchol_test.o tensor.o xxt.o xxt_test2.o xxt_test.o

CC=gcc -fopenmp -std=c99 
#--pedantic
#CFLAGS+=-DMPI
#CFLAGS+=-DPREFIX=jl_
CFLAGS+=-DNO_NEK_EXITT 
CFLAGS+=-DGLOBAL_LONG -DUSE_LONG
LDFLAGS+=-lm

#CFLAGS+=-DPRINT_MALLOCS=1

CFLAGS+=-DUSE_NAIVE_BLAS
#CFLAGS+=-DUSE_CBLAS
#LDFLAGS+=-lcblas

#CFLAGS+=-DAMG_DUMP
CFLAGS+=-DGS_TIMING -DGS_BARRIER

CFLAGS+=-g -O0 -march=native

CFLAGS+=-W -Wall -Wno-unused-function -Wno-unused-parameter
#CFLAGS+=-Minform=warn

CCCMD=$(CC) $(G) $(CFLAGS)

all: serial_amg

serial_amg: serial_amg.o amg_tools.o amg_setup.o fail.o sort.o sarray_sort.o 

	$(CC) $(CFLAGS) -o $@ serial_amg.o amg_tools.o amg_setup.o fail.o sort.o sarray_sort.o -lm


%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	-rm serial_amg *.o

.PHONY: clean
