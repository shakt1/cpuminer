
if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST	= example-cfg.json

SUBDIRS		= compat

INCLUDES	= $(PTHREAD_FLAGS) $(JANSSON_INCLUDES)

bin_PROGRAMS	= minerd

if HAVE_x86_64
minerd_SOURCES	= elist.h miner.h compat.h			\
		  cpu-miner.c util.c scrypt.c sha256-helpers.h	\
		  scrypt-simd-helpers.h scrypt-x86.S scrypt-x64.S
endif

minerd_LDFLAGS	= $(PTHREAD_FLAGS)
minerd_LDADD	= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@
minerd_CPPFLAGS = @LIBCURL_CPPFLAGS@

if HAVE_CELL_SPU

minerd_SOURCES	= elist.h miner.h compat.h			\
		  cpu-miner.c util.c scrypt.c sha256-helpers.h	\
		  scrypt-simd-helpers.h 

scrypt-cell-spu.o: scrypt-cell-spu.c sha256-helpers.h \
		   scrypt-simd-helpers.h scrypt-cell-spu.h
	$(SPU_ELF_GCC) -O3 -fstrict-aliasing -Wall -Wstrict-aliasing \
		       -o scrypt-cell-spu.elf scrypt-cell-spu.c
	$(EMBEDSPU) scrypt_spu scrypt-cell-spu.elf scrypt-cell-spu.o

minerd_LDADD += scrypt-cell-spu.o @SPE2_LIBS@

endif
