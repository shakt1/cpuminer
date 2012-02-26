/*-
 * Copyright 2009 Colin Percival, 2011 ArtForz, 2011 Siarhei Siamashka
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 */

/*
 * This is the performance critical part of scrypt key derivation function [1],
 * implemented using gcc vector extensions [2]. The parameters are set
 * to N = 1024, r = 1, p = 1 as used for litecoin proof of work [3].
 *
 * The drawback is that these extensions are only supported by gcc and a few
 * other compilers, which are trying to be gcc-compatible (clang, path64, ...)
 *
 * The advantage is that this code works on any SIMD capable hardware
 * (x86 SSE2, PowerPC Altivec, Cell SPU, ARM NEON, ARM iWMMXt, ...) without
 * modifications when compiled with gcc 4.7. The older compiler versions are
 * missing bits and pieces, but still can work for Altivec, SPU and SSE2 with
 * a bit of intrinsic band aid.
 *
 * 1. http://www.tarsnap.com/scrypt.html
 * 2. http://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html
 * 3. https://github.com/coblee/litecoin/wiki/Scrypt-proof-of-work
 */

#ifndef __SCRYPT_SIMD_HELPERS_H__
#define __SCRYPT_SIMD_HELPERS_H__

#include <stdint.h>
//#include "sha256-helpers.h"

#if defined(__GNUC__) && \
    ((defined(__SSE2__) || defined(__ALTIVEC__) || defined(__SPU__)) || \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)))

#define HAVE_SCRYPT_SIMD_HELPERS

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __ALTIVEC__
#include <altivec.h>
#include <vec_types.h>
#endif

#ifdef __SPU__
#include <spu_intrinsics.h>
#endif

typedef uint32_t uint32x4 __attribute__ ((vector_size(16), aligned(16)));
typedef uint8_t uint8x16 __attribute__ ((vector_size(16), aligned(16)));

/*
 * Define two helper functions ('rol_32x4' and 'shuffle_32x4') to ensure
 * better support for old gcc versions and gcc-compatible compilers
 */
static __attribute__((always_inline)) uint32x4
rol_32x4(uint32x4 a, uint32_t b)
{
#ifdef __ALTIVEC__
	return vec_rl(a, vec_splats(b));
#elif defined(__SPU__)
	return spu_rl(a, b);
#elif defined(__SSE2__)
	return (uint32x4)_mm_slli_epi32((__m128i)a, b) ^
	       (uint32x4)_mm_srli_epi32((__m128i)a, 32 - b);
#else
	return (a << b) ^ (a >> (32 - b));
#endif
}

#if defined(__clang__)
# define shuffle_32x4(a, p1, p2, p3, p4) \
	__builtin_shufflevector(a, a, p1, p2, p3, p4)
#elif defined(__SSE2__)
# define shuffle_32x4(a, p1, p2, p3, p4) \
	(uint32x4)_mm_shuffle_epi32((__m128i)a, _MM_SHUFFLE(p4, p3, p2, p1))
#else
static __attribute__((always_inline)) uint32x4
shuffle_32x4(uint32x4 a, const int p1, const int p2, const int p3, const int p4)
{
#if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7))
	/* gcc 4.7 introduces '__builtin_shuffle' */
	const uint32x4 mask = { p1, p2, p3, p4 };
	return __builtin_shuffle(a, mask);
#elif defined(__SPU__)
	const uint8x16 mask = {
		p1 * 4, p1 * 4 + 1, p1 * 4 + 2, p1 * 4 + 3,
		p2 * 4, p2 * 4 + 1, p2 * 4 + 2, p2 * 4 + 3,
		p3 * 4, p3 * 4 + 1, p3 * 4 + 2, p3 * 4 + 3,
		p4 * 4, p4 * 4 + 1, p4 * 4 + 2, p4 * 4 + 3
	};
	return spu_shuffle(a, a, mask);
#elif defined(__ALTIVEC__)
	const uint8x16 mask = {
		p1 * 4, p1 * 4 + 1, p1 * 4 + 2, p1 * 4 + 3,
		p2 * 4, p2 * 4 + 1, p2 * 4 + 2, p2 * 4 + 3,
		p3 * 4, p3 * 4 + 1, p3 * 4 + 2, p3 * 4 + 3,
		p4 * 4, p4 * 4 + 1, p4 * 4 + 2, p4 * 4 + 3
	};
	return vec_perm(a, a, mask);
#else
# error Have no implementation for 'shuffle_32x4' inline function
#endif
}
#endif

/*****************************************************************************/

static __attribute__((always_inline)) void
blkcpy128(uint32x4 * __restrict D, const uint32x4 * __restrict S)
{
	D[0] = S[0]; D[1] = S[1]; D[2] = S[2]; D[3] = S[3];
	D[4] = S[4]; D[5] = S[5]; D[6] = S[6]; D[7] = S[7];
}

static __attribute__((always_inline)) void
blkxor128(uint32x4 * __restrict D, const uint32x4 * __restrict S)
{
	D[0] ^= S[0]; D[1] ^= S[1]; D[2] ^= S[2]; D[3] ^= S[3];
	D[4] ^= S[4]; D[5] ^= S[5]; D[6] ^= S[6]; D[7] ^= S[7];
}

static __attribute__((always_inline)) void
salsa20_8_xor3(uint32x4 * __restrict B, const uint32x4 * __restrict Bx,
               uint32x4 * __restrict C, const uint32x4 * __restrict Cx,
               uint32x4 * __restrict D, const uint32x4 * __restrict Dx)
{
        register uint32x4 X0, X1, X2, X3;
        register uint32x4 Y0, Y1, Y2, Y3;
        register uint32x4 Z0, Z1, Z2, Z3;
        int i;

        X0 = (B[0] ^= Bx[0]);
        X1 = (B[1] ^= Bx[1]);
        X2 = (B[2] ^= Bx[2]);
        X3 = (B[3] ^= Bx[3]);
        Y0 = (C[0] ^= Cx[0]);
        Y1 = (C[1] ^= Cx[1]);
        Y2 = (C[2] ^= Cx[2]);
        Y3 = (C[3] ^= Cx[3]);
        Z0 = (D[0] ^= Dx[0]);
        Z1 = (D[1] ^= Dx[1]);
        Z2 = (D[2] ^= Dx[2]);
        Z3 = (D[3] ^= Dx[3]);
#pragma unroll 
        for (i = 0; i < 8; i += 2) {
                /* Operate on "columns". */
                X1 ^= rol_32x4(X0 + X3, 7);
                Y1 ^= rol_32x4(Y0 + Y3, 7);
                Z1 ^= rol_32x4(Z0 + Z3, 7);
                X2 ^= rol_32x4(X1 + X0, 9);
                Y2 ^= rol_32x4(Y1 + Y0, 9);
                Z2 ^= rol_32x4(Z1 + Z0, 9);
                X3 ^= rol_32x4(X2 + X1, 13);
                Y3 ^= rol_32x4(Y2 + Y1, 13);
                Z3 ^= rol_32x4(Z2 + Z1, 13);
                X0 ^= rol_32x4(X3 + X2, 18);
                Y0 ^= rol_32x4(Y3 + Y2, 18);
                Z0 ^= rol_32x4(Z3 + Z2, 18);

                /* Rearrange data. */
                X1 = shuffle_32x4(X1, 3, 0, 1, 2);
                Y1 = shuffle_32x4(Y1, 3, 0, 1, 2);
                Z1 = shuffle_32x4(Z1, 3, 0, 1, 2);
                X2 = shuffle_32x4(X2, 2, 3, 0, 1);
                Y2 = shuffle_32x4(Y2, 2, 3, 0, 1);
                Z2 = shuffle_32x4(Z2, 2, 3, 0, 1);
                X3 = shuffle_32x4(X3, 1, 2, 3, 0);
                Y3 = shuffle_32x4(Y3, 1, 2, 3, 0);
                Z3 = shuffle_32x4(Z3, 1, 2, 3, 0);

                /* Operate on "rows". */
                X3 ^= rol_32x4(X0 + X1, 7);
                Y3 ^= rol_32x4(Y0 + Y1, 7);
                Z3 ^= rol_32x4(Z0 + Z1, 7);
                X2 ^= rol_32x4(X3 + X0, 9);
                Y2 ^= rol_32x4(Y3 + Y0, 9);
                Z2 ^= rol_32x4(Z3 + Z0, 9);
                X1 ^= rol_32x4(X2 + X3, 13);
                Y1 ^= rol_32x4(Y2 + Y3, 13);
                Z1 ^= rol_32x4(Z2 + Z3, 13);
                X0 ^= rol_32x4(X1 + X2, 18);
                Y0 ^= rol_32x4(Y1 + Y2, 18);
                Z0 ^= rol_32x4(Z1 + Z2, 18);

                /* Rearrange data. */
                X1 = shuffle_32x4(X1, 1, 2, 3, 0);
                Y1 = shuffle_32x4(Y1, 1, 2, 3, 0);
                Z1 = shuffle_32x4(Z1, 1, 2, 3, 0);
                X2 = shuffle_32x4(X2, 2, 3, 0, 1);
                Y2 = shuffle_32x4(Y2, 2, 3, 0, 1);
                Z2 = shuffle_32x4(Z2, 2, 3, 0, 1);
                X3 = shuffle_32x4(X3, 3, 0, 1, 2);
                Y3 = shuffle_32x4(Y3, 3, 0, 1, 2);
                Z3 = shuffle_32x4(Z3, 3, 0, 1, 2);
        }

        B[0] += X0;
        B[1] += X1;
        B[2] += X2;
        B[3] += X3;
        C[0] += Y0;
        C[1] += Y1;
        C[2] += Y2;
        C[3] += Y3;
        D[0] += Z0;
        D[1] += Z1;
        D[2] += Z2;
        D[3] += Z3;
}

/* Helps to prevent the violation of strict aliasing rules */
typedef union { uint32x4 q[8]; uint32_t w[32]; } XY;

static inline
void scrypt_simd_core_3way(uint32_t * databufA, uint32_t * databufB, uint32_t * databufC, void * scratch)
{
//	uint32_t * databufA = (uint32_t *)&databuf[0];
//	uint32_t * databufB = (uint32_t *)&databuf[32];
//	uint32_t * databufC = (uint32_t *)&databuf[64];
	XY       * XA = (XY *)((uintptr_t)(scratch + 128*0));
	XY       * XB = (XY *)((uintptr_t)(scratch + 128*1));
	XY       * XC = (XY *)((uintptr_t)(scratch + 128*2));
	uint32x4 * VA = (uint32x4 *)((uintptr_t)(scratch + 128*3));
	uint32x4 * VB = (uint32x4 *)((uintptr_t)(scratch + 128*4));
	uint32x4 * VC = (uint32x4 *)((uintptr_t)(scratch + 128*5));
	int i, jA, jB, jC;

	/* 1: X <-- B */
	for (i = 0; i < 16; i++) {
		XA->w[i]      = databufA[i * 5 % 16];
		XA->w[16 + i] = databufA[16 + (i * 5 % 16)];
		XB->w[i]      = databufB[i * 5 % 16];
		XB->w[16 + i] = databufB[16 + (i * 5 % 16)];
		XC->w[i]      = databufC[i * 5 % 16];
		XC->w[16 + i] = databufC[16 + (i * 5 % 16)];
	}

	/* 2: for i = 0 to N - 1 do */
	for (i = 0; i < 1024; i+=2) {
		blkcpy128(&VA[i * 8*3], &XA->q[0]);
		blkcpy128(&VB[i * 8*3], &XB->q[0]);
		blkcpy128(&VC[i * 8*3], &XC->q[0]);
		salsa20_8_xor3(&XA->q[0], &XA->q[4], &XB->q[0], &XB->q[4], &XC->q[0], &XC->q[4] );
		salsa20_8_xor3(&XA->q[4], &XA->q[0], &XB->q[4], &XB->q[0], &XC->q[4], &XC->q[0] );
		blkcpy128(&VA[(i+1) * 8*3], &XA->q[0]);
		blkcpy128(&VB[(i+1) * 8*3], &XB->q[0]);
		blkcpy128(&VC[(i+1) * 8*3], &XC->q[0]);
		salsa20_8_xor3(&XA->q[0], &XA->q[4], &XB->q[0], &XB->q[4], &XC->q[0], &XC->q[4] );
		salsa20_8_xor3(&XA->q[4], &XA->q[0], &XB->q[4], &XB->q[0], &XC->q[4], &XC->q[0] );
	}

	/* 6: for i = 0 to N - 1 do */
	for (i = 0; i < 1024; i+=2) {
		jA = XA->w[16] & 1023; /* j <-- Integerify(X) mod N */
		jB = XB->w[16] & 1023; /* j <-- Integerify(X) mod N */
		jC = XC->w[16] & 1023; /* j <-- Integerify(X) mod N */
		blkxor128(XA->q, &VA[jA * 8*3]);
		blkxor128(XB->q, &VB[jB * 8*3]);
		blkxor128(XC->q, &VC[jC * 8*3]);
		salsa20_8_xor3(&XA->q[0], &XA->q[4], &XB->q[0], &XB->q[4], &XC->q[0], &XC->q[4] );
		salsa20_8_xor3(&XA->q[4], &XA->q[0], &XB->q[4], &XB->q[0], &XC->q[4], &XC->q[0] );
		jA = XA->w[16] & 1023; /* j <-- Integerify(X) mod N */
		jB = XB->w[16] & 1023; /* j <-- Integerify(X) mod N */
		jC = XC->w[16] & 1023; /* j <-- Integerify(X) mod N */
		blkxor128(XA->q, &VA[jA * 8*3]);
		blkxor128(XB->q, &VB[jB * 8*3]);
		blkxor128(XC->q, &VC[jC * 8*3]);
		salsa20_8_xor3(&XA->q[0], &XA->q[4], &XB->q[0], &XB->q[4], &XC->q[0], &XC->q[4]);
		salsa20_8_xor3(&XA->q[4], &XA->q[0], &XB->q[4], &XB->q[0], &XC->q[4], &XC->q[0]);
	}

	/* 10: B' <-- X */
	for (i = 0; i < 16; i++) {
		databufA[i * 5 % 16] = XA->w[i];
		databufA[16 + (i * 5 % 16)] = XA->w[16 + i];
		databufB[i * 5 % 16] = XB->w[i];
		databufB[16 + (i * 5 % 16)] = XB->w[16 + i];
		databufC[i * 5 % 16] = XC->w[i];
		databufC[16 + (i * 5 % 16)] = XC->w[16 + i];
	}

}

static __attribute__((always_inline)) void
salsa20_8_xor2(uint32x4 * __restrict B, const uint32x4 * __restrict Bx,
               uint32x4 * __restrict C, const uint32x4 * __restrict Cx)
{
        uint32x4 X0, X1, X2, X3;
        uint32x4 Y0, Y1, Y2, Y3;
        int i;

        X0 = (B[0] ^= Bx[0]);
        X1 = (B[1] ^= Bx[1]);
        X2 = (B[2] ^= Bx[2]);
        X3 = (B[3] ^= Bx[3]);
        Y0 = (C[0] ^= Cx[0]);
        Y1 = (C[1] ^= Cx[1]);
        Y2 = (C[2] ^= Cx[2]);
        Y3 = (C[3] ^= Cx[3]);

        for (i = 0; i < 8; i += 2) {
                /* Operate on "columns". */
                X1 ^= rol_32x4(X0 + X3, 7);
                Y1 ^= rol_32x4(Y0 + Y3, 7);
                X2 ^= rol_32x4(X1 + X0, 9);
                Y2 ^= rol_32x4(Y1 + Y0, 9);
                X3 ^= rol_32x4(X2 + X1, 13);
                Y3 ^= rol_32x4(Y2 + Y1, 13);
                X0 ^= rol_32x4(X3 + X2, 18);
                Y0 ^= rol_32x4(Y3 + Y2, 18);

                /* Rearrange data. */
                X1 = shuffle_32x4(X1, 3, 0, 1, 2);
                Y1 = shuffle_32x4(Y1, 3, 0, 1, 2);
                X2 = shuffle_32x4(X2, 2, 3, 0, 1);
                Y2 = shuffle_32x4(Y2, 2, 3, 0, 1);
                X3 = shuffle_32x4(X3, 1, 2, 3, 0);
                Y3 = shuffle_32x4(Y3, 1, 2, 3, 0);

                /* Operate on "rows". */
                X3 ^= rol_32x4(X0 + X1, 7);
                Y3 ^= rol_32x4(Y0 + Y1, 7);
                X2 ^= rol_32x4(X3 + X0, 9);
                Y2 ^= rol_32x4(Y3 + Y0, 9);
                X1 ^= rol_32x4(X2 + X3, 13);
                Y1 ^= rol_32x4(Y2 + Y3, 13);
                X0 ^= rol_32x4(X1 + X2, 18);
                Y0 ^= rol_32x4(Y1 + Y2, 18);

                /* Rearrange data. */
                X1 = shuffle_32x4(X1, 1, 2, 3, 0);
                Y1 = shuffle_32x4(Y1, 1, 2, 3, 0);
                X2 = shuffle_32x4(X2, 2, 3, 0, 1);
                Y2 = shuffle_32x4(Y2, 2, 3, 0, 1);
                X3 = shuffle_32x4(X3, 3, 0, 1, 2);
                Y3 = shuffle_32x4(Y3, 3, 0, 1, 2);
        }

        B[0] += X0;
        B[1] += X1;
        B[2] += X2;
        B[3] += X3;
        C[0] += Y0;
        C[1] += Y1;
        C[2] += Y2;
        C[3] += Y3;
}

static inline
void scrypt_simd_core_2way(uint32_t *databufA, uint32_t * databufB, void * scratch)
{
        XY       * XA = (XY *)((uintptr_t)scratch);
        XY       * XB = (XY *)((uintptr_t)scratch + 128 );
        uint32x4 * VA = (uint32x4 *)((uintptr_t)XA + 128*2);
        uint32x4 * VB = (uint32x4 *)((uintptr_t)XA + 128*3);
        int i, jA, jB;

        /* 1: X <-- B */
        for (i = 0; i < 16; i++) {
                XA->w[i]      = databufA[i * 5 % 16];
                XA->w[16 + i] = databufA[16 + (i * 5 % 16)];
                XB->w[i]      = databufB[i * 5 % 16];
                XB->w[16 + i] = databufB[16 + (i * 5 % 16)];
        }

        /* 2: for i = 0 to N - 1 do */
        for (i = 0; i < 1024; i++) {
                blkcpy128(&VA[i * 8*2], &XA->q[0]);
                blkcpy128(&VB[i * 8*2], &XB->q[0]);
                salsa20_8_xor2(&XA->q[0], &XA->q[4], &XB->q[0], &XB->q[4]);
                salsa20_8_xor2(&XA->q[4], &XA->q[0], &XB->q[4], &XB->q[0]);
        }

        /* 6: for i = 0 to N - 1 do */
        for (i = 0; i < 1024; i++) {
                jA = XA->w[16] & 1023; /* j <-- Integerify(X) mod N */
                jB = XB->w[16] & 1023; /* j <-- Integerify(X) mod N */
                blkxor128(XA->q, &VA[jA * 8*2]);
                blkxor128(XB->q, &VB[jB * 8*2]);
                salsa20_8_xor2(&XA->q[0], &XA->q[4], &XB->q[0], &XB->q[4]);
                salsa20_8_xor2(&XA->q[4], &XA->q[0], &XB->q[4], &XB->q[0]);
        }

        /* 10: B' <-- X */
        for (i = 0; i < 16; i++) {
                databufA[i * 5 % 16] = XA->w[i];
                databufA[16 + (i * 5 % 16)] = XA->w[16 + i];
                databufB[i * 5 % 16] = XB->w[i];
                databufB[16 + (i * 5 % 16)] = XB->w[16 + i];
        }
}

static __attribute__((always_inline)) void
salsa20_8_xor(uint32x4 * __restrict B, const uint32x4 * __restrict Bx)
{
        uint32x4 X0, X1, X2, X3;
        int i;

        X0 = (B[0] ^= Bx[0]);
        X1 = (B[1] ^= Bx[1]);
        X2 = (B[2] ^= Bx[2]);
        X3 = (B[3] ^= Bx[3]);

        for (i = 0; i < 8; i += 2) {
                /* Operate on "columns". */
                X1 ^= rol_32x4(X0 + X3, 7);
                X2 ^= rol_32x4(X1 + X0, 9);
                X3 ^= rol_32x4(X2 + X1, 13);
                X0 ^= rol_32x4(X3 + X2, 18);

                /* Rearrange data. */
                X1 = shuffle_32x4(X1, 3, 0, 1, 2);
                X2 = shuffle_32x4(X2, 2, 3, 0, 1);
                X3 = shuffle_32x4(X3, 1, 2, 3, 0);

                /* Operate on "rows". */
                X3 ^= rol_32x4(X0 + X1, 7);
                X2 ^= rol_32x4(X3 + X0, 9);
                X1 ^= rol_32x4(X2 + X3, 13);
                X0 ^= rol_32x4(X1 + X2, 18);

                /* Rearrange data. */
                X1 = shuffle_32x4(X1, 1, 2, 3, 0);
                X2 = shuffle_32x4(X2, 2, 3, 0, 1);
                X3 = shuffle_32x4(X3, 3, 0, 1, 2);
        }
        B[0] += X0;
        B[1] += X1;
        B[2] += X2;
        B[3] += X3;
}


static inline
void scrypt_simd_core(uint32_t * databufA, void * scratch)
{
        XY       * X = (XY *)((uintptr_t)scratch + 0);
        uint32x4 * V = (uint32x4 *)((uintptr_t)scratch + 128);
        int i, j;

        /* 1: X <-- B */
        for (i = 0; i < 16; i++) {
                X->w[i]      = databufA[i * 5 % 16];
                X->w[16 + i] = databufA[16 + (i * 5 % 16)];
        }

        /* 2: for i = 0 to N - 1 do */
        for (i = 0; i < 1024; i++) {
                blkcpy128(&V[i * 8], &X->q[0]);
                salsa20_8_xor(&X->q[0], &X->q[4]);
                salsa20_8_xor(&X->q[4], &X->q[0]);
        }

        /* 6: for i = 0 to N - 1 do */
        for (i = 0; i < 1024; i++) {
                j = X->w[16] & 1023; /* j <-- Integerify(X) mod N */
                blkxor128(X->q, &V[j * 8]);
                salsa20_8_xor(&X->q[0], &X->q[4]);
                salsa20_8_xor(&X->q[4], &X->q[0]);
        }

        /* 10: B' <-- X */
        for (i = 0; i < 16; i++) {
                databufA[i * 5 % 16] = X->w[i];
                databufA[16 + (i * 5 % 16)] = X->w[16 + i];
        }
}



#endif
#endif
