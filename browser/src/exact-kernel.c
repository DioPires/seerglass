// SPDX-License-Identifier: Apache-2.0
// The bank search of EmbeddingRecognizer.topAndOther (recognizer.ts), for a batch of looks, as a
// WebAssembly SIMD kernel. scripts/build-kernel.mjs compiles it into src/exact-kernel-wasm.ts.
//
// Exact by construction: every (row, look) similarity is the float64 sum, in ascending k, of the
// products of the row's and the look's float32 values, which is the loop's own arithmetic. A
// float32 widened to float64 is exact, the product of two of them is exact in float64, and
// f64x2.add rounds as a float64 addition does. The SIMD lanes hold different looks, never
// different k, so no sum is reordered. No relaxed SIMD and no contraction (-ffp-contract=off):
// WebAssembly SIMD128 has no fused multiply-add.
//
// best, bestRow and other are kept online with the loop's rules, per look: the first strict
// maximum in row order, starting from NO_OTHER_IDENTITY (-2) with no identity (-1), and the best
// similarity among rows of a different identity. When a row beats the best and is of another
// identity, the old best is the largest similarity seen so far and of another identity, so it is
// the new `other`. When it beats the best within the same identity, `other` stands.

#include <stdint.h>
#include <wasm_simd128.h>

#define D 128 /* values per row; the recognizer checks the bank's dim before it uses the kernel */
#define G 8   /* looks per group: four f64x2 lanes pairs */

static inline void keep(int32_t row, int32_t id, double x, double *best, double *other, int32_t *bestRow,
                        int32_t *bestIdent) {
  if (x > *best) {
    if (id != *bestIdent) *other = *best;
    *best = x;
    *bestRow = row;
    *bestIdent = id;
  } else if (id != *bestIdent && x > *other) {
    *other = x;
  }
}

// bank: n rows of D float32, row-major. rowIdent: n identity indices (0 or more). q: `groups`
// groups of G looks, as float64, q[g * D * G + k * G + j] = value k of look g * G + j.
// best, other, bestRow, bestIdent: groups * G entries each, set by the caller to -2, -2, -1, -1.
__attribute__((export_name("score"))) void score(const float *bank, int32_t n, const int32_t *rowIdent,
                                                 const double *q, int32_t groups, double *best, double *other,
                                                 int32_t *bestRow, int32_t *bestIdent) {
  int32_t r = 0;
  // Two rows at a time: more independent chains of additions, the same sums.
  for (; r + 1 < n; r += 2) {
    const float *b = bank + (long)r * D;
    const float *c = b + D;
    const int32_t idb = rowIdent[r], idc = rowIdent[r + 1];
    for (int32_t g = 0; g < groups; g++) {
      const double *qg = q + (long)g * D * G;
      v128_t a0 = wasm_f64x2_const(0.0, 0.0), a1 = a0, a2 = a0, a3 = a0;
      v128_t c0 = a0, c1 = a0, c2 = a0, c3 = a0;
      for (int32_t k = 0; k < D; k++) {
        const v128_t bk = wasm_f64x2_splat((double)b[k]);
        const v128_t ck = wasm_f64x2_splat((double)c[k]);
        const double *qk = qg + k * G;
        const v128_t q0 = wasm_v128_load(qk), q1 = wasm_v128_load(qk + 2);
        const v128_t q2 = wasm_v128_load(qk + 4), q3 = wasm_v128_load(qk + 6);
        a0 = wasm_f64x2_add(a0, wasm_f64x2_mul(bk, q0));
        a1 = wasm_f64x2_add(a1, wasm_f64x2_mul(bk, q1));
        a2 = wasm_f64x2_add(a2, wasm_f64x2_mul(bk, q2));
        a3 = wasm_f64x2_add(a3, wasm_f64x2_mul(bk, q3));
        c0 = wasm_f64x2_add(c0, wasm_f64x2_mul(ck, q0));
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(ck, q1));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(ck, q2));
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(ck, q3));
      }
      double d[2 * G];
      wasm_v128_store(d, a0);
      wasm_v128_store(d + 2, a1);
      wasm_v128_store(d + 4, a2);
      wasm_v128_store(d + 6, a3);
      wasm_v128_store(d + 8, c0);
      wasm_v128_store(d + 10, c1);
      wasm_v128_store(d + 12, c2);
      wasm_v128_store(d + 14, c3);
      // Row r before row r + 1 for every look, as the loop visits them.
      for (int32_t j = 0; j < G; j++) {
        const int32_t l = g * G + j;
        keep(r, idb, d[j], best + l, other + l, bestRow + l, bestIdent + l);
        keep(r + 1, idc, d[G + j], best + l, other + l, bestRow + l, bestIdent + l);
      }
    }
  }
  // The last row of an odd count, one look at a time.
  for (; r < n; r++) {
    const float *b = bank + (long)r * D;
    const int32_t id = rowIdent[r];
    for (int32_t g = 0; g < groups; g++) {
      const double *qg = q + (long)g * D * G;
      for (int32_t j = 0; j < G; j++) {
        double x = 0.0;
        for (int32_t k = 0; k < D; k++) x += (double)b[k] * qg[k * G + j];
        const int32_t l = g * G + j;
        keep(r, id, x, best + l, other + l, bestRow + l, bestIdent + l);
      }
    }
  }
}
