#pragma once
#include "interval.cuh"

// Reference profile: bounds on RATIONAL_BF16_V1, not a PyTorch/FA bridge.
// Host code tests arithmetic only. Device instruction correctness still relies
// on the documented CUDA directed-rounding semantics and compiler contract.
namespace statecut {
struct Moment {
  uint32_t n;
  double vmin, vmax;
  Interval sum;
};

SC_HD inline Interval abs_interval(Interval x) {
  if (!valid(x)) return invalid();
  if (x.lo >= 0) return x;
  if (x.hi <= 0) return neg(x);
  return {0.0, mx(-x.lo, x.hi)};
}

SC_HD inline Interval residual_moment(Moment s, Interval w, double threshold) {
  if (!s.n || !finite(s.vmin) || !finite(s.vmax) || s.vmin > s.vmax ||
      !valid(s.sum) || !valid(w) || w.lo < 0 || !finite(threshold)) return invalid();
  const Interval n = point(double(s.n)); // uint32 exactly representable in FP64.
  const Interval low_sum=mul(n,point(s.vmin)), high_sum=mul(n,point(s.vmax));
  // This only detects obvious inconsistencies, not forged summary provenance.
  // An ingestion algorithm must establish sum/range correctness from real rows.
  if (s.sum.hi < low_sum.lo || s.sum.lo > high_sum.hi) return invalid();
  if (s.vmin == s.vmax) {
    return mul(mul(n,w),sub(point(s.vmin),point(threshold)));
  }
  Interval denominator=sub(point(s.vmax),point(s.vmin));
  if (!valid(denominator) || denominator.lo <= 0) return invalid();
  Interval left=sub(high_sum,s.sum), right=sub(s.sum,low_sum);
  Interval edge_lo=abs_interval(sub(point(s.vmin),point(threshold)));
  Interval edge_hi=abs_interval(sub(point(s.vmax),point(threshold)));
  Interval chord=divide_positive(add(mul(left,edge_lo),mul(right,edge_hi)),denominator);
  if (!valid(chord)) return invalid();
  const Interval m=mul(add(point(w.lo),point(w.hi)),point(0.5));
  const Interval h=mul(sub(point(w.hi),point(w.lo)),point(0.5));
  Interval center=mul(m,sub(s.sum,mul(n,point(threshold))));
  const double radius=up_mul(mx(0.0,h.hi),mx(0.0,chord.hi));
  if (!valid(center) || !finite(radius)) return invalid();
  return add(center,{-radius,radius});
}

SC_HD inline double rne_e24_grid(double x) {
  // Exact power-of-two scaling, floor, and exact integer/parity operations.
  // When scaled >= 2^53 it is already an even integer in binary64. Below
  // that range every chosen integer and the rescaling are representable.
  if (!finite(x) || x < 0.0) return NAN;
  const double scaled=x*0x1p24;
  if (!finite(scaled)) return NAN;
  const double lower=floor(scaled), fraction=scaled-lower;
  const bool odd=lower-2.0*floor(lower*0.5)!=0.0;
  return (lower+(fraction>0.5 || (fraction==0.5 && odd) ? 1.0 : 0.0))*0x1p-24;
}

SC_HD inline Interval e24_weights(Interval score) {
  // E24(z) = 2^-24 RNE(2^24 exp(z)). Half an E24 quantum is a
  // global ABSOLUTE quantization bound. No common score shift is applied.
  if (!valid(score)) return invalid();
  if (score.lo == 0.0 && score.hi == 0.0) return point(1.0);
  // Since e > 2, x <= -25 implies exp(x) < 2^-25 and E24(x)=0.
  // This also handles arbitrarily negative finite scores without a shift.
  Interval a=score.lo<=-25.0 ? point(0.0) : exp_real(score.lo);
  Interval b=score.hi<=-25.0 ? point(0.0) : exp_real(score.hi);
  if (!valid(a)||!valid(b)) return invalid();
  // Monotonicity permits rounding both enclosure endpoints to the actual
  // target lattice. This is tighter than adding a global half-quantum.
  return {rne_e24_grid(a.lo),rne_e24_grid(b.hi)};
}

struct Cell { double lo, hi; bool closed; bool ok; };
SC_HD inline double bf16_exact(uint16_t bits) {
  const unsigned magnitude=bits&0x7fff;
  const unsigned exponent=(magnitude>>7)&255, mantissa=magnitude&127;
  if (exponent==255) return NAN;
  double value=exponent==0 ? ldexp(double(mantissa),-133)
                           : ldexp(double(128+mantissa),int(exponent)-134);
  return bits&0x8000 ? -value : value;
}
SC_HD inline Cell exact_bf16_cell(uint16_t bits) {
  const unsigned c=bits&0x7fff;
  if (c>=0x7f80 || bits==0x8000) return {0,0,false,false};
  if (!c) return {-0x1p-134,0x1p-134,true,true};
  const double v=bf16_exact(uint16_t(c)), p=bf16_exact(uint16_t(c-1));
  const double next=c==0x7f7f ? v+(v-p) : bf16_exact(uint16_t(c+1));
  // All BF16 adjacent midpoints are exactly representable in FP64.
  const double lo=(p+v)*0.5, hi=(v+next)*0.5;
  return bits&0x8000 ? Cell{-hi,-lo,(c&1)==0,true}
                     : Cell{lo,hi,(c&1)==0,true};
}
SC_HD inline bool residual_cell_ok(Interval lo, Interval hi, Cell cell) {
  if (!cell.ok || !valid(lo) || !valid(hi)) return false;
  return cell.closed ? lo.lo>=0 && hi.hi<=0 : lo.lo>0 && hi.hi<0;
}
} // namespace statecut
