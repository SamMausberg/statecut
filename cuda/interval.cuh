#pragma once
#include <cmath>
#include <cstdint>
#include <limits>

// CUDA uses explicit directed-rounding intrinsics. The host emulation widens
// correctly-rounded binary64 operations by one ULP; it is a test path only.
// Compile host code without fast math or contraction. See docs/VERIFICATION.md.
#ifdef __CUDACC__
#define SC_HD __host__ __device__
#else
#define SC_HD
#endif

namespace statecut {
struct Interval { double lo, hi; };
SC_HD inline double down_add(double x,double y) {
#ifdef __CUDA_ARCH__
  return __dadd_rd(x,y);
#else
  return std::nextafter(x+y,-std::numeric_limits<double>::infinity());
#endif
}
SC_HD inline double up_add(double x,double y) {
#ifdef __CUDA_ARCH__
  return __dadd_ru(x,y);
#else
  return std::nextafter(x+y,std::numeric_limits<double>::infinity());
#endif
}
SC_HD inline double down_mul(double x,double y) {
#ifdef __CUDA_ARCH__
  return __dmul_rd(x,y);
#else
  return std::nextafter(x*y,-std::numeric_limits<double>::infinity());
#endif
}
SC_HD inline double up_mul(double x,double y) {
#ifdef __CUDA_ARCH__
  return __dmul_ru(x,y);
#else
  return std::nextafter(x*y,std::numeric_limits<double>::infinity());
#endif
}
SC_HD inline double down_div(double x,double y) {
#ifdef __CUDA_ARCH__
  return __ddiv_rd(x,y);
#else
  return std::nextafter(x/y,-std::numeric_limits<double>::infinity());
#endif
}
SC_HD inline double up_div(double x,double y) {
#ifdef __CUDA_ARCH__
  return __ddiv_ru(x,y);
#else
  return std::nextafter(x/y,std::numeric_limits<double>::infinity());
#endif
}
SC_HD inline bool finite(double x) {
#ifdef __CUDA_ARCH__
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}
SC_HD inline bool valid(Interval x) { return finite(x.lo)&&finite(x.hi)&&x.lo<=x.hi; }
SC_HD inline Interval invalid() { return {NAN,NAN}; }
SC_HD inline double mn(double a,double b) { return a<b?a:b; }
SC_HD inline double mx(double a,double b) { return a>b?a:b; }
SC_HD inline Interval point(double x) { return {x,x}; }
SC_HD inline Interval add(Interval a,Interval b) {
  if(!valid(a)||!valid(b)) return invalid();
  return {down_add(a.lo,b.lo),up_add(a.hi,b.hi)};
}
SC_HD inline Interval neg(Interval x) { return {-x.hi,-x.lo}; }
SC_HD inline Interval sub(Interval a,Interval b) { return add(a,neg(b)); }
SC_HD inline Interval mul(Interval a,Interval b) {
  if(!valid(a)||!valid(b)) return invalid();
  double l=mn(mn(down_mul(a.lo,b.lo),down_mul(a.lo,b.hi)),mn(down_mul(a.hi,b.lo),down_mul(a.hi,b.hi)));
  double u=mx(mx(up_mul(a.lo,b.lo),up_mul(a.lo,b.hi)),mx(up_mul(a.hi,b.lo),up_mul(a.hi,b.hi)));
  return {l,u};
}
SC_HD inline Interval divide_positive(Interval a,Interval b) {
  if(!valid(a)||!valid(b)||!(b.lo>0)) return invalid();
  return mul(a,{down_div(1.0,b.hi),up_div(1.0,b.lo)});
}
SC_HD inline Interval exp_real(double x) {
  // TRUE real exp enclosure, not libdevice exp and not the Python E_24.
  // Positive Taylor series at |x|/128 <= 1/2, followed by seven squarings.
  // Unsupported domains return invalid; callers must fall back.
  if(!finite(x)||x < -64.0 || x > 64.0) return invalid();
  if(x==0.0) return point(1.0);
  const double ax=x<0?-x:x;
  const Interval t={down_mul(ax,0.0078125),up_mul(ax,0.0078125)};
  Interval term=point(1.0),sum=point(1.0);
  constexpr int m=20;
  for(int k=1;k<=m;++k) {
    term=divide_positive(mul(term,t),point(double(k)));
    sum=add(sum,term);
  }
  const Interval next=divide_positive(mul(term,t),point(double(m+1)));
  const Interval denom=sub(point(1.0),divide_positive(t,point(double(m+2))));
  const Interval tail=divide_positive(next,denom);
  if(!valid(tail)) return invalid();
  Interval result={sum.lo,up_add(sum.hi,tail.hi)};
  for(int k=0;k<7;++k) result=mul(result,result);
  if(x<0) result=divide_positive(point(1.0),result);
  return valid(result)?result:invalid();
}
SC_HD inline Interval signed_numerator(double a,double b,Interval p,Interval m) {
  if(!finite(a)||!finite(b)||a<0||a>b||!valid(p)||!valid(m)||p.lo<0||m.hi>0)
    return invalid();
  return {down_add(down_mul(a,p.lo),down_mul(b,m.lo)),
          up_add(up_mul(b,p.hi),up_mul(a,m.hi))};
}
} // namespace statecut
