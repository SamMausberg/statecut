#include "interval.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Research CUDA path: mathematical real-softmax enclosures only.
// NOT a pretrained-model binding, NOT a proof of FlashAttention equivalence.
// No raw KV pointer is accepted by evaluate_summaries, so it cannot read raw KV.
// One query/head, d <= 128, summary arrays [blocks,d].
namespace statecut {

__global__ void evaluate_summaries(
    const double* __restrict__ q,
    const double* __restrict__ key_lo,const double* __restrict__ key_hi,
    const Interval* __restrict__ positive,const Interval* __restrict__ negative,
    const uint32_t* __restrict__ count,int d,int blocks,
    Interval* __restrict__ denominator,Interval* __restrict__ numerator) {
  const int b=blockIdx.x,j=threadIdx.x;
  if(b>=blocks) return;
  __shared__ Interval scores[128];
  __shared__ double a,upper;
  if(j<128) scores[j]=point(0.0);
  __syncthreads();
  if(j<d) {
    const Interval key={key_lo[b*d+j],key_hi[b*d+j]};
    scores[j]=mul(point(q[j]),key);
  }
  __syncthreads();
  for(int stride=64;stride;stride>>=1) {
    if(j<stride) scores[j]=add(scores[j],scores[j+stride]);
    __syncthreads();
  }
  if(j==0) {
    Interval l=exp_real(scores[0].lo),u=exp_real(scores[0].hi);
    if(!valid(l)||!valid(u)||count[b]==0) {
      a=upper=NAN;
      denominator[b]=invalid();
    } else {
      a=l.lo; upper=u.hi;
      denominator[b]={down_mul(double(count[b]),a),up_mul(double(count[b]),upper)};
    }
  }
  __syncthreads();
  if(j<d) numerator[b*d+j]=signed_numerator(a,upper,positive[b*d+j],negative[b*d+j]);
}

__global__ void reduce_summaries(const Interval* denominator,const Interval* numerator,
                               int blocks,int d,Interval* attention) {
  const int j=threadIdx.x;
  __shared__ Interval den;
  if(j==0) {
    den=point(0.0);
    for(int b=0;b<blocks;++b) den=add(den,denominator[b]);
  }
  __syncthreads();
  if(j<d) {
    Interval num=point(0.0);
    for(int b=0;b<blocks;++b) num=add(num,numerator[b*d+j]);
    attention[j]=divide_positive(num,den);
  }
}

__device__ inline double bf16_as_double(uint16_t code) {
  return double(__int_as_float(uint32_t(code)<<16));
}

// A fast conversion only PROPOSES a BF16 code. Exact binary64 midpoint
// comparisons verify it, so double rounding in the proposal is harmless.
__device__ inline bool strict_bf16_cell(Interval x,uint16_t* code) {
  if(!valid(x)) return false;
  const double midpoint=x.lo*0.5+x.hi*0.5;
  uint16_t c=__bfloat16_as_ushort(__float2bfloat16_rn(float(midpoint)));
  const uint16_t magnitude=c&0x7fff;
  if(magnitude>=0x7f7f) return false; // excludes infinities, NaNs, overflow cell
  double before,after,center;
  if(magnitude==0) {
    c=0;center=0;
    before=bf16_as_double(0x8001);after=bf16_as_double(1);
  } else {
    center=bf16_as_double(c);
    before=bf16_as_double((c&0x8000)?c+1:c-1);
    after=bf16_as_double((c&0x8000)?c-1:c+1);
  }
  // Adjacent BF16 midpoints are exactly representable in binary64.
  const double l=(before+center)*0.5,u=(center+after)*0.5;
  if(l<x.lo && x.hi<u) { *code=c; return true; }
  return false; // midpoint ties are conservatively rejected in this CUDA path
}

// error[j] must be a SOUND bridge to a named backend's pre-round output.
// A negative/NaN/missing bridge is invalid. Supplying zero only establishes
// the correctly-rounded mathematical real-softmax reference, NOT PyTorch.
__global__ void check_cells(const Interval* mathematical,const double* error,
                           int d,uint16_t* codes,uint32_t* coordinate_ok) {
  const int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(j>=d) return;
  coordinate_ok[j]=0;
  const double e=error[j];
  if(!finite(e)||e<0) return;
  Interval expanded=add(mathematical[j],{-e,e});
  uint16_t code=0;
  if(strict_bf16_cell(expanded,&code)) {
    codes[j]=code;
    coordinate_ok[j]=1;
  }
  // Host must require EVERY coordinate flag before committing a vector.
}

cudaError_t launch_summary_attention(
    const double* q,const double* kl,const double* ku,const Interval* p,const Interval* m,
    const uint32_t* n,int d,int blocks,Interval* den,Interval* num,Interval* out,
    cudaStream_t stream) {
  if(d<1||d>128||blocks<1||!q||!kl||!ku||!p||!m||!n||!den||!num||!out)
    return cudaErrorInvalidValue;
  evaluate_summaries<<<blocks,128,0,stream>>>(q,kl,ku,p,m,n,d,blocks,den,num);
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) return e;
  reduce_summaries<<<1,128,0,stream>>>(den,num,blocks,d,out);
  return cudaGetLastError();
}
} // namespace statecut
