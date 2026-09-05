#include "residual.cuh"
#include <cuda_runtime.h>

// No raw K or V pointer is accepted. Every successful call is a local cut
// certificate for the named E24 reference, conditional on trusted summaries.
namespace statecut {
__global__ void node_residuals(const double* q,const double* kl,const double* ku,
    const uint32_t* count,const double* vmin,const double* vmax,const Interval* sum,
    const uint16_t* candidates,int heads,int nodes,int d,int dv,
    Interval* den,Interval* lower,Interval* upper) {
  const int sid=blockIdx.x, head=sid/nodes, j=threadIdx.x;
  if (head>=heads) return;
  __shared__ Interval score[128];
  __shared__ Interval weights;
  score[j]=j<d ? mul(point(q[head*d+j]),{kl[sid*d+j],ku[sid*d+j]}) : point(0.0);
  __syncthreads();
  for (int stride=64;stride;stride>>=1) {
    if (j<stride) score[j]=add(score[j],score[j+stride]);
    __syncthreads();
  }
  if (j==0) {
    weights=e24_weights(score[0]);
    den[sid]=count[sid] ? mul(point(double(count[sid])),weights) : invalid();
  }
  __syncthreads();
  if (j<dv) {
    const int offset=sid*dv+j;
    Cell cell=exact_bf16_cell(candidates[head*dv+j]);
    Moment m{count[sid],vmin[offset],vmax[offset],sum[offset]};
    lower[offset]=cell.ok ? residual_moment(m,weights,cell.lo) : invalid();
    upper[offset]=cell.ok ? residual_moment(m,weights,cell.hi) : invalid();
  }
}

__global__ void frontier_gate(const Interval* den,const Interval* lower,const Interval* upper,
    const uint16_t* candidates,int nodes,int dv,int* accepted) {
  const int head=blockIdx.x,j=threadIdx.x;
  __shared__ int ok[128];
  __shared__ int positive;
  if (j==0) {
    Interval d=point(0.0);
    for (int b=0;b<nodes;++b) d=add(d,den[head*nodes+b]);
    positive=valid(d)&&d.lo>0;
  }
  __syncthreads();
  int good=1;
  if (j<dv) {
    Interval lo=point(0),hi=point(0);
    for (int b=0;b<nodes;++b) {
      const int index=(head*nodes+b)*dv+j;
      lo=add(lo,lower[index]);hi=add(hi,upper[index]);
    }
    good=positive && residual_cell_ok(lo,hi,exact_bf16_cell(candidates[head*dv+j]));
  }
  ok[j]=good;
  __syncthreads();
  for (int stride=64;stride;stride>>=1) {
    if (j<stride) ok[j]&=ok[j+stride];
    __syncthreads();
  }
  if (j==0) accepted[head]=ok[0];
}
} // namespace statecut

extern "C" int statecut_residual_launch(const double* q,const double* kl,const double* ku,
    const uint32_t* count,const double* vmin,const double* vmax,const statecut::Interval* sum,
    const uint16_t* candidates,int heads,int nodes,int d,int dv,
    statecut::Interval* den,statecut::Interval* lower,statecut::Interval* upper,
    int* accepted,cudaStream_t stream) {
  if (!q||!kl||!ku||!count||!vmin||!vmax||!sum||!candidates||!den||!lower||!upper||!accepted||
      heads<1||nodes<1||d<1||d>128||dv<1||dv>128) return int(cudaErrorInvalidValue);
  // This reference ABI uses signed-int indexing for flat array offsets.
  if ((int64_t)heads*nodes*128>INT32_MAX) return int(cudaErrorInvalidValue);
  statecut::node_residuals<<<heads*nodes,128,0,stream>>>(q,kl,ku,count,vmin,vmax,sum,candidates,
      heads,nodes,d,dv,den,lower,upper);
  cudaError_t status=cudaGetLastError();
  if (status!=cudaSuccess) return int(status);
  statecut::frontier_gate<<<heads,128,0,stream>>>(den,lower,upper,candidates,nodes,dv,accepted);
  return int(cudaGetLastError());
}
