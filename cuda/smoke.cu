#include "summary_attention.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { auto err=(call); if(err!=cudaSuccess) { \
 std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); return 1; } } while(0)

template<class T> cudaError_t allocate(T** p,size_t n) {
  return cudaMalloc(reinterpret_cast<void**>(p),n*sizeof(T));
}
int main() {
  constexpr int d=128,blocks=32,rows_per_block=128;
  double *q=nullptr,*kl=nullptr,*ku=nullptr;
  statecut::Interval *p=nullptr,*m=nullptr,*den=nullptr,*num=nullptr,*out=nullptr;
  uint32_t* count=nullptr;
  CHECK(allocate(&q,d));CHECK(allocate(&kl,blocks*d));CHECK(allocate(&ku,blocks*d));
  CHECK(allocate(&p,blocks*d));CHECK(allocate(&m,blocks*d));CHECK(allocate(&count,blocks));
  CHECK(allocate(&den,blocks));CHECK(allocate(&num,blocks*d));CHECK(allocate(&out,d));
  std::vector<double> zeros(blocks*d,0.0);
  std::vector<statecut::Interval> pos(blocks*d,{64.0,64.0}),neg(blocks*d,{0.0,0.0}),result(d);
  std::vector<uint32_t> counts(blocks,rows_per_block);
  CHECK(cudaMemcpy(q,zeros.data(),d*sizeof(double),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(kl,zeros.data(),zeros.size()*sizeof(double),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(ku,zeros.data(),zeros.size()*sizeof(double),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(p,pos.data(),pos.size()*sizeof(pos[0]),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(m,neg.data(),neg.size()*sizeof(neg[0]),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(count,counts.data(),counts.size()*sizeof(counts[0]),cudaMemcpyHostToDevice));
  CHECK(statecut::launch_summary_attention(q,kl,ku,p,m,count,d,blocks,den,num,out,0));
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(result.data(),out,d*sizeof(result[0]),cudaMemcpyDeviceToHost));
  for(auto x:result) {
    if(!statecut::valid(x)||x.lo>0.5||x.hi<0.5) {
      std::fprintf(stderr,"incorrect enclosure [%g,%g]\n",x.lo,x.hi);return 2;
    }
  }
  // Also exercise candidate rounding-cell checks on device.
  double* bridge=nullptr;uint16_t* codes=nullptr;uint32_t* flags=nullptr;
  CHECK(allocate(&bridge,d));CHECK(allocate(&codes,d));CHECK(allocate(&flags,d));
  CHECK(cudaMemset(bridge,0,d*sizeof(double)));
  statecut::check_cells<<<1,128>>>(out,bridge,d,codes,flags);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> hf(d);std::vector<uint16_t> hc(d);
  CHECK(cudaMemcpy(hf.data(),flags,d*sizeof(uint32_t),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hc.data(),codes,d*sizeof(uint16_t),cudaMemcpyDeviceToHost));
  for(int j=0;j<d;++j) if(hf[j]!=1||hc[j]!=0x3f00) return 3;
  CHECK(cudaFree(q));CHECK(cudaFree(kl));CHECK(cudaFree(ku));CHECK(cudaFree(p));CHECK(cudaFree(m));
  CHECK(cudaFree(count));CHECK(cudaFree(den));CHECK(cudaFree(num));CHECK(cudaFree(out));
  CHECK(cudaFree(bridge));CHECK(cudaFree(codes));CHECK(cudaFree(flags));
  std::puts("PASS: constant-key mathematical reference; no raw KV pointer used. Not a pretrained benchmark.");
  return 0;
}
