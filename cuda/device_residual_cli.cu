#include "residual.cuh"
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Test-only batched transport. All checked arithmetic executes on the device.
struct Request {
  int op;
  statecut::Moment moment;
  statecut::Interval weights;
  double threshold;
  uint16_t bits;
};
struct Answer { double lo, hi; int closed, ok; };

__global__ void probe(const Request* requests, Answer* answers, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const Request r = requests[i];
  if (r.op == 1) {
    auto c = statecut::exact_bf16_cell(r.bits);
    answers[i] = {c.lo, c.hi, int(c.closed), int(c.ok)};
  } else {
    auto v = r.op == 0 ? statecut::residual_moment(r.moment, r.weights, r.threshold)
                       : statecut::e24_weights(r.weights);
    answers[i] = {v.lo, v.hi, 0, int(statecut::valid(v))};
  }
}

static void checked(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    std::exit(1);
  }
}
int main() {
  std::vector<Request> requests;
  std::string op;
  while (std::cin >> op) {
    Request r{};
    if (op == "moment") {
      r.op = 0;
      if (!(std::cin >> r.moment.n >> r.moment.sum.lo >> r.moment.sum.hi
            >> r.moment.vmin >> r.moment.vmax >> r.weights.lo >> r.weights.hi
            >> r.threshold)) return 2;
    } else if (op == "cell") {
      r.op = 1;
      unsigned bits;
      if (!(std::cin >> bits) || bits > 65535) return 2;
      r.bits = uint16_t(bits);
    } else if (op == "weights") {
      r.op = 2;
      if (!(std::cin >> r.weights.lo >> r.weights.hi)) return 2;
    } else return 2;
    requests.push_back(r);
  }
  if (requests.empty()) return 2;
  std::vector<Answer> answers(requests.size());
  Request* device_requests = nullptr;
  Answer* device_answers = nullptr;
  checked(cudaMalloc(&device_requests, requests.size() * sizeof(Request)));
  checked(cudaMalloc(&device_answers, answers.size() * sizeof(Answer)));
  checked(cudaMemcpy(device_requests, requests.data(), requests.size() * sizeof(Request), cudaMemcpyHostToDevice));
  probe<<<(requests.size() + 127) / 128, 128>>>(device_requests, device_answers, requests.size());
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
  checked(cudaMemcpy(answers.data(), device_answers, answers.size() * sizeof(Answer), cudaMemcpyDeviceToHost));
  checked(cudaFree(device_requests));
  checked(cudaFree(device_answers));
  std::cout << std::setprecision(17);
  for (size_t i = 0; i < answers.size(); ++i) {
    const auto a = answers[i];
    std::cout << a.lo << ' ' << a.hi;
    if (requests[i].op == 1) std::cout << ' ' << a.closed << ' ' << a.ok;
    std::cout << '\n';
  }
}
