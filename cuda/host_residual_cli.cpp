#include "residual.cuh"
#include <iostream>
#include <iomanip>
#include <string>
int main() {
  using namespace statecut;
  std::cout << std::setprecision(17);
  std::string op;
  while (std::cin >> op) {
    if (op=="moment") {
      Moment s{}; Interval w{}; double t;
      if (!(std::cin>>s.n>>s.sum.lo>>s.sum.hi>>s.vmin>>s.vmax>>w.lo>>w.hi>>t)) return 2;
      Interval r=residual_moment(s,w,t);
      std::cout<<r.lo<<" "<<r.hi<<"\n";
    } else if (op=="cell") {
      unsigned code;
      if (!(std::cin>>code) || code>65535) return 2;
      Cell c=exact_bf16_cell(uint16_t(code));
      std::cout<<c.lo<<" "<<c.hi<<" "<<c.closed<<" "<<c.ok<<"\n";
    } else if (op=="weights") {
      Interval s;
      if (!(std::cin>>s.lo>>s.hi)) return 2;
      Interval r=e24_weights(s);
      std::cout<<r.lo<<" "<<r.hi<<"\n";
    } else return 2;
  }
  return 0;
}
