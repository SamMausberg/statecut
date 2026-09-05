#include "interval.cuh"
#include <iomanip>
#include <iostream>
#include <string>

// Line protocol for independent Fraction/mpmath differential checks.
int main() {
  std::string op;
  double a,b,c,d;
  std::cout << std::setprecision(17);
  while(std::cin>>op) {
    statecut::Interval out;
    if(op=="exp") {
      if(!(std::cin>>a)) return 2;
      out=statecut::exp_real(a);
    } else {
      if(!(std::cin>>a>>b>>c>>d)) return 2;
      if(op=="add") out=statecut::add({a,b},{c,d});
      else if(op=="mul") out=statecut::mul({a,b},{c,d});
      else if(op=="div") out=statecut::divide_positive({a,b},{c,d});
      else return 2;
    }
    std::cout<<out.lo<<' '<<out.hi<<'\n';
  }
  return std::cin.eof()?0:2;
}
