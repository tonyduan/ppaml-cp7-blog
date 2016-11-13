#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
  {
    mat A = randu<mat>(4,5);
      mat B = randu<mat>(4,5);
        
          cout << A << endl;
          cout << A(0,0) << endl;
          cout << A[1,1] << endl;
            
              return 0;
                }
