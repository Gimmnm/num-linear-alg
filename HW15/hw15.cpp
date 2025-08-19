#include "../Com/Matrix.h"
#include "../Com/common.h"

void task(int n) {
    std::cout << "Case " << n << " : \n";
    Matrix<double> A(n, n);
    for (size_t i = 0; i < n; ++i) {
        if (i != 0) A(i, i - 1) = 1;
        if (i != n - 1) A(i, i + 1) = 1;
        A(i, i) = 4;
    }
    Matrix<double>::Jacobi_classic(A);
}

int main() {
    for (size_t i = 50; i <= 50; ++i) {
        task(i);
    }
    return 0;
}