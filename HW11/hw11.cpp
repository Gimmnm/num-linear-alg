#include "../Com/Matrix.h"
#include "../Com/common.h"

void task_Hilbert(int n) {
    Matrix<double> A = Matrix<double>::HilbertMatrix(n);
    std::vector<double> xx(n);
    for (size_t i = 0; i < n; ++i) {
        xx[i] = 1.0 / 3.0;
    }
    std::vector<double> x0(n);
    for (size_t i = 0; i < n; ++i) {
        x0[i] = 0;
    }
    std::vector<double> b = A.MulWithVector(xx);
    std::vector<double> x = A.ConjugateGradientMethod(b, x0);
    VectorPrint(x);
}

void task_example() {
    Matrix<double> A = {{10, 1, 2, 3, 4}, {1, 9, -1, 2, -3}, {2, -1, 7, 3, -5}, {3, 2, 3, 12, -1}, {4, -3, -5, -1, 15}};
    std::vector<double> x0(5);
    for (size_t i = 0; i < 5; ++i) {
        x0[i] = 0;
    }
    std::vector<double> b = {12, -27, 14, -17, 12};
    std::vector<double> x_co = A.ConjugateGradientMethod(b, x0);
    VectorPrint(x_co);
    std::vector<double> x_ja = A.JacobiIterativeMethod(b);
    VectorPrint(x_co);
    std::vector<double> x_gs = A.GaussSeidelIterativeMethod(b);
    VectorPrint(x_gs);
}

int main() {

    for (size_t i = 0; i < 100; ++i) {
        task_Hilbert(i);
    }

    task_example();

    return 0;
}