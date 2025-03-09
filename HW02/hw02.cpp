#include "../Com/Matrix.h"
#include "../Com/common.h"

template <typename T>
void SetVal(Matrix<T> &m, std::vector<T> &b, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        if (i != 0) m(i, i - 1) = (long double)8;
        m(i, i) = (long double)6;
        if (i != N - 1) m(i, i + 1) = (long double)1;
    }
    for (size_t i = 0; i < N; ++i) {
        b[i] = (long double)15;
        if (i == 0) b[i] = (long double)7;
        if (i == N - 1) b[i] = (long double)14;
    }
}

template <typename T>
void SetValTest(Matrix<T> &m, std::vector<T> &b) {
    m = { {1, 4, 7}, {2, 5, 8}, {3, 6, 10} };
    b = {1, 1, 1};
}

void LUsolve() {

    // Set the matrix
    // size_t N = 3;
    size_t N = 84;
    Matrix<long double> m(N, N);
    std::vector<long double> b(N);
    SetVal(m, b, N);
    // SetValTest(m, b);

    // Solve mx=b N*N
    Matrix<long double> LU = m.LUdecomposition();
    Matrix<long double> L(LU), U(LU);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < i; ++j)
            U(i, j) = (long double)(0);
        for (size_t j = i + 1; j < N; ++j)
            L(i, j) = (long double)(0);
        L(i, i) = (long double)(1);
    }
    std::vector<long double> y = L.SolveLowerTriangular(b);
    std::vector<long double> x = U.SolveUpperTriangular(y);

    // Print the solution
    std::cout << "Solution by A=LU\n";
    for (auto val : x) {
        std::cout << val << std::endl;
    }
    std::cout << std::endl;
}

void PALUsolve() {

    // Set the matrix
    // size_t N = 3;
    size_t N = 84;
    Matrix<long double> m(N, N);
    std::vector<long double> b(N);
    SetVal(m, b, N);
    // SetValTest(m, b);

    // Solve mx=b N*N
    
    Matrix<long double> LU = m.PALUdecomposition(b).first;
    Matrix<long double> L(LU), U(LU);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < i; ++j)
            U(i, j) = (long double)0;
        for (size_t j = i + 1; j < N; ++j)
            L(i, j) = (long double)0;
        L(i, i) = (long double)1;
    } 
    std::vector<long double> y = L.SolveLowerTriangular(b);
    std::vector<long double> x = U.SolveUpperTriangular(y);

    // Print the solution
    std::cout << "Solution by PA=LU\n";
    for (auto val : x) {
        std::cout << val << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(8);
    LUsolve();
    PALUsolve();
    return 0;
}