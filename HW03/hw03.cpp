#include "../Com/Matrix.h"
#include "../Com/common.h"

template <typename T = long double>
std::pair<Matrix<T>, std::vector<T>> SetTriVal(size_t N) {
    Matrix<T> m = Matrix<T>::TridiagonalMatrix(N, 1.0, 10.0, 1.0);
    std::vector<T> b(N);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<T> dist(1.0L, 150.0L);
    // for (size_t i = 1; i < N - 1; ++i)
    //     b[i] = 12.0;
    // b[0] = 11.0;
    // b[N - 1] = 11.0;
    for (size_t i = 0; i < N; ++i)
        b[i] = dist(gen);
    return std::make_pair(m, b);
}

template <typename T = long double>
std::pair<Matrix<T>, std::vector<T>> SetHilbVal(size_t N) {
    Matrix<T> m = Matrix<T>::HilbertMatrix(N);
    std::vector<T> b(N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            b[i] += 1.0L / (i + j + 1.0L);
    return std::make_pair(m, b);
}

template <typename T = long double>
void PrintDiff(Matrix<T> &m, std::vector<T> &x, std::vector<T> &b) {
    std::vector<T> bb = m.MulWithVector(x);
    size_t n = m.get_col();
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::fabs(b[i] - bb[i]) << std::endl;
    }
    std::cout << std::endl;
}

void LLTsolveTri() {

    size_t N = 100;
    auto mb = SetTriVal(N);
    Matrix<long double> m = mb.first;
    std::vector<long double> b = mb.second;

    Matrix<long double> L = m.CholeskyDecomposition();
    std::vector<long double> y = L.SolveLowerTriangular(b);
    Matrix<long double> LT = L.transpose();
    std::vector<long double> x = LT.SolveUpperTriangular(y);

    std::cout << "Solution by LL^T (Cholesky) on Tridiagonal Matrix:" << std::endl;
    for (auto val : x)
        std::cout << val << std::endl;
    std::cout << std::endl;
    PrintDiff(m, x, b);
}

void LDLTsolveTri() {

    size_t N = 100;
    auto mb = SetTriVal(N);
    Matrix<long double> m = mb.first;
    std::vector<long double> b = mb.second;

    auto ldlt = m.LDLTDecomposition();
    Matrix<long double> L = ldlt.first;
    std::vector<long double> D = ldlt.second;
    std::vector<long double> y = L.SolveLowerTriangular(b);
    std::vector<long double> z(N);
    for (size_t i = 0; i < N; ++i) {
        if (D[i] == 0)
            throw std::runtime_error("Zero pivot in LDLTsolveTri");
        z[i] = y[i] / D[i];
    }
    Matrix<long double> LT = L.transpose();
    std::vector<long double> x = LT.SolveUpperTriangular(z);

    std::cout << "Solution by LDL^T on Tridiagonal Matrix:" << std::endl;
    for (auto val : x)
        std::cout << val << std::endl;
    std::cout << std::endl;
    PrintDiff(m, x, b);
}

void LUsolveTri() {

    size_t N = 100;
    auto mb = SetTriVal(N);
    Matrix<long double> m = mb.first;
    std::vector<long double> b = mb.second;

    Matrix<long double> LU = m.LUdecomposition();
    Matrix<long double> L(LU), U(LU);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (j > i)
                L(i, j) = 0;
            else if (j < i)
                U(i, j) = 0;
        }
        L(i, i) = 1;
    }
    std::vector<long double> y = L.SolveLowerTriangular(b);
    std::vector<long double> x = U.SolveUpperTriangular(y);

    std::cout << "Solution by LU (no pivoting) on Tridiagonal Matrix:" << std::endl;
    for (auto val : x)
        std::cout << val << std::endl;
    std::cout << std::endl;
    PrintDiff(m, x, b);
}

void PALUsolveTri() {

    size_t N = 100;
    auto mb = SetTriVal(N);
    Matrix<long double> m = mb.first;
    std::vector<long double> b = mb.second;
    std::vector<long double> ob = b;

    std::vector<int> p;
    Matrix<long double> LU = m.PALUdecomposition(b).first;
    Matrix<long double> L(LU), U(LU);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (j > i)
                L(i, j) = 0;
            else if (j < i)
                U(i, j) = 0;
        }
        L(i, i) = 1;
    }
    std::vector<long double> y = L.SolveLowerTriangular(b);
    std::vector<long double> x = U.SolveUpperTriangular(y);

    std::cout << "Solution by PA=LU on Tridiagonal Matrix:" << std::endl;
    for (auto val : x)
        std::cout << val << std::endl;
    std::cout << std::endl;
    PrintDiff(m, x, ob);
}

void LLTsolveHilb() {

    size_t N = 13;
    auto mb = SetHilbVal(N);
    Matrix<long double> m = mb.first;
    std::vector<long double> b = mb.second;

    Matrix<long double> L = m.CholeskyDecomposition();
    std::vector<long double> y = L.SolveLowerTriangular(b);
    Matrix<long double> LT = L.transpose();
    std::vector<long double> x = LT.SolveUpperTriangular(y);

    std::cout << "Solution by LL^T (Cholesky) on Hilbert Matrix:" << std::endl;
    for (auto val : x)
        std::cout << val << std::endl;
    std::cout << std::endl;
}

void LDLTsolveHilb() {

    size_t N = 13;
    auto mb = SetHilbVal(N);
    Matrix<long double> m = mb.first;
    std::vector<long double> b = mb.second;

    auto ldlt = m.LDLTDecomposition();
    Matrix<long double> L = ldlt.first;
    std::vector<long double> D = ldlt.second;
    std::vector<long double> y = L.SolveLowerTriangular(b);
    std::vector<long double> z(N);
    for (size_t i = 0; i < N; ++i) {
        if (D[i] == 0)
            throw std::runtime_error("Zero pivot in LDLTsolveHilb");
        z[i] = y[i] / D[i];
    }
    Matrix<long double> LT = L.transpose();
    std::vector<long double> x = LT.SolveUpperTriangular(z);

    std::cout << "Solution by LDL^T on Hilbert Matrix:" << std::endl;
    for (auto val : x)
        std::cout << val << std::endl;
    std::cout << std::endl;
}

void LUsolveHilb() {

    size_t N = 40;
    auto mb = SetHilbVal(N);
    Matrix<long double> m = mb.first;
    std::vector<long double> b = mb.second;

    Matrix<long double> LU = m.LUdecomposition();
    Matrix<long double> L(LU), U(LU);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (j > i)
                L(i, j) = 0;
            else if (j < i)
                U(i, j) = 0;
        }
        L(i, i) = 1;
    }
    std::vector<long double> y = L.SolveLowerTriangular(b);
    std::vector<long double> x = U.SolveUpperTriangular(y);

    std::cout << "Solution by LU (no pivoting) on Hilbert Matrix:" << std::endl;
    for (auto val : x)
        std::cout << val << std::endl;
    std::cout << std::endl;
}

void PALUsolveHilb() {

    size_t N = 13;
    auto mb = SetHilbVal(N);
    Matrix<long double> m = mb.first;
    std::vector<long double> b = mb.second;

    std::vector<int> p;
    Matrix<long double> LU = m.PALUdecomposition(b).first;
    Matrix<long double> L(LU), U(LU);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (j > i)
                L(i, j) = 0;
            else if (j < i)
                U(i, j) = 0;
        }
        L(i, i) = 1;
    }
    std::vector<long double> y = L.SolveLowerTriangular(b);
    std::vector<long double> x = U.SolveUpperTriangular(y);

    std::cout << "Solution by PA=LU on Hilbert Matrix:" << std::endl;
    for (auto val : x)
        std::cout << val << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(8);
    LLTsolveTri();
    LDLTsolveTri();
    LUsolveTri();
    PALUsolveTri();
    LLTsolveHilb();
    LDLTsolveHilb();
    LUsolveHilb();
    PALUsolveHilb();
    return 0;
}