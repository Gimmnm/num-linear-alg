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
template <typename T = double>
std::pair<Matrix<T>, std::vector<T>> SetTriVal(size_t N) {
    Matrix<T> m = Matrix<T>::TridiagonalMatrix(N, 1.0, 10.0, 1.0);
    std::vector<T> b(N);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<T> dist(1.0L, 150.0L);
    for (size_t i = 1; i < N - 1; ++i)
        b[i] = 12.0;
    b[0] = 11.0;
    b[N - 1] = 11.0;
    // for (size_t i = 0; i < N; ++i)
    // b[i] = dist(gen);
    return std::make_pair(m, b);
}

template <typename T = double>
std::pair<Matrix<T>, std::vector<T>> SetHilbVal(size_t N) {
    Matrix<T> m = Matrix<T>::HilbertMatrix(N);
    std::vector<T> b(N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            b[i] += 1.0L / (i + j + 1.0L);
    return std::make_pair(m, b);
}

void Soloneone() {
    size_t N = 50;
    std::cout << "Solving system for N = " << N << " using SetVal function.\n";
    Matrix<double> m(N, N);
    std::vector<double> b(N);
    SetVal(m, b, N);
    std::vector<double> sol = m.QRSolveLinearSystem(b);
    std::cout << "Solution for the system (first 10 values):\n";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "x[" << i << "] = " << sol[i] << "\n";
    }
    std::cout << std::endl;
}

void Solonetwo() {
    size_t N = 40;
    std::cout << "Solving system for N = " << N << " using SetHilbVal function.\n";
    auto mb = SetHilbVal(N);
    Matrix<double> m = mb.first;
    std::vector<double> b = mb.second;
    std::vector<double> sol = m.QRSolveLinearSystem(b);
    std::cout << "Solution for the system (first 10 values):\n";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "x[" << i << "] = " << sol[i] << "\n";
    }
    std::cout << std::endl;
}

void Solonethree() {
    size_t N = 100;
    std::cout << "Solving system for N = " << N << " using SetTriVal function.\n";
    auto mb = SetTriVal(N);
    Matrix<double> m = mb.first;
    std::vector<double> b = mb.second;
    std::vector<double> sol = m.QRSolveLinearSystem(b);
    std::cout << "Solution for the system (first 10 values):\n";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "x[" << i << "] = " << sol[i] << "\n";
    }
    std::cout << std::endl;
}

void Soltwo() {
    std::vector<double> t = {-1, -0.75, -0.5, 0, 0.25, 0.5, 0.75};
    Matrix<double> A(7, 3);
    for (size_t i = 0; i < 7; ++i) {
        A(i, 0) = t[i] * t[i];
        A(i, 1) = t[i];
        A(i, 2) = 1;
    }
    std::vector<double> b = {1.00, 0.8125, 0.75, 1.00, 1.3125, 1.75, 2.3125};
    std::vector<double> sol = A.SolveLeastSquares(b);

    std::cout << "Fitting results for y = ax^2 + bx + c (first 3 coefficients):\n";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "Coefficient " << i << ": " << sol[i] << "\n";
    }
    std::cout << std::endl;
}

void Solthree() {
    Matrix<double> A = {
        {1, 4.9176, 1, 3.472, 0.998, 1, 7, 4, 42, 3, 1, 0},
        {1, 5.0208, 1, 3.531, 1.5, 2, 7, 4, 62, 1, 1, 0},
        {1, 4.5429, 1, 2.275, 1.175, 1, 6, 3, 40, 2, 1, 0},
        {1, 4.5573, 1, 4.05, 1.232, 1, 6, 3, 54, 4, 1, 0},
        {1, 5.0597, 1, 4.455, 1.121, 1, 6, 3, 42, 3, 1, 0},
        {1, 3.891, 1, 4.455, 0.988, 1, 6, 3, 56, 2, 1, 0},
        {1, 5.898, 1, 5.85, 1.24, 1, 7, 3, 51, 2, 1, 1},
        {1, 5.6039, 1, 9.52, 1.501, 0, 6, 3, 32, 1, 1, 0},
        {1, 15.4202, 2.5, 9.8, 3.42, 2, 10, 5, 42, 2, 1, 1},
        {1, 14.4598, 2.5, 12.8, 3, 2, 9, 5, 14, 4, 1, 1},
        {1, 5.8282, 1, 6.435, 1.225, 2, 6, 3, 32, 1, 1, 0},
        {1, 5.3003, 1, 4.9883, 1.552, 1, 6, 3, 30, 1, 2, 0},
        {1, 6.2712, 1, 5.52, 0.975, 1, 5, 2, 30, 1, 2, 0},
        {1, 5.9592, 1, 6.666, 1.121, 2, 6, 3, 32, 2, 1, 0},
        {1, 5.05, 1, 5, 1.02, 0, 5, 2, 46, 4, 1, 1},
        {1, 5.6039, 1, 9.52, 1.501, 0, 6, 3, 32, 1, 1, 0},
        {1, 8.2462, 1.5, 5.15, 1.664, 2, 8, 4, 50, 4, 1, 0},
        {1, 6.6969, 1.5, 6.092, 1.488, 1.5, 7, 3, 22, 1, 1, 1},
        {1, 7.7841, 1.5, 7.102, 1.376, 1, 6, 3, 17, 2, 1, 0},
        {1, 9.0384, 1, 7.8, 1.5, 1.5, 7, 3, 23, 3, 3, 0},
        {1, 5.9894, 1, 5.52, 1.256, 2, 6, 3, 40, 4, 1, 1},
        {1, 7.5422, 1.5, 4, 1.69, 1, 6, 3, 22, 1, 1, 0},
        {1, 8.7951, 1.5, 9.89, 1.82, 2, 8, 4, 50, 1, 1, 1},
        {1, 6.0931, 1.5, 6.7265, 1.652, 1, 6, 3, 44, 4, 1, 0},
        {1, 8.3607, 1.5, 9.15, 1.777, 2, 8, 4, 48, 1, 1, 1},
        {1, 8.14, 1, 8, 1.504, 2, 7, 3, 3, 1, 3, 0},
        {1, 9.1416, 1.5, 7.3262, 1.831, 1.5, 8, 4, 31, 4, 1, 0},
        {1, 12, 1.5, 5, 1.2, 2, 6, 3, 30, 3, 1, 1}};

    std::vector<double> b = {25.9, 29.5, 27.9, 25.9, 29.9, 29.9, 30.9, 28.9, 84.9, 82.9, 35.9, 31.5, 31.0, 30.9, 30.0, 28.9, 36.9, 41.9, 40.5, 43.9, 37.5, 37.9, 44.5, 37.9, 38.9, 36.9, 45.8, 41.0}; // 28
    std::vector<double> sol = A.SolveLeastSquares(b);

    std::cout << "Solution for the housing price model (first 12 coefficients):\n";
    for (size_t i = 0; i < 12; ++i) {
        std::cout << "Coefficient " << i << ": " << sol[i] << "\n";
    }
    std::cout << std::endl;
}

void Householdtest() {
    std::vector<double> x = {3, 4};
    auto house = Householder(x);
    Matrix<double> Hi(2, 2);
    for (size_t j = 0; j < 2; ++j) {
        Hi(j, j) = 1;
    }

    for (size_t j = 0; j < 2; ++j) {
        for (size_t k = 0; k < 2; ++k) {
            Hi(j, k) -= house.second * house.first[j] * house.first[k];
        }
    }
    Hi.print();
}
void QRDtest() {
    Matrix<double> A{{1, 2}, {3, 4}, {5, 6}};
    auto QR_de = A.QRDecomposition();
    auto QR = QR_de.first;
    std::vector<double> d = QR_de.second;
    size_t m = 3;
    size_t n = 2;
    Matrix<double> R(m, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            R(i, j) = QR(i, j);
        }
    }

    Matrix<double> Q(m, m);
    for (size_t i = 0; i < m; ++i) {
        Q(i, i) = 1;
    }

    for (size_t i = 0; i < n; ++i) {
        std::vector<double> v(m - i);
        for (size_t j = i + 1; j < m; ++j) {
            v[j - i] = QR(j, i);
        }
        v[0] = 1.0;
        double beta = d[i];

        Matrix<double> Hi(m, m);
        for (size_t j = 0; j < m; ++j) {
            Hi(j, j) = 1;
        }

        // 更新 Hi 矩阵
        for (size_t j = 0; j < m - i; ++j) {
            for (size_t k = 0; k < m - i; ++k) {
                Hi(i + j, i + k) -= beta * v[j] * v[k];
            }
        }
        Q = Q.MulWithMatrix(Hi);
    }
    Q.print();
    R.print();
    Matrix<double> M(3, 2);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {

            for (size_t k = 0; k < 3; ++k) {
                M(i, j) += Q(i, k) * R(k, j);
            }
        }
    }
    M.print();
}
void LSQtest() {
    Matrix<double> A = {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}};
    std::vector<double> b = {1.2, 1.9, 3.1, 3.9, 5.1};
    std::vector<double> sol = A.SolveLeastSquares(b);

    for (size_t i = 0; i < 2; ++i) {
        std::cout << sol[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

int main() {
    // Householdtest();
    // QRDtest();
    Soloneone();
    Solonetwo();
    Solonethree();
    Soltwo();
    Solthree();
    // LSQtest();
    return 0;
}