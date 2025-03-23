#include "../Com/Matrix.h"
#include "../Com/common.h"

void taskone() {
    for (size_t i = 5; i <= 20; ++i) {
        Matrix<double> hilb = Matrix<double>::HilbertMatrix(i).transpose();
        std::cout << "||hilb_" << i << "||_∞ = " << hilb.blindHillClimbingInfinityNorm() << std::endl;
    }
    for (size_t i = 5; i <= 20; ++i) {
        Matrix<double> hilb = Matrix<double>::HilbertMatrix(i).transpose();
        std::cout << "condition number of Hilbert Matrix of order " << i << ": " << hilb.blindHillClimbingInfinityNorm() * hilb.blindHillClimbingInfinityNorm(1) << std::endl;
    }
}

void tasktwo() {
    for (size_t n = 5; n <= 30; ++n) {
        std::vector<double> xn(n, 0.0);
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_real_distribution<double> dist(1.0L, 150.0L);
        for (size_t j = 0; j < n; ++j) {
            xn[j] = dist(gen);
        }
        Matrix<double> An = Matrix<double>::HwfiveSecondAnMatrix(n);
        std::vector<double> b = An.MulWithVector(xn);
        std::vector<double> xhat = An.PALUsolve(b);
        std::vector<double> r(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            r[i] = xn[i] - xhat[i];
        }
        Matrix<double> AnT = An.transpose();
        r = An.MulWithVector(r);
        double niu = AnT.blindHillClimbingInfinityNorm(1), gamma = VectorInfinityNorm(r), beta = VectorInfinityNorm(b), miu = AnT.blindHillClimbingInfinityNorm();
        // std::cout << niu << " " << gamma << " " << beta << " " << miu << std::endl;
        std::cout << "A" << n << ": " << niu * miu * gamma / beta << std::endl;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(16);
    // taskone();
    tasktwo();
    return 0;
}