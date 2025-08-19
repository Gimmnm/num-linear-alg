#include "../Com/Matrix.h"
#include "../Com/common.h"

double calculate_y(double a, double epsilon, double x) {
    return (1 - a) / (1 - exp(-1 / epsilon)) * (1 - exp(-x / epsilon)) + a * x;
}

void task(double eps, double w) {
    std::cout << "epsilon = " << eps << std::endl;
    double ep = eps, a = 0.5, h = 0.01;
    int n = 99;
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
        y[i] = calculate_y(a, ep, (double)(i + 1) * h);
    }
    // for (size_t i = 0; i < n; ++i)
    //     std::cout << y[i] << " ";
    // std::cout << std::endl;
    Matrix<double> A(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (j == i - 1) A(i, j) = ep;
            if (j == i) A(i, j) = -(2 * ep + h);
            if (j == i + 1) A(i, j) = ep + h;
        }
    }
    // std::cout << A(0, 0) << " " << A(0, 1) << std::endl;
    // std::vector<double> bb = A.MulWithVector(y);
    // for (size_t i = 0; i < n; ++i)
    //     std::cout << bb[i] << " ";
    // std::cout << std::endl;
    std::vector<double> b(n);
    for (size_t i = 0; i < n; ++i)
        b[i] = a * h * h;
    b[n - 1] = a * h * h - ep - h;

    double err = 0;

    std::cout << "Jocobi迭代结果: " << std::endl;
    std::vector<double> y1 = A.JacobiIterativeMethod(b);
    std::cout << (double)0 << " ";
    err = 0;
    for (size_t i = 0; i < n; ++i) {
        err = std::max(err, std::fabs(y1[i] - y[i]));
        std::cout << y1[i] << " ";
    }
    std::cout << (double)1 << " ";
    std::cout << std::endl;
    std::cout << "Jocobi误差: " << err << std::endl;
    std::cout << std::endl;

    std::cout << "G-S迭代结果: " << std::endl;
    std::vector<double> y2 = A.GaussSeidelIterativeMethod(b);
    std::cout << (double)0 << " ";
    err = 0;
    for (size_t i = 0; i < n; ++i) {
        err = std::max(err, std::fabs(y2[i] - y[i]));
        std::cout << y2[i] << " ";
    }
    std::cout << (double)1 << " ";
    std::cout << std::endl;
    std::cout << "G-S误差: " << err << std::endl;
    std::cout << std::endl;

    std::cout << "SOR迭代结果: " << std::endl;
    std::vector<double> y3 = A.SuccessiveOverRelaxationMethod(b, w);
    std::cout << (double)0 << " ";
    err = 0;
    for (size_t i = 0; i < n; ++i) {
        err = std::max(err, std::fabs(y3[i] - y[i]));
        std::cout << y3[i] << " ";
    }
    std::cout << (double)1 << " ";
    std::cout << std::endl;
    std::cout << "SOR误差: " << err << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(5);
    task(1, 1.45);
    std::cout << std::endl;
    task(0.1, 1.5);
    std::cout << std::endl;
    task(0.01, 1.5);
    std::cout << std::endl;
    task(0.0001, 1.4);
    return 0;
}