#include "../Com/Matrix.h"
#include "../Com/common.h"

int main() {
    Matrix<double> A1 = {{0, 0, -3}, {1, 0, 5}, {0, 1, -1}};
    Matrix<double> A2 = {{0, 0, 1}, {1, 0, 3}, {0, 1, 0}};
    Matrix<double> A3 = {{0, 0, 0, 0, 0, 0, 0, 1000}, {1, 0, 0, 0, 0, 0, 0, -790}, {0, 1, 0, 0, 0, 0, 0, 99902}, {0, 0, 1, 0, 0, 0, 0, -79108.9}, {0, 0, 0, 1, 0, 0, 0, -9802.08}, {0, 0, 0, 0, 1, 0, 0, -10891.01}, {0, 0, 0, 0, 0, 1, 0, -208.01}, {0, 0, 0, 0, 0, 0, 1, -101}};
    std::cout << A1.power_iteration_method().first << std::endl;
    std::cout << A2.power_iteration_method().first << std::endl;
    std::cout << A3.power_iteration_method().first << std::endl;
    return 0;
}