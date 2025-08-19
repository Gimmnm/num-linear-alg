#pragma once
#include "common.h"

// 模板化的向量点积 (内积)
template <typename T>
T dot_product(const std::vector<T> &v1, const std::vector<T> &v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("向量维度必须匹配才能进行点积运算");
    }
    // std::inner_product 计算两个范围的点积
    // 第三个参数是第二个范围的起始，第四个参数是初始和
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), T(0));
}

// 模板化的标量乘以向量
template <typename T>
std::vector<T> scalar_multiply(T scalar, const std::vector<T> &vec) {
    std::vector<T> result = vec; // 创建一个副本
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] *= scalar;
    }
    return result;
}

template <typename T>
std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for addition");
    }
    std::vector<T> result;
    result.reserve(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] + b[i]);
    }
    return result;
}

// 复合赋值版本，支持 a += b
template <typename T>
std::vector<T> &operator+=(std::vector<T> &a, const std::vector<T> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for addition");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] += b[i];
    }
    return a;
}

// 元素按位相减，要求两个向量长度相同
template <typename T>
std::vector<T> operator-(const std::vector<T> &a, const std::vector<T> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for subtraction");
    }
    std::vector<T> result;
    result.reserve(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] - b[i]);
    }
    return result;
}

// 复合赋值版本，支持 a -= b
template <typename T>
std::vector<T> &operator-=(std::vector<T> &a, const std::vector<T> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for subtraction");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] -= b[i];
    }
    return a;
}

template <typename T>
T VectorL1Norm(const std::vector<T> &vec) {
    T norm = 0;
    for (const auto &v : vec)
        norm += std::fabs(v);
    return norm;
}

template <typename T>
T VectorL2Norm(const std::vector<T> &vec) {
    T norm = 0;
    for (const auto &v : vec)
        norm += v * v;
    return std::sqrt(norm);
}

template <typename T>
T VectorInfinityNorm(const std::vector<T> &vec) {
    return abs(*std::max_element(vec.begin(), vec.end(), [](T a, T b) { return std::abs(a) < std::abs(b); }));
}

template <typename T>
void VectorPrint(const std::vector<T> &vec) {
    size_t n = vec.size();
    for (size_t i = 0; i < n; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
std::pair<T, size_t> VectorInfinityNormWithIndex(const std::vector<T> &vec) {
    if (vec.empty()) {
        throw std::runtime_error("Vector is empty.");
    }

    size_t max_index = 0;
    T max_value = std::abs(vec[0]);

    for (size_t i = 1; i < vec.size(); ++i) {
        if (std::abs(vec[i]) > max_value) {
            max_value = std::abs(vec[i]);
            max_index = i;
        }
    }

    return {max_value, max_index};
}

template <typename T>
std::pair<std::vector<T>, T> Householder(std::vector<T> x) {
    T beta = 0;
    size_t n = x.size();
    T eta = VectorInfinityNorm(x);
    if (eta == 0) {
        throw std::runtime_error("Zero vector encountered, cannot perform Householder transformation.");
    }
    for (size_t i = 0; i < n; ++i)
        x[i] = x[i] / eta;
    std::vector<T> v = x;

    T sigma = 0;
    for (size_t i = 1; i < n; ++i)
        sigma += x[i] * x[i];

    if (sigma == 0) {
        beta = 0;
    } else {
        T alpha = std::sqrt(x[0] * x[0] + sigma);
        if (x[0] <= 0) {
            v[0] = x[0] - alpha;
        } else {
            v[0] = -sigma / (x[0] + alpha);
        }
        beta = 2 * (v[0] * v[0]) / (sigma + v[0] * v[0]);
        T div = v[0];
        for (size_t i = 0; i < n; ++i)
            v[i] = v[i] / div;
    }

    return std::make_pair(v, beta);
}

template <typename T>
std::pair<T, T> Givens(T a, T b) {
    T c = 0;
    T s = 0;

    if (b == 0) {
        c = 1;
        s = 0;
    } else {
        if (std::fabs(b) > std::fabs(a)) {
            T tau = a / b;
            s = 1 / std::sqrt(1 + tau * tau);
            c = s * tau;
        } else {
            T tau = b / a;
            c = 1 / std::sqrt(1 + tau * tau);
            s = c * tau;
        }
    }
    return std::make_pair(c, s);
}
template <typename T = double>
class Matrix {
private:
    std::vector<T> data;
    size_t mat_row;
    size_t mat_col;

public:
    Matrix() : mat_row(0), mat_col(0) {}
    Matrix(size_t row, size_t col) : data(row * col, T()), mat_row(row), mat_col(col) {}
    Matrix(std::initializer_list<std::initializer_list<T>> initList) {
        mat_row = initList.size();
        mat_col = mat_row ? initList.begin()->size() : 0;
        data.resize(mat_row * mat_col, T());
        size_t i = 0;
        for (auto &r : initList) {
            size_t j = 0;
            for (auto v : r) {
                (*this)(i, j) = v;
                j++;
            }
            i++;
        }
    }
    Matrix(const Matrix &other) : data(other.data), mat_row(other.mat_row), mat_col(other.mat_col) {}
    Matrix &operator=(const Matrix &other) {
        if (this == &other)
            return *this;
        data = other.data;
        mat_row = other.mat_row;
        mat_col = other.mat_col;
        return *this;
    }
    Matrix(Matrix &&other) noexcept
        : data(std::move(other.data)), mat_row(other.mat_row), mat_col(other.mat_col) {
        other.mat_row = 0;
        other.mat_col = 0;
    }
    Matrix &operator=(Matrix &&other) noexcept {
        if (this == &other)
            return *this;
        data = std::move(other.data);
        mat_row = other.mat_row;
        mat_col = other.mat_col;
        other.mat_row = 0;
        other.mat_col = 0;
        return *this;
    }
    T &operator()(size_t row, size_t col) {
        return data[row * mat_col + col];
    }
    const T &operator()(size_t row, size_t col) const {
        return data[row * mat_col + col];
    }
    size_t get_row() const { return mat_row; }
    size_t get_col() const { return mat_col; }
    void print(const std::string &msg = "") const {
        if (!msg.empty()) {
            std::cout << msg << std::endl;
        }
        for (size_t i = 0; i < mat_row; ++i) {
            for (size_t j = 0; j < mat_col; ++j) {
                std::cout << std::setw(8) << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    Matrix<T> transpose() const {
        Matrix<T> result(mat_col, mat_row);
        for (size_t i = 0; i < mat_row; ++i) {
            for (size_t j = 0; j < mat_col; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    T normL1() const {
        T max_sum = 0;
        for (size_t j = 0; j < mat_col; ++j) {
            T col_sum = 0;
            for (size_t i = 0; i < mat_row; ++i) {
                col_sum += std::abs((*this)(i, j));
            }
            max_sum = std::max(max_sum, col_sum);
        }
        return max_sum;
    }

    T normLInfinity() const {
        T max_sum = 0;
        for (size_t i = 0; i < mat_row; ++i) {
            T row_sum = 0;
            for (size_t j = 0; j < mat_col; ++j) {
                row_sum += std::abs((*this)(i, j));
            }
            max_sum = std::max(max_sum, row_sum);
        }
        return max_sum;
    }

    std::vector<T> SolveLowerTriangular(const std::vector<T> &b) const {
        std::vector<T> x(mat_row, 0.0);
        for (size_t i = 0; i < mat_row; ++i) {
            T sum = b[i];
            for (size_t j = 0; j < i; ++j) {
                sum -= (*this)(i, j) * x[j];
            }
            x[i] = sum; // L(i, i) = 1
        }
        return x;
    }
    std::vector<T> SolveUpperTriangular(const std::vector<T> &b) const {
        std::vector<T> x(mat_row, 0.0);
        for (int i = (int)mat_row - 1; i >= 0; --i) {
            T sum = b[i];
            for (int j = i + 1; j < (int)mat_row; ++j) {
                sum -= (*this)(i, j) * x[j];
            }
            x[i] = sum / (*this)(i, i);
        }
        return x;
    }
    std::vector<T> MulWithVector(const std::vector<T> &b) const {
        if (b.size() != mat_col)
            throw std::runtime_error("Vector dimension mismatch");
        std::vector<T> x(mat_row, T(0));
        for (size_t i = 0; i < mat_row; ++i) {
            for (size_t j = 0; j < mat_col; ++j) {
                x[i] += (*this)(i, j) * b[j];
            }
        }
        return x;
    }
    Matrix<T> MulWithMatrix(const Matrix<T> &B) const {
        size_t n = get_col();
        Matrix<T> C(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    C(i, j) += (*this)(i, k) * B(k, j);
                }
            }
        }
        return C;
    }
    Matrix<T> LUdecomposition() const {
        Matrix<T> LU(*this);
        size_t n = mat_row;
        for (size_t k = 0; k < n - 1; ++k) {
            if (LU(k, k) == 0)
                throw std::runtime_error("Pivot is zero");
            for (size_t j = k + 1; j < n; ++j) {
                LU(j, k) /= LU(k, k);
            }
            for (size_t j = k + 1; j < n; ++j) {
                for (size_t i = k + 1; i < n; ++i) {
                    LU(j, i) -= LU(j, k) * LU(k, i);
                }
            }
        }
        return LU;
    }
    std::pair<Matrix<T>, Matrix<T>> PALUdecomposition(std::vector<T> &b) const {
        Matrix<T> LU(*this);
        size_t n = mat_row;
        std::vector<int> p(n);
        Matrix<T> P(n, n);
        std::vector<T> bb = b;
        for (size_t i = 0; i < n; ++i) {
            p[i] = (int)i;
        }
        for (size_t k = 0; k < n - 1; ++k) {
            T maxx = std::fabs(LU(k, k));
            size_t pivot = k;
            for (size_t j = k + 1; j < n; ++j) {
                T tmp = std::fabs(LU(j, k));
                if (tmp > maxx) {
                    // std::cout << k << " " << j << std::endl;
                    maxx = tmp;
                    pivot = j;
                }
            }
            if (maxx < 1e-15)
                throw std::runtime_error("Matrix is singular");
            if (pivot != k) {
                std::swap(p[k], p[pivot]);
                for (size_t col = 0; col < n; ++col) {
                    std::swap(LU(k, col), LU(pivot, col));
                }
            }
            for (size_t j = k + 1; j < n; ++j) {
                LU(j, k) /= LU(k, k);
                for (size_t i = k + 1; i < n; ++i) {
                    LU(j, i) -= LU(j, k) * LU(k, i);
                }
            }
        }
        for (size_t i = 0; i < n; ++i)
            P(i, p[i]) = 1.0, bb[i] = b[p[i]];
        b = bb;
        return std::make_pair(LU, P);
    }
    Matrix<T> CholeskyDecomposition() const {
        if (mat_row != mat_col)
            throw std::runtime_error("Cholesky decomposition requires a square matrix.");
        size_t n = mat_row;
        Matrix<T> L(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                T sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += L(i, k) * L(j, k);
                }
                if (i == j) {
                    T diag = (*this)(i, i) - sum;
                    if (diag < 1e-15) {
                        throw std::runtime_error("Matrix is not positive definite.");
                    }
                    L(i, j) = std::sqrt(std::fabs(diag));
                } else {
                    L(i, j) = (((*this)(i, j) - sum) / L(j, j));
                }
            }
        }
        return L;
    }
    std::pair<Matrix<T>, std::vector<T>> LDLTDecomposition() const {
        if (mat_row != mat_col)
            throw std::runtime_error("LDLᵀ decomposition requires a square matrix.");
        size_t n = mat_row;
        Matrix<T> L(n, n);
        std::vector<T> D(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            L(i, i) = 1.0;
        }
        for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < j; ++k) {
                sum += L(j, k) * L(j, k) * D[k];
            }
            D[j] = (*this)(j, j) - sum;
            if (std::fabs(D[j]) < 1e-15)
                throw std::runtime_error("Matrix is not positive definite (zero pivot encountered in D).");
            for (size_t i = j + 1; i < n; ++i) {
                T sum2 = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum2 += L(i, k) * L(j, k) * D[k];
                }
                L(i, j) = (((*this)(i, j) - sum2) / D[j]);
            }
        }
        return std::make_pair(L, D);
    }

    std::pair<Matrix<T>, std::vector<T>> QRDecomposition() const {
        Matrix<T> QR = *this;
        size_t m = get_row();
        size_t n = get_col();
        std::vector<T> d(n);

        for (size_t i = 0; i < n; ++i) {
            if (i < m) {
                std::vector<T> x(m - i);
                for (size_t j = i; j < m; ++j) {
                    x[j - i] = QR(j, i); // 修正了索引
                }

                auto house = Householder(x); // v = house.first; beta = house.second

                // 构造 Hi 矩阵
                Matrix<T> Hi(m - i, m - i);
                for (size_t j = 0; j < m - i; ++j) {
                    Hi(j, j) = 1;
                }

                // 更新 Hi 矩阵
                for (size_t j = 0; j < m - i; ++j) {
                    for (size_t k = 0; k < m - i; ++k) {
                        Hi(j, k) -= house.second * house.first[j] * house.first[k];
                    }
                }

                // 构造 B 矩阵
                Matrix<T> B(m - i, n - i);
                for (size_t j = 0; j < m - i; ++j) {
                    for (size_t k = 0; k < n - i; ++k) {
                        for (size_t s = 0; s < m - i; ++s) {
                            B(j, k) += Hi(j, s) * QR(i + s, i + k);
                        }
                    }
                }

                // 更新 QR 矩阵
                for (size_t j = 0; j < m - i; ++j) {
                    for (size_t k = 0; k < n - i; ++k) {
                        QR(i + j, i + k) = B(j, k);
                    }
                }

                d[i] = house.second;

                // 更新 QR 的下三角部分
                for (size_t j = i + 1; j < m; ++j) {
                    QR(j, i) = house.first[j - i];
                }
            }
        }

        return std::make_pair(QR, d);
    }

    std::vector<T> PALUsolve(std::vector<T> b) const {
        if (mat_row != mat_col) {
            throw std::runtime_error("Matrix must be square for PALU decomposition.");
        }
        std::vector<T> bb = b;
        auto LU = PALUdecomposition(bb).first;
        std::vector<T> y = LU.SolveLowerTriangular(bb);
        std::vector<T> x = LU.SolveUpperTriangular(y);

        return x;
    }

    std::vector<T> QRSolveLinearSystem(const std::vector<T> &b) const {
        // 使用 QR 分解求解线性方程组 Ax = b
        auto qr_decomp = QRDecomposition();
        Matrix<T> QR = qr_decomp.first;      // 获取 QR 矩阵
        std::vector<T> d = qr_decomp.second; // 获取 d 向量（beta）

        size_t m = get_row();
        size_t n = get_col();
        Matrix<T> R(m, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                R(i, j) = QR(i, j);
            }
        }

        Matrix<T> Q(m, m);
        for (size_t i = 0; i < m; ++i) {
            Q(i, i) = 1;
        }

        for (size_t i = 0; i < n; ++i) {
            std::vector<T> v(m - i);
            for (size_t j = i + 1; j < m; ++j) {
                v[j - i] = QR(j, i);
            }
            v[0] = 1.0;
            T beta = d[i];

            Matrix<T> Hi(m, m);
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

        // 计算 Q^T * b
        std::vector<T> y = Q.transpose().MulWithVector(b);

        // 解 Rx = y（上三角系统）
        return R.SolveUpperTriangular(y);
    }

    std::vector<T> SolveLeastSquares(const std::vector<T> &b) const {
        // 使用 QR 分解求解线性方程组 Ax = b
        auto qr_decomp = QRDecomposition();
        Matrix<T> QR = qr_decomp.first;      // 获取 QR 矩阵
        std::vector<T> d = qr_decomp.second; // 获取 d 向量（beta）

        size_t m = get_row();
        size_t n = get_col();
        Matrix<T> R(m, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                R(i, j) = QR(i, j);
            }
        }

        Matrix<T> Q(m, m);
        for (size_t i = 0; i < m; ++i) {
            Q(i, i) = 1;
        }

        for (size_t i = 0; i < n; ++i) {
            std::vector<T> v(m - i);
            for (size_t j = i + 1; j < m; ++j) {
                v[j - i] = QR(j, i);
            }
            v[0] = 1.0;
            T beta = d[i];

            Matrix<T> Hi(m, m);
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
        Matrix<T> QQ(m, n);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                QQ(i, j) = Q(i, j);
            }
        }
        Matrix<T> RR(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                RR(i, j) = R(i, j);
            }
        }
        // 计算 Q^T * b
        std::vector<T> y = QQ.transpose().MulWithVector(b);

        // 解 Rx = y（上三角系统）
        return RR.SolveUpperTriangular(y);
    }
    static Matrix<T> HilbertMatrix(size_t n) {
        Matrix<T> H(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                H(i, j) = T(1) / T(i + j + 1.0L);
            }
        }
        return H;
    }

    static Matrix<T> TridiagonalMatrix(size_t n, T subVal, T diagVal, T supVal) {
        Matrix<T> M(n, n);
        for (size_t i = 0; i < n; ++i) {
            M(i, i) = diagVal;
            if (i > 0)
                M(i, i - 1) = subVal;
            if (i < n - 1)
                M(i, i + 1) = supVal;
        }
        return M;
    }

    static Matrix<T> HwfiveSecondAnMatrix(size_t n) {
        Matrix<T> M(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                M(i, j) = 0;
                if (i == j) M(i, j) = 1;
                if (j == n - 1) M(i, j) = 1;
                if (i > j) M(i, j) = -1;
            }
        }
        return M;
    }
    T blindHillClimbingInfinityNorm(bool inverse = false) const {
        size_t n = mat_row;
        std::vector<T> x(n, 1.0 / (T)n); // 初始向量
        size_t iter = 1000;
        if (!inverse) {
            while (--iter) {
                std::vector<T> w(n, 0.0);
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < mat_col; ++j) {
                        w[i] += (*this)(i, j) * x[j];
                    }
                }
                std::vector<T> v(n, 0.0);
                for (size_t i = 0; i < n; ++i) {
                    v[i] = (w[i] >= 0) ? 1 : -1;
                }
                std::vector<T> z(n, 0.0);
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < mat_col; ++j) {
                        z[i] += (*this)(j, i) * v[j];
                    }
                }
                auto zinfinitynorm = VectorInfinityNormWithIndex(z);
                T ztranx = 0.0;
                for (size_t i = 0; i < n; ++i)
                    ztranx += z[i] * x[i];

                if (zinfinitynorm.first <= ztranx) {
                    return VectorL1Norm(w);
                } else {
                    std::vector<T> ej(n, 0.0);
                    ej[zinfinitynorm.second] = 1;
                    x = ej;
                }
            }
        } else {
            while (--iter) {
                std::vector<T> w = PALUsolve(x);
                std::vector<T> v(n, 0.0);
                for (size_t i = 0; i < n; ++i) {
                    if (w[i] > 0) v[i] = 1;
                    if (w[i] < 0) v[i] = -1;
                }
                std::vector<T> z = PALUsolve(v);
                auto zinfinitynorm = VectorInfinityNormWithIndex(z);
                T ztranx = 0.0;
                for (size_t i = 0; i < n; ++i)
                    ztranx += z[i] * x[i];
                if (zinfinitynorm.first <= ztranx) {
                    return VectorL1Norm(w);
                } else {
                    std::vector<T> ej(n, 0.0);
                    ej[zinfinitynorm.second] = 1.0;
                    x = ej;
                }
                if (iter == 1) {
                    return VectorL1Norm(w);
                }
            }
        }
        return 0;
    }

    std::vector<T> JacobiIterativeMethod(const std::vector<T> &b, size_t max_iter = 10000000, T tol = 1e-7) const {
        size_t n = mat_row;
        std::vector<T> x(n, T(0));     // 初始解向量
        std::vector<T> x_old(n, T(0)); // 上一轮解向量

        for (size_t k = 0; k < max_iter; ++k) {
            for (size_t i = 0; i < n; ++i) {
                T sum = b[i];
                for (size_t j = 0; j < n; ++j) {
                    if (i != j) {
                        sum -= (*this)(i, j) * x_old[j];
                    }
                }
                x[i] = sum / (*this)(i, i);
            }

            // 检查收敛性
            T error = 0.0;
            for (size_t i = 0; i < n; ++i) {
                error = std::max(error, std::fabs(x[i] - x_old[i]));
            }

            if (error < tol) {
                std::cout << "jocobi iter: " << k << std::endl;
                break;
            }

            // 更新 x_old
            x_old = x;
        }

        return x;
    }

    std::vector<T> GaussSeidelIterativeMethod(const std::vector<T> &b, size_t max_iter = 10000000, T tol = 1e-7) const {
        size_t n = mat_row;
        std::vector<T> x(n, T(0)); // 初始解向量

        for (size_t k = 0; k < max_iter; ++k) {
            for (size_t i = 0; i < n; ++i) {
                T sum = b[i];
                for (size_t j = 0; j < n; ++j) {
                    if (i != j) {
                        sum -= (*this)(i, j) * x[j];
                    }
                }
                x[i] = sum / (*this)(i, i);
            }

            // 检查收敛性
            T error = 0.0;
            for (size_t i = 0; i < n; ++i) {
                error = std::max(error, std::fabs(b[i] - this->MulWithVector(x)[i]));
            }

            if (error < tol) {
                std::cout << "G-S iter: " << k << std::endl;
                break;
            }
        }

        return x;
    }

    std::vector<T> SuccessiveOverRelaxationMethod(const std::vector<T> &b, T omega = 1.1, size_t max_iter = 100000, T tol = 1e-7) const {
        size_t n = mat_row;
        std::vector<T> x(n, T(0)); // 初始解向量

        for (size_t k = 0; k < max_iter; ++k) {
            for (size_t i = 0; i < n; ++i) {
                T sum = b[i];
                for (size_t j = 0; j < n; ++j) {
                    if (i != j) {
                        sum -= (*this)(i, j) * x[j];
                    }
                }
                T new_x = sum / (*this)(i, i);
                x[i] = x[i] + omega * (new_x - x[i]);
            }

            // 检查收敛性
            T error = 0.0;
            for (size_t i = 0; i < n; ++i) {
                error = std::max(error, std::fabs(b[i] - this->MulWithVector(x)[i]));
            }

            if (error < tol) {
                std::cout << "松弛迭代 iter: " << k << std::endl;
                break;
            }
        }

        return x;
    }

    std::pair<T, std::vector<T>> power_iteration_method(int max_iterations = 1000000, T tolerance = 1e-8) {
        size_t n = this->get_row();

        std::vector<T> xk(n), xk1(n), xk2(n), yk(n), yk1(n), yk2(n), xk3(n), yk3(n);
        std::mt19937 rng(std::random_device{}());         // 随机数引擎
        std::uniform_real_distribution<T> dist(0.0, 1.0); // 均匀分布
        for (size_t i = 0; i < n; ++i) {
            // b_k[i] = dist(rng); // 生成随机初始向量
            xk[i] = T(1.0); // 或者使用全1向量作为初始向量
        }

        T norm_x_k_initial = VectorInfinityNorm(xk); // 使用您提供的L2范数函数
        if (norm_x_k_initial == T(0)) {              // 理论上不太可能，但作为健壮性检查
            if (n > 0)
                xk[0] = T(1.0); // 如果是零向量，尝试设置一个非零分量
            else
                return {T(0), {}}; // 如果是0x0矩阵或无法设置，返回
        }
        // 初始归一化
        for (size_t i = 0; i < n; ++i) {
            yk[i] = xk[i] / norm_x_k_initial;
        }
        T lambda_max;
        for (int k_iter = 0; k_iter < max_iterations; ++k_iter) {
            xk3 = xk2;
            yk3 = yk2;
            xk2 = xk1;
            yk2 = yk1;
            xk1 = xk;
            yk1 = yk;
            xk = this->MulWithVector(yk1);
            T norm_x_k_initial = VectorInfinityNorm(xk);
            // for (size_t i = 0; i < n; ++i) {
            // std::cout << xk[i] << " ";
            // }
            // std::cout << std::endl;
            // std::cout << norm_x_k_initial << std::endl;
            for (size_t i = 0; i < n; ++i) {
                yk[i] = xk[i] / norm_x_k_initial;
                // std::cout << yk[i] << " ";
            }
            // std::cout << std::endl;

            // case 1.1
            if (k_iter > 10 && VectorInfinityNorm(yk - yk1) < tolerance && VectorInfinityNorm(yk1 - yk2) < tolerance) {
                lambda_max = VectorInfinityNorm(this->MulWithVector(yk));
                return std::make_pair(lambda_max, yk);
            }
            // case 1.2
            if (k_iter > 10 && VectorInfinityNorm(yk - yk2) < tolerance && VectorInfinityNorm(yk1 - yk3) < tolerance && VectorInfinityNorm(yk + yk1) < tolerance) {
                lambda_max = -VectorInfinityNorm(this->MulWithVector(yk));
                return std::make_pair(lambda_max, yk);
            }
            // case2
            if (k_iter > 10 && VectorInfinityNorm(yk - yk2) < tolerance && VectorInfinityNorm(yk1 - yk3) < tolerance) {
                xk1 = this->MulWithVector(yk);
                xk2 = this->MulWithVector(xk1);
                lambda_max = std::sqrt(VectorInfinityNorm(xk2));
                for (size_t i = 0; i < n; ++i) {
                    xk1[i] = xk1[i] * lambda_max;
                }
                return std::make_pair(lambda_max, xk2 + xk1);
            }
        }
        return std::make_pair(0, xk);
    }

    std::vector<T> ConjugateGradientMethod(const std::vector<T> &b, const std::vector<T> &x0, int max_iterations = -1, T tolerance = 1e-7) const {
        size_t n = mat_row;
        if (max_iterations < 0) {
            max_iterations = n + 1; // 默认最大迭代次数为矩阵维度
        }

        std::vector<T> x = x0;
        std::vector<T> r = b - this->MulWithVector(x); // r_0 = b - A*x_0
        std::vector<T> p = r;                          // p_0 = r_0
        T rsold = dot_product(r, r);                   // r_0^T * r_0

        if (std::sqrt(rsold) < tolerance) { // 如果初始解已经满足要求
            // std::cout << "共轭梯度法: 初始解已满足容差。" << std::endl;
            return x;
        }

        // std::cout << "共轭梯度法迭代过程:" << std::endl;
        // std::cout << "迭代 | 残差L2范数" << std::endl;
        // std::cout << "--------------------" << std::endl;
        // std::cout << std::setw(4) << 0 << " | " << std::scientific << std::setprecision(6) << std::sqrt(rsold) << std::endl;

        for (int k = 0; k < max_iterations; ++k) {
            std::vector<T> Ap = this->MulWithVector(p); // A * p_k
            T p_dot_Ap = dot_product(p, Ap);

            T alpha = rsold / p_dot_Ap; // alpha_k = (r_k^T * r_k) / (p_k^T * A * p_k)

            x = x + scalar_multiply(alpha, p);  // x_{k+1} = x_k + alpha_k * p_k
            r = r - scalar_multiply(alpha, Ap); // r_{k+1} = r_k - alpha_k * A * p_k

            T rsnew = dot_product(r, r); // r_{k+1}^T * r_{k+1}

            // std::cout << std::setw(4) << k + 1 << " | " << std::scientific << std::setprecision(6) << std::sqrt(rsnew) << std::endl;

            if (std::sqrt(rsnew) < tolerance) { // 检查收敛性
                std::cout << "共轭梯度法在 " << k + 1 << " 次迭代后收敛。" << std::endl;
                return x;
            }

            T beta = rsnew / rsold;           // beta_k = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
            p = r + scalar_multiply(beta, p); // p_{k+1} = r_{k+1} + beta_k * p_k
            rsold = rsnew;
        }

        // std::cout << "共轭梯度法达到最大迭代次数 " << max_iterations << "，可能未完全收敛。" << std::endl;
        std::cout << "残差L2范数: " << std::scientific << std::setprecision(6) << std::sqrt(rsold) << std::endl;
        return x;
    }

    static std::vector<T> Jacobi_rotation(const Matrix<T> &A, int p, int q) {
        T tau, t, c, s;
        if (A(p, q) != 0) {
            tau = (A(q, q) - A(p, p)) / (2 * A(p, q));
            t = 0;
            if (tau >= 0) {
                t = 1 / (tau + std::sqrt(1 + tau * tau));
            } else {
                t = 1 / (tau - std::sqrt(1 + tau * tau));
            }
            c = 1 / (std::sqrt(1 + t * t));
            s = c * t;
        } else {
            c = 1, s = 0, t = 0;
        }
        std::cout << t << std::endl;
        std::vector<T> res = {c, s, t};
        return res;
    }

    static T fornorm(const Matrix<T> &A) {
        T res = 0;
        size_t n = A.get_col();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                res += A(i, j) * A(i, j);
            }
        }
        return res;
    }

    static void Jacobi_classic(Matrix<T> A) {
        size_t n = A.get_col();
        Matrix<T> V(n, n);
        for (size_t i = 0; i < n; ++i) {
            V(i, i) = 1;
        }
        T delta = 1e-10 * fornorm(A);
        T sigma1 = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                sigma1 += 2 * A(i, j) * A(i, j);
            }
        }
        while (sigma1 > delta) {
            T maxx = std::abs(A(0, 1));
            int p = 0, q = 1;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    if (std::abs(A(i, j)) > maxx) {
                        p = i, q = j;
                        maxx = std::abs(A(i, j));
                    }
                }
            }
            sigma1 -= 2 * A(p, q) * A(p, q);
            std::vector<T> cst = Jacobi_rotation(A, p, q);
            T c = cst[0], s = cst[1], t = cst[2];
            for (size_t i = 0; i < n; ++i) {
                if (i != p && i != q) {
                    T tmp1 = c * A(p, i) - s * A(q, i);
                    T tmp2 = c * A(q, i) + s * A(p, i);
                    A(i, p) = tmp1;
                    A(p, i) = tmp1;
                    A(i, q) = tmp2;
                    A(q, i) = tmp2;
                }
            }
            A(p, p) = A(p, p) - t * A(p, q);
            A(q, q) = A(q, q) + t * A(p, q);
            A(p, q) = 0;
            A(q, p) = 0;
            for (size_t i = 0; i < n; ++i) {
                T tmp1 = c * V(i, p) - s * V(i, q);
                T tmp2 = c * V(i, q) + s * V(i, p);
                V(i, p) = tmp1;
                V(i, q) = tmp2;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            std::cout << "特征值" << i + 1 << ":" << " ";
            std::cout << A(i, i) << " " << "特征向量: ";
            for (size_t j = 0; j < 5; ++j) {
                std::cout << V(j, i) << " ";
            }
            std::cout << "....";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};
