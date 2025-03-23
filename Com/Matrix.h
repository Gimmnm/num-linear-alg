#pragma once
#include "common.h"

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
    return *std::max_element(vec.begin(), vec.end(), [](T a, T b) { return std::abs(a) < std::abs(b); });
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
};