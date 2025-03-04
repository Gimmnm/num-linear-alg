#pragma once
#include "common.h"

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
    void print() const {
        for (size_t i = 0; i < mat_row; ++i) {
            for (size_t j = 0; j < mat_col; ++j) {
                std::cout << (*this)(i, j) << " ";
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
    std::vector<T> SolveLowerTriangular(const std::vector<T> &b) const {
        std::vector<T> x(mat_row, T(0));
        for (size_t i = 0; i < mat_row; ++i) {
            T sum = 0;
            for (size_t j = 0; j < i; ++j) {
                sum += (*this)(i, j) * x[j];
            }
            if ((*this)(i, i) == 0)
                throw std::runtime_error("Pivot is zero");
            x[i] = (b[i] - sum) / (*this)(i, i);
        }
        return x;
    }
    std::vector<T> SolveUpperTriangular(const std::vector<T> &b) const {
        std::vector<T> x(mat_row, T(0));
        for (int i = (int)mat_row - 1; i >= 0; --i) {
            T sum = 0;
            for (int j = i + 1; j < (int)mat_row; ++j) {
                sum += (*this)(i, j) * x[j];
            }
            if ((*this)(i, i) == 0)
                throw std::runtime_error("Pivot is zero");
            x[i] = (b[i] - sum) / (*this)(i, i);
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
    Matrix<T> PALUdecomposition(std::vector<int> &p) const {
        Matrix<T> LU(*this);
        size_t n = mat_row;
        p.resize(n);
        for (size_t i = 0; i < n; ++i) {
            p[i] = (int)i;
        }
        for (size_t k = 0; k < n - 1; ++k) {
            T maxx = std::abs(LU(k, k));
            size_t pivot = k;
            for (size_t j = k + 1; j < n; ++j) {
                T tmp = std::abs(LU(j, k));
                if (tmp > maxx) {
                    maxx = tmp;
                    pivot = j;
                }
            }
            if (maxx == 0)
                throw std::runtime_error("Matrix is singular");
            if (pivot != k) {
                std::swap(p[k], p[pivot]);
                for (size_t col = 0; col < n; ++col) {
                    std::swap(LU(k, col), LU(pivot, col));
                }
            }
            if (LU(k, k) == 0)
                throw std::runtime_error("Pivot is zero");
            for (size_t j = k + 1; j < n; ++j) {
                LU(j, k) /= LU(k, k);
                for (size_t i = k + 1; i < n; ++i) {
                    LU(j, i) -= LU(j, k) * LU(k, i);
                }
            }
        }
        return LU;
    }
    Matrix<T> CholeskyDecomposition() const {
        if (mat_row != mat_col)
            throw std::runtime_error("Cholesky decomposition requires a square matrix.");
        size_t n = mat_row;
        Matrix<T> L(n, n);
        T tolerance = std::numeric_limits<T>::epsilon() * 100;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < j; ++k) {
                    sum += L(i, k) * L(j, k);
                }
                if (i == j) {
                    T diag = (*this)(i, i) - sum;
                    if (diag <= tolerance) {
                        std::cout << diag << std::endl;
                        throw std::runtime_error("Matrix is not positive definite.");
                    }
                    L(i, j) = std::sqrt(diag);
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
        T tolerance = std::numeric_limits<T>::epsilon() * 100;
        std::vector<T> D(n, T(0));
        for (size_t i = 0; i < n; ++i) {
            L(i, i) = T(1);
        }
        for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < j; ++k) {
                sum += L(j, k) * L(j, k) * D[k];
            }
            D[j] = (*this)(j, j) - sum;
            if (std::abs(D[j] - T(0)) < tolerance)
                throw std::runtime_error("Matrix is not positive definite (zero pivot encountered in D).");
            for (size_t i = j + 1; i < n; ++i) {
                T sum2 = T(0);
                for (size_t k = 0; k < j; ++k) {
                    sum2 += L(i, k) * L(j, k) * D[k];
                }
                L(i, j) = (((*this)(i, j) - sum2) / D[j]);
            }
        }
        return std::make_pair(L, D);
    }

    static Matrix<T> HilbertMatrix(size_t n) {
        Matrix<T> H(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                H(i, j) = T(1) / T(i + j + 1);
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
};