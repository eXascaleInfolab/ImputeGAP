//
// Created by quentin nater on 12/01/25.
//

#include <cassert>
#include <chrono>

#include "SharedLibGROUSE.h"

#include "../Algorithms/GROUSE.h"


// Non-exposed functions
arma::mat
marshal_as_arma(double *matrixNative, size_t dimN, size_t dimM)
{
    return arma::mat(matrixNative, dimN, dimM, false, true);
}

void
marshal_as_native(const arma::mat &matrixArma, double *container)
{
    std::copy_n(matrixArma.memptr(), matrixArma.n_rows * matrixArma.n_cols, container);
}

void
marshal_as_failed(double *container, size_t dimN, size_t dimM)
{
    std::fill_n(container, dimN * dimM, std::numeric_limits<double>::quiet_NaN());
}

void
verifyRecovery(arma::mat &mat)
{
    for (uint64_t j = 0; j < mat.n_cols; ++j)
    {
        for (uint64_t i = 0; i < mat.n_rows; ++i)
        {
            if (std::isnan(mat.at(i, j)))
            {
                mat.at(i, j) = std::sqrt(std::numeric_limits<double>::max() / 100000.0);
            }
        }
    }
}

// Exposed functions

extern "C"
{

void
recoveryGROUSE(double *matrixNative, size_t dimN, size_t dimM, size_t rank)
{
    arma::mat input = marshal_as_arma(matrixNative, dimN, dimM);

    input = input.t();

    //std::cout << "(NATERQ:___) increment = (" << input.n_rows << ", " << input.n_cols << ")" << std::endl;
    //std::cout << "(NATERQ:___) maxrank = (" << rank << ")" << std::endl;

    // Local
    int64_t result;

    Algorithms::GROUSE grouse;

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Recovery
    begin = std::chrono::steady_clock::now();
    grouse.doGROUSE(input, rank);
    end = std::chrono::steady_clock::now();

    result = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    input = input.t();

    verifyRecovery(input);

    (void)result;
}

}