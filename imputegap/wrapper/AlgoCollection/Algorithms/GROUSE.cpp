//
// Created on 04/01/19.
//

//
// Code translated to C++ from the original: https://gitlab.eecs.umich.edu/girasole/ProcIEEEstreamingPCA/blob/master/algorithms.py
//

#include <cmath>
#include <iostream>

#include "GROUSE.h"

namespace Algorithms
{

void Algorithms::GROUSE::doGROUSE(arma::mat &input, uint64_t maxrank)
{
    arma::arma_rng::set_seed(1921);
    
    std::vector<arma::uvec> indices;

    std::cout << "\t\t(C++) GROUSE: Matrix Shape: (" << input.n_rows << ", " << input.n_cols << ") for Max Rank " << maxrank << "..." << std::endl;

    for (uint64_t i = 0; i < input.n_cols; ++i)
    {
        indices.emplace_back(arma::find_finite(input.col(i)));
    }
    
    arma::mat U = arma::orth(arma::randn<arma::mat>(input.n_rows, maxrank));
    
    for (uint64_t outiter = 0; outiter < maxCycles; ++outiter)
    {
        for (uint64_t k = 0; k < input.n_cols; ++k)
        {
            // Pull out the relevant indices and revealed entries for this column
            arma::uvec &idx = indices[k];//find(Indicator(:,col_order(k)));
            arma::vec currentCol = input.col(k);
            arma::vec v_Omega = currentCol.elem(idx);
            arma::mat U_Omega = U.rows(idx);
            
            // Predict the best approximation of v_Omega by u_Omega.
            // That is, find weights to minimize ||U_Omega*weights-v_Omega||^2
            
            arma::vec weights;
            bool success = arma::solve(weights, U_Omega, v_Omega);
            
            if (!success)
            {
                std::cout << "arma::solve has failed, aborting remaining recovery" << std::endl;
                return;
            }
            
            //arma::vec weights = arma::pinv(U_Omega) * v_Omega;
            double norm_weights = arma::norm(weights); (void) norm_weights;
            
            // Compute the residual not predicted by the current estmate of U.
            
            arma::vec p = U_Omega * weights;
            arma::vec residual = v_Omega - p;
            double norm_residual = arma::norm(residual);
            
            //if (norm_residual < 0.000000001)
            //{
            //    norm_residual = 0.000000001;
            //}
    
            // This step-size rule is given by combining Edelman's geodesic
            // projection algorithm with a diminishing step-size rule from SGD.  A
            // different step size rule could suffice here...
            
            // this is matlab version
            #if true
            {
                double sG = norm_residual*norm_weights;
                if (norm_residual < 0.000000001)
                {
                    sG = 0.000000001 * norm_weights;
                }
                //err_reg((outiter-1)*numc + k) = norm_residual/norm(v_Omega);
                double t = step_size*sG/(double)( (outiter)*input.n_cols + k + 1 );
    
                // Take the gradient step.
                if (t < (arma::datum::pi / 2.0)) // drop big steps
                {
                    double alpha = (cos(t) - 1.0) / std::pow(norm_weights, 2);
                    double beta = sin(t) / sG;
        
                    arma::vec step = U * (alpha * weights);
        
                    step.elem(idx) += (beta * residual);
        
                    U = U + step * weights.t();
                }
            }
            #endif
            
            // this is python version
            #if false
            {
                double norm_p = arma::norm(p);
                double theta = std::atan(norm_residual/norm_p); //r_norm/p_norm
    
                arma::vec compA = (std::cos(theta) - 1.0) * p / norm_p;
                arma::vec compB = std::sin(theta) * residual / norm_residual;
                arma::vec compC = weights / norm_p; // \|w\|_2 = \|p\|_2
                
                arma::rowvec compC_T = compC.t();
                
                arma::mat out_AC = compA * compC_T;
    
                auto i = idx.begin();
                uint64_t iouter = 0;
                for ( ; i < idx.end(); ++i, ++iouter)
                {
                    for (uint64_t j = 0; j < U.n_cols; ++j)
                    {
                        U.at(*i, j) += out_AC.at(iouter, j);
                    }
                };
                //U = U + out_AC;
                
                arma::mat out_BC = compB * compC_T;
    
                //U[Omega, :] += np.outer(compB, compC)
                i = idx.begin();
                iouter = 0;
                for ( ; i < idx.end(); ++i, ++iouter)
                {
                    for (uint64_t j = 0; j < U.n_cols; ++j)
                    {
                        U.at(*i, j) += out_BC.at(iouter, j);
                    }
                };
            }
            #endif
        }
    }
    
    // generate R
    
    arma::mat R(input.n_cols, maxrank);
    
    for (uint64_t k = 0; k < input.n_cols; ++k)
    {
        // Pull out the relevant indices and revealed entries for this column
        arma::uvec &idx = indices[k]; //idx = find(Indicator(:,k));
        arma::vec currentCol = input.col(k);
        arma::vec v_Omega = currentCol.elem(idx);
        arma::mat U_Omega = U.rows(idx);
        // solve a simple least squares problem to populate R
        arma::vec sol = arma::solve(U_Omega, v_Omega);
    
        for (uint64_t i = 0; i < sol.n_rows; ++i)
        {
            R(k, i) = sol[i];
        }
    }
    
    arma::mat recon = U * R.t();
    
    for (uint64_t j = 0; j < input.n_cols; ++j)
    {
        for (uint64_t i = 0; i < input.n_rows; ++i)
        {
            if (std::isnan(input.at(i, j)))
            {
                input.at(i, j) = recon.at(i, j);
            }
        }
    }
}

} // namespace Algorithms
