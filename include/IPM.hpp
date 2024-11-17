#ifndef IPM_H
#define IPM_H

#include <iostream>
#include <random>

#include <Eigen/SparseCore>


const double GAMMA = 2/3;

enum class ErCode{
    NO_ERROR,
    LU_FACTARIZATION,
    SOLVE_LIN_SYS
};

struct TermCrit {   // Terminate criteria
    int countMax;   
    double sq_eps; // eps^2
};


class IPM 
{   
    using SpMat = Eigen::SparseMatrix<double> ;
    using Vec = Eigen::VectorXd;
    using SolverLU = Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>>;

    SpMat& m_A;
    Vec& m_b, m_c;

    TermCrit m_crit;
    Vec m_x;  // неизвестный вектор
    Vec m_r; // невязка
    int m_countfor = 0;

    SolverLU m_solverLU;

    void _initRandX(int len){
        //init gamma and random x 
        std::random_device rd;
        std::mt19937 gen(rd());  //here you could set the seed, but std::random_device already does that
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        m_x = Vec::NullaryExpr(len, [&](){return dis(gen);});
    }

    bool _checkTermCriteria(){

        if(m_countfor >= m_crit.countMax || m_r.squaredNorm() < m_crit.sq_eps){
            return true;
        }

        return false;
    }

    
    ErCode _solveLinSysLU(SpMat& A, Vec& b, Vec& out){

        //ONLY FIRST TIME
        if(m_countfor == 0){
            m_solverLU.analyzePattern(A);
        }

        // Compute the numerical factorization 
        m_solverLU.factorize(A); 
        if(m_solverLU.info()!=Eigen::Success) {
            return ErCode::LU_FACTARIZATION;
        }

        out = m_solverLU.solve(b);
        if(m_solverLU.info()!=Eigen::Success) {
            return ErCode::SOLVE_LIN_SYS;
        }

        return ErCode::NO_ERROR;

    }

public:
    IPM(SpMat& A, Vec& b, Vec& c, TermCrit& crit)
                        : m_A(A), m_c(c), m_b(b){ 
        m_crit = crit; 
    };

    Vec solve(ErCode& err){
        _initRandX(m_A.cols());

        //r = b - A * x;

        err = ErCode::NO_ERROR; 
        return m_x;//COPY
    };
};


#endif  // IPM_H