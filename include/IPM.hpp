#ifndef IPM_H
#define IPM_H

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#include <Eigen/SparseCore>
#include<Eigen/SparseCholesky>
#include<Eigen/SparseLU>

const double GAMMA = 0.3;


enum class ErCode{
    NO_ERROR,
    FACTARIZATION_ER,
    SOLVE_LIN_SYS_ER,
    SPECIAL_SITUATION,
    FACTARIZATION_LU
};

struct TermCrit {   // Terminate criteria
    int countMax;   
    double sq_eps; // eps^2
    double sq_delt;
};


class IPM 
{   
    using SpMat = Eigen::SparseMatrix<double> ;
    using Vec = Eigen::VectorXd;
    using Solver = Eigen::SimplicialLDLT<SpMat, Eigen::Lower>;

    SpMat& m_A;
    Vec& m_b, m_c;

    TermCrit m_crit;
    Vec m_x;  // неизвестный вектор
    Vec m_r; // невязка          //use for crit
    Vec m_g;  // g = c - A.T*u   //use for crit
    double m_sqnorm_r; //квадрат нормы невязки 
    int m_countfor = 0; //use for crit

    std::vector<double> m_vec_lambds; //вектор нужен для вычисления lambda

    Solver m_solver;

    void _initRandX(int len){
        //init gamma and random x 
        std::random_device rd;
        std::mt19937 gen(42);  // rd() // here you could set the seed, but std::random_device already does that
        std::uniform_real_distribution<double> dis(0., 10.0); //TODO set range

        m_x = Vec::NullaryExpr(len, [&](){return 1;}); //return dis(gen);
    }

    bool _checkTermCriteria(){

        return m_countfor >= m_crit.countMax 
               || (m_sqnorm_r < m_crit.sq_eps && m_g.squaredNorm() < m_crit.sq_delt);
    }

    
    ErCode _solveLinSys(SpMat& A, Vec& b, Vec& out){

        //ONLY FIRST TIME
        if(m_countfor == 0){
            m_solver.analyzePattern(A);
        }

        // Compute the numerical factorization 
        m_solver.factorize(A); 
        if(m_solver.info()!=Eigen::Success) {
            //zero pivot
            // so use LU
            Eigen::SparseLU<SpMat> solverLU;
            solverLU.compute(A);
            if(solverLU.info()!=Eigen::Success) {
                return ErCode::FACTARIZATION_LU;
            }

            solverLU.solve(b);
            out = m_solver.solve(b);

            return  ErCode::NO_ERROR;
        }

        out = m_solver.solve(b);
        if(m_solver.info()!=Eigen::Success) {
            return ErCode::SOLVE_LIN_SYS_ER;
        }

        return ErCode::NO_ERROR;

    }

    inline double _calculateLambda(Vec& s, Vec& x, ErCode& err){
        double lambda;
        err = ErCode::NO_ERROR;

        //find s_i < 0
        for(int i=0; i < s.size(); i++){
            if (s(i)<0){
                m_vec_lambds.push_back(-1*x(i)/s(i));
            };
        } 

        if (m_vec_lambds.size()==0){
            //not s_i < 0
            if(m_sqnorm_r <= m_crit.sq_eps) err = ErCode::SPECIAL_SITUATION; //if r = 0
            return 1.0;
        } else{
            lambda = *std::min_element(m_vec_lambds.begin(), m_vec_lambds.end()); 
            lambda *= GAMMA;
            m_vec_lambds.clear();

            if(m_sqnorm_r <= m_crit.sq_eps) {
                //r = 0
                return lambda;
            } else {
                //r != 0
                return std::min(1. , lambda);
            }
        }

    }

public:
    IPM(SpMat& A, Vec& b, Vec& c, TermCrit& crit)
                        : m_A(A), m_c(c), m_b(b){ 
        m_crit = crit; 
        m_vec_lambds.reserve(m_A.cols());
    };

    Vec solve(ErCode& err){
        _initRandX(m_A.cols());
        m_r = m_b - m_A * m_x;
        m_sqnorm_r = m_r.squaredNorm();
        //std::cout << "x = " << m_x<< std::endl; // DEBUG
        std::cout << "Невязка = " << m_sqnorm_r << std::endl; // DEBUG
        std::cout << "-------------------------------" << m_countfor <<
            "-------------------------------" << std::endl; // DEBUG

        int n = m_x.size();
        double lamb;
        Vec d(n), q(n), u(n), s(n);
        SpMat T(n,n);
        err = ErCode::NO_ERROR; 
        m_g = Vec::Zero(n);

        while (!_checkTermCriteria()){
            d = m_x.cwiseProduct(m_x); 

            T = m_A * d.asDiagonal(); 
            q = m_r + T * m_c;
            T = (T * m_A.transpose()).pruned(); //does a lot of unnecessary calculations TODO

            //T.triangularView<Eigen::Lower>(); = T * m_A.transpose() //dont work TODO
            //std::cout << m_x << std::endl;

            err = _solveLinSys(T, q, u);

            if (err!=ErCode::NO_ERROR){
                std::cout << Eigen::MatrixXd(T) << std::endl; //
                return m_x;
            }

            m_g = m_c - m_A.transpose() * u;
            s = -1 * d.asDiagonal() * m_g; //m_g need for termCrit

            lamb = _calculateLambda(s, m_x, err);

            if (err!=ErCode::NO_ERROR){
                return m_x;
            }

            m_x += lamb * s; 
            m_r = (1 - lamb) * m_r;
            m_sqnorm_r = m_r.squaredNorm();
            std::cout << "Невязка = " << m_sqnorm_r << std::endl; // DEBUG
            std::cout << "g.norm = " << m_g.squaredNorm() << std::endl; // DEBUG
            std::cout << "cT*x = " << m_c.transpose() * m_x << std::endl; // DEBUG
            std::cout << "bT*u = " << m_b.transpose() * u << std::endl; // DEBUG
            std::cout << "lambda = " << lamb << std::endl; // DEBUG
            std::cout << "-------------------------------" << m_countfor+1 <<
            "-------------------------------" << std::endl; // DEBUG

            m_countfor += 1;
        }
        

        
        return m_x;//COPY
    };
};


#endif  // IPM_H