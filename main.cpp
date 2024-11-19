#include <iostream>
#include <filesystem>
#include <string>

#include "CoinMpsIO.hpp"
#include <Eigen/SparseCore>

#include "IPM.hpp"


using SpMat = Eigen::SparseMatrix<double> ;
using Vec = Eigen::VectorXd;


/**
 * @brief Функция инициализирует данные задачи линейного программирования`eg_A`,`eg_b`,`eg_c`. 
 *     Данные считывааються из файла *.mps который должен располагаться по пути `filepath`
 * 
 * @param filepath путь к файлу *.mps
 * @param eg_A разреженная не иниц. матрица А
 * @param eg_b плотный не иниц. вектор b
 * @param eg_c плотный не иниц. вектор с
 * @return Код завершения
 */
int initLpProblem(std::string& filepath, SpMat& eg_A, Vec& eg_b, Vec& eg_c){
    // init reader 
    CoinMpsIO m;

    //read data
    int numErr = m.readMps(filepath.c_str(), "mps");
    
    if(numErr){
        //std::cout << "Error read mps. numErr = " << numErr << std::endl;
        return numErr;
    }

    // get b and c
    int len_c = m.getNumCols();
    int len_b = m.getNumRows();

    const double * ptr_c = m.getObjCoefficients();
    const double * ptr_b = m.getRightHandSide();

    eg_c = Eigen::Map<const Vec>(ptr_c, len_c); // here COPY data Map->VectorXd
    eg_b = Eigen::Map<const Vec>(ptr_b, len_b); // here COPY data Map->VectorXd


    // get A
    const CoinPackedMatrix * A = m.getMatrixByCol();

    int nnz = m.getNumElements();
    const double* ptr_val = A->getElements(); 
    const int * ptr_indices = A->getIndices();
    const int * ptr_starts = A->getVectorStarts();
    int len_starts = A->getMajorDim() + 1;

    eg_A = Eigen::Map<const SpMat>(A->getNumRows(), A->getNumCols(), // here COPY data Map->SpMat
                             nnz, ptr_starts, ptr_indices, ptr_val);

    return 0;
}


int main(int, char**){
    namespace Eg = Eigen;
    using std::cout, std::endl;

    //read data Lp
    SpMat A;
    Vec b, c;

    std::string curr_dir = std::filesystem::current_path();
    std::string fp = curr_dir+"/test.mps";

    int err_read = initLpProblem(fp, A, b, c);

    if(err_read){
        cout << "Error read mps file. Code error = " << err_read << endl;
        exit(0);
    }

    //solve Lp problem
    Vec ans;
    ErCode errIPM;
    TermCrit crit{.countMax=200, .sq_eps=0.1, .sq_delt=0.1};
    IPM ipm_solver(A, b, c, crit);

    ans = ipm_solver.solve(errIPM);
 
    cout << "ErCode = " << static_cast<std::underlying_type<ErCode>::type>(errIPM) << endl ;
    
    cout << "ans=" << ans << endl ;

    return 0;
}