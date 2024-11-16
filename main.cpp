#include <iostream>
#include <filesystem>
#include <string>

#include <Eigen/SparseCore>
#include "CoinMpsIO.hpp"

using std::cout, std::endl;

typedef Eigen::SparseMatrix<double> SpMat;
using Eigen::VectorXd;

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
int initLpProblem(std::string& filepath, SpMat& eg_A, VectorXd& eg_b, VectorXd& eg_c){
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

    eg_c = Eigen::Map<const VectorXd>(ptr_c, len_c); // here COPY data Map->VectorXd
    eg_b = Eigen::Map<const VectorXd>(ptr_b, len_b); // here COPY data Map->VectorXd


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
    
    std::string curr_dir = std::filesystem::current_path();
    std::string fp = curr_dir+"/test.mps";
    
    SpMat A;
    VectorXd b, c;

    int err = initLpProblem(fp, A, b,c);

    if(err){
        std::cout << "Error when read mps. Code error = " << err << std::endl;
        exit(0);
    }

    cout << A << endl;
}