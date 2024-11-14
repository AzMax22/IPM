#include <iostream>
#include <filesystem>
#include <vector>

#include "CoinMpsIO.hpp"

template<typename T>
void print_carray(T * arr, int len, std::string name){
     std::cout << name << " = [";

    for(int i=0; i < len; i++){
        std::cout << arr[i] << " ";
    }

    std::cout << "]" << std::endl;
}


int main(int, char**){
    // init reader 
    CoinMpsIO m;

    std::string curr_dir = std::filesystem::current_path();
    std::string fn = curr_dir+"/test.mps";
    int numErr = m.readMps(fn.c_str(),"mps");
    
    if(numErr){
        std::cout << "Error read mps. numErr = " << numErr << std::endl;
        exit(0);
    }

    // get data
    int len_c = m.getNumCols();
    int len_b = m.getNumRows();
    int len_el_A = m.getNumElements();

    const double * c = m.getObjCoefficients();
    const double * b = m.getRightHandSide();
    const CoinPackedMatrix * A = m.getMatrixByCol();

    const double* el_A = A->getElements(); // stored by columns

    //std::vector<double> vec_c(c, c + len_c);

    print_carray(c, len_c, "c");
    print_carray(b, len_b, "b");
    print_carray(el_A, len_el_A, "el_A");
    print_carray(A->getIndices(), len_el_A, "ind");
    print_carray(A->getVectorStarts(), A->getMajorDim() + 1, "vstarts");

    //std::cout << "c = " << A->getIndices() << std::endl;
    //std::cout << "len_b = " << len_b << std::endl;

    //std::cout << "nr = " <<  m.getNumElements() << std::endl;
}
