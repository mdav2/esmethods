#include <fstream>
#include <iostream>
#include <string>
#include "phf.h"

namespace phf {
namespace butter {

template <typename T>
class fint4 
    //stores a tensor4 
    //
    //for e.g. tijab:
    //nd1 = nocc
    //nd2 = nvir
    //
    //for e.g. moeri:
    //nd1 = nbf
    //nd2 = nbf
    private:
        int sz;
        int fnum; //<-- e.g. fnum = 1, stored in file cio.1 
        int nd1; //<-- set these in constructor (
        int nd2; //<-- 
        int szab; //<-- (size of a 2D array of a,b (nd1^2))
        int szij; //<-- (size of a 2D array of i,j (nd2^2))
        bool keepopen;
        std::string fname;
        std::fstream file;
        int ij ( int i, int j ) {
            return i*this->nd1 + j;
        }
        int ab ( int a, int b ) {
            return a*this->nd2 + b; 
        }
        int ijab ( int i, int j, int a, int b ) {
            int ij = this->ij(i,j);
            int ab = this->ab(a,b);
            return ij*szab + ab;
        }
        void autopen (void) {
            if (!file.is_open()) {
                this->file.open (fname.c_string, ios::binary);
            }
            if (!this->file) {
                std::cout << "Error opening file!\n";
            }
        }
        void autoclose (void) {
            if (!this->keepopen) {
                this->file.close();
            }
        }

    public:
    
        void set ( int i, int j, int a, int b, double val ) {
            //simplest implementation, just a random access file
            this->autopen();
            int ijab = this->ijab(i,j,a,b);
            this->file.seekp (ijab*this->sz);
            this->file.write (&val, this->sz);
            this->autoclose();
        }
        
        T get ( int i, int j, int a, int b ) {
            //simplest implementation, just a random access file
            this->autopen();
            int ijab = this->ijab(i,j,a,b);
            double result = 0.0;
            this->file.seekg (ijab*this->sz);
            this->file.read (&result, this->sz);
            this->autoclose();
            return result;
        }    
        
        fint4 (int num) {
            this->fnum = num;
            this->sz = sizeof(T);
        }

        ~fint4 ( ) {
        }
}
}
