#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>

//Function prototypes
float read_enuc (std::string fname);
void read_2D (std::string fname, gsl_matrix * A, int nbf);
void read_ERI (std::string fname, double ****A, int nbf);
void build_D (gsl_matrix * D, gsl_matrix * C, int nocc, int nbf);
void build_F (gsl_matrix * F, gsl_matrix * D, 
              gsl_matrix * Hcore, double ****A,
              int nbf);
double compute_E (gsl_matrix * F, gsl_matrix * D, gsl_matrix * Hcore, 
                  int nbf); 


int main () {
//    int natom = 3; //water
    int nbf = 7; //sto-3g water
    int nocc = 5;
    int maxiter = 50;
    double enuc = read_enuc ("enuc.dat"); //nuclear repulsion
    double Em = 0.0;
    double E = 0.0;

    //>Memory allocation
    //>2D arrays
    gsl_vector * eval = gsl_vector_alloc (nbf);
    gsl_matrix * AO_overlap = gsl_matrix_alloc (nbf, nbf);
    gsl_matrix * AO_potential = gsl_matrix_alloc (nbf, nbf);
    gsl_matrix * AO_kinetic = gsl_matrix_alloc (nbf, nbf);
    gsl_matrix * evec = gsl_matrix_alloc (nbf, nbf);
    gsl_matrix * lambda = gsl_matrix_alloc (nbf, nbf);
    gsl_matrix * lambdapf = gsl_matrix_alloc (nbf, nbf);
    gsl_matrix * F = gsl_matrix_alloc (nbf, nbf);
    gsl_matrix * Fp = gsl_matrix_alloc (nbf, nbf);
    gsl_matrix * Hcore = gsl_matrix_alloc (nbf, nbf);
    gsl_matrix * D = gsl_matrix_alloc (nbf, nbf);

    gsl_vector_set_zero (eval);
    gsl_matrix_set_zero (AO_overlap);
    gsl_matrix_set_zero (AO_potential);
    gsl_matrix_set_zero (AO_kinetic);
    gsl_matrix_set_zero (evec);
    gsl_matrix_set_zero (lambda);
    gsl_matrix_set_zero (lambdapf);
    gsl_matrix_set_zero (F);
    gsl_matrix_set_zero (Fp);
    gsl_matrix_set_zero (Hcore);
    gsl_matrix_set_zero (D);

    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (nbf);
    //<2D arrays

    //>4D AO ERI
    double ****AO_eri = new double*** [nbf];
    for (int i = 0; i < nbf; i ++ ) {
        AO_eri[i] = new double** [nbf];
        for (int j = 0 ; j< nbf; j++) {
            AO_eri[i][j] = new double* [nbf];
            for (int k = 0; k < nbf; k++) {
                AO_eri[i][j][k] = new double[nbf];
            }
        }
    }
    //<4D AO ERI
    //<Memory allocation

    //> read integrals
    read_2D ("overlap.dat", AO_overlap, nbf);
    read_2D ("kinetic.dat", AO_kinetic, nbf);
    read_2D ("potential.dat", AO_potential, nbf);
    read_ERI ("eri.dat", AO_eri, nbf);
    //< read integrals

    //>build orthogonalizer
    gsl_eigen_symmv (AO_overlap, eval, evec, w);
    for (int i = 0; i < nbf; i++) {
        gsl_matrix_set (lambda, i, i, 1.0/std::sqrt(gsl_vector_get (eval, i)));
    }
    
    gsl_blas_dgemm (CblasNoTrans, CblasTrans,
                    1.0, lambda, evec,
                    0.0, lambdapf);
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                    1.0, evec, lambdapf,
                    0.0, lambda);

    //>round to zero w/in eps and make Hcore from v + t
    double v,t;
    for (int i = 0; i < nbf; i++) {
        for (int j = 0; j < nbf; j++) {
            if (std::fabs(gsl_matrix_get (lambda, i, j)) < 1E-14) {
                gsl_matrix_set (lambda, i, j, 0.0);
            }
            t = gsl_matrix_get (AO_kinetic, i, j);
            v = gsl_matrix_get (AO_potential, i, j);
            gsl_matrix_set (Hcore, i, j, t + v);
        }
    }
    //<round to zero w/in eps and make Hcore from v + t
    //<build orthogonalizer

    build_F ( F, D, Hcore, AO_eri, nbf);
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                    1.0, F, lambda,
                    0.0, lambdapf);

    gsl_blas_dgemm (CblasTrans, CblasNoTrans,
                    1.0, lambda, lambdapf,
                    0.0, Fp);
    std::cout << gsl_matrix_get (Fp, 0, 0);
    //<build guess fock matrix

    //>Diag Fock matrix
    gsl_eigen_symmv (Fp, eval, evec, w);
    gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);
    //<Diag Fock matrix
    
    //>Transform C' (orthog) -> C (non-orthog)
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                    1.0, lambda, evec,
                    0.0, lambdapf);
    //<Transform C' (orthog) -> C (non-orthog)
    
    //>Build Density matrix
    //double tsum = 0.0;
    build_D (D, lambdapf, nocc, nbf);
    //<Build Density matrix


    //>Compute initial energy
    Em = compute_E ( F, D, Hcore, nbf);
    std::cout << "Energy " << Em << std::endl;
    //<Compute initial energy

    for (int i = 0; i < maxiter; i++) {
        //>Build new Fock matrix
        build_F ( F, D, Hcore, AO_eri, nbf);
        //<Build new Fock matrix
        //>Transform new Fock matrix
        gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                        1.0, F, lambda,
                        0.0, lambdapf);
        gsl_blas_dgemm (CblasTrans, CblasNoTrans,
                        1.0, lambda, lambdapf,
                        0.0, Fp);
        //<Transform new Fock matrix
        
        //>Diag new Fock matrix
        gsl_eigen_symmv (Fp, eval, evec, w);
        gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);

        //>Transform C' (orthog) -> C (non-orthog)
        gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                        1.0, lambda, evec,
                        0.0, lambdapf);
        //<Transform C' (orthog) -> C (non-orthog)

        //>Build density matrix
        build_D ( D, lambdapf, nocc, nbf);
        //<Build density matrix

        E = compute_E (F, D, Hcore, nbf);
        std::cout << std::setprecision(15) << "Energy " << E + enuc << std::endl;
    }


    //>Memory deallocation
    //> 2D matrices
    gsl_vector_free (eval);
    gsl_matrix_free (AO_overlap);
    gsl_matrix_free (AO_kinetic);
    gsl_matrix_free (AO_potential);
    gsl_matrix_free (evec);
    gsl_matrix_free (lambda);
    gsl_matrix_free (lambdapf);
    gsl_matrix_free (F);
    gsl_matrix_free (Fp);
    gsl_matrix_free (Hcore);
    gsl_matrix_free (D);
    
    gsl_eigen_symmv_free (w); 
    //< 2D matrices
    //>4D AO ERI
    for (int i = 0; i < nbf; i++) {
        for (int j = 0; j < nbf; j++) {
            for (int k = 0; k < nbf; k++) {
                delete[] AO_eri[i][j][k];
            }
            delete[] AO_eri[i][j];
        }
        delete[] AO_eri[i];
    }
    delete[] AO_eri;

    //<4D AO ERI
    //<Memory deallocation
    return 0;
}

float read_enuc (std::string fname) {
    //input std::string fname; filename for Vnuc
    //output double enuc; Vnuc from filename
    //I'm not clear whether it is ok to return 
    //a local variable (enuc) by value or  not.
    //Seems that by reference/pointer is bad?
    float enuc;
    std::ifstream fnuc(fname);
    fnuc >> enuc;
    return enuc;
}

void read_2D (std::string fname, gsl_matrix * A, int nbf) {
    //input double **A; pre-allocated nbf X nbf array
    //      int nbf; number of basis functions
    //output void;
    std::ifstream arr2d(fname);
    int ti,tj;
    float s;

    for (int i = 0; i < nbf; i++) {
        for (int j = 0; j <= i; j++) {
            arr2d >> ti >> tj >> s;
            gsl_matrix_set (A, i, j, s);
            gsl_matrix_set (A, j, i, s);
        }
    }
}

//TODO: figure out how to use the 1D compound index
void read_ERI (std::string fname, double ****A, int nbf) {
    //input *A; 1D array in compressed notation
    //      fname; filename
    //output void;
    //notation
    // A[u][v][l][s] = (uv|ls)
    // 8fold permutational symmetry:
    // (uv|ls) = (vu|ls) = (uv|sl) = (vu|sl) = (ls|uv) = (sl|uv) = (sl|vu) = (ls|vu)

    std::ifstream eri(fname);
    int u,v,l,s;
    double temp;
    for (int i = 0; i < 228; i++) {
        eri >> u >> v >> l >> s >> temp;
        std::cout << u << v << l << s << temp << "\n";
        u -= 1;
        v -= 1;
        l -= 1;
        s -= 1;
        A[u][v][l][s] = temp;
        A[u][v][s][l] = temp;

        A[v][u][l][s] = temp;
        A[v][u][s][l] = temp;
        
        A[l][s][u][v] = temp;
        A[l][s][v][u] = temp;

        A[s][l][u][v] = temp;
        A[s][l][v][u] = temp;
    } 
}

void build_D (gsl_matrix * D, gsl_matrix * C, int nocc, int nbf) {
    std::cout << "D \n";
    double tsum = 0.0;
    for (int u = 0.0; u < nbf; u++) {
        for (int v = 0.0; v < nbf; v++) {
            tsum = 0.0;
            for (int m = 0; m < nocc; m++) {
                tsum += gsl_matrix_get (C, u, m)
                      * gsl_matrix_get (C, v, m);
            }
            if (std::fabs(tsum) < 1E-14) {
                tsum = 0.0;
            }
            gsl_matrix_set (D, u, v, tsum);
            std::cout << gsl_matrix_get (D, u, v) << " ";
        }
        std::cout << "\n";
    }
}

void build_F (gsl_matrix * F, gsl_matrix * D, 
              gsl_matrix * Hcore, double ****AO_eri,
              int nbf) {
    std::cout << "F \n";
    double tsum = 0.0;
    for (int u = 0; u < nbf; u++) {
        for (int v = 0; v < nbf; v++) {
            tsum = 0.0;
            for (int l = 0; l < nbf; l++) {
                for (int s = 0; s < nbf; s++) {
                    tsum += ( gsl_matrix_get (D, l, s)
                            * (2.0*AO_eri[u][v][l][s] - AO_eri[u][l][v][s]));
                }
            }
            gsl_matrix_set (F, u, v,
                            gsl_matrix_get (Hcore, u, v) + tsum);
            std::cout << gsl_matrix_get (F, u, v) << " ";

        }
        std::cout << "\n";
    }
}
    
double compute_E (gsl_matrix * F, gsl_matrix * D, gsl_matrix * Hcore, 
                  int nbf) {
    double Em = 0.0;
    for (int u = 0; u < nbf; u++) {
        for (int v = 0; v < nbf; v++) {
            Em += gsl_matrix_get (D, v, u)
                * ( gsl_matrix_get (Hcore, u, v)
                  + gsl_matrix_get (F, u, v));
        }
    }
    return Em; 
}
