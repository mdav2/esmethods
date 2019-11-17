#include "HF.h"

HF::tensor4p::tensor4p(int parraysize)
    : //arraysize{parraysize}
    myarray{ new double*** [parraysize] }
{
    this->arraysize = parraysize;
    for (int i = 0; i < this->arraysize; i++) {
        myarray[i] = new double** [this->arraysize];
            for (int j = 0; j < this->arraysize; j++) {
                myarray[i][j] = new double* [this->arraysize];
                for (int k = 0; k < this->arraysize; k++) {
                    myarray[i][j][k] =  new double[this->arraysize];
                    //for (int l = 0; l < this->arraysize; l++) {
                    //    myarray[i][j][k][l] = 0.0;
                    //}
                }
            }
    }
}
HF::tensor4p::~tensor4p() {
    for (int i = 0; i < this->arraysize; i++) {
        for (int j = 0; j < this->arraysize; j++) {
            for (int k = 0; k < this->arraysize; k++) {
                delete[] this->myarray[i][j][k];
            }
            delete[] this->myarray[i][j];
        }
        delete[] this->myarray[i];
    }
}

HF::wfn::wfn(int nbf) 
{  
    this->eval = gsl_vector_alloc(nbf);
    this->AO_overlap = gsl_matrix_alloc (nbf, nbf);
    this->AO_potential = gsl_matrix_alloc (nbf, nbf);
    this->AO_kinetic = gsl_matrix_alloc (nbf, nbf);
    this->evec = gsl_matrix_alloc (nbf, nbf);
    this->lambda = gsl_matrix_alloc (nbf, nbf);
    this->lambdapf = gsl_matrix_alloc (nbf, nbf);
    this->F = gsl_matrix_alloc (nbf, nbf);
    this->Fp = gsl_matrix_alloc (nbf, nbf);
    this->FPROTECT = gsl_matrix_alloc (nbf, nbf);
    this->Hcore = gsl_matrix_alloc (nbf, nbf);
    this->D = gsl_matrix_alloc (nbf, nbf);
    this->C = gsl_matrix_alloc (nbf, nbf);
    this->AO_eri = new tensor4p (nbf);
    this->MO_eri = new tensor4p (nbf);
    this->w = gsl_eigen_symmv_alloc (nbf);
    gsl_vector_set_zero (this->eval);
    gsl_matrix_set_zero (this->AO_overlap);
    gsl_matrix_set_zero (this->AO_potential);
    gsl_matrix_set_zero (this->AO_kinetic);
    gsl_matrix_set_zero (this->evec);
    gsl_matrix_set_zero (this->lambda);
    gsl_matrix_set_zero (this->lambdapf);
    gsl_matrix_set_zero (this->F);
    gsl_matrix_set_zero (this->Fp);
    gsl_matrix_set_zero (this->FPROTECT);
    gsl_matrix_set_zero (this->Hcore);
    gsl_matrix_set_zero (this->D);
    gsl_matrix_set_zero (this->C);
    this->nbf = nbf;
    this->nocc = 5;
    this->natom = 3;
    this->maxiter = 50;
}

HF::wfn::~wfn() 
{
    gsl_eigen_symmv_free(this->w);
    gsl_vector_free(this->eval);
    gsl_matrix_free(this->AO_overlap);
    gsl_matrix_free(this->AO_potential);
    gsl_matrix_free(this->AO_kinetic);
    gsl_matrix_free(this->evec);
    gsl_matrix_free(this->lambda);
    gsl_matrix_free(this->lambdapf);
    gsl_matrix_free(this->F);
    gsl_matrix_free(this->Fp);
    gsl_matrix_free(this->FPROTECT);
    gsl_matrix_free(this->Hcore);
    gsl_matrix_free(this->D);
    gsl_matrix_free(this->C);
    delete this->AO_eri;
    delete this->MO_eri;
} 

double HF::wfn::do_SCF (void) {
    //int natom = 3; //water
    //int nbf = 7; //sto-3g water
    //int nocc = 5;
    //int maxiter = 50;
    std::string enuc = "enuc.dat";
    this->read_enuc (enuc); //nuclear repulsion
    this->E = 0.0;

    std::string overlap = "overlap.dat";
    std::string kinetic = "kinetic.dat";
    std::string potential = "potential.dat";
    std::string eri = "eri.dat";
    this->read_2D (overlap, this->AO_overlap);
    this->read_2D (kinetic, this->AO_kinetic);
    this->read_2D (potential, this->AO_potential);
    this->read_ERI (eri);//, AO_eri, nbf);
    //< read integrals

    //>build orthogonalizer
    gsl_eigen_symmv (this->AO_overlap, this->eval, this->evec, this->w);
    for (int i = 0; i < this->nbf; i++) {
        gsl_matrix_set (this->lambda, i, i, 1.0/std::sqrt(gsl_vector_get (this->eval, i)));
    }
    
    gsl_blas_dgemm (CblasNoTrans, CblasTrans,
                    1.0, this->lambda, this->evec,
                    0.0, this->lambdapf);
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                    1.0, this->evec, this->lambdapf,
                    0.0, this->lambda);

    //>round to zero w/in eps and make Hcore from v + t
    double v,t;
    for (int i = 0; i < this->nbf; i++) {
        for (int j = 0; j < this->nbf; j++) {
            if (std::fabs(gsl_matrix_get (this->lambda, i, j)) < 1E-14) {
                gsl_matrix_set (this->lambda, i, j, 0.0);
            }
            t = gsl_matrix_get (this->AO_kinetic, i, j);
            v = gsl_matrix_get (this->AO_potential, i, j);
            gsl_matrix_set (this->Hcore, i, j, t + v);
        }
    }
    //<round to zero w/in eps and make Hcore from v + t
    //<build orthogonalizer


    //>build guess fock matrix
    this->build_F (); 
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                    1.0, this->F, this->lambda,
                    0.0, this->lambdapf);

    gsl_blas_dgemm (CblasTrans, CblasNoTrans,
                    1.0, this->lambda, this->lambdapf,
                    0.0, this->Fp);

    //>Diag Fock matrix
    gsl_matrix_memcpy(this->FPROTECT, this->Fp);
    gsl_eigen_symmv (this->Fp, this->eval, this->evec, this->w);
    gsl_eigen_symmv_sort (this->eval, this->evec, GSL_EIGEN_SORT_VAL_ASC);
    gsl_matrix_memcpy(this->Fp, this->FPROTECT);
    //<Diag Fock matrix
    
    //>Transform C' (orthog) -> C (non-orthog)
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                    1.0, this->lambda, this->evec,
                    0.0, this->C);
    //<Transform C' (orthog) -> C (non-orthog)
    

    //>Build Density matrix
    this->build_D ();
    //<Build Density matrix


    //>Compute initial energy
    this->E = this->compute_E () + this->enuc;
    //<Compute initial energy

    for (int i = 0; i < this->maxiter; i++) {
        //>Build new Fock matrix
        this->build_F ();
        //<Build new Fock matrix

        //>Transform new Fock matrix
        gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                        1.0, this->F, this->lambda,
                        0.0, this->lambdapf);
        gsl_blas_dgemm (CblasTrans, CblasNoTrans,
                        1.0, this->lambda, this->lambdapf,
                        0.0, this->Fp);
        //<Transform new Fock matrix
        
        //>Diag new Fock matrix
        gsl_eigen_symmv (this->Fp, this->eval, this->evec, this->w);
        gsl_eigen_symmv_sort (this->eval, this->evec, GSL_EIGEN_SORT_VAL_ASC);

        //>Transform C' (orthog) -> C (non-orthog)
        gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                        1.0, this->lambda, this->evec,
                        0.0, this->C);
        //<Transform C' (orthog) -> C (non-orthog)

        //>Build density matrix
        this->build_D ();
        //<Build density matrix

        this->E = this->compute_E () + this->enuc;
    }
    
    return this->E + this->enuc;
}


void HF::wfn::AOtoMOnoddy (void) {
    double tsum = 0.0;
    for(int i = 0; i < this->nbf; i++) {
        for (int j = 0; j < this->nbf; j++) {
            for (int k = 0; k < this->nbf; k++) {
                for (int l = 0; l < this->nbf; l++) {
                    for (int p = 0; p < this->nbf; p++) {
                        for (int q = 0; q < this->nbf; q++) {
                            for (int r = 0; r < this->nbf; r++) {
                                for (int s = 0; s < this->nbf; s++) {
                                    this->MO_eri->myarray[i][j][k][l] += gsl_matrix_get(this->C,p,i)
                                          * gsl_matrix_get(this->C,q,j)
                                          * this->AO_eri->myarray[p][q][r][s]
                                          * gsl_matrix_get(this->C,r,k)
                                          * gsl_matrix_get(this->C,s,l);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void HF::wfn::do_MP2 (void) {
    this->AOtoMOnoddy();
    this->Ecorr = 0;
    for (int i = 0; i < this->nocc; i++) {
        for (int j = 0; j < this->nocc; j++) {
            for (int a = (this->nocc); a < this->nbf; a++) {
                for (int b = (this->nocc); b < this->nbf; b++) {
                    this->Ecorr +=    this->MO_eri->myarray[i][a][j][b]
                                 * (2*this->MO_eri->myarray[i][a][j][b]
                                    - this->MO_eri->myarray[i][b][j][a])
                                 / (gsl_vector_get (this->eval, i) 
                                   +gsl_vector_get (this->eval, j) 
                                   -gsl_vector_get (this->eval, a)
                                   -gsl_vector_get (this->eval, b));
                }
            }
        }
    }
}

void HF::wfn::read_enuc (std::string fname) {
    //input std::string fname; filename for Vnuc
    //output double enuc; Vnuc from filename
    //I'm not clear whether it is ok to return 
    //a local variable (enuc) by value or  not.
    //Seems that by reference/pointer is bad?
    std::ifstream fnuc(fname);
    fnuc >> this->enuc;
}

void HF::wfn::read_2D (std::string fname, gsl_matrix * A) {
    //input double **A; pre-allocated nbf X nbf array
    //      int nbf; number of basis functions
    //output void;
    std::ifstream arr2d(fname);
    int ti,tj;
    double s;

    for (int i = 0; i < this->nbf; i++) {
        for (int j = 0; j <= i; j++) {
            arr2d >> std::setprecision(15) >> ti >> tj >> s;
            gsl_matrix_set (A, i, j, s);
            gsl_matrix_set (A, j, i, s);
        }
    }
}

void HF::wfn::read_ERI (std::string fname) {
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
        u -= 1;
        v -= 1;
        l -= 1;
        s -= 1;
        this->AO_eri->myarray[u][v][l][s] = temp;
        this->AO_eri->myarray[u][v][s][l] = temp;

        this->AO_eri->myarray[v][u][l][s] = temp;
        this->AO_eri->myarray[v][u][s][l] = temp;

        this->AO_eri->myarray[l][s][u][v] = temp;
        this->AO_eri->myarray[l][s][v][u] = temp;

        this->AO_eri->myarray[s][l][u][v] = temp;
        this->AO_eri->myarray[s][l][v][u] = temp;
    } 
}

void HF::wfn::build_D (void) {
    double tsum = 0.0;
    for (int u = 0.0; u < this->nbf; u++) {
        for (int v = 0.0; v < this->nbf; v++) {
            tsum = 0.0;
            for (int m = 0; m < this->nocc; m++) {
                tsum += gsl_matrix_get (this->C, u, m)
                      * gsl_matrix_get (this->C, v, m);
            }
            if (std::fabs(tsum) < 1E-14) {
                tsum = 0.0;
            }
            gsl_matrix_set (this->D, u, v, tsum);

        }
    }
}

void HF::wfn::build_F (void) {
    double tsum = 0.0;
    for (int u = 0; u < this->nbf; u++) {
        for (int v = 0; v < this->nbf; v++) {
            tsum = 0.0;
            for (int l = 0; l < this->nbf; l++) {
                for (int s = 0; s < this->nbf; s++) {
                    tsum += ( gsl_matrix_get (this->D, l, s)
                            * (2.0*this->AO_eri->myarray[u][v][l][s] - this->AO_eri->myarray[u][l][v][s]));
                }
            }
            gsl_matrix_set (this->F, u, v,
                            gsl_matrix_get (this->Hcore, u, v) + tsum);

        }
    }
}
    
double HF::wfn::compute_E (void) {
    double Em = 0.0;
    for (int u = 0; u < this->nbf; u++) {
        for (int v = 0; v < this->nbf; v++) {
            Em += gsl_matrix_get (this->D, v, u)
                * ( gsl_matrix_get (this->Hcore, u, v)
                  + gsl_matrix_get (this->F, u, v));
        }
    }
    return Em; 
}
