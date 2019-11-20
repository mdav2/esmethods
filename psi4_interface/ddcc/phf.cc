#include "phf.h"
#include "psi4/libmints/matrix.h"

double phf::kron ( int p, int q ) {
    return (p == q );
}

phf::tensor4p::tensor4p(int parraysize)
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
                    for (int l = 0; l < this->arraysize; l++) {
                        myarray[i][j][k][l] = 0.0;
                    }
                }
            }
    }
}
phf::tensor4p::~tensor4p() {
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

phf::phfwfn::phfwfn(int nbf, int nocc, int natom) 
{  
    int nmo = nbf*2;
    this->nbf = nbf;
    this->nocc = nocc; //5
    this->noccso = this->nocc*2;
    this->natom = natom;//3
    this->nmo = nmo;
    this->maxiter = 50;
    SharedMatrix this->evec(rows=nbf,cols=nbf);
    SharedMatrix this->SO_eri(rows=nmo,cols=nmo);
//    this->evec = gsl_matrix_alloc (nbf, nbf);
//    this->lambda = gsl_matrix_alloc (nbf, nbf);
//    this->lambdapf = gsl_matrix_alloc (nbf, nbf);
//    this->F = gsl_matrix_alloc (nbf, nbf);
//    this->FSO = gsl_matrix_alloc (nmo, nmo);
//    this->Fp = gsl_matrix_alloc (nbf, nbf);
//    this->FPROTECT = gsl_matrix_alloc (nbf, nbf);
//    this->Hcore = gsl_matrix_alloc (nbf, nbf);
//    this->HcoreSO = gsl_matrix_alloc (nmo, nmo);
//    this->D = gsl_matrix_alloc (nbf, nbf);
//    this->C = gsl_matrix_alloc (nbf, nbf);
//    this->FF = gsl_matrix_alloc (nmo, nmo);
//    this->tautijab = new tensor4p (nmo);
//    this->tauijab = new tensor4p (nmo);
//    this->tia = gsl_matrix_alloc (nmo, nmo);
//    this->tia_new = gsl_matrix_alloc (nmo, nmo);
//    this->tijab = new tensor4p (nmo);
//    this->tijab_new = new tensor4p (nmo);
//    this->W = new tensor4p (nmo);
//    this->Dia = gsl_matrix_alloc (nmo, nmo);
//    this->Dijab = new tensor4p (nmo);

//    this->w = gsl_eigen_symmv_alloc (nbf);
//    gsl_vector_set_zero (this->eval);
//    gsl_matrix_set_zero (this->AO_overlap);
//    gsl_matrix_set_zero (this->AO_potential);
//    gsl_matrix_set_zero (this->AO_kinetic);
//    gsl_matrix_set_zero (this->evec);
//    gsl_matrix_set_zero (this->lambda);
//    gsl_matrix_set_zero (this->lambdapf);
//    gsl_matrix_set_zero (this->F);
//    gsl_matrix_set_zero (this->FSO);
//    gsl_matrix_set_zero (this->Fp);
//    gsl_matrix_set_zero (this->FPROTECT);
//    gsl_matrix_set_zero (this->Hcore);
//    gsl_matrix_set_zero (this->HcoreSO);
//    gsl_matrix_set_zero (this->D);
//    gsl_matrix_set_zero (this->C);
//    gsl_matrix_set_zero (this->FF);
//    gsl_matrix_set_zero (this->tia);
//    gsl_matrix_set_zero (this->tia_new);
//    gsl_matrix_set_zero (this->Dia);
}

phf::phfwfn::~phfwfn() 
{
//    gsl_eigen_symmv_free(this->w);
//    gsl_vector_free(this->eval);
//    gsl_matrix_free(this->AO_overlap);
//    gsl_matrix_free(this->AO_potential);
//    gsl_matrix_free(this->AO_kinetic);
//    gsl_matrix_free(this->evec);
//    gsl_matrix_free(this->lambda);
//    gsl_matrix_free(this->lambdapf);
//    gsl_matrix_free(this->F);
//    gsl_matrix_free(this->FSO);
//    gsl_matrix_free(this->Fp);
//    gsl_matrix_free(this->FPROTECT);
//    gsl_matrix_free(this->HcoreSO);
//    gsl_matrix_free(this->Hcore);
//    gsl_matrix_free(this->D);
//    gsl_matrix_free(this->C);
//    gsl_matrix_free(this->FF);
//    gsl_matrix_free(this->tia);
//    gsl_matrix_free(this->tia_new);
//    gsl_matrix_free(this->Dia);
//    delete this->AO_eri;
//    delete this->MO_eri;
//    delete this->SO_eri;
//    delete this->tautijab;
//    delete this->tauijab;
//    delete this->tijab;
//    delete this->tijab_new;
//    delete this->W;
//    delete this->Dijab;
} 
<<-- ENERGY COMPUTATIONS -->>//
double phf::wfn::do_SCF (void) {
    std::cout << "Doing SCF\n";
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
    std::cout << "Reading overlap\n";
    this->read_2D (overlap, this->AO_overlap);
    std::cout << "Reading kinetic\n";
    this->read_2D (kinetic, this->AO_kinetic);
    std::cout << "Reading potential\n";
    this->read_2D (potential, this->AO_potential);
    std::cout << "Reading ERI\n";
    this->read_ERI (eri);//, AO_eri, nbf);
    std::cout << "done Reading integrals\n";
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
    std::cout << "making hcore\n";
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
    std::cout << "making fock matrix\n";
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
    std::cout << std::setprecision(15) << this->E << "\n";

    for (int i = 0; i < 35; i++) {
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
        gsl_matrix_memcpy(this->FPROTECT, this->Fp);
        gsl_eigen_symmv (this->Fp, this->eval, this->evec, this->w);
        gsl_eigen_symmv_sort (this->eval, this->evec, GSL_EIGEN_SORT_VAL_ASC);
        gsl_matrix_memcpy(this->Fp, this->FPROTECT);
        //for ( int i = 0; i < this->nbf; i++ ) {
        //    for ( int j = 0; j < this->nbf; j++ ) {
        //        std::cout << std::setprecision(15) << gsl_matrix_get ( this->Fp, i, j ) << " ";
        //    }
        //    std::cout << "\n";
        //}
        //>Transform C' (orthog) -> C (non-orthog)
        gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                        1.0, this->lambda, this->evec,
                        0.0, this->C);
        //<Transform C' (orthog) -> C (non-orthog)

        //>Build density matrix
        this->build_D ();
        //<Build density matrix

        this->E = this->compute_E () + this->enuc;
        std::cout << std::setprecision(15) << this->E << "\n";
    }
    //this->build_F();
    //this->print_MO_F();
    
    return this->E + this->enuc;
}

double phf::wfn::compute_E (void) {
    double Em = 0.0;
    for (int u = 0; u < this->nbf; u++) {
        for (int v = 0; v < this->nbf; v++) {
            Em += gsl_matrix_get (this->D, u, v)
                * ( gsl_matrix_get (this->Hcore, u, v)
                  + gsl_matrix_get (this->F, u, v));
        }
    }
    return Em; 
}

void phf::wfn::do_MP2 (void) {
    this->AOtoMOnoddy();
    //this->print_MO_F_alt ();
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

double phf::wfn::MP2viaCC ( void ) {

    std::cout << "in MP2viaCC\n";
    double tsum = 0.0;
    for ( int i = 0; i < this->noccso; i++) {
        for ( int j = 0; j < this->noccso; j++) {
            for ( int a = this->noccso; a < this->nmo; a++) {
                for ( int b = this->noccso; b < this->nmo; b++) {
                    tsum += this->SO_eri->myarray[i][j][a][b]
                            * this->tijab->myarray[i][j][a][b];
                            //  * ( this->SO_eri->myarray[i][a][j][b])
                            //  / ( gsl_vector_get ( this->eval, i/2)
                            //    + gsl_vector_get ( this->eval, j/2)
                            //    - gsl_vector_get ( this->eval, a/2)
                            //    - gsl_vector_get ( this->eval, b/2)
                            //  );
                }
            }
        }
    }
    return tsum*0.25; 
}

void phf::wfn::ccenergy (void) {
    double Ecc = 0.0;
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int a = this->noccso; a < this->nmo; a++ ) {
            Ecc += gsl_matrix_get ( this->FSO, i, a )
                 * gsl_matrix_get ( this->tia, i, a );
        }
    }
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    Ecc += 0.25*this->SO_eri->myarray[i][j][a][b]
                               *this->tijab->myarray[i][j][a][b]; 
                    Ecc += 0.5*this->SO_eri->myarray[i][j][a][b]
                              *gsl_matrix_get ( this->tia, i, a )
                              *gsl_matrix_get ( this->tia, j, b );
                }
            }
        }
    }
    this->Ecorr =  Ecc;
}

void phf::wfn::cciter (void) {
    this->build_tautijab(); //Eqn 9
    this->build_tauijab(); //Eqn10
    this->build_Fae(); //Eqn 3
    this->build_Fmi(); //Eqn 4
    this->build_Fme(); //Eqn 5
    this->build_Wmnij(); //Eqn 6
    this->build_Wabef(); //Eqn 7
    this->build_Wmbej(); //Eqn 8
    this->build_tia(); //Eqn 1
    this->build_tijab(); //Eqn 2
    this->tiacpy();
    this->tijabcpy();
    this->ccenergy();
    std::cout << "E(CC) " << this->Ecorr << "\n";
}
    

void phf::wfn::do_CCSD (void) {
    //CHECKED:
    //ITER 0 : ALL
    //ITER 1 : tautijab, tauijab, Fae, Fmi, Fme, Wmnij, Wabef, Wmbej
    //         ... tia,
    //this->print_MO_F ();
    std::cout << "Transforming MO -> SO ... \n";
    this->MOtoSO ();    
    std::cout << "Transforming F -> FSO ... \n";
    this->FtoSO ();
    this->build_Dia(); //Eqn 12
    this->build_Dijab(); //Eqn 13
    //T2
    this->build_tijab_MP2(); //Eqn 2
    this->ccenergy();
    std::cout << "E[MP2](CC) " << this->MP2viaCC() << "\n";
    std::cout << "E[MP2](CC) " << this->Ecorr << "\n";
    for ( int i = 0; i < 38 ; i++ ) {
        this->cciter();
    }
}

//-->> ENERGY COMPUTATIONS <<--//


//<<-- INTEGRAL TRANSFORMATIONS -->>//
void phf::wfn::HcoretoSO (void ) {
    double spinint;
    int pp,qq;
    for ( int p = 0; p < this->nmo; p++) {
        for ( int q = 0; q < this->nmo; q++) {
            spinint = 0.0;
            pp = p/2;
            qq = q/2;
            for ( int mu = 0; mu < this->nbf; mu++ ) {
                for ( int  nu = 0; nu < this->nbf; nu++ ) {
                    spinint += gsl_matrix_get ( this->C, mu, qq)
                            * gsl_matrix_get (this->C, nu, pp)
                            * gsl_matrix_get (this->Hcore,mu,nu);
                }
            }

            spinint *= ( p%2 == q%2 );
            gsl_matrix_set ( this->HcoreSO, p, q, spinint );
        }
    }
}
void phf::wfn::FtoSO (void ) {
    double tsum = 0.0;
    std::cout << "Transforming Hcore -> HcoreSO\n";
    this->HcoretoSO ();
    std::cout << "Transformed Hcore -> HcoreSO\n";
    for ( int p = 0; p < this->nmo; p++) {
        for ( int q = 0; q < this->nmo; q++) {
            tsum = 0.0;
            tsum += gsl_matrix_get (this->HcoreSO, p, q);
            for ( int m = 0; m < this->noccso; m++) {
                tsum += this->SO_eri->myarray[p][m][q][m];
                //tsum -= this->SO_eri->myarray[p][m][q][m];
            }
            gsl_matrix_set (this->FSO, p, q, tsum);
            //std::cout << tsum << " ";
        }
        //std::cout << "\n";
    }
}
void phf::wfn::MOtoSO (void) {
    double spinint1;
    double spinint2;
    int pp,qq,rr,ss;
    for ( int p = 0; p < this->nmo; p++) {
        for ( int q = 0; q < this->nmo; q++) {
            for ( int r = 0; r < this->nmo; r++) {
                for ( int s = 0; s < this->nmo; s++) {
                    pp = p/2;
                    qq = q/2;
                    rr = r/2;
                    ss = s/2;
                    spinint1 = this->MO_eri->myarray[pp][qq][rr][ss] * ( p%2 == q%2) * (r%2 == s%2);
                    spinint2 = this->MO_eri->myarray[pp][ss][qq][rr] * ( p%2 == s%2) * ( q%2 == r%2);
                    this->SO_eri->myarray[p][r][q][s] = spinint1 - spinint2;
                
                }
            }
        }
    } 
}
void phf::wfn::AOtoMOnoddy (void) {
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

//<<-- INTEGRAL TRANSFORMATIONS -->>//


//<<-- BUILDS (FOCK ETC) -->>//
void phf::wfn::tiacpy (void) {
    for ( int i =0; i < this->noccso; i++ ) {
        for ( int a = this->noccso; a < this->nmo; a++ ) {
            gsl_matrix_set ( this->tia , i , a , gsl_matrix_get (this->tia_new , i , a ));
        }
    }
}
void phf::wfn::tijabcpy (void) {
    for ( int i =0; i < this->noccso; i++ ) {
        for ( int j =0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    this->tijab->myarray[i][j][a][b] = this->tijab_new->myarray[i][j][a][b];
                }
            }
        }
    }
}
void phf::wfn::build_D (void) {
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

void phf::wfn::build_F (void) {
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

void phf::wfn::build_tia (void) {
    //Equation 1 from Stanton90
    double tiatmp = 0.0; //will be assigned via gsl_matrix_set to this->tia
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    double tsum3 = 0.0;
    double tsum4 = 0.0;
    double tsum5 = 0.0;
    double tsum6 = 0.0;
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int a = this->noccso; a < this->nmo; a++ ) {

            //Summation 1: over e ( vir )
            tsum1 = 0.0;
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                tsum1 += gsl_matrix_get ( this->tia, i, e )
                       * gsl_matrix_get ( this->FF, a, e ); 
            }

            //Summation 2: over m ( occ )
            tsum2 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                tsum2 += gsl_matrix_get ( this->tia, m, a )
                       * gsl_matrix_get ( this->FF, m, i );
            }

            //Summation 3: over m, e ( occ, vir )
            tsum3 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    tsum3 += this->tijab->myarray[i][m][a][e]
                           * gsl_matrix_get ( this->FF, m, e );        
                }
            }

            //Summation 4: over n, f ( occ, vir )
            tsum4 = 0.0;
            for ( int n = 0; n < this->noccso; n++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    tsum4 += gsl_matrix_get ( this->tia, n, f )
                           * this->SO_eri->myarray[n][a][i][f]; 
                }
            }

            //Summation 5: over m, e, f ( occ, vir, vir )
            tsum5 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum5 += this->tijab->myarray[i][m][e][f]
                               * this->SO_eri->myarray[m][a][e][f]; 
                    }
                }
            }
            tsum5 *= 0.5;

            //Summation 6: over m, e , n ( occ, vir, occ )
            tsum6 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    for ( int n = 0; n < this->noccso; n++ ) {
                        tsum6 += this->tijab->myarray[m][n][a][e]
                               * this->SO_eri->myarray[n][m][e][i]; 
                    }
                }
            }
            tsum6 *= 0.5;
            
            tiatmp = gsl_matrix_get ( this->FSO, i , a ) 
                   + tsum1 - tsum2 + tsum3 - tsum4 - tsum5 - tsum6;
            tiatmp /= gsl_matrix_get ( Dia, i, a );            
            gsl_matrix_set ( this->tia_new, i, a, tiatmp );
        }
    }
}

void phf::wfn::build_tijab (void) {
    double tijabtmp = 0.0;
    double tsum1 = 0.0;
    double tsum1a = 0.0;
    double tsum2 = 0.0;
    double tsum2a = 0.0;
    double tsum3 = 0.0;
    double tsum4 = 0.0;
    double tsum5 = 0.0;
    double tsum6 = 0.0;
    double tsum7 = 0.0;
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    tijabtmp = this->SO_eri->myarray[i][j][a][b];
                    //term #2
                    //<<--PERMUTATION P_(ab)-->>//
                    //P_(ab) = p(ab) - p(ba)
                    //p(ab) part
                    //Summation 1: over e ( vir )
                    tsum1 = 0.0;
                    for ( int e = this->noccso; e < this->nmo; e++ ) { 
                        //Sub-Summation 1a: over m ( occ )
                        tsum1a = 0.0;
                        for ( int m = 0; m < this->noccso; m++ ) {
                            tsum1a += gsl_matrix_get ( this->tia, m, b )
                                  * gsl_matrix_get ( this->FF, m, e ); 
                        }
                        tsum1a *= 0.5;
                        tsum1 += this->tijab->myarray[i][j][a][e]
                               * ( gsl_matrix_get ( this->FF, b, e )
                                 - tsum1a
                                 );
                    }
                    //at this point tsum1 = p(ab) part
                    //p(ba) part
                    //Summation 1': over e ( vir )
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        //Sub-Summation 1a': over m ( occ )
                        tsum1a = 0.0;
                        for ( int m = 0; m < this->noccso; m++ ) {
                            tsum1a += gsl_matrix_get ( this->tia, m, a )
                                    * gsl_matrix_get ( this->FF, m, e );
                        }
                        tsum1a *= 0.5;
                        // -= here ...
                        tsum1 -= this->tijab->myarray[i][j][b][e]
                              * ( gsl_matrix_get ( this->FF, a, e )
                                - tsum1a
                                );
                    }
                    // ... means that at this point tsum1 = p(ab) - p(ba)
                     
                    //-->>PERMUTATION P_(ab)<<--//

                    //term #3
                    //<<--PERMUTATION P_(ij)-->>//
                    //P_(ij) = p(ij) - p(ji)
                    //p(ij) part
                    //Summation 2: over m ( occ )
                    tsum2 = 0.0;
                    for ( int m = 0; m < this->noccso; m++ ) { 
                        //Sub-Summation 2a: over e ( vir )
                        tsum2a = 0.0;
                        for ( int e = this->noccso; e < this->nmo; e++ ) {
                            tsum2a += gsl_matrix_get ( this->tia, j, e )
                                  * gsl_matrix_get ( this->FF, m, e ); 
                        }
                        tsum2a *= 0.5;
                        tsum2 += this->tijab->myarray[i][m][a][b]
                               * ( gsl_matrix_get ( this->FF, m, j )
                                 + tsum2a
                                 );
                    }
                    //at this point tsum2 = p(ij) part
                    //p(ji) part
                    //Summation 2': over m ( occ )
                    for ( int m = 0; m < this->noccso; m++ ) {
                        //Sub-Summation 2a': over e ( vir )
                        tsum2a = 0.0;
                        for ( int e = this->noccso; e < this->nmo; e++ ) {
                            tsum2a += gsl_matrix_get ( this->tia, i, e )
                                    * gsl_matrix_get ( this->FF, m, e );
                        }
                        tsum2a *= 0.5;
                        // -= here ...
                        tsum2 -= this->tijab->myarray[j][m][a][b]
                              * ( gsl_matrix_get ( this->FF, m, i )
                                + tsum2a
                                );
                    }
                    // ... means that at this point tsum2 = p(ij) - p(ji)
                     
                    //-->>PERMUTATION P_(ab)<<--//

                    //term #4
                    //Summation : over m, n ( occ, occ )
                    tsum3 = 0.0;
                    for ( int m = 0; m < this->noccso; m++ ) {
                        for ( int n = 0; n < this->noccso; n++ ) {
                            tsum3 += this->tauijab->myarray[m][n][a][b]
                                   * this->W->myarray[m][n][i][j];
                        }
                    } 
                    tsum3 *= 0.5;
        
                    //term #5
                    //Summation : over e, f ( vir, vir )
                    tsum4 = 0.0;
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        for ( int f = this->noccso; f < this->nmo; f++ ) {
                            tsum4 += this->tauijab->myarray[i][j][e][f]
                                   * this->W->myarray[a][b][e][f];
                        }
                    }
                    tsum4 *= 0.5;
            
                    //term #6
                    //<<--PERMUTATION P_(ij)P_(ab)-->>//
                    //this is a nested permutation
                    //P_(ij)P_(ab)[f] = P_(ij)[p(ab)f - p(ba)f]
                    // = p(ij)[p(ab)f - p(ba)f] - p(ji)[p(ab)f - p(ba)f]
                    // = p(ij)p(ab)f - p(ij)p(ba)f - p(ji)p(ab)f + p(ji)p(ba)f
                    
                    //Summation : over m, e ( occ , vir )
                    tsum5 = 0.0;
                    for ( int m = 0; m < this->noccso; m++ ) {
                        for ( int e = this->noccso; e < this->nmo; e++ ) {
                            //p(ij)p(ab) part
                            tsum5 += this->tijab->myarray[i][m][a][e]
                                  * this->W->myarray[m][b][e][j];
                            tsum5 -= gsl_matrix_get ( this->tia, i, e )
                                  * gsl_matrix_get ( this->tia, m, a )
                                  * this->SO_eri->myarray[m][b][e][j];

                            //p(ij)p(ba)f part
                            tsum5 -= this->tijab->myarray[i][m][b][e]
                                  * this->W->myarray[m][a][e][j];
                            tsum5 += gsl_matrix_get ( this->tia, i, e )
                                  * gsl_matrix_get ( this->tia, m, b )
                                  * this->SO_eri->myarray[m][a][e][j];

                            //p(ji)p(ab)f part
                            tsum5 -= this->tijab->myarray[j][m][a][e]
                                  * this->W->myarray[m][b][e][i];
                            tsum5 += gsl_matrix_get ( this->tia, j, e )
                                  * gsl_matrix_get ( this->tia, m, a )
                                  * this->SO_eri->myarray[m][b][e][i];
                    
                            //p(ji)p(ba)f part
                            tsum5 += this->tijab->myarray[j][m][b][e]
                                  * this->W->myarray[m][a][e][i];
                            tsum5 -= gsl_matrix_get ( this->tia, j, e )
                                  * gsl_matrix_get ( this->tia, m, b )
                                  * this->SO_eri->myarray[m][a][e][i];
                            
                        }
                    }
                    //-->>PERMUTATION P_(ij)P_(ab)<<--//

                    //term #7
                    //<<--PERMUTATION P_(ij)-->>//
                    tsum6 = 0.0;
                    //Summation : over e (vir)
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        //p(ij) part 
                        tsum6 += gsl_matrix_get ( this->tia, i, e )
                               * this->SO_eri->myarray[a][b][e][j];
                        
                        //p(ji) part
                        tsum6 -= gsl_matrix_get ( this->tia, j, e )
                               * this->SO_eri->myarray[a][b][e][i];
                    }
                    //<<--PERMUTATION P_(ij)-->>//

                    //term #8
                    //<<--PERMUTATION P_(ab)-->>//
                    //Summation : over m ( occ )
                    tsum7 = 0.0;
                    for ( int m = 0; m < this->noccso; m++ ) {
                        //p(ab) part
                        tsum7 += gsl_matrix_get ( this->tia, m, a )
                               * this->SO_eri->myarray[m][b][i][j]; 
                        
                        //p(ba) part
                        tsum7 -= gsl_matrix_get ( this->tia, m, b )
                               * this->SO_eri->myarray[m][a][i][j];
                    }
                    tijabtmp += tsum1;
                    tijabtmp -= tsum2;
                    tijabtmp += tsum3;
                    tijabtmp += tsum4;
                    tijabtmp += tsum5;
                    //std::cout << "TIJAB " << tijabtmp << "\n";
                    tijabtmp += tsum6;
                    tijabtmp -= tsum7;
                    tijabtmp /= this->Dijab->myarray[i][j][a][b];
                    this->tijab_new->myarray[i][j][a][b] = tijabtmp;
                }
            }
        }
    }
}
void phf::wfn::build_tijab_MP2 (void) {
    for ( int i = 0; i < this->noccso; i++ ){
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++) {
                    this->tijab->myarray[i][j][a][b] = 
                        this->SO_eri->myarray[i][j][a][b]
                      / (
                        gsl_vector_get ( this -> eval, i/2)
                      + gsl_vector_get ( this -> eval, j/2)
                      - gsl_vector_get ( this -> eval, a/2)
                      - gsl_vector_get ( this -> eval, b/2)
                        );
                }
            }
        }
    }
}

void phf::wfn::build_tautijab (void) {
    //Eqn 9 from Stanton90
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    tautijab->myarray[i][j][a][b] = this->tijab->myarray[i][j][a][b]
                                               + 0.5 * 
                                               ( 
                                                  gsl_matrix_get ( this->tia, i, a)
                                                * gsl_matrix_get ( this->tia, j, b)
                                                - gsl_matrix_get ( this->tia, i, b)
                                                * gsl_matrix_get ( this->tia, j, a)
                                               );
                    
                }
            }
        }
    }
}

void phf::wfn::build_tauijab (void) {
    //Eqn 10 from Stanton90
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int a = this->noccso; a < this->nmo; a++ ) {
                for ( int b = this->noccso; b < this->nmo; b++ ) {
                    tauijab->myarray[i][j][a][b] = this->tijab->myarray[i][j][a][b]
                                             + (
                                                  gsl_matrix_get ( this->tia, i, a )
                                                * gsl_matrix_get ( this->tia, j, b )
                                                - gsl_matrix_get ( this->tia, i, b )
                                                * gsl_matrix_get ( this->tia, j, a )
                                               );    
                }
            }
        }
    }
}
void phf::wfn::build_Dia (void) {
    for ( int i = 0; i < this->nmo; i++ ) {
        for ( int a = 0; a < this->nmo; a++ ) {
            gsl_matrix_set (this->Dia, i, a, 
                            gsl_matrix_get ( this->FSO, i, i)
                            - gsl_matrix_get ( this->FSO, a, a));
        }
    }
}

void phf::wfn::build_Dijab (void) {
    for ( int i = 0; i < this->nmo; i++ ){
        for ( int j = 0; j < this->nmo; j++ ) {
            for ( int a = 0; a < this->nmo; a++ ) {
                for ( int b = 0; b < this->nmo; b++) {
                    this->Dijab->myarray[i][j][a][b] = 
                        gsl_matrix_get ( this -> FSO, i, i)
                      + gsl_matrix_get ( this -> FSO, j, j)
                      - gsl_matrix_get ( this -> FSO, a, a)
                      - gsl_matrix_get ( this -> FSO, b, b);
                }
            }
        }
    }
}

void phf::wfn::build_Fae (void) {
    //Eqn 3 from Stanton90
    //Not sure about array bounds for these first two loops
    double tsum1 = 0;
    double tsum2 = 0;
    double tsum3 = 0;
    for ( int a = this->noccso; a < this->nmo; a++ ) {
        for ( int e = this->noccso; e < this->nmo; e++ ) {
            //First summation: over m (occ)
            tsum1 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                tsum1 +=  gsl_matrix_get ( this->FSO, m, e )
                        * gsl_matrix_get ( this->tia, m, a );   
            } 
            tsum1 *= 0.5;
            //Second summation: over m, f (occ, vir)
            tsum2 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    tsum2 +=  gsl_matrix_get ( this->tia, m, f)
                            * this->SO_eri->myarray[m][a][f][e];
                }
            }
            //Third summation: over m, n, f (occ, occ, vir)
            tsum3 = 0.0;
            for ( int m = 0; m < this->noccso; m++ ) {
                for ( int n = 0; n < this->noccso; n++ ) {
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum3 +=  this->tautijab->myarray[m][n][a][f]
                                * this->SO_eri->myarray[m][n][e][f]; 
                    }
                }
            }
            tsum3 *= 0.5;
            gsl_matrix_set ( this->FF , a, e, 
                             (1.0 - kron(a,e))*gsl_matrix_get ( this->FSO, a, e)
                            - tsum1 + tsum2 - tsum3
                           );

        }
    }
}

void phf::wfn::build_Fmi (void) {
    //Equation 4 from Stanton90
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    double tsum3 = 0.0;
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int i = 0; i < this->noccso; i++ ) {
            //First summation: over e (vir)
            tsum1 = 0.0;
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                tsum1 +=  gsl_matrix_get ( this->tia, i, e )
                       * gsl_matrix_get ( this->FSO, m, e );
            }
            tsum1 *= 0.5;

            //Second summation: over n, e (occ, vir)
            tsum2 = 0.0;
            for ( int n = 0; n < this->noccso; n++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    tsum2 +=  gsl_matrix_get ( this->tia, n, e )
                            * this->SO_eri->myarray[m][n][i][e]; 
                }
            }
            
            //Third summation: over n, e , f (occ, vir, vir)
            tsum3 = 0.0;
            for ( int n = 0; n < this->noccso; n++ ) {
                for ( int e = this->noccso; e < this->nmo; e++ ) {
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum3 +=  this->tautijab->myarray[i][n][e][f]
                                * this->SO_eri->myarray[m][n][e][f];
                    }
                }
            }
            tsum3 *= 0.5;
            gsl_matrix_set ( this-> FF, m, i,
                        (1 - kron(m,i))*gsl_matrix_get ( this->FSO, m, i )
                        + tsum1 + tsum2 + tsum3 );
        }
    }
}

void phf::wfn::build_Fme (void) {
    //Equation 5 in Stanton90
    double tsum = 0.0;
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int e = this->noccso; e < this->nmo; e++ ) {
            //Summation: over n, f (occ, vir)
            tsum = 0.0;
            for ( int n = 0; n < this->noccso; n++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    tsum +=   gsl_matrix_get (this->tia, n, f)
                            * this->SO_eri->myarray[m][n][e][f];
                }
            }
            gsl_matrix_set ( this->FF, m, e,
                             gsl_matrix_get ( this->FSO, m, e )
                           + tsum );
        }
    }
}
         
void phf::wfn::build_Wmnij (void) {
    //Equation 6 in Stanton90
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int n = 0; n < this->noccso; n++ ) {
            for ( int i = 0; i < this->noccso; i++ ) {
                for ( int j = 0; j < this->noccso; j++ ) {
                    //Summation 1: over e (vir)
                    tsum1 = 0.0;
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        //p(ij) - p(ji)
                        tsum1 += gsl_matrix_get ( this->tia, j, e )
                              * this->SO_eri->myarray[m][n][i][e];
                        tsum1 -= gsl_matrix_get ( this->tia, i, e )
                              * this->SO_eri->myarray[m][n][j][e];
                    }                
                    
                    //Summation 2: over e, f (vir, vir)
                    tsum2 = 0.0;
                    for ( int e = this->noccso; e < this->nmo; e++ ) {
                        for ( int f = this->noccso; f < this->nmo; f++ ) {
                            tsum2 += this->tauijab->myarray[i][j][e][f]
                                   * this->SO_eri->myarray[m][n][e][f];       
                        }
                    } 
                    tsum2 *= 0.25;
                    
                    this->W->myarray[m][n][i][j] = this->SO_eri->myarray[m][n][i][j]
                                                   + tsum1 + tsum2;
                }
            }
        }
    }
}

void phf::wfn::build_Wabef (void) {
    //Equation 7 from Stanton90
    //with a large virtual space, I think this will be the chonkiest term.
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    for ( int a = this->noccso; a < this->nmo; a++ ) {
        for ( int b = this->noccso; b < this->nmo; b++ ) {
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                //Summation 1: over m (occ)
                tsum1 = 0.0;
                for ( int m = 0; m < this->noccso; m++ ) {
                    tsum1 += gsl_matrix_get ( tia, m, b )
                           * this->SO_eri->myarray[a][m][e][f];
                    tsum1 -= gsl_matrix_get ( tia, m, a )
                           * this->SO_eri->myarray[b][m][e][f];
                }
            
                //Summation 2: over m, n (occ, occ)
                tsum2 = 0.0;
                for ( int m = 0; m < this->noccso; m++ ) {
                    for ( int n = 0; n < this->noccso; n++ ) {
                        tsum2 += this->tauijab->myarray[m][n][a][b]
                               * this->SO_eri->myarray[m][n][e][f];
                    }
                }
                tsum2 *= 0.25;
                
                this->W->myarray[a][b][e][f] = this->SO_eri->myarray[a][b][e][f]
                                             - tsum1 + tsum2;
                }
            }
        }
    }
} 

void phf::wfn::build_Wmbej (void) {
    //Equation 8 from Stanton90
    double tsum1 = 0.0;
    double tsum2 = 0.0;
    double tsum3 = 0.0;
    for ( int m = 0; m < this->noccso; m++ ) {
        for ( int b = this->noccso; b < this->nmo; b++ ) {
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int j = 0; j < this->noccso; j++ ) {
                    //Summation 1: over f (vir) 
                    tsum1 = 0.0;
                    for ( int f = this->noccso; f < this->nmo; f++ ) {
                        tsum1 += gsl_matrix_get ( this->tia, j, f )
                              * this->SO_eri->myarray[m][b][e][f];
                    }

                    //Summation 2: over n (occ)
                    tsum2 = 0.0;
                    for ( int n = 0; n < this->noccso; n++ ) {
                        tsum2 += gsl_matrix_get ( this->tia, n, b )
                               * this->SO_eri->myarray[m][n][e][j];
                    }
                    
                    //Summation 3: over n, f ( occ , vir )
                    tsum3 = 0.0;
                    for ( int n = 0; n < this->noccso; n++ ) {
                        for ( int f = this->noccso; f < this->nmo; f++ ) {
                            tsum3 += (0.5*this->tijab->myarray[j][n][f][b]
                                      + gsl_matrix_get ( this->tia, j, f)
                                      * gsl_matrix_get ( this->tia, n, b)
                                     )
                                   * this->SO_eri->myarray[m][n][e][f];
                        }
                    }
                    this->W->myarray[m][b][e][j] = this->SO_eri->myarray[m][b][e][j]
                                                  + tsum1 - tsum2 - tsum3;
                }
            }
        }
    } 
}
//-->> BUILDS (FOCK ETC) <<--//

//<<-- I/O -->>//
//
void phf::wfn::print_F ( void ) {
    for ( int i = 0; i < this->nmo; i++ ) {
        for ( int j = 0; j < this->nmo; j++ ) {
            std::cout << gsl_matrix_get (this->FSO, i, j) << " ";
        }
        std::cout << "\n";
    }
}

void phf::wfn::print_Fmi ( void ) {
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            std::cout << gsl_matrix_get ( this->FF, i, j ) << " ";        
        }
        std::cout << "\n";
    }
}
void phf::wfn::print_Fme ( void ) {
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = this->noccso; j < this->nmo; j++ ) {
            std::cout << gsl_matrix_get ( this->FF, i, j ) << " ";        
        }
        std::cout << "\n";
    }
}
void phf::wfn::print_Fae ( void ) {
    for ( int i = this->noccso; i < this->nmo; i++ ) {
        for ( int j = this->noccso; j < this->nmo; j++ ) {
            std::cout << gsl_matrix_get ( this->FF, i, j ) << " ";        
        }
        std::cout << "\n";
    }
}

void phf::wfn::print_tia (void) {
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int a = this->noccso; a < this->nmo; a++ ) {
            std::cout << gsl_matrix_get ( this->tia, i , a ) << " ";        
        }
        std::cout << "\n";
    }
}
void phf::wfn::print_tijab (void) {
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    std::cout << this->tijab->myarray[i][j][e][f] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
void phf::wfn::print_tauijab (void) {
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    std::cout << this->tauijab->myarray[i][j][e][f] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
void phf::wfn::print_tautijab (void) {
    for ( int i = 0; i < this->noccso; i++ ) {
        for ( int j = 0; j < this->noccso; j++ ) {
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    std::cout << this->tautijab->myarray[i][j][e][f] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void phf::wfn::print_Wabef (void) {
    for ( int a = this->noccso; a < this->nmo; a++ ) {
        for ( int b = this->noccso; b < this->nmo; b++ ) {
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int f = this->noccso; f < this->nmo; f++ ) {
                    std::cout << this->W->myarray[a][b][e][f] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
void phf::wfn::print_Wmnij (void) {
    for ( int a = 0; a < this->noccso; a++ ) {
        for ( int b = 0; b < this->noccso; b++ ) {
            for ( int e = 0; e < this->noccso; e++ ) {
                for ( int f = 0; f < this->noccso; f++ ) {
                    std::cout << this->W->myarray[a][b][e][f] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
void phf::wfn::print_Wmbej (void) {
    for ( int a = 0; a < this->noccso; a++ ) {
        for ( int b = this->noccso; b < this->nmo; b++ ) {
            for ( int e = this->noccso; e < this->nmo; e++ ) {
                for ( int f = 0; f < this->noccso; f++ ) {
                    std::cout << this->W->myarray[a][b][e][f] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void phf::wfn::print_MO_F_alt (void) {
    std::cout << "MO_F (spinint)\n";
    double tsum = 0.0;
    for ( int p = 0; p < this->nbf; p++ ) {
        for ( int q = 0; q < this->nbf; q++ ) {
            tsum = 0.0;
            for ( int mu = 0; mu < this-> nbf; mu++ ) {
                for ( int nu = 0; nu < this->nbf; nu++ ) {
                    tsum += gsl_matrix_get ( this->C, mu, q)
                            * gsl_matrix_get ( this->C, nu, p)
                            * gsl_matrix_get ( this->Hcore,mu,nu);
                }
            }
            //tsum += gsl_matrix_get ( this->Hcore, p, q );
            for ( int m = 0; m < this->nocc; m++ ) {
                tsum += 2*this->MO_eri->myarray[p][q][m][m];
                tsum -= this->MO_eri->myarray[p][m][q][m];
            }
            std::cout << tsum << " ";
        }
        std::cout << "\n";
    }
}
void phf::wfn::print_MO_F (void) {
    std::cout << "MO_F (mo coeff)\n";
    double tsum = 0.0;
    for ( int i = 0; i < this->nbf; i++ ) {
        for ( int j = 0; j < this->nbf; j++) {
            tsum = 0.0;
            for ( int mu = 0; mu < this->nbf; mu++) {
                for ( int nu = 0; nu < this->nbf; nu++) {
                    tsum += gsl_matrix_get (this->C,mu,j)
                            * gsl_matrix_get (this->C,nu,i)
                            * ( gsl_matrix_get (this->F,mu,nu)
                                - gsl_matrix_get (this->Hcore,mu,nu));
                }
            } 
            std::cout << tsum << " ";
        }
        std::cout << "\n";
    }
   }


void phf::wfn::read_enuc (std::string fname) {
    //input std::string fname; filename for Vnuc
    //output double enuc; Vnuc from filename
    //I'm not clear whether it is ok to return 
    //a local variable (enuc) by value or  not.
    //Seems that by reference/pointer is bad?
    std::ifstream fnuc(fname);
    fnuc >> this->enuc;
}

void phf::wfn::read_2D (std::string fname, gsl_matrix * A) {
    //input double **A; pre-allocated nbf X nbf gsl_matrix 
    //      int nbf; number of basis functions
    //output void;
    std::ifstream arr2d(fname);
    int i = 0;
    int j = 0;
    double s;

    while (!arr2d.eof()) {
            arr2d >> std::setprecision(15) >> i >> j >> s;
            if ( arr2d.eof() ) {
                break;
            }
            std::cout << i << " " << j << "\n";
            i -= 1;
            j -= 1;
            gsl_matrix_set (A, i, j, s);
            gsl_matrix_set (A, j, i, s);
    }

    //for (int i = 0; i < this->nbf; i++) {
    //    for (int j = 0; j <= i; j++) {
    //        std::cout << i << " " << j << "\n";
    //        arr2d >> std::setprecision(15) >> ti >> tj >> s;
    //        gsl_matrix_set (A, i, j, s);
    //        gsl_matrix_set (A, j, i, s);
    //    }
    //}
}

void phf::wfn::read_ERI (std::string fname) {
    //input *A; 1D array in compressed notation
    //      fname; filename
    //output void;
    //notation
    // A[u][v][l][s] = (uv|ls)

    std::ifstream eri(fname);
    int u,v,l,s;
    double temp;
    while (!eri.eof()) {
        eri >> u >> v >> l >> s >> temp;
        u -= 1;
        v -= 1;
        l -= 1;
        s -= 1;
        // 8fold permutational symmetry:
        // (uv|ls) = (vu|ls) = (uv|sl) = (vu|sl) = (ls|uv) = (sl|uv) = (sl|vu) = (ls|vu)
        this->AO_eri->myarray[u][v][l][s] = temp;
        this->AO_eri->myarray[u][v][s][l] = temp;

        this->AO_eri->myarray[v][u][l][s] = temp;
        this->AO_eri->myarray[v][u][s][l] = temp;

        this->AO_eri->myarray[l][s][u][v] = temp;
        this->AO_eri->myarray[l][s][v][u] = temp;

        this->AO_eri->myarray[s][l][u][v] = temp;
        this->AO_eri->myarray[s][l][v][u] = temp;
    } 
    eri.close();
}
//-->> I/O <<--//
    
