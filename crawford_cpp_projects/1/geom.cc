#include <iostream>
#include <fstream>
#include <gsl/gsl_math.h>
#include <iomanip>
#include <cmath>

using namespace std;
int main () {
    ifstream input("geom.dat");
    int natom;
    input >> natom; //get the number of atoms

    //>memory allocation
    int *zval = new int[natom];
    double *x = new double[natom];
    double *y = new double[natom];
    double *z = new double[natom];
    double **R = new double* [natom]; //distance array
    for(int i=0; i < natom; i++)
        R[i] = new double[natom]; 
    double ***phi = new double** [natom]; //angle array
    for(int i=0; i < natom; i++) {
        phi[i] = new double* [natom];
        for(int j=0; j<natom; j++){
            phi[i][j] = new double[natom];
        }
    }
    double ****oop = new double*** [natom]; //out of plane angle array
    double ****t = new double*** [natom]; //torsional angle array
    for(int i = 0; i < natom; i++) {
        oop[i] = new double** [natom]; //do both 4 index arrays at the same time
        t[i] = new double** [natom];
        for(int j = 0; j < natom; j++) {
            oop[i][j] = new double* [natom];
            t[i][j] = new double* [natom];
            for(int k = 0; k < natom; k++) {
                oop[i][j][k] = new double[natom];
                t[i][j][k] = new double[natom];
            }
        }
    }
    double **ex = new double* [natom];
    double **ey = new double* [natom];
    double **ez = new double* [natom];
    for(int i=0; i < natom; i++) {
        ex[i] = new double[natom];
        ey[i] = new double[natom];
        ez[i] = new double[natom];
    }
    //<memory allocation

    for(int i=0; i < natom; i++)
     input >> zval[i] >> x[i] >> y[i] >> z[i]; //reading in atom positions
    input.close();

    cout << "Number of atoms:" << natom << endl;
    cout << "Input Cartesian coordinates:\n";
    for(int i=0; i < natom; i++)
     printf("%d %20.12f %20.12f %20.12f\n", (int) zval[i], x[i], y[i], z[i]);

    cout << "Printing distance matrix R[i][j]\n";

    //>Evaluate distance matrix
    for(int i=0; i < natom; i++) {
        for(int j=0; j < natom; j++) {
            R[i][j] = sqrt( (x[i] - x[j])*(x[i] - x[j])
                       + (y[i] - y[j])*(y[i] - y[j])
                       + (z[i] - z[j])*(z[i] - z[j]));
        }
    }
    //<Evaluate distance matrix

    //>print distance matrix
    for(int i=0; i<natom; i++)
        for(int j=0; j<i; j++)
            printf("i%d j%d %20.12f\n", (int) i, (int) j, R[i][j]);
    //<print distance matrix

    //>Evaluate unit vector matrices (ex, ey, ez)
    for(int i=0; i < natom; i++) {
        for(int j=0; j < i; j++) {
            ex[i][j] = -(x[i] - x[j])/R[i][j];
            ey[i][j] = -(y[i] - y[j])/R[i][j];
            ez[i][j] = -(z[i] - z[j])/R[i][j];
            ex[j][i] = -ex[i][j]; //e[i][j] = -e[j][i]
            ey[j][i] = -ey[i][j];
            ez[j][i] = -ez[i][j];
        }
    }
    //<Evaluate unit vector matrices (ex, ey, ez)
 
    //>Evaluate angle matrix
    float sum;
    for(int i=0; i<natom; i++) {
        for(int j=0; j < natom; j++) {
            for(int k=0; k < natom; k++) {
                sum = 0.0;
                sum += ex[j][i]*ex[j][k]; 
                sum += ey[j][i]*ey[j][k];
                sum += ez[j][i]*ez[j][k];
                phi[i][j][k] = acos(sum);
                phi[k][j][i] = phi[i][j][k];
            }
        }
    }
    //<Evaluate angle matrix

    //>Print angle matrix
    cout << "Printing angle matrix phi[i][j][k]\n";
    for(int i=0; i < natom; i++) {
        for(int j=0; j < i; j++)   {
            for(int k=0; k < j; k++) {
                if (R[j][i] < 4.0 && R[j][k] < 4.0) {
                    printf("i%d j%d k%d %5.6f\n", i, j, k, phi[i][j][k]);
                }
            }
        }
    }
    //<Print angle matrix
                        
    //>Evaluate OOP matrix
    float tx,ty,tz;
    float f;
    for(int i=0; i < natom; i++) {
        for(int j=0; j < i; j++) {
            for(int k=0; k < j; k++) {
                for(int l=0; l < k; l++) {
                    tx = ey[k][j]*ez[k][l] - ez[k][j]*ey[k][l];  
                    ty = ez[k][j]*ex[k][l] - ex[k][j]*ez[k][l];
                    tz = ex[k][j]*ey[k][l] - ey[k][j]*ex[k][l];
                    f = sin(phi[i][j][k]);
                    tx /= f;
                    ty /= f;
                    tz /= f;
                    sum = 0.0;
                    sum += tx*ex[k][i];
                    sum += ty*ey[k][i];
                    sum += tz*ez[k][i];
                    if(sum < -1.0) oop[i][j][k][l] = asin(-1.0);
                    else if(sum > 1.0) oop[i][j][k][l] = asin(1.0);
                    else oop[i][j][k][l] = asin(sum);
                    
                }
            }
        }
    }
    //<Evaluate OOP matrix
    
    //>Evaluate torsional matrix
//    for(int i=0; i < natom; i++) {
//        for(int j=0; j < i; j++) {
//            for(int k=0; k < j; k++) {
//                for(int l=0; l < k; l++) {
//		    

     
    


    //>memory deallocation
    //>>deallocate 1 index arrays
    delete[] zval;
    delete[] x;
    delete[] y;
    delete[] z;
    //<<deallocate 1 index arrays

    //>>deallocate 2 index arrays
    for(int i=0; i < natom; i++) {
        delete[] R[i];
        delete[] ex[i];
        delete[] ey[i];
        delete[] ez[i];
    }
    delete[] R;
    delete[] ex;
    delete[] ey;
    delete[] ez;
    //<<deallocate 2 index arrays

    //>>deallocate 3 index arrays
    for(int i=0; i < natom; i++) {
        for(int j=0; j<natom; j++) {
            delete[] phi[i][j];
        }
        delete[] phi[i];
    }
    delete[] phi;
    //<<deallocate 3 index arrays

    //<<deallocate 4 index arrays
    for(int i=0; i < natom; i++) {
        for(int j = 0; j < natom; j++) {
            for(int k = 0; k < natom; k++) {
                delete[] oop[i][j][k];
                delete[] t[i][j][k];
            }
            delete[] oop[i][j];
            delete[] t[i][j];
        }
        delete[] oop[i];
        delete[] t[i];
    }
    delete[] oop;  
    delete[] t;
    //<<deallocate 4 index arrays

    //<memory deallocation
    return 0;
}
