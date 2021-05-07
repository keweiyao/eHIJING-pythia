#include <iostream>
#include <fstream>
#include "eHIJING.h"
#include <vector>
#include <cmath>
#include <random>

double func(std::vector<double> & X){
    double x = X[0], y = X[1], z = X[2];
    return x*y*z+std::exp(-x*y*std::cos(z));
}

int test_table(){
    int N = 21;
    std::vector<double> grid_max({1,1,1});
    std::vector<double> grid_min({0., 0., 0.});
    std::vector<int> grid_shape({N, N, N});
    double step = 1./(N-1);
    int grid_dim = 3;
    table_nd T1(grid_dim, grid_shape, grid_min, grid_max);
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            for (int k=0; k<N; k++){
                std::vector<int> IX({i,j,k});
                std::vector<double> X({step*i, step*j, step*k});
                T1.set(IX, func(X));
            }
        }
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> R(0.,1.);
    std::cout << "True\tInterp\tErrorRel" << std::endl;
    for (int i=0; i<10; i++){
        double x = R(gen), y = R(gen), z = R(gen);
        std::vector<double> X({x,y,z});
        double Y0 = func(X);
        double Y1 = T1.interpolate(X);
        std::cout << Y0 << "    " << Y1 << "    " << 2*(Y1-Y0)/(Y1+Y0) << std::endl;
    } 
    return 0;
}

int test_table_gen(){
    eHIJING gen(0, 1.0);
    gen.Tabulate();
    std::cout << "qhatF = " << gen.qhatF(0.1, 2.25, gen.rho0()*1.0*5.076) * 5.076 << " [GeV^2/fm]" << std::endl;
    std::cout << "qhatF = " << gen.qhatF(0.1, 2.25, gen.rho0()*3.0*5.076) * 5.076 << " [GeV^2/fm]" << std::endl;
    std::cout << "qhatF = " << gen.qhatF(0.1, 2.25, gen.rho0()*5.0*5.076) * 5.076 << " [GeV^2/fm]" << std::endl;
    std::cout << "qhatF = " << gen.qhatF(0.01, 2.25, gen.rho0()*3.0*5.076) * 5.076 << " [GeV^2/fm]" << std::endl;
    std::ofstream f("test.dat");
    double kt2=1.0;
    double x = 0.1, Q2 = 2.0,  L = 5*5.076;
    for (double z=1e-2; z<1.0; z+=0.005){
        f << z << " " << gen.induced_dFdkt2dz(x, Q2, L, 10.0, kt2, z) << std::endl;
    }
    return 0;
}

int main(int argc, char*argv[]){
    int mode = atoi(argv[1]);
    double K = atof(argv[2]);
    double A = atof(argv[3]);
    eHIJING gen(mode, K);
    gen.Tabulate();
    std::ofstream f("test.dat");
    double E0 = 10, Q2 = 2.25;
    double x = Q2/2/E0/0.938;
    double kt2min = 0.16;
    int pid = 1;
    for (int i=0; i<100000; i++){
        double L = gen.sample_L(A);
        double E = E0;
        double kt2 = Q2;
        while(kt2 > kt2min){
            if (!gen.next_kt2(kt2, pid, E, L, kt2min, x, Q2)) continue;
            if (kt2 < kt2min) break;
            double z;
            if (!gen.sample_z(z, pid, E, L, kt2, x, Q2)) continue;
            if (z*E < std::sqrt(kt2)) continue;
            f << z*E << " ";
            E = E*(1.-z);
            if (E<.4) break;
        }
        f << E << std::endl; 
    }
    return 0;
}

