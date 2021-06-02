#include <iostream>
#include <fstream>
#include "eHIJING.h"
#include <vector>
#include <cmath>
#include <random>

using namespace EHIJING;

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
                T1.set_with_linear_index(k, func(X));
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
    gen.Tabulate("./Tables/");
    std::cout << "qhatF = " << gen.qhatF(0.1, 2.25, gen.rho0()*1.0*5.076) * 5.076 << " [GeV^2/fm]" << std::endl;
    std::cout << "qhatF = " << gen.qhatF(0.1, 2.25, gen.rho0()*3.0*5.076) * 5.076 << " [GeV^2/fm]" << std::endl;
    std::cout << "qhatF = " << gen.qhatF(0.1, 2.25, gen.rho0()*5.0*5.076) * 5.076 << " [GeV^2/fm]" << std::endl;
    std::cout << "qhatF = " << gen.qhatF(0.01, 2.25, gen.rho0()*3.0*5.076) * 5.076 << " [GeV^2/fm]" << std::endl;
    std::ofstream f("test.dat");
    double kt2 = 1.0;
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
    gen.Tabulate("./Tables/");
    //std::ofstream f("test.dat");
    double E0 = 10, Q2 = 2.25;
    double x = Q2/2/E0/0.938;
    double kt2min = 0.25;
    int pid = 1;
    double avg_qhat = 0.;
    int N = 100000;
    /*for (int i=0; i<N; i++){
        double L = gen.sample_L(A);
        avg_qhat += gen.qhatF(x, Q2, gen.rho0()*std::max(L,5.076));
        double E = E0;
        double kt2 = Q2;
        double z;
        while(kt2 > kt2min){
            if (!gen.next_kt2(kt2, pid, E, L, kt2min, x, Q2)) continue;
            if (kt2 < kt2min) break;
  
            if (!gen.sample_z(z, pid, E, L, kt2, x, Q2)) continue;
            f << z*E << " ";
            E = E*(1.-z);
        }
        f << E << std::endl; 
    }
    std::cout << "Avg. qhatF("<<A<<") = " << K*avg_qhat / N * 5.076 << " [GeV^2/fm]" << std::endl;*/

    // test multiple collisions
    double L = 5.076 * 4;
    std::ofstream f2("q2s.dat");
    double correct_qhat = gen.qhatF(x, Q2, gen.rho0()*L);
    double cumq2 = 0., Ncoll=0.;
    for (int i=0; i<N; i++){
        std::vector<double> q2s, ts;
        gen.sample_all_qt2(1, E0, L, x, Q2, q2s, ts);
        for (int j=0; j<ts.size(); j++){
            f2 << q2s[j] << " " << ts[j] << std::endl;
            cumq2 += q2s[j];
            Ncoll += 1.;
        }
    }
    std::cout << "Number of collision / fm = " << Ncoll/L/N*5.076 << std::endl;
    std::cout << "Simu. qhatF("<<A<<") = " << cumq2 / L / N * 5.076 << " [GeV^2/fm]" << std::endl;
    std::cout << "Calc. qhatF("<<A<<") = " << K * correct_qhat * 5.076 << " [GeV^2/fm]" << std::endl;

    return 0;
}

