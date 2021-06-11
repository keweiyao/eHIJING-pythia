#include <iostream>
#include <fstream>
#include "Pythia8/Pythia.h"
#include "Pythia8/eHIJING.h"
#include <vector>
#include <cmath>
#include <random>


int main(int argc, char*argv[]){

    int mode = atoi(argv[1]);
    double K = atof(argv[2]);
    double A = atof(argv[3]);
    EHIJING::eHIJING gen(mode, K, 4, -0.5, 0.5);
    EHIJING::InMediumFragmentation gen2(mode, K, 4, -0.5, 0.5);
    gen.Tabulate("./Tables/");
    gen2.Tabulate("./Tables/");
    std::ofstream f("test.dat");
    double E0 = 12, Q2 = 2.25;
    double x = Q2/2/E0/0.938;
    double kt2min = .4*.4, lambda2 = 0.01;
    int pid = 1;
    double avg_qhat = 0.;
    int N = 10000;
    double L = 1.12*std::pow(A,1./3.)*5.076 * 0.75;
    for (int i=0; i<N; i++){
        if (i%1000==0) std::cout << i << std::endl;
        avg_qhat += gen.qhatF(x, Q2, EHIJING::rho0*std::max(L,5.076));
        double E = E0;
        double kt2 = Q2;
        double z;
        // high-Q shower, kT-ordered
        while(kt2 > kt2min && E>1){
            if (!gen.next_kt2(kt2, pid, E, L, kt2min, x, Q2)) continue;
            if (kt2 < kt2min) break;

            if (!gen.sample_z(z, pid, E, L, kt2, x, Q2)) continue;
            f << z*E << " ";
            E = E*(1.-z);
        }
        // low-Q shower, tauf-ordered
        double omegaL_min = lambda2*L/(.5*E);
        double omegaL = std::sqrt(kt2min)*L / 2.;
        while(omegaL > omegaL_min && E>1){
            double z;
            bool status = gen2.next_radiation(pid, E, L, x, Q2, kt2min, lambda2, omegaL, z);
            if (omegaL < omegaL_min) break;
            if (!status) continue;
            f << z*E << " ";
            E = E*(1.-z);
            omegaL_min = lambda2*L/(.5*E);
        }
        f << E << std::endl;
    }
    std::cout << "Avg. qhatF("<<A<<") = " << avg_qhat / N * 5.076 << " [GeV^2/fm]" << std::endl;
    std::cout << "Avg. Qs("<<A<<") = " << std::sqrt(gen.Qs2(0.01, 4, EHIJING::rho0*std::max(L,5.076))) << " [GeV]" << std::endl;
    return 0;
}
