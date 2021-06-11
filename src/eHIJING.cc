#include "eHIJING/eHIJING.h"
#include <iostream>
#include <fstream>
#include "eHIJING/integrator.h"
#include <cmath>
#include <thread>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_gamma.h>
#include <string>
#include <sstream>
#include <filesystem>
#include <atomic>

namespace EHIJING {

// color algebra constnats
const double CA = 3;
const double dA = 8;
const double CF = 4./3.;
// Lambda QCD and Lambda^2
const double mu = 0.25; // [GeV]
const double mu2 = std::pow(mu, 2); // GeV^2
// minimum and maximum nuclear thickness function
// for tabulation
const double TAmin = 0.05/5.076/5.076;
const double TAmax = 2.8/5.076/5.076;
// The b parameter for 3-flavor QCD
// b0 = 11 - 2/3*Nf
const double b0 = 9./2.;
const double twoPioverb0 = 2.*M_PI/b0;
const double piCAoverdA = M_PI * CA / dA;
const double Mproton = 0.938;
const double rho0 = 0.17/std::pow(5.076,3);
const double r0 = 1.12*5.076;

// coupling in eHIJING
double alphas(double Q2){
    double Q2overmu2 = Q2/mu2;
    if (Q2overmu2<2.71828) return twoPioverb0;
    else return twoPioverb0/std::log(Q2/mu2);
}
// MultipleCollision related functions
double normG(double Q2x, double Qs2, double powerG, double lambdaG, double avgxG){
    double BetaG = gsl_sf_beta(1.+powerG, 1.+lambdaG);
    if (Q2x<Qs2) return  twoPioverb0*avgxG/BetaG/std::log(Q2x/mu2);
    else         return  twoPioverb0*avgxG/BetaG/(.5*std::log(Q2x*Qs2/mu2/mu2));
}
double PhiG(double x, double q2, double Qs2, double Q2x, double powerG, double lambdaG, double avgxG){
    double result;
    if (q2>Qs2) result = std::pow(1.-x, powerG)*std::pow(x, lambdaG) / (q2*alphas(q2));
    else        result = std::pow(1.-x, powerG)*std::pow(x, lambdaG) / (Qs2*alphas(Qs2));
    return result * normG(Q2x, Qs2, powerG, lambdaG, avgxG);
}
double alphas_PhiG(double x, double q2, double Qs2, double Q2x, double powerG, double lambdaG, double avgxG){
    double result;
    if (q2>Qs2) result = std::pow(1.-x, powerG)*std::pow(x, lambdaG) / q2;
    else        result = std::pow(1.-x, powerG)*std::pow(x, lambdaG) / Qs2;
    return result * normG(Q2x, Qs2, powerG, lambdaG, avgxG);
}
// induced radiation related functions
double CHT_F2(double u){
    return - gsl_sf_Ci(u) + std::log(u) + std::sin(u)/u;
}

double FiniteZcorr(double a){
    return -1.76*std::pow(a, 0.966) * std::exp(-0.907*std::pow(a, 2.871));
}

double CHT_F1(double x){
    return 1.0 - sin(x)/x;
}

///////// Class: Multiple Collision /////////////////////
// initializer
MultipleCollision::MultipleCollision(double Kfactor, double powerG,
                                     double lambdaG, double avgxG):
Kfactor_(Kfactor),
powerG_(powerG),
lambdaG_(lambdaG),
avgxG_(avgxG),
rd(),
gen(rd()),
flat_gen(0.,1.),
Qs2Table(2, {21,21}, // TA, ln(Q2/x) --> size and grid for Qs table
           {TAmin, std::log(1.0)},
           {TAmax, std::log(1e3)}
       ),
RateTable(3, {21,21,21}, // TA, ln(Q2/x), ln(l2) --> size and grid for Qs table
          {TAmin, std::log(1.0), std::log(0.01)},
          {TAmax, std::log(1e3), std::log(4.0)}
      )
{
}
// Tabulate Qs
void MultipleCollision::Tabulate(std::filesystem::path table_path){
    std::filesystem::path fname = table_path/std::filesystem::path("Qs.dat");
    if (std::filesystem::exists(fname)) {
        std::cout << "Loading Qs Table" << std::endl;
        std::ifstream f(fname.c_str());
        int count = 0;
        double entry;
        std::string line;
        while(getline(f, line)){
            std::istringstream in(line);
            in >> entry;
            if (count>=Qs2Table.size()){
                std::cerr << "Loading table Qs: mismatched size - 1" << std::endl;
                exit(-1);
            }
            Qs2Table.set_with_linear_index(count, entry);
            count ++;
        }
        if (count<Qs2Table.size()){
            std::cerr << "Loading table Qs: mismatched size - 2" << std::endl;
            exit(-1);
        }
    }
    else {
        if (std::filesystem::create_directory(table_path) ) std::cout << "Generating Qs^2 table" << std::endl;
        else std::cout << "Create dir failed" << std::endl;
        std::ofstream f(fname.c_str());

        // Table Qs as a function of lnx, lnQ2, TA
        for (int c=0; c<Qs2Table.size(); c++) {
            auto index = Qs2Table.LinearIndex2ArrayIndex(c);
            auto xvals = Qs2Table.ArrayIndex2Xvalues(index);
            double TA = xvals[0];
            double Q2xB = std::exp(xvals[1]);
            double Qs2 = compute_Qs2(TA, Q2xB);
            Qs2Table.set_with_linear_index(c, Qs2);
            f << Qs2  << std::endl;
        }
    }
    // conditioned rate table for gluons
    std::filesystem::path fname2 = table_path/std::filesystem::path("Rg.dat");
    if (std::filesystem::exists(fname2)) {
        std::cout << "Loading Rg Table" << std::endl;
        std::ifstream f(fname2.c_str());
        int count = 0;
        double entry;
        std::string line;
        while(getline(f, line)){
            std::istringstream in(line);
            in >> entry;
            if (count>=RateTable.size()){
                std::cerr << "Loading table Rg: mismatched size - 1" << std::endl;
                exit(-1);
            }
            RateTable.set_with_linear_index(count, entry);
            count ++;
        }
        if (count<RateTable.size()){
            std::cerr << "Loading table Rg: mismatched size - 2" << std::endl;
            exit(-1);
        }
    }
    else {
        if (std::filesystem::create_directory(table_path) ) std::cout << "Generating Rg table" << std::endl;
        else std::cout << "Create dir failed" << std::endl;
        std::ofstream f(fname2.c_str());

        // Table Qs as a function of lnx, lnQ2, TA
        for (int c=0; c<RateTable.size(); c++) {
            auto index = RateTable.LinearIndex2ArrayIndex(c);
            auto xvals = RateTable.ArrayIndex2Xvalues(index);
            double TA = xvals[0];
            double Q2xB = std::exp(xvals[1]);
            double l2 = std::exp(xvals[2]);
            double Rg = compute_Rg(TA, Q2xB, l2) * l2;
            RateTable.set_with_linear_index(c, Rg);
            f << Rg << std::endl;
        }
    }
}
// self-consisten euqation for Qs2
double MultipleCollision::Qs2_self_consistent_eq(double Qs2, double TA, double Q2x){
    double LHS = 0.;
    double scaledTA = piCAoverdA * TA;
    auto dfdq2 = [this, Qs2, Q2x](double ln1_q2oQs2) {
        double q2 = Qs2*(std::exp(ln1_q2oQs2)-1);
        double Jacobian = Qs2+q2;
        double x = q2/Q2x;
        return alphas_PhiG(x, q2, Qs2, Q2x, this->powerG_, this->lambdaG_, this->avgxG_) * Jacobian;
    };
    double error;
    double res =  normG(Q2x, Qs2, powerG_, lambdaG_, avgxG_) * scaledTA
                * quad_1d(dfdq2, {std::log(1+mu2/Qs2), std::log(1+Q2x/Qs2)}, error);
    return res - Qs2;
}
// Solver of the Qs2 self-consistent equation, using a simple bisection
double MultipleCollision::compute_Qs2(double TA, double Q2xB){
    // a naive bisection
    double xleft = mu2*.1, xright = Q2xB*10.0;
    const double EPS = 1e-4;
    double yleft = Qs2_self_consistent_eq(xleft, TA, Q2xB),
           yright = Qs2_self_consistent_eq(xright, TA, Q2xB);
    if (yleft*yright>0) {
        std::cout << "eHIJING warning: setting Qs2 = mu2*0.1" << std::endl;
        return xleft;
    } else {
        do {
            double xmid = (xright+xleft)/2.;
            double ymid = Qs2_self_consistent_eq(xmid, TA, Q2xB);
            if (yleft*ymid<0) {
                yright = ymid;
                xright = xmid;
            } else{
                yleft = ymid;
                xleft = xmid;
            }
        } while (xright-xleft > EPS);
        return (xright+xleft)/2.;
    }
}
double MultipleCollision::compute_Rg(double TA, double Q2xB, double l2){
  if (l2>Q2xB) return 0.;
  double qs2 = Qs2(Q2xB, TA);
  auto dfdlnq2 = [this, qs2, Q2xB](double lnq2) {
      double q2 = std::exp(lnq2);
      double xg = q2/Q2xB;
      return alphas_PhiG(xg, q2, qs2, Q2xB, this->powerG_, this->lambdaG_, this->avgxG_);
  };
  double error;
  double scaledrho = piCAoverdA * rho0;
  double res =  normG(Q2xB, qs2, powerG_, lambdaG_, avgxG_) * scaledrho
              * quad_1d(dfdlnq2, {std::log(l2), std::log(Q2xB)}, error);
  return res;
}
// Sample elastic collisio, without radiation
void MultipleCollision::sample_all_qt2(int pid, double E, double L, double xB, double Q2,
                             std::vector<double> & q2_list, std::vector<double> & t_list) {
    q2_list.clear();
    t_list.clear();
    double q2max = Q2/xB; // WK:but shouldn't we only allow multiple scatterings to be softer than Q2?
    double TA = rho0*L;
    double CR = (pid==21)? CA : CF;
    double qs2 = Qs2(xB, Q2, TA);
    double tildeTA = Kfactor_*normG(q2max, qs2, powerG_, lambdaG_, avgxG_)*M_PI*CR*TA/dA;
    double qs2overCTA = qs2/tildeTA;
    double q2 = q2max;
    while (q2 > mu2) {
        // sample the next hard multiple collision
        double lnr = std::log(flat_gen(gen));
        if (q2 < qs2) {
            q2 = q2*std::pow(1.0 + lambdaG_*lnr*qs2overCTA*std::pow(q2max/q2, lambdaG_),1./lambdaG_);
        }
        else{
            double Pc = tildeTA/(lambdaG_-1.)*( std::pow(q2/q2max, lambdaG_) / q2
                                             - std::pow(qs2/q2max, lambdaG_) / qs2
                                             );
            if (lnr > -Pc) {
                q2 = q2 * std::pow(1.0 + (lambdaG_-1.)*lnr*q2/tildeTA*std::pow(q2max/q2, lambdaG_),
                                   1./(lambdaG_-1.));
            }
            else {
                q2 = qs2 * std::pow(1.0 + lambdaG_*(lnr+Pc)*qs2overCTA*std::pow(q2max/qs2, lambdaG_),
                                   1./lambdaG_);
            }
        }
        double xg = q2/q2max;
        if (q2 > mu2 && flat_gen(gen) < std::pow(1.-xg, powerG_)) {
            q2_list.push_back(q2);
            t_list.push_back(flat_gen(gen)*L);
        }
    }
}

/////////// Class: eHIJING
// initializer
eHIJING::eHIJING(int mode, double Kfactor,
                 double powerG, double lambdaG, double avgxG):
MultipleCollision(Kfactor, powerG, lambdaG, avgxG),
mode_(mode),
Kfactor_(Kfactor),
powerG_(powerG),
lambdaG_(lambdaG),
avgxG_(avgxG),
rd(),
gen(rd()),
flat_gen(0.,1.),
GHT_z_kt2_Table(4, {11,41,41,201}, // TA, ln(Q2/x), ln(kt2), ln(3+L/tauf) --> size and grid for H-T table -- 1
           {TAmin, std::log(1.0), std::log(2*mu2), std::log(5+.05)},
           {TAmax, std::log(1e2), std::log(1e2),  std::log(5+500)}
       ),
GHT_kt2_Table(4, {11,41,41,201}, // TA, ln(Q2/x), ln(kt2), ln(3+L*kt2/2E) --> size and grid for H-T table -- 2
           {TAmin, std::log(1.0), std::log(2*mu2), std::log(5+.05)},
           {TAmax, std::log(1e2), std::log(1e2),  std::log(5+500)}
       )
{
}
// Tabulate the Qs and (if necessary) the generlized H-T / GLV table (collinear H-T do not need separate table)
void eHIJING::Tabulate(std::filesystem::path table_path){
    MultipleCollision::Tabulate(table_path);
    if (mode_==1) {
        // Generlized higher-twist able is at least 4-dimensional with each entry a 2D integral
        // the following routine will use the max number of hard ware concurrency of your computer
        // to parallel the computation of the table
        std::filesystem::path fname = table_path/std::filesystem::path("GHT.dat");
        if (std::filesystem::exists(fname)) {
            std::cout << "Loading GHT Table" << std::endl;
            std::ifstream f(fname.c_str());
            int count = 0;
            double entry1, entry2;
            std::string line;
            while(getline(f, line)){
                std::istringstream in(line);
                in >> entry1 >> entry2;
               //std::cout << entry1 << " " << entry2 << std::endl;
                if (count>=GHT_kt2_Table.size()){
                    std::cerr << "Loading table GHT: mismatched size - 1" << std::endl;
                    exit(-1);
                }
                GHT_z_kt2_Table.set_with_linear_index(count, entry1);
                GHT_kt2_Table.set_with_linear_index(count, entry2);
                count ++;
            }
            if (count<GHT_kt2_Table.size()){
                std::cerr << "Loading table GHT: mismatched size - 2" << std::endl;
                exit(-1);
            }
        } else {
            std::filesystem::create_directory(table_path);
            std::ofstream f(fname.c_str());
            // Table Qs as a function of lnx, lnQ2, TA
            std::atomic_int counter =  0;
            int percentbatch = int(GHT_z_kt2_Table.size()/100.);
            auto code = [this, percentbatch](int start, int end) {
                static std::atomic_int counter;
                for (int c=start; c<end; c++) {
                    counter ++;

                    if (counter%percentbatch==0) {
                      std::cout <<std::flush << "\r" << counter/percentbatch << "% done";
                    }
                    auto index = GHT_z_kt2_Table.LinearIndex2ArrayIndex(c);
                    auto xvals = GHT_z_kt2_Table.ArrayIndex2Xvalues(index);
                    double TA = xvals[0];
                    double Q2x = std::exp(xvals[1]);
                    double kt2 = std::exp(xvals[2]);
                    double Ltauf = std::exp(xvals[3])-5;
                    double Qs2 = MultipleCollision::Qs2(Q2x, TA);
                    double entry1 = 0., entry2 = 0.;
                    if (kt2<Q2x) {
                        entry1 = kt2/Qs2*compute_GHT_z_kt2(TA, Q2x, kt2, Ltauf, Qs2);
                        entry2 = kt2/Qs2*compute_GHT_kt2(TA, Q2x, kt2, Ltauf, Qs2);
                    }
                    GHT_z_kt2_Table.set_with_linear_index(c, entry1);
                    GHT_kt2_Table.set_with_linear_index(c, entry2);
                }
            };
            std::vector<std::thread> threads;
            int nthreads = std::thread::hardware_concurrency();
            int padding = int(std::ceil(GHT_z_kt2_Table.size()*1./nthreads));
            std::cout << "Generating GHT tables with " << nthreads << " thread" << std::endl;
            for(auto i=0; i<nthreads; ++i) {
                int start = i*padding;
                int end = std::min(padding*(i+1), GHT_z_kt2_Table.size());
                threads.push_back( std::thread(code, start, end) );
            }
            for(auto& t : threads) t.join();
            for (int c=0; c<GHT_z_kt2_Table.size(); c++) {
                f << GHT_z_kt2_Table.get_with_linear_index(c) << " "
                   << GHT_kt2_Table.get_with_linear_index(c)   << std::endl;
            }
        }
        std::cout << "... done" << std::endl;
    }
}

// computation of generalized HT table
double eHIJING::compute_GHT_z_kt2(double TA, double Q2xB, double kt2, double Ltauf, double Qs2) {
    double prefactor = 2.0*piCAoverdA*TA; // integrate phi from 0 to pi,i.e, the factor two
    auto dF1 = [this, Qs2, Q2xB, kt2, Ltauf](const double * x){
        double lnq2 = x[0], phi = x[1];
        double q2 = std::exp(lnq2);
        double cphi = std::cos(phi);
        double Ltaufq2 = q2/kt2*Ltauf;
        double A = 2*std::sqrt(Ltauf*Ltaufq2)*cphi;
        double B = Ltauf + Ltaufq2 - A;
        double Phase = A/B*(1.-std::sin(B)/B);
        double xg = q2/Q2xB;
        double aPhiG = alphas_PhiG(xg, q2, Qs2, Q2xB, this->powerG_, this->lambdaG_, this->avgxG_);
        std::vector<double> res{Phase*aPhiG/M_PI};
        return res;
    };
    // it is numerically more stable to apply quaduature separately to
    // two different domins of the integration 0<q2<kt2 and kt2<q2<Q^2/xB
    double xmin1[2] = {std::log(mu2), 0};
    double xmax1[2] = {std::log(kt2), M_PI};
    double xmin2[2] = {std::log(kt2), 0};
    double xmax2[2] = {std::log(Q2xB), M_PI};
    double err;
    double P1 = quad_nd(dF1, 2, 1, xmin1, xmax1, err)[0];
    double P2 = 0;
    if (kt2<Q2xB) P2 = quad_nd(dF1, 2, 1, xmin2, xmax2, err)[0];
    return prefactor*(P1+P2);
}

double eHIJING::compute_GHT_kt2(double TA, double Q2xB, double kt2, double Lk2_2E, double Qs2) {
    double prefactor = 2.0*M_PI*CA/dA*TA; // integrate phi from 0 to pi,i.e, the factor two
    auto dF1 = [this, Qs2, Q2xB, kt2, Lk2_2E](const double * x){
        double Emax = Q2xB/2/Mproton;
        double zmin = .2/Emax;
        double zmax = 1.0-zmax;
        double lnq2 = x[0], phi = x[1];
        double q2 = std::exp(lnq2);
        double cphi = std::cos(phi);
        double Lq2_2E = q2/kt2*Lk2_2E;
        double A = 2*std::sqrt(Lk2_2E*Lq2_2E)*cphi;
        double B = Lk2_2E + Lq2_2E - A;
        double Phase = A/B*( CHT_F2(B/zmin) - CHT_F2(B/zmax) + FiniteZcorr(B) );
        double xg = q2/Q2xB;
        double aPhiG = alphas_PhiG(xg, q2, Qs2, Q2xB, this->powerG_, this->lambdaG_, this->avgxG_);
        std::vector<double> res{Phase*aPhiG/M_PI};
        return res;
    };
    // it is numerically more stable to apply quaduature separately to
    // two different domins of the integration 0<q2<kt2 and kt2<q2<Q^2/xB
    double xmin1[2] = {std::log(mu2), 0};
    double xmax1[2] = {std::log(kt2), M_PI};
    double xmin2[2] = {std::log(kt2), 0};
    double xmax2[2] = {std::log(Q2xB), M_PI};
    double err;
    double P1 = quad_nd(dF1, 2, 1, xmin1, xmax1, err)[0];
    double P2 = 0;
    if (kt2<Q2xB) P2 = quad_nd(dF1, 2, 1, xmin2, xmax2, err)[0];
    return prefactor*(P1+P2);
}



// sampling
bool eHIJING::next_kt2(double & kt2, int pid, double E, double L,
                       double kt2min, double xB, double Q20) {
    double TA = rho0*L;
    double CR = (pid==21)? CA : CF;
    double Lover2E = L/2./E;
    double CR_2overb0 = CR*2.0/b0;
    if (mode_==0) { // Collinear HT
        double qhat_g = MultipleCollision::qhatA(xB, Q20, std::max(TA, 1.0*5.076*rho0));
        double qhat_g_2L = qhat_g*2.0*L;
        double zmin = std::min(.2/E, .5);
        double zmax = 1. - zmin;
        double logvac = std::log(1./zmin - 1.);
        double acceptance = 0.;
        while (acceptance<flat_gen(gen) && kt2>kt2min) {
            double Cvac = CR_2overb0 * logvac;
            double DeltaFmax = CHT_F2(kt2*Lover2E/zmin) - CHT_F2(kt2min*Lover2E/zmax);
            double Cmed = CR_2overb0 * qhat_g_2L/kt2min * DeltaFmax;
            double r = flat_gen(gen);
            double Ctot = Cmed + Cvac;
            kt2 = mu2 * std::pow(kt2/mu2, std::pow(r, 1.0/Ctot) );

            double DeltaFcorr = CHT_F2(kt2*Lover2E/zmin) - CHT_F2(kt2*Lover2E/zmax) + FiniteZcorr(kt2*Lover2E);
            acceptance = (Cvac + CR_2overb0 * qhat_g_2L/kt2 * DeltaFcorr) / Ctot;
        }
    }
    else{
        double Emax = Q20/xB/2/Mproton;
        double zmin = std::min(.2/Emax, .5);
        double zmax = 1. - zmin;
        double logvac = std::log(1./zmin - 1.);
        double qs2 = MultipleCollision::Qs2(xB, Q20, TA);
        double qs = std::sqrt(qs2);
        double kt2_c = 2*std::max(Kfactor_*2*M_PI*qs2 * 2*(CHT_F2(Mproton*L/zmin)-CHT_F2(Mproton*L/zmax)) / logvac,
                                  4*mu2);
        double lnkt2_c = std::log(kt2_c/mu2);
        double acceptance = 0.;
        while (acceptance<flat_gen(gen) && (kt2 > kt2min)) {
            double induced = induced_dFdkt2(xB, Q20, L, E, kt2)/logvac;
            double r = flat_gen(gen);
            double lninvr = std::log(1./r);
            double Cvac = CR_2overb0 * logvac;
            if ( kt2_c < kt2 || Kfactor_ < 1e-3) {
                double Pc = 2*Cvac*std::log(std::log(kt2/mu2)/lnkt2_c);
                if (lninvr < Pc || Kfactor_ < 1e-3) {
                    kt2 = mu2 * std::pow(kt2/mu2, std::pow(r, 1.0/Cvac/2.0) );
                    acceptance = ( 1 + induced)/2;
                    if (acceptance > 1.) std::cout << "warn-A " << acceptance << std::endl;
                } else {
                    kt2 = mu2*std::pow(kt2_c/mu2, std::pow(r*std::exp(Pc), 1./Cvac/(1+Kfactor_*L*qs)) );
                    acceptance = ( 1 + induced)/(1.0+Kfactor_*L*qs);
                    if (acceptance > 1.) std::cout << "warn-B " << acceptance << std::endl;
                }
            } else {
                kt2 = mu2*std::pow(kt2/mu2, std::pow(r, 1./Cvac/(1+4*Kfactor_*L*qs)) );
                acceptance = ( 1 + induced)/(1.0+4*Kfactor_*L*qs);
                if (acceptance > 1.) std::cout << "warn-C " << acceptance << std::endl;
            }
        }
    }
    return (kt2>kt2min);
}


bool eHIJING::sample_z(double & z, int pid, double E, double L, double kt2, double xB, double Q20) {
    double TA = rho0*L;
    double Lover2E = L/2./E;
    double W0 = 0.;
    if (mode_==0) { // Collinear HT
        double qhat_g = MultipleCollision::qhatA(xB, Q20, std::max(TA, 1.0*5.076*rho0));
        double qhat_g_2L_over_kt2 = qhat_g*2.0*L/kt2;
        double zmin = .2/E;
        double zmax = 1. - zmin;
        double acceptance = 0.;
        while (acceptance<flat_gen(gen)) {
            z = zmin*std::pow(zmax/zmin, flat_gen(gen));
            acceptance = (1.0 + qhat_g_2L_over_kt2 * CHT_F1(kt2*Lover2E/z/(1.-z)) )
                       / (1.0 + qhat_g_2L_over_kt2 * 1.22);
        }
    }
    else {
        double Emax = Q20/xB/2/Mproton;
        double zmin = .2/Emax;
        double zmax = 1. - zmin;
        double acceptance = 0.;
        double qs2 = MultipleCollision::Qs2(xB, Q20, TA);
        while (acceptance<flat_gen(gen)) {
            z = zmin*std::pow(zmax/zmin, flat_gen(gen));
            W0 = induced_dFdkt2dz(xB, Q20, L, E, kt2, z);
            acceptance = (1.0 + W0)
                       / (1.0 + 4*Kfactor_ * qs2/kt2 * 8*M_PI);
            if (acceptance > 1.) std::cout << "warn-D " << acceptance << std::endl;
        }
    }

    double acceptance = (pid==21)? (1.+std::pow(1.-z,3))/2. : (1.+std::pow(1.-z,2))/2.;
    double cut = .25*kt2/E/E;
    return (acceptance * (1.0 + (1.0-z/2.)*W0 ) / (1.0+W0) > flat_gen(gen))
        && (z*(1-z)>cut)
        && (std::pow(1-z, 2)>cut)
        && (std::pow(z,2)>cut)
        && (z*E > 0.5)
        && ((1-z)*E > 0.5);
}

// In medium fragmentation process
// initializer
InMediumFragmentation::InMediumFragmentation(int mode, double Kfactor,
                 double powerG, double lambdaG, double avgxG):
MultipleCollision(Kfactor, powerG, lambdaG, avgxG),
mode_(mode),
Kfactor_(Kfactor),
powerG_(powerG),
lambdaG_(lambdaG),
avgxG_(avgxG),
rd(),
gen(rd()),
flat_gen(0.,1.)
{
}

void InMediumFragmentation::Tabulate(std::filesystem::path table_path){
    MultipleCollision::Tabulate(table_path);
}

// provide these values in the nuclear rest frame!
bool InMediumFragmentation::next_radiation_CHT(int pid, double E, double L,
                    double xB, double Q20,
                    double kt2_max, double kt2_min,
                    double & omegaL, double & z){
  double zmin = 0.2/E;
  double zmax = 1.-zmin;
  if (zmin>zmax) {
    omegaL=0.;
    return false;
  }
  double TA = rho0*L;
  double CR = (pid==21)? CA : CF;
  double alphas0 = alphas(kt2_max);
  double qhat_g = MultipleCollision::qhatA(xB, Q20, std::max(TA, 1.0*5.076*rho0));
  double prefactor = alphas0*CR/M_PI * qhat_g*L*L/E
                  * ( 1./zmin-1./zmax
                   + std::log( (1.-zmin)/(1.-zmax) * zmax/zmin )
                     );
  double omegaL_max = std::sqrt(kt2_max)*L / 2.;
  double omegaL_min = kt2_min*L/(.5*E);
  // determine the initial condition for the omegaL evolution
  double omegaL_c = 2.705;

  // no phase-space left for low-Q radiation
  if (omegaL < omegaL_min) return false;
  // sample the next omegaL = L/tauf
  double acceptance = 0.;
  while (acceptance < flat_gen(gen) && omegaL > omegaL_min) {
      // try the next omegaL
      double r = flat_gen(gen);
      double lninvr = std::log(1./r);
      if (omegaL < omegaL_c ) {
          // probablity of no radiation
          double P1 = prefactor/12.*(std::pow(omegaL,2) - std::pow(omegaL_min,2));
          double Pveto = std::exp(-P1);
          if (r<Pveto) {
              omegaL = 0.;
              return false; // no radiation
          }
          omegaL = std::sqrt(std::pow(omegaL,2) - lninvr/prefactor*12.);
          acceptance = CHT_F1(omegaL)/(std::pow(omegaL,2)/6.);
      }else{
          if (omegaL_c < omegaL_min){
              // probablity of no next_radiation
              double P1 = prefactor * std::pow(omegaL_c,2)/6. * std::log(omegaL/omegaL_min);
              double Pveto = std::exp(-P1);
              if (r<Pveto) {
                  omegaL = 0.;
                  return false; // no radiation
              }
              omegaL = omegaL * std::exp(-lninvr * 6./std::pow(omegaL_c,2)/prefactor);
              acceptance = CHT_F1(omegaL)/(std::pow(omegaL_c,2)/6.);
          }
          else{
              // Find the probablity at the kink
              double P1c = prefactor * std::pow(omegaL_c,2)/6. * std::log(omegaL/omegaL_c);
              // probablity of no next_radiation
              double P1 = P1c
                        + prefactor/12. * (std::pow(omegaL_c,2) - std::pow(omegaL_min,2));
              double Pveto = std::exp(-P1), Pc = std::exp(-P1c);
              if (r<Pveto) {
                  omegaL = 0.;
                  return false; // no radiation
              } else if (r < Pc) {
                  omegaL = std::sqrt(std::pow(omegaL_c,2) - (lninvr - P1c)*12./prefactor);
                  acceptance = CHT_F1(omegaL)/(std::pow(omegaL,2)/6.);
              } else {
                  omegaL = omegaL * std::exp(-lninvr * 6./std::pow(omegaL_c,2)/prefactor);
                  acceptance = CHT_F1(omegaL)/(std::pow(omegaL_c,2)/6.);
              }
          }
      }
  }
  if (omegaL<omegaL_min) return false; // no radiation
  // sample z or equivalently kt2, reject cases where kT2 and z are out of bounds
  // dz/(z^2(1-z))
  acceptance = 0.;
  double Nhalf = 2.*(1/zmin-2.);
  double Ntot = Nhalf + 4.*std::log(2./(1.-zmax));
  double Phalf = Nhalf/Ntot;
  while (acceptance < flat_gen(gen)){
      double r = flat_gen(gen);
      if (r<Phalf){
          z = 1./(1./zmin - r*Ntot/2.);
          acceptance = 1./(2.*(1.-z));
      } else {
          z = 1. - 2.*std::exp( - (r*Ntot - Nhalf)/4. );
          acceptance = 1./(z*z*4.);
      }
      acceptance *= (pid==21)? (1.+std::pow(1.-z,3))/2. : (1.+std::pow(1.-z,2))/2.;
  }
  double kt2 = omegaL*(2.*z*(1.-z)*E)/L;
  double kt = std::sqrt(kt2);
  if (kt2<kt2_min || kt2>kt2_max || kt > z*E || kt  > (1.-z)*E)  return false;
  return true;
}


bool InMediumFragmentation::next_radiation_GHT(int pid, double E, double L,
                    double xB, double Q20,
                    double kt2_max, double kt2_min,
                    double & omegaL, double & z){
  double zmin = 0.2/E;
  double zmax = 1.-zmin;
  if (zmin>zmax) {
    omegaL=0.;
    return false;
  }
  double TA = rho0*L;
  double CR = (pid==21)? CA : CF;
  double alphas0 = alphas(kt2_max);
  double qhat_geff_max = 2*MultipleCollision::conditioned_qhat_gluon(
              kt2_min, std::max(TA, 1.0*5.076*rho0), xB, Q20);

  double prefactor = alphas0*CR/M_PI * qhat_geff_max*L*L/E
                  * ( 1./zmin-1./zmax
                   + std::log( (1.-zmin)/(1.-zmax) * zmax/zmin )
                     );
  double omegaL_max = std::sqrt(kt2_max)*L / 2.;
  double omegaL_min = kt2_min*L/(.5*E);
  // determine the initial condition for the omegaL evolution
  double omegaL_c = 2.705;

  // no phase-space left for low-Q radiation
  if (omegaL < omegaL_min) return false;
  // sample the next omegaL = L/tauf
  double acceptance = 0.;
  while (acceptance < flat_gen(gen) && omegaL > omegaL_min) {
      // try the next omegaL
      double r = flat_gen(gen);
      double lninvr = std::log(1./r);
      if (omegaL < omegaL_c ) {
          // probablity of no radiation
          double P1 = prefactor/12.*(std::pow(omegaL,2) - std::pow(omegaL_min,2));
          double Pveto = std::exp(-P1);
          if (r<Pveto) {
              omegaL = 0.;
              return false; // no radiation
          }
          omegaL = std::sqrt(std::pow(omegaL,2) - lninvr/prefactor*12.);
          acceptance = CHT_F1(omegaL)/(std::pow(omegaL,2)/6.);
      }else{
          if (omegaL_c < omegaL_min){
              // probablity of no next_radiation
              double P1 = prefactor * std::pow(omegaL_c,2)/6. * std::log(omegaL/omegaL_min);
              double Pveto = std::exp(-P1);
              if (r<Pveto) {
                  omegaL = 0.;
                  return false; // no radiation
              }
              omegaL = omegaL * std::exp(-lninvr * 6./std::pow(omegaL_c,2)/prefactor);
              acceptance = CHT_F1(omegaL)/(std::pow(omegaL_c,2)/6.);
          }
          else{
              // Find the probablity at the kink
              double P1c = prefactor * std::pow(omegaL_c,2)/6. * std::log(omegaL/omegaL_c);
              // probablity of no next_radiation
              double P1 = P1c
                        + prefactor/12. * (std::pow(omegaL_c,2) - std::pow(omegaL_min,2));
              double Pveto = std::exp(-P1), Pc = std::exp(-P1c);
              if (r<Pveto) {
                  omegaL = 0.;
                  return false; // no radiation
              } else if (r < Pc) {
                  omegaL = std::sqrt(std::pow(omegaL_c,2) - (lninvr - P1c)*12./prefactor);
                  acceptance = CHT_F1(omegaL)/(std::pow(omegaL,2)/6.);
              } else {
                  omegaL = omegaL * std::exp(-lninvr * 6./std::pow(omegaL_c,2)/prefactor);
                  acceptance = CHT_F1(omegaL)/(std::pow(omegaL_c,2)/6.);
              }
          }
      }
  }
  if (omegaL<omegaL_min) return false; // no radiation
  // sample z or equivalently kt2, reject cases where kT2 and z are out of bounds
  // dz/(z^2(1-z))
  acceptance = 0.;
  double Nhalf = 2.*(1/zmin-2.);
  double Ntot = Nhalf + 4.*std::log(2./(1.-zmax));
  double Phalf = Nhalf/Ntot;
  while (acceptance < flat_gen(gen)){
      double r = flat_gen(gen);
      if (r<Phalf){
          z = 1./(1./zmin - r*Ntot/2.);
          acceptance = 1./(2.*(1.-z));
      } else {
          z = 1. - 2.*std::exp( - (r*Ntot - Nhalf)/4. );
          acceptance = 1./(z*z*4.);
      }
      acceptance *= (pid==21)? (1.+std::pow(1.-z,3))/2. : (1.+std::pow(1.-z,2))/2.;
  }
  double kt2 = omegaL*(2.*z*(1.-z)*E)/L;
  double kt = std::sqrt(kt2);
  if (kt2<kt2_min || kt2>kt2_max || kt > z*E || kt  > (1.-z)*E)  return false;
  acceptance =  MultipleCollision::conditioned_qhat_gluon(
                kt2, std::max(TA, 1.0*5.076*rho0), xB, Q20) / qhat_geff_max;
  if (acceptance>1.) std::cout << "warn-Z " << acceptance << " " << kt2 << " " << kt2_min << std::endl;
  if (acceptance < flat_gen(gen))  return false;
  return true;
}

} //End eHIJING namespace
