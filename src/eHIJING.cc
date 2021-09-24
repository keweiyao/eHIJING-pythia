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
const double TAmin = 0.1/5.076/5.076;
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
     return 1;
     //double BetaG = gsl_sf_beta(1.+powerG, 1.+lambdaG);
    //if (Q2x<Qs2) return  twoPioverb0*avgxG/BetaG*Qs2/Q2x/std::log(Qs2/mu2);
    //else         return  twoPioverb0*avgxG/BetaG/std::log(Qs2/mu2)/(1.+std::log(Q2x/Qs2));
    //double GammaG = gsl_sf_gamma(powerG+1.)/std::pow(1.+lambdaG, 1.+powerG);
    //if (Q2x<Qs2) return  twoPioverb0*avgxG/BetaG/std::log(Q2x/mu2);
    //else         return  twoPioverb0*avgxG/BetaG/(.5*std::log(Q2x*Qs2/mu2/mu2));
}
double PhiG(double x, double q2, double Qs2, double Q2x, double powerG, double lambdaG, double avgxG){
    double result;
    if (q2>Qs2) result = std::pow(1.-x, powerG)*std::pow(x, lambdaG) / (q2*alphas(Qs2));
    else        result = std::pow(1.-x, powerG)*std::pow(x, lambdaG) / (Qs2*alphas(Qs2));
    return result * normG(Q2x, Qs2, powerG, lambdaG, avgxG);
    //if (q2>Qs2) result = std::pow(-std::log(x), powerG)*std::pow(x, lambdaG) / (q2*alphas(q2));
    //else        result = std::pow(-std::log(x), powerG)*std::pow(x, lambdaG) / (Qs2*alphas(Qs2));
    //return result * normG(Q2x, Qs2, powerG, lambdaG, avgxG);

}
double alphas_PhiG(double x, double q2, double Qs2, double Q2x, double powerG, double lambdaG, double avgxG){
    double result;
    if (q2>Qs2) result = std::pow(1.-x, powerG)*std::pow(x, lambdaG) / q2;
    else        result = std::pow(1.-x, powerG)*std::pow(x, lambdaG) / Qs2;
    return result * normG(Q2x, Qs2, powerG, lambdaG, avgxG);
    //if (q2>Qs2) result = std::pow(-std::log(x), powerG)*std::pow(x, lambdaG) / q2;
    //else        result = std::pow(-std::log(x), powerG)*std::pow(x, lambdaG) / Qs2;
    //return result * normG(Q2x, Qs2, powerG, lambdaG, avgxG);
}

// integrate du (1-cos(1/u)) / u
double inte_C(double u){
    return gsl_sf_Ci(1./u) + std::log(u);
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
Qs2Table(2, {51,51}, // TA, ln(Q2/x) --> size and grid for Qs table
           {TAmin, std::log(1.0)},
           {TAmax, std::log(1e5)}
       ),
RateTable(3, {21,21,21}, // TA, ln(Q2/x), ln(l2) --> size and grid for Qs table
          {TAmin, std::log(1.0), std::log(0.01)},
          {TAmax, std::log(1e5), std::log(4.0)}
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
    double xleft = mu2*.25, xright = Q2xB*10.0;
    const double EPS = 1e-4;
    double yleft = Qs2_self_consistent_eq(xleft, TA, Q2xB),
           yright = Qs2_self_consistent_eq(xright, TA, Q2xB);
    if (yleft*yright>0) {
        std::cout << "eHIJING warning: setting Qs2 = mu2/4" << std::endl;
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
// Sample all elastic collisio, without radiation, ordered from high scale to low
int MultipleCollision::sample_all_qt2(int pid, double E, double L, double xB, double Q2,
                             std::vector<double> & q2_list, std::vector<double> & t_list,
                             std::vector<double> & phi_list) {
    q2_list.clear();
    t_list.clear();
    phi_list.clear();
    double q2max = Q2/xB; // WK: but shouldn't we only allow multiple scatterings to be softer than Q2?
    double TA = rho0*L;
    double CR = (pid==21)? CA : CF;
    double qs2 = Qs2(xB, Q2, TA);
    double tildeTA = Kfactor_*normG(q2max, qs2, powerG_, lambdaG_, avgxG_)*M_PI*CR*TA/dA;
    if (tildeTA<1e-15) return q2_list.size();
    double qs2overCTA = qs2/tildeTA;
    double q2 = q2max;
    double q2min = EHIJING::mu2/16.;
    while (q2 > q2min) {
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
        double t = flat_gen(gen)*L;
        if (   q2>q2min
            && flat_gen(gen) < std::pow(1.-xg, powerG_) // correct the distribution at large xg
            && t > r0 // exclude the collision with the original nucleon from hard collision
           ) {
            q2_list.push_back(q2);
            t_list.push_back(t);
            phi_list.push_back(2.*M_PI*flat_gen(gen));
        }
    }
    return q2_list.size();
}

// Sample exact one single collision; therefore, not a rate sampling!
void MultipleCollision::sample_single_qt2(int pid, double E, double L, double xB, double Q2,
                                         double & qx, double & qy, double & xg, double & tq,
                                         double minimum_q2) {
    double q2max = Q2/xB; // WK:but shouldn't we only allow multiple scatterings to be softer than Q2?
    double TA = rho0*L;
    double CR = (pid==21)? CA : CF;
    double qs2 = Qs2(xB, Q2, TA);
    double q2 = 0., phi = 0.;
    double minimum_q2_over_Qs2 = minimum_q2 / qs2;
    double maximum_q2_over_Qs2 = q2max / qs2;
    if (minimum_q2 > qs2) {
        double Ntot = (std::pow(minimum_q2_over_Qs2, lambdaG_-1)
                     - std::pow(maximum_q2_over_Qs2, lambdaG_-1))/(1.-lambdaG_);
        double r = flat_gen(gen);
        q2 = qs2*std::pow(1./(std::pow(minimum_q2_over_Qs2, lambdaG_-1)
                          - r*Ntot*(1.-lambdaG_)),
                          1./(1.-lambdaG_));
    }else{
        double N1 = (1. - std::pow(minimum_q2_over_Qs2, lambdaG_))/lambdaG_;
        double N2 = (1. - std::pow(maximum_q2_over_Qs2, lambdaG_-1))/(1.-lambdaG_);
        double Ntot = N1 + N2;
        double r = flat_gen(gen);
        if (r<N1/Ntot) {
            q2 = qs2*std::pow(Ntot * r * lambdaG_ + std::pow(minimum_q2, lambdaG_), 1./lambdaG_);
        }else{
            q2 = qs2*std::pow(1./(1. - (r*Ntot-N1)*(1.-lambdaG_)), 1./(1.-lambdaG_));
        }
    }
    // step 2, sample phi2
    phi = 2*M_PI*flat_gen(gen);
    double q = std::sqrt(q2);
    qx = q*std::cos(phi);
    qy = q*std::sin(phi);
    xg = q2/Q2*xB;
    tq = L*flat_gen(gen);
    return;
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
GHT_Angular_Table(2, {51, 51}, // X = delta = 2kq/(k^2+q^22),
                               // ln(1+Y) = log(1+ (|k|-|q|)^2*t/(2*z*(1-z)*E) )
                               // 0.5<delta<1, ln(1)<ln(1+Y)<ln(11)
           {0.5, 1e-3},
           {0.99, std::log(11.0)}
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
                in >> entry1;
                if (count>=GHT_Angular_Table.size()){
                    std::cerr << "Loading table GHT: mismatched size - 1" << std::endl;
                    exit(-1);
                }
                GHT_Angular_Table.set_with_linear_index(count, entry1);
                count ++;
            }
            if (count<GHT_Angular_Table.size()){
                std::cerr << "Loading table GHT: mismatched size - 2" << std::endl;
                exit(-1);
            }
        } else {
            std::filesystem::create_directory(table_path);
            std::ofstream f(fname.c_str());
            // Table Qs as a function of lnx, lnQ2, TA
            std::atomic_int counter =  0;
            int percentbatch = int(GHT_Angular_Table.size()/100.);
            auto code = [this, percentbatch](int start, int end) {
                static std::atomic_int counter;
                for (int c=start; c<end; c++) {
                    counter ++;

                    if (counter%percentbatch==0) {
                      std::cout <<std::flush << "\r" << counter/percentbatch << "% done";
                    }
                    auto index = GHT_Angular_Table.LinearIndex2ArrayIndex(c);
                    auto xvals = GHT_Angular_Table.ArrayIndex2Xvalues(index);
                    double X = xvals[0];
                    double ln1Y = xvals[1];
                    double Y = std::exp(ln1Y)-1.;
                    double entry1 = 0.;
                    entry1 = compute_GHT_Angular_Table(X, Y);
                    GHT_Angular_Table.set_with_linear_index(c, entry1);
                }
            };
            std::vector<std::thread> threads;
            int nthreads = std::thread::hardware_concurrency();
            int padding = int(std::ceil(GHT_Angular_Table.size()*1./nthreads));
            std::cout << "Generating GHT angular tables with " << nthreads << " thread" << std::endl;
            for(auto i=0; i<nthreads; ++i) {
                int start = i*padding;
                int end = std::min(padding*(i+1), GHT_Angular_Table.size());
                threads.push_back( std::thread(code, start, end) );
            }
            for(auto& t : threads) t.join();
            for (int c=0; c<GHT_Angular_Table.size(); c++) {
                f << GHT_Angular_Table.get_with_linear_index(c) << std::endl;
            }
        }
        std::cout << "... done" << std::endl;
    }
}

// computation of generalized HT table
double eHIJING::compute_GHT_Angular_Table(double X, double Y) {
    double A = Y/(1.-X);
    double B = Y*X/(1.-X);
    auto dfdphi = [X, Y, A, B](double phi) {
        double cosphi = std::cos(phi);
        double xcphi = X*cosphi;
        return xcphi/(1.-xcphi) * (1.-std::cos(A-B*cosphi));
    };
    double error;
    double result = quad_1d(dfdphi, {0., M_PI}, error);
    result /= M_PI;
    return result;
}

bool eHIJING::next_kt2_stochastic(double & kt2, int pid,
                        double E,
                        double kt2min,
                        std::vector<double> qt2s,
                        std::vector<double> ts) {
    double CR = (pid==21)? CA : CF;
    double CAoverCR = CA/CR;
    double CR_2overb0 = CR*2.0/b0;
    double zmin = std::min(.4/E, .4);
    double zmax = 1. - zmin;
    double logvac = std::log(zmax/zmin);
    int Ncolls = ts.size();
    double acceptance = 0.;

    if (mode_ == 0){
        while (acceptance<flat_gen(gen) && kt2>kt2min) {
            double maxlogmed = 0.;
            for (int i=0; i<Ncolls; i++){
                double q2 = qt2s[i], t = ts[i];
                if (q2>kt2) continue;
                double phasemax = inte_C((2*zmax*E)/(t*kt2min)) - inte_C((2*zmin*E)/(t*kt2));
                maxlogmed += q2 * phasemax ;
            }
            maxlogmed *= 2. / kt2min * CAoverCR;
            double Crad = CR_2overb0 * (logvac + maxlogmed);
            double r = flat_gen(gen);
            kt2 = mu2 * std::pow(kt2/mu2, std::pow(r, 1.0/Crad) );
            double logmed = 0.;
            for (int i=0; i<Ncolls; i++){
                double q2 = qt2s[i], t = ts[i];
                if (q2>kt2) continue;
                double phase = inte_C((2*zmax*E)/(t*kt2)) - inte_C((2*zmin*E)/(t*kt2));
                logmed += q2 * phase;
            }
            logmed *= 2./kt2*CAoverCR;
            acceptance = (logvac + logmed) / (logvac + maxlogmed);
        }
    } else {
        // Genearlized formula
        double maxdiffz = 1./zmin - 1./zmax + 2.*logvac;
        while (acceptance<flat_gen(gen) && kt2>kt2min) {
            double maxmedcoeff = 0.;
            for (int i=0; i<Ncolls; i++){
                double q2 = qt2s[i], t = ts[i];
                maxmedcoeff += t*(kt2 + q2);
            }
            maxmedcoeff *= CAoverCR/(2.*E)*maxdiffz;
            double Crad = CR_2overb0 * (logvac + maxmedcoeff);
            double r = flat_gen(gen);
            kt2 = mu2 * std::pow(kt2/mu2, std::pow(r, 1.0/Crad) );
            // compute acceptance
            double medcoeff = 0.;
            for (int i=0; i<Ncolls; i++){
                double q2 = qt2s[i], t = ts[i];
                medcoeff += t*(kt2 + q2);
            }
            medcoeff *= CAoverCR/(2.*E)*maxdiffz;
            acceptance = (logvac + medcoeff) / (logvac + maxmedcoeff);
        }
    }
    return (kt2>kt2min);
}

double eHIJING::sample_z_stochastic(double & z, int pid,
                        double E,
                        double kt2,
                        std::vector<double> qt2s,
                        std::vector<double> ts,
                        std::vector<double> phis) {
    double CR = (pid==21)? CA : CF;
    double zmin = std::min(.4/E, .4);
    double zmax = 1. - zmin;

    int Ncolls = ts.size();
    double acceptance = 0.;
    double weight = 1.0;
    if (mode_==0){
        while (acceptance<flat_gen(gen)) {
            z = zmin*std::pow(zmax/zmin, flat_gen(gen));
            double tauf = 2.*z*(1.0-z)*E/kt2;
            double w = 1.0, wmax = 1.0;
            for (int i=0; i<Ncolls; i++){
                double q2 = qt2s[i], t = ts[i];
                if (q2>kt2) continue;
                w += 2.*q2/kt2 * CA / CR * (1.-cos(t/tauf));
                wmax += 4.*q2/kt2 * CA / CR;
            }
            acceptance = w/wmax;
        }
    } else {
        double a = 0.;
        for (int i=0; i<Ncolls; i++){
            double q2 = qt2s[i], t = ts[i];
            a += (kt2 +q2)*t;
        }
        a *= CA/CR/(2.*E);
        double Norm = (1. + 2.*a)*std::log(zmax/zmin) + a*(1./zmin-1/zmax);
        // sample z ~ 1/z + a/[z^2(1-z)]
      	double left = zmin;
      	double right = zmax;
      	double mid = (left+right)/2.;
      	double r = flat_gen(gen);
      	while(right-left > 1e-3) {
        		mid = (left+right)/2.;
        		double fmid = ( (1+a)*std::log(mid/zmin)
        			    + a*(1./zmin-1/mid)
        			    + a*std::log(zmax/(1-mid)) ) / Norm;
        		if (fmid<r) left = mid;
        		else right = mid;
      	}
        z = mid;
        if (Ncolls==0) weight=1.;
        else {
           double wmax = CR/CA;
           for (int i=0; i<Ncolls; i++){
               double q2 = qt2s[i], t = ts[i];
               wmax += (kt2+q2)*t;
           }
           wmax /= (2.*z*(1-z)*E);

           double w = CR/CA;
           for (int i=0; i<Ncolls; i++){
               double dw;
               double q2 = qt2s[i], t = ts[i], phi = phis[i];
               double A = (kt2+q2)*t/(2.*z*(1-z)*E),
                      B = 2.*std::sqrt(kt2*q2)*t/(2.*z*(1-z)*E);
               double X = B/A;
               double Y = A*(1.-X);
               if (X<.5){
                    double jv0 = std::cyl_bessel_j(0,B),
                           jv1 = std::cyl_bessel_j(1,B);
                    dw = - X*std::sin(A)*jv1
                         + X*X * ( .5 + X*std::cos(A) * (jv1/B - jv0) );
               } else {
                   if (Y>10.){
                       dw = 1./std::sqrt(1.-X*X)-1;
                   }
                   else {
                       dw = GHT_Angular_Table.interpolate({X, std::log(1.+Y)});
                   }
               }
               w += dw;
           }
           weight = w/wmax;
        }
    }
    return std::min(std::max(weight,0.),1.);
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
  double alphas0 = alphas(std::sqrt(kt2_max*mu2));
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
  double alphas0 = alphas(std::sqrt(kt2_max*mu2));
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
  if (kt2<z*(1.-z)*kt2_min || kt2>kt2_max || kt > z*E || kt  > (1.-z)*E)  return false;
  acceptance =  MultipleCollision::conditioned_qhat_gluon(
                kt2, std::max(TA, .2*5.076*rho0), xB, Q20) / qhat_geff_max;
  if (acceptance>1.) std::cout << "warn-Z " << acceptance << " " << kt2 << " " << kt2_min << std::endl;
  if (acceptance < flat_gen(gen))  return false;
  return true;
}

} //End eHIJING namespace
