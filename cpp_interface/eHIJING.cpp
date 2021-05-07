#include "eHIJING.h"
#include <iostream>
#include <fstream>
#include "integrator.h"
#include <cmath>
#include <thread>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_expint.h>
double CA = 3;
double dA = 8;
double CF = 4./3.;
double mu = 0.2;
double mu2 = 0.2*0.2;
double TAmin = 0.1/5.076/5.076;
double TAmax = 2.7/5.076/5.076;
double b0 = 9./2.;

eHIJING::eHIJING(int mode, double Kfactor):
mode_(mode), 
Kfactor_(Kfactor),
rho0_(0.17/std::pow(5.076,3)),
Mproton_(0.938),
rd(),
gen(rd()),
flat_gen(0.,1.),
Qs2Table(2, {21,21}, // TA, ln(Q2/x)
           {TAmin, std::log(1)}, 
           {TAmax, std::log(1e3)}
       ),
GHT_z_kt2_Table(4, {11,21,21,31}, // TA, ln(Q2/x), ln(kt2), ln(3+L/tauf)
           {TAmin, std::log(1),    std::log(1), std::log(5+.05)}, 
           {TAmax, std::log(1e3), std::log(1e2),  std::log(5+500)}
       ),
GHT_kt2_Table(4, {11,21,21,31}, // TA, ln(Q2/x), ln(kt2), ln(3+L/tauf)
           {TAmin, std::log(1),    std::log(1), std::log(5+.05)}, 
           {TAmax, std::log(1e3), std::log(1e2),  std::log(5+500)}
       )
{
}

void eHIJING::Tabulate(){
    std::ofstream f("Qs.dat");
    std::cout << "Generating Qs^2 table" << std::endl;
    // Table Qs as a function of lnx, lnQ2, TA
    for (int c=0; c<Qs2Table.size(); c++) {
        auto index = Qs2Table.LinearIndex2ArrayIndex(c);
        auto xvals = Qs2Table.ArrayIndex2Xvalues(index);
        double TA = xvals[0];
        double Q2xB = std::exp(xvals[1]);
        double Qs2 = compute_Qs2(TA, Q2xB);
        Qs2Table.set(index, Qs2);
        f << Qs2  << std::endl;
    }

    if (mode_==1) {
        std::ofstream f2("GHT.dat");
        // Table Qs as a function of lnx, lnQ2, TA
        auto code = [this](int start, int end) { 
            for (int c=start; c<end; c++) {
                auto index = GHT_z_kt2_Table.LinearIndex2ArrayIndex(c);
                auto xvals = GHT_z_kt2_Table.ArrayIndex2Xvalues(index);
                double TA = xvals[0];
                double Q2x = std::exp(xvals[1]);
                double kt2 = std::exp(xvals[2]);
                double Ltauf = std::exp(xvals[3])-5;
                double Qs2 = Qs2Table.interpolate({TA, std::log(Q2x)});
                double entry1 = 0., entry2 = 0.;
                if (kt2<Q2x) {
                    entry1 = kt2/Qs2*compute_GHT_z_kt2(TA, Q2x, kt2, Ltauf, Qs2);
                    entry2 = kt2/Qs2*compute_GHT_kt2(TA, Q2x, kt2, Ltauf, Qs2);
                }
                GHT_z_kt2_Table.set(index, entry1);
                GHT_kt2_Table.set(index, entry2);
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
        std::cout << "Tables done" << std::endl;
        for (int c=0; c<GHT_z_kt2_Table.size(); c++) {
            f2 << GHT_z_kt2_Table.get_with_linear_index(c) << " " 
               << GHT_kt2_Table.get_with_linear_index(c)   << std::endl;
        }
    }
}

// related to in-medium couplin, and the computation of Qs
double eHIJING::alphas(double Q2){
    return 2.*M_PI/b0/std::log(std::max(Q2, 4.0*mu2)/mu2);
}

double eHIJING::PhiG(double x, double q2, double Qs2){
    double normG = 0.5, lambdaG = 0.3;
    if (q2>Qs2) return normG/q2/alphas(q2) * std::pow(1-x, 4)/std::pow(x, lambdaG);
    else return normG/Qs2/alphas(Qs2) * std::pow(1-x, 4)/std::pow(x, lambdaG);
}

double eHIJING::alphas_PhiG(double x, double q2, double Qs2){
    double normG = 0.5, lambdaG = 0.3;
    if (q2>Qs2) return normG/q2 * std::pow(1-x, 4)/std::pow(x, lambdaG);
    else return normG/Qs2 * std::pow(1-x, 4)/std::pow(x, lambdaG);
}

double eHIJING::Qs2_self_consistent_eq(double Qs2, double TA, double Q2x){
    double LHS = 0.;
    double scaledTA = M_PI * CA / dA * TA;

    auto dfdq2 = [this, Qs2, Q2x](double ln1_q2oQs2) {
        double q2 = Qs2*(std::exp(ln1_q2oQs2)-1);
        double Jacobian = Qs2+q2;
        return this->alphas_PhiG(q2/Q2x, q2, Qs2) * Jacobian;
    };
    double error;
    double res = scaledTA * quad_1d(dfdq2, {0., std::log(1+Q2x/Qs2)}, error);
    return res - Qs2;
}

double eHIJING::compute_Qs2(double TA, double Q2xB){
    // a naive bisection
    double xleft = mu2*.1, xright = Q2xB*4.0;
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

// computation of generalized HT table
double eHIJING::compute_GHT_z_kt2(double TA, double Q2xB, double kt2, double Ltauf, double Qs2) {
    double prefactor = 2.0*M_PI*CA/dA*TA; // integrate phi from 0 to pi,i.e, the factor two
    auto dF1 = [this, Qs2, Q2xB, kt2, Ltauf](const double * x){
        double lnq2 = x[0], phi = x[1];
        double q2 = std::exp(lnq2);
        double cphi = std::cos(phi);
        double Ltaufq2 = q2/kt2*Ltauf;
        double A = 2*std::sqrt(Ltauf*Ltaufq2)*cphi;
        double B = Ltauf + Ltaufq2 - A;
        double Phase = A/B*(1.-std::sin(B)/B);
        double alphas_PhiG = this->alphas_PhiG(q2/Q2xB, q2, Qs2);
        std::vector<double> res{Phase*alphas_PhiG};
        return res;
    };
    // it is numerically more stable to apply quaduature separately to 
    // two different domins of the integration 0<q2<kt2 and kt2<q2<Q^2/xB
    double xmin1[2] = {std::log(1e-2), 0};
    double xmax1[2] = {std::log(kt2), M_PI};
    double xmin2[2] = {std::log(kt2), 0};
    double xmax2[2] = {std::log(Q2xB), M_PI};
    double err;
    double P1 = quad_nd(dF1, 2, 1, xmin1, xmax1, err)[0];
    double P2 = quad_nd(dF1, 2, 1, xmin2, xmax2, err)[0];
    return prefactor*(P1+P2);
}


double eHIJING::compute_GHT_kt2(double TA, double Q2xB, double kt2, double Lk2_2E, double Qs2) {
    double prefactor = 2.0*M_PI*CA/dA*TA; // integrate phi from 0 to pi,i.e, the factor two
    auto dF1 = [this, Qs2, Q2xB, kt2, Lk2_2E](const double * x){
        double zmin = 1e-2;
        double lnq2 = x[0], phi = x[1];
        double q2 = std::exp(lnq2);
        double cphi = std::cos(phi);
        double Lq2_2E = q2/kt2*Lk2_2E;
        double A = 2*std::sqrt(Lk2_2E*Lq2_2E)*cphi;
        double B = Lk2_2E + Lq2_2E - A;
        double Phase = A/B*( gsl_sf_Ci(B)-gsl_sf_Ci(B/zmin)
                           - std::log(zmin) 
                           - std::sin(B)/B + std::sin(B/zmin) / (B/zmin)
                            );
        double alphas_PhiG = this->alphas_PhiG(q2/Q2xB, q2, Qs2);
        std::vector<double> res{Phase*alphas_PhiG};
        return res;
    };
    // it is numerically more stable to apply quaduature separately to 
    // two different domins of the integration 0<q2<kt2 and kt2<q2<Q^2/xB
    double xmin1[2] = {std::log(1e-2), 0};
    double xmax1[2] = {std::log(kt2), M_PI};
    double xmin2[2] = {std::log(kt2), 0};
    double xmax2[2] = {std::log(Q2xB), M_PI};
    double err;
    double P1 = quad_nd(dF1, 2, 1, xmin1, xmax1, err)[0];
    double P2 = quad_nd(dF1, 2, 1, xmin2, xmax2, err)[0];
    return prefactor*(P1+P2);
}

// sampling
double CHT_F2(double x){
    return - gsl_sf_Ci(x) + std::log(x) + std::sin(x)/x;
}
double CHT_F1(double x){
    return 1.0 - sin(x)/x;
}

bool eHIJING::next_kt2(double & kt2, int pid, double E, double L, 
                       double kt2min, double xB, double Q20) {
    double TA = rho0_*L;
    double CR = (pid==21)? CA : CF;
    double Lover2E = L/2./E;
    double CR_2overb0 = CR*2.0/b0;
    double qhat_g = qhatA(xB, Q20, TA);
    double qhat_g_2L = qhat_g*2.0*L;
    double zmin = .01;
    double zmax = 1. - zmin;
    double logvac = std::log(1./zmin - 1.);
    if (mode_==0) { // Collinear HT       
        double acceptance = 0.;
        while (acceptance<flat_gen(gen) && kt2>kt2min) {
            double Cvac = CR_2overb0 * logvac;
            double DeltaFmax = CHT_F2(kt2*Lover2E/zmin) - CHT_F2(kt2min*Lover2E/zmax);
            double Cmed = Kfactor_* CR_2overb0 * qhat_g_2L/kt2min * DeltaFmax;
            double r = flat_gen(gen);
            double Ctot = Cmed + Cvac;
            //std::cout << kt2 << " --- " << r << " " << Ctot << std::endl;
            kt2 = mu2 * std::pow(kt2/mu2, std::pow(r, 1.0/Ctot) );

            double DeltaFcorr = CHT_F2(kt2*Lover2E/zmin) - CHT_F2(kt2*Lover2E/zmax);
            acceptance = (Cvac + Kfactor_* CR_2overb0 * qhat_g_2L/kt2 * DeltaFcorr) / Ctot;
            //std::cout << kt2 << " " <<acceptance << std::endl;
        }
    }
    else{
        double qs2 = Qs2(xB, Q20, TA);
        double qs = 2*std::sqrt(mu2+qs2);
        double kt2_c = Kfactor_*2*M_PI*qs2 * 2*(CHT_F2(Mproton_*L/zmin)) / logvac;
        double lnkt2_c = std::log(lnkt2_c);
        double acceptance = 0.;
        while (acceptance<flat_gen(gen) && kt2>kt2min) {
            double r = flat_gen(gen);
            double lninvr = std::log(1./r);
            double Cvac = CR_2overb0 * logvac;
            //std::cout << kt2_c;
            if ( kt2_c < kt2 || L<.5 ) {
                double Pc = 2*Cvac*std::log(std::log(kt2/mu2)/lnkt2_c);
                if (lninvr < Pc || L<.5) {
                    kt2 = mu2 * std::pow(kt2/mu2, std::pow(r, 1.0/Cvac/2.0) );
                    acceptance = ( 1 + Kfactor_*induced_dFdkt2(xB, Q20, L, E, kt2) /logvac)/2;
                    if (acceptance > 1.) std::cout << "warn-A " << acceptance << std::endl;
                } else {
                    kt2 = mu2*std::pow(kt2_c/mu2, std::pow(r*std::exp(Pc), 1./Cvac/(1+Kfactor_*L*qs)) ); 
                    acceptance = ( 1 + Kfactor_*induced_dFdkt2(xB, Q20, L, E, kt2) /logvac)/(1.0+Kfactor_*L*qs);
                    if (acceptance > 1.) std::cout << "warn-B " << acceptance << std::endl;                    
                }
            } else {
                kt2 = mu2*std::pow(kt2/mu2, std::pow(r, 1./Cvac/(1+Kfactor_*L*qs)) ); 
                acceptance = ( 1 + Kfactor_*induced_dFdkt2(xB, Q20, L, E, kt2) /logvac)/(1.0+Kfactor_*L*qs);
                if (acceptance > 1.) std::cout << "warn-C " << acceptance << std::endl;          
            }
        }
    }
    return (kt2>kt2min);
}


bool eHIJING::sample_z(double & z, int pid, double E, double L, double kt2, double xB, double Q20) {
    double TA = rho0_*L;
    double qhat_g = qhatA(xB, Q20, TA);
    double qhat_g_2L_over_kt2 = qhat_g*2.0*L/kt2;
    double zmin = 1e-3;
    double zmax = 1. - zmin;
    double Lover2E = L/2./E;
    if (mode_==0) { // Collinear HT   
        double acceptance = 0.;
        while (acceptance<flat_gen(gen)) {
            z = zmin*std::pow(zmax/zmin, flat_gen(gen));
            acceptance = (1.0 + Kfactor_*qhat_g_2L_over_kt2 * CHT_F1(kt2*Lover2E/z) ) 
                       / (1.0 + Kfactor_*qhat_g_2L_over_kt2 * 1.22);
        }
    }
    else {
        double acceptance = 0.;
        double qs2 = Qs2(xB, Q20, TA);
        while (acceptance<flat_gen(gen)) {
            z = zmin*std::pow(zmax/zmin, flat_gen(gen));
            acceptance = (1.0 + Kfactor_*induced_dFdkt2dz(xB, Q20, L, E, kt2, z))
                       / (1.0 + Kfactor_*qs2/kt2 * 8*M_PI);
            if (acceptance > 1.) std::cout << "warn-D " << acceptance << std::endl;  
        }
    }
    
    double acceptance = (pid==21)? (1.+std::pow(1.-z,3))/2. : (1.+std::pow(1.-z,2))/2.;
    return (acceptance>flat_gen(gen));
}









