#ifndef EHIJING_H
#define EHIJING_H
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <filesystem>


// A light-weighted N-dimensional table class that supports
// 1) Setting and retriving data using linear or N-dim indices
// 2) Interpolating within the range of the grid
// 3) Only equally-spaced grid.
class table_nd{
private:
    // the dimensional D of the table and the size of a unit cube 2^D
    const int dim_, power_dim_;
    // total number of data points
    int total_size_;
    // Linear data arrat, grid step, grid minimum and maximum
    std::vector<double> data_, step_, grid_min_, grid_max_;
    // Shape of each dimension [N1, ... ND], total_size = N1*N2...*ND 
    // and the conjugate block size [N2*...*ND, N3*...ND, ..., ND, 1]
    std::vector<int> shape_, conj_size_;
public:
    // initializer, given dimension, shape, grid minimum and maximum
    table_nd(int dim, std::vector<int> shape, std::vector<double> gridmin, std::vector<double> gridmax):
    dim_(dim), power_dim_(std::pow(2,dim)), shape_(shape), grid_min_(gridmin), grid_max_(gridmax){
        step_.resize(dim_);
        for (int i=0; i<dim_; i++) step_[i] = (grid_max_[i]-grid_min_[i])/(shape_[i]-1);
        conj_size_.resize(dim_);
        conj_size_[dim_-1] = 1;
        for (int i=1; i<dim_; i++) conj_size_[dim_-i-1] = conj_size_[dim_-i] * shape[dim_-i];
        total_size_ = conj_size_[0] * shape[0];
        data_.resize(total_size_);
    }
    // Convert a N-dim index to a linear index
    int ArrayIndex2LinearIndex(std::vector<int> & index){
        int k = 0;
        for (int i=0; i<dim_; i++) k += index[i] * conj_size_[i];
        return k;
    }
    // Convert a linear index to a N-dim index
    std::vector<int> LinearIndex2ArrayIndex(int k){
        std::vector<int> index;
        int K = k;
        for (int i=0; i<dim_; i++) {
            index.push_back(int(floor(K/conj_size_[i])));
            K = K%conj_size_[i];
        }
        return index;
    }
    // Convert a N-dim index to the physical values of each independent variable
    std::vector<double> ArrayIndex2Xvalues(std::vector<int> index){
        std::vector<double> xvals;
        for (int i=0; i<dim_; i++) xvals.push_back(grid_min_[i]+step_[i]*index[i]);
        return xvals;
    }
    int dim() const {return dim_;}; // return dimension
    int size() const {return total_size_;}; // return total size
    std::vector<int> shape() const { return shape_; }    // return shape
    std::vector<double> step() const{ return step_; }    // return step
    // set an element with a N-dim index
    void set_with_index(std::vector<int> index, double value) { data_[ArrayIndex2LinearIndex(index)] = value; }
    // set an element with a linear index
    void set_with_linear_index(int k, double value) { data_[k] = value; }
    // get an element with a N-dim index   
    double get_with_index(std::vector<int> index) { return data_[ArrayIndex2LinearIndex(index)]; }
    // get an element with a linear index   
    double get_with_linear_index(int k) { return data_[k]; }
    // Interpolating at a given array of physical varaibles X = [x1, x2, ..., xD]
    double interpolate(std::vector<double> Xinput) {
       // First, find the unit cube that contains the input point
       std::vector<int> start_index; start_index.resize(dim_);
       std::vector<double> w; w.resize(dim_);
       for(auto i=0; i<dim_; i++) {
           double x = (Xinput[i] - grid_min_[i])/step_[i];
           x = std::min(std::max(x, 0.), shape_[i]-1.);
           size_t nx = size_t(std::floor(x));
           double rx = x - nx;
           start_index[i] = nx;
           w[i] = rx;
       }
       // Second, use linear interpolation within the unit cube
       std::vector<int> index(dim_);
       double result = 0.0;
       for (auto i=0; i<power_dim_; i++) {
           auto W = 1.0;
           for (auto j=0; j<dim_; j++) {
               index[j] = start_index[j] + ((i & ( 1 << j )) >> j);
               W *= (index[j]==start_index[j])?(1.-w[j]):w[j];
           }
           result += data_[ArrayIndex2LinearIndex(index)]*W;
       }
       return result;
    }
};


// The main eHIJING class
// Main usage / interfaces:
// 1) Sample dP/dkt2, given the previous kt2
// 2) Sample dP/dkt2/dz, given the current kt2
// 3) Tabulate the static&soft generlized-HT / GLV table
// 4) Tabulate the saturation scale and compute the effective qhat parameter
class eHIJING{
private:
    // mode=0: collinear H-T;
    // mode=1: static&soft generalized H-T / GLV
    const int mode_;
    // A K factor enhancing the medium-induced term for testing,
    // and density of nuclear matter and mass of proton, everything in units of GeV^{...}
    const double Kfactor_, rho0_, Mproton_;
    // Random number generators
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> flat_gen;
    // table for the Qs2, and generalized Higher-twist / GLV
    table_nd Qs2Table, GHT_z_kt2_Table, GHT_kt2_Table; 
    // LO running coupling constant
    double alphas(double Q2);
    // Shape of a un parametrized kT-dependent gluon distribution function, only a function of x, q2 and Qs2
    // The normalization at given x and Q2 is choosen such that <x> of gluon is the user given value (~0.3--0.5)
    double PhiG(double x, double q2, double Qs2);
    // alphas*PhiG
    double alphas_PhiG(double x, double q2, double Qs2);
    // the self consistent equation for Qs2: Qs2[Phi_G(Qs2), TA, x, Q2] - Qs2 = 0
    double Qs2_self_consistent_eq(double Qs2, double TA, double Q2x);
    // self-consistently solve for Qs2(Q^2/x, TA)
    double compute_Qs2(double TA, double Q2xB);
    // compute the induced term in generalized HT:
    // a): this one computes F(z, kt2) in terms of a table as function of TA, Q2/x, kt2, L/tauf=L*kt^2/(2z(1-z)E)
    //     where F is dP/dz/dkt2 = alphas*CR*P(z)/(2*pi) * 1/kt2 * (1 + F(z, kt2))
    //     and save F*kt2/Qs2 to the table
    double compute_GHT_z_kt2(double TA, double Q2xB, double kt2, double Ltauf, double Qs2);
    // a): this one computes F(kt2) in terms of a table as function of TA, Q2/x, kt2, L/tauf=L*kt^2/(2E)
    //     where F is dP/dz/dkt2 = alphas*CR/pi * 1/kt2 * (ln(zmax/zmin) + F(kt2; zmax, zmin))
    //     and save F*kt2/Qs2 to the table
    double compute_GHT_kt2(double TA, double Q2xB, double kt2, double Ltauf, double Qs2);
public:
    // constructor, given mode and the K factor
    eHIJING(int mode, double K);
    // Tabulate Qs and (if nessesary) genralized HT table
    void Tabulate(std::filesystem::path table_path);
    // Sample the location of the hard vertex in a nuclei A
    // and compute the path length [GeV^{-1}]
    double sample_L(double A){
        double RA = (1.12*5.076) * std::pow(A, 1./3.);
        double r = RA * std::pow(flat_gen(gen), 1./3.);
        double costheta = flat_gen(gen)*2.0 - 1.0;
        double rz = r*costheta;
        double L = - rz + std::sqrt((RA*RA-r*r)+rz*rz);
        return L;
    };
    // return the nulcear matter density
    double rho0() const {return rho0_;};
    // interpolate the saturation scale table
    double Qs2(double x, double Q2, double TA) {
        return Qs2Table.interpolate({TA, std::log(Q2/x)});
    };
    // compute the qhat of a gluon
    double qhatA(double x, double Q2, double TA) {
        return rho0_*Qs2(x, Q2, TA)/TA;
    };
    // compute the qhat of a quark
    double qhatF(double x, double Q2, double TA) {
        return 4./9.*qhatA(x, Q2, TA);
    };
    // interpolate F(z, kt2)
    double induced_dFdkt2(double x, double Q2, double L, double E, double kt2){
        double TA = rho0_*L;
        double Q2x = Q2/x;
        double Lkt2_over_2E = L*kt2/2/E;
        return Qs2Table.interpolate({TA, std::log(Q2x)})/kt2 
             * GHT_kt2_Table.interpolate({TA, std::log(Q2x), std::log(kt2), std::log(5+Lkt2_over_2E)});
    }
    // interpolate F(kt2)
    double induced_dFdkt2dz(double x, double Q2, double L, double E, double kt2, double z){
        double TA = rho0_*L;
        double Q2x = Q2/x;
        double LoverTauf = L*kt2/(2*(1.0-z)*z*E);
        return Qs2Table.interpolate({TA, std::log(Q2x)})/kt2 
             * GHT_z_kt2_Table.interpolate({TA, std::log(Q2x), std::log(kt2), std::log(5+LoverTauf)});
    }
    // Main routinue for sampling the next kt2, if the returned status is false, 
    // the splitting should be reject and only keep the evolution in kt2
    bool next_kt2(double & kt2, int pid, double E, double L, 
                       double kt2min, double xB, double Q20);
    // Main routinue for sampling the z, if the returned status is false, 
    // the splitting should be reject and only keep the evolution in kt2
    bool sample_z(double & z, int pid, double E, double L, double kt2, double xB, double Q20);
    // Main routinue for sampling the next q2 of the multiple collisons, to be done
    void Sample_q2(double & q2, double & phi, double kt2, double z) {};
};

#endif
