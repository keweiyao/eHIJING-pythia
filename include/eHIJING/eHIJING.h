#ifndef EHIJING_H
#define EHIJING_H
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <filesystem>

namespace EHIJING{

// eHIJING running alphas and other constants
extern const double rho0, Mproton, r0, CA, dA, CF, TAmax, TAmin, b0, mu2;
// LO running coupling constant
double alphas(double Q2);
// Shape of a un parametrized kT-dependent gluon distribution function, only a function of x, q2 and Qs2
// The normalization at given x and Q2 is choosen such that <x> of gluon is the user given value (~0.3--0.5)
double PhiG(double x, double q2, double Qs2);
// alphas*PhiG
double alphas_PhiG(double x, double q2, double Qs2, double Q2x, double powerG, double lambdaG, double avgxG);

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

class NuclearGeometry{
private:
    const int A_, Z_, N_;
    const double R_, R2_;
    // Random number generators
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> flat_gen;
public:
    NuclearGeometry(int A, int Z): A_(A), Z_(Z), N_(A-Z),
    R_(r0*std::pow(A*1., 1./3.)), R2_(R_*R_),
    rd(), gen(rd()), flat_gen(0.,1.)
    {
    };
    int A() const {return A_; };
    int Z() const {return Z_; };
    int N() const {return N_; };
    double R() const {return R_; };
    // Sample the location of the hard vertex in a nuclei A
    void sample_HardVertex(double & Rx, double & Ry, double & Rz){
        double r = R_ * std::pow(flat_gen(gen), 1./3.);
        double costheta = flat_gen(gen)*2.0 - 1.0;
        double sintheta = std::sqrt(1.-costheta*costheta);
        double phi = flat_gen(gen)*2.*M_PI-M_PI; // (-pi, pi)
        Rz = r*costheta;
        Rx = r*sintheta*std::cos(phi);
        Ry = r*sintheta*std::sin(phi);
    };
    // Compute the path length, given current location in the nuclus and the velocity in the nuclear rest frame!
    double compute_L(double rx, double ry, double rz, double vx, double vy, double vz){
        double r2 = rx*rx + ry*ry + rz*rz;
        double rdotv = rx*vx + ry*vy + rz*vz;
        return -rdotv + std::sqrt((R2_-r2)+rdotv*rdotv);
    };
};



// A saturation physics based multiple collision model
class MultipleCollision{
private:
  const double Kfactor_;
  // Parameters for kT-dependent gluon distribution
  // PhiG(x, qt2) = N * (1-x)^powerG * x^lambdaG / qt2 / alphas(max(Qs2, qt2))
  // N is determined such that integrate dx PhiG = integrate dN/dx x dx = <xg> =  momentum fraction of gluon
  // Qs will be determined self-consistenly
  const double powerG_;
  const double lambdaG_;
  // avg momentum fraction of gluon
  const double avgxG_;
  // Random number generators
  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<double> flat_gen;
  table_nd Qs2Table, RateTable;
  // the self consistent equation for Qs2: Qs2[Phi_G(Qs2), TA, x, Q2] - Qs2 = 0
  double Qs2_self_consistent_eq(double Qs2, double TA, double Q2x);
  // self-consistently solve for Qs2(Q^2/x, TA)
  double compute_Qs2(double TA, double Q2xB);
public:
  MultipleCollision(double K, double pG, double lG, double xG);
  // Tabulate Qs table
  void Tabulate(std::filesystem::path table_path);
  double Qs2(double x, double Q2, double TA) {
      return Qs2Table.interpolate({TA, std::log(Q2/x)});
  };
  double Qs2(double Q2x, double TA) {
      return Qs2Table.interpolate({TA, std::log(Q2x)});
  };
  // compute the qhat of a gluon
  double qhatA(double x, double Q2, double TA) {
      return rho0*Qs2(x, Q2, TA)/TA;
  };
  // compute the qhat of a quark
  double qhatF(double x, double Q2, double TA) {
      return CF/CA*qhatA(x, Q2, TA);
  };
  // Sample pure multiple collisions
  int sample_all_qt2(int pid, double E, double L, double xB, double Q2,
                      std::vector<double> & q2_list, std::vector<double> & t_list,
                     std::vector<double> & phi_list);
  // sample a collision anyway with q2 > a given minimum
  void sample_single_qt2(int pid, double E, double L, double xB, double Q2,
                         double & qx, double & qy, double & xg, double & tq,
                         double minimum_q2);
};

// The main eHIJING class
class eHIJING: public MultipleCollision{
private:
    // mode=0: collinear H-T;
    // mode=1: static&soft generalized H-T / GLV
    const int mode_;
    // A K factor enhancing the medium-induced term for testing,
    const double Kfactor_;
    // Parameters for kT-dependent gluon distribution
    // PhiG(x, qt2) = N * (1-x)^powerG * x^lambdaG / qt2 / alphas(max(Qs2, qt2))
    // N is determined such that integrate dx PhiG = integrate dN/dx x dx = <xg> =  momentum fraction of gluon
    // Qs will be determined self-consistenly
    const double powerG_;
    const double lambdaG_;
    // avg momentum fraction of gluon
    const double avgxG_;
    // Random number generators
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> flat_gen;
    table_nd GHT_Angular_Table;
    double compute_GHT_Angular_Table(double X, double Y);
public:
    // constructor, given mode, the K factor, powerG, lambdaG, and xG
    eHIJING(int mode, double K, double pG, double lG, double xG);
    // Tabulate (if nessesary) genralized HT table
    void Tabulate(std::filesystem::path table_path);
    bool next_kt2_stochastic(double & kt2, int pid,
                                     double E,     double kt2min,
                                     std::vector<double> qt2s,
                                     std::vector<double> ts);
    double sample_z_stochastic(double & z, int pid,
                           double E,
                           double kt2,
                           std::vector<double> qt2s,
                           std::vector<double> ts,
                           std::vector<double> phis);
};



}

#endif
