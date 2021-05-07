#ifndef EHIJING_H
#define EHIJING_H
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

class table_nd{
private:
    const int dim_, power_dim_;
    int total_size_;
    std::vector<double> data_, step_, grid_min_, grid_max_;
    std::vector<int> shape_, conj_size_;

public:
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
    int ArrayIndex2LinearIndex(std::vector<int> & index){
        int k = 0;
        for (int i=0; i<dim_; i++) k += index[i] * conj_size_[i];
        return k;
    }
    std::vector<int> LinearIndex2ArrayIndex(int k){
        std::vector<int> index;
        int K = k;
        for (int i=0; i<dim_; i++) {
            index.push_back(int(floor(K/conj_size_[i])));
            K = K%conj_size_[i];
        }
        return index;
    }
    std::vector<double> ArrayIndex2Xvalues(std::vector<int> index){
        std::vector<double> xvals;
        for (int i=0; i<dim_; i++) xvals.push_back(grid_min_[i]+step_[i]*index[i]);
        return xvals;
    }
    int dim() const {return dim_;};
    int size() const {return total_size_;};
    std::vector<int> shape() const { return shape_; }   
    std::vector<double> step() const{ return step_; }   
    void set(std::vector<int> index, double value) { data_[ArrayIndex2LinearIndex(index)] = value; }
    double get_with_index(std::vector<int> index) { return data_[ArrayIndex2LinearIndex(index)]; }
    double get_with_linear_index(int k) { return data_[k]; }
    double interpolate(std::vector<double> Xinput) {
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



class eHIJING{
private:
    const int mode_;
    const double Kfactor_, rho0_, Mproton_;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> flat_gen;
    table_nd Qs2Table, GHT_z_kt2_Table, GHT_kt2_Table; // table for the Qs2, and generalized Higher-twist
    // related to in-medium couplin, and the computation of Qs
    double alphas(double Q2);
    double PhiG(double x, double q2, double Qs2);
    double alphas_PhiG(double x, double q2, double Qs2);
    double Qs2_self_consistent_eq(double Qs2, double TA, double Q2x);
    double compute_Qs2(double TA, double Q2xB);
    double compute_GHT_z_kt2(double TA, double Q2xB, double kt2, double Ltauf, double Qs2);
    double compute_GHT_kt2(double TA, double Q2xB, double kt2, double Ltauf, double Qs2);

public:
    eHIJING(int mode, double K);
    void Tabulate();
    void LoadTable() {};
    double sample_L(double A){
        double RA = (1.12*5.086) * std::pow(A, 1./3.);
        double r = RA * std::pow(flat_gen(gen), 1./3.);
        double costheta = flat_gen(gen)*2.0 - 1.0;
        double rz = r*costheta;
        double L = - rz + std::sqrt((RA*RA-r*r)+rz*rz);
        return L;
    };
    double rho0() const {return rho0_;};
    double Qs2(double x, double Q2, double TA) {
        return Qs2Table.interpolate({TA, std::log(Q2/x)});
    };
    double qhatA(double x, double Q2, double TA) {
        return rho0_*Qs2(x, Q2, TA)/TA;
    };
    double qhatF(double x, double Q2, double TA) {
        return 4./9.*qhatA(x, Q2, TA);
    };
    double induced_dFdkt2(double x, double Q2, double L, double E, double kt2){
        double TA = rho0_*L;
        double Q2x = Q2/x;
        double Lkt2_over_2E = L*kt2/2/E;
        return Qs2(x, Q2, TA)/kt2 * GHT_kt2_Table.interpolate({TA, std::log(Q2x), std::log(kt2), std::log(5+Lkt2_over_2E)});
    }
    double induced_dFdkt2dz(double x, double Q2, double L, double E, double kt2, double z){
        double TA = rho0_*L;
        double Q2x = Q2/x;
        double LoverTauf = L*kt2/(2*z*E);
        return Qs2(x, Q2, TA)/kt2 * GHT_z_kt2_Table.interpolate({TA, std::log(Q2x), std::log(kt2), std::log(5+LoverTauf)});
    }
    bool next_kt2(double & kt2, int pid, double E, double L, 
                       double kt2min, double xB, double Q20);
    bool sample_z(double & z, int pid, double E, double L, double kt2, double xB, double Q20);
    void Sample_q2(double & q2, double & phi, double kt2, double z) {};
};

#endif
