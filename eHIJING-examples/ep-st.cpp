#include "Pythia8/Pythia.h"
using namespace Pythia8;
#include <fstream>
#include <random>
#include <sstream>
#include <algorithm>
class hadronizer{
public:
   hadronizer():pythia(),rd(),gen(rd()),dist(0.,1.){
    pythia.readString("Tune:pp = 19");
    pythia.readString("PDF:pSet = 12");
    pythia.readString("ProcessLevel:all = off");
    pythia.readString("Print:quiet = on");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");
    pythia.readString("HadronLevel:all = on");
    pythia.readString("HadronLevel:Decay = on");
    pythia.readString("StringFragmentation:stopMass = 0.0");
    pythia.readString("HadronLevel:mStringMin = 0.5");
    pythia.readString("111:mayDecay=off");
    pythia.readString("211:mayDecay=off");
    pythia.readString("321:mayDecay=off");
    pythia.init();
   }

    std::vector<Particle> hadronize(Pythia & pythiaIn, int Z, int A){
       double ZoverA = Z*1./A;
       std::vector<Particle> FinalParticles;
       FinalParticles.clear();
       int hardid = pythiaIn.event[5].id();
       pythia.event.reset();
       for (int i=0; i<pythiaIn.event.size(); i++){
         auto & p = pythiaIn.event[i];
         if (! ( p.isFinal() && p.isParton() ) ) continue;

         if (p.status()==63 && 1000<p.idAbs() && p.idAbs()<3000) {
                 // valence stuff, the remnants will contain the rest flavor compoennt.
                 // note that the hard quark has already been sampled accorrding to the
                 // the isospin content of the nuclear PDF; however, the remanent is generated
                 // assuming the rest stuff comes from a proton. Therefore, we need to resample
                 // it according to the Z/A ratio this nuclei
                 // 1) decide wither it is from a neutron or proton
                 if (dist(gen) < ZoverA) { // From a proton 2212
                     if (hardid==1) { // produce 2203
                       p.id(2203);
                     }
                     if (hardid==2) { // produce 2101 and 2103 with ratio 3:1
                       if (dist(gen) < 0.75) p.id(2101);
                       else p.id(2103);
                     }
                 } else { // From a neutron 2112
                     if (hardid==1) { // produce 2101 and 2103 with ratio 3:1
                       if (dist(gen) < 0.75) p.id(2101);
                       else p.id(2103);
                    }
                     if (hardid==2) { // produce 1103
                       p.id(1103);
                     }
                 }
         }
         pythia.event.append(p.id(), 23, p.col(), p.acol(),
                       p.px(), p.py(), p.pz(), p.e(), p.m());
       }
       pythia.next();
       for (int i=0; i<pythia.event.size(); i++) {
         auto & p = pythia.event[i];
         if (p.isFinal()) FinalParticles.push_back(p);
      }
      return FinalParticles;
    }

private:
    Pythia pythia;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen; //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dist;
};

template <class T>
void add_arg(Pythia & pythia, std::string name, T value){
  std::stringstream ss;
  ss << name << " = " << value;
  std::cout << ss.str() << std::endl;
  pythia.readString(ss.str());
}

bool trigger(Pythia & pythia) {
    Vec4 pProton = pythia.event[1].p(); // four-momentum of proton
    Vec4 peIn    = pythia.event[4].p(); // incoming electron
    Vec4 peOut   = pythia.event[6].p(); // outgoing electron
    Vec4 pGamma = peIn - peOut; // virtual boson photon/Z^0/W^+-

    // Q2, W2, Bjorken x, y.
    double Q2 = - pGamma.m2Calc(); // hard scale square

    double W2 = (pProton + pGamma).m2Calc();
    double x  = Q2 / (2. * pProton * pGamma); // Bjorken x
    double nu = pGamma.e();

    // In Breit frame, where gamma ~ (0,0,0,-Q),
    double y     = (pProton * pGamma) / (pProton * peIn);
    return (1.0<Q2) & (W2>10.) & (nu>0.) & (y<0.85);
}

template<typename T>
void output(Pythia & pythia, T & plist, ofstream & f){
    // Compute four-momenta of proton, electron, virtual
    Vec4 pProton = pythia.event[1].p(); // four-momentum of proton
    Vec4 peIn    = pythia.event[4].p(); // incoming electron
    Vec4 peOut   = pythia.event[6].p(); // outgoing electron
    Vec4 pGamma = peIn - peOut; // virtual boson photon/Z^0/W^+-
    // Q2, W2, Bjorken x, y.
    double Q2 = - pGamma.m2Calc(); // hard scale square
    double W2 = (pProton + pGamma).m2Calc();
    double x  = Q2 / (2. * pProton * pGamma); // Bjorken x
    double nu = pGamma.e();
    Vec4 pCoM = pGamma + pProton;
    Vec4 pGamma2 = pGamma;
    pGamma2.bstback(pCoM);
    // rotate to gamma direction
    double theta = - pGamma.theta();
    double phi = - pGamma.phi();

    double z1=-1;
    double z2=-1;
    int c1=0, c2=0;
    for (int j=0; j<plist.size(); j++){
      auto p = plist[j];
      if (p.isFinal() && p.isHadron()){
         auto pbst = p.p();
         pbst.bstback(pCoM);
         double xF = dot3(pGamma2, pbst);
         double z = p.e()/nu;
         //if (xF<0) continue;
         //if (z>z1) { z1 = z; c1 = p.charge();}
         //else if (z>z2) { z2 = z; c2 = p.charge();}
         auto prot = p.p();
         prot.rot(0, phi);
         prot.rot(theta, 0);
         f << p.id() << " " << z << " " << prot.pT() << " " << nu << " " << Q2 << std::endl;
        }
    }
    // exclude +- pairs
    //if (c1*c2>-.1) f << z1 << " " << z2 << " " << Q2 << " " << nu << " " << pythia.info.sigmaGen() << std::endl;
}

int main(int argc, char *argv[]) {
  //

  // commandline args
  int nEvent = atoi(argv[1]);
  int Z = atoi(argv[2]);
  int A = atoi(argv[3]);
  int mode = atof(argv[4]);
  double K = atof(argv[5]);
  double ZoverA = Z*1./A;
  auto header = std::string(argv[6]);
  int ishadow = 0;
  double pTmin = .5;
  double pT2min = std::pow(pTmin, 2);

  std::stringstream  ss;
  ss << header << "/" << Z << "-" << A << "-cutRA.dat";
  std::ofstream f(ss.str());

  std::stringstream  ss2;
  ss2 << header << "/" << Z << "-" << A << "-stat.dat";
  std::ofstream fstat(ss2.str());

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> dist(0.,1.);

  int inuclei = 100000000
              +   Z*10000
              +   A*10;
  hadronizer HZ;
  EHIJING::NuclearGeometry  eHIJING_Geometry(A, Z);
  // Initialize
  Pythia pythia;        // Generator
  Event& event = pythia.event; // Event record
  pythia.readFile("pythia-ep-settings.txt"); // read settings
  add_arg<double>(pythia, "TimeShower:pTmin", pTmin);
  add_arg<int>(pythia, "eHIJING:Mode", mode);
  add_arg<int>(pythia, "PDF:nPDFSetA", (A>14)?3:0);
  add_arg<int>(pythia, "PDF:nPDFBeamA", inuclei);
  add_arg<int>(pythia, "eHIJING:AtomicNumber", A);
  add_arg<int>(pythia, "eHIJING:ChargeNumber", Z);
  add_arg<double>(pythia, "eHIJING:Kfactor", K);
  double alpha_fix = EHIJING::alphas(pT2min);
  double alphabar = alpha_fix * EHIJING::CA/M_PI;
  pythia.init();
  // output

  // Begin event loop.
  int Ntriggered = 0;
  int count = 0, failed=0;
  while(Ntriggered<nEvent){
      count++;
      if (!pythia.next()) {
          failed++;
          continue;
      }; // skip bad events
      if (!trigger(pythia)) continue; // only study triggered events
      Ntriggered ++;
      if (Ntriggered%1000==0)  std::cout << "# of trigged events: "
                               << Ntriggered << std::endl;

      Vec4 pProton = event[1].p(); // four-momentum of proton
      Vec4 peIn    = event[4].p(); // incoming electron
      Vec4 peOut   = event[6].p(); // outgoing electron
      Vec4 pGamma = peIn - peOut; // virtual boson photon/Z^0/W^+-
      double Q20 = - pGamma.m2Calc(); // hard scale square
      double xB  = Q20 / (2. * pProton * pGamma); // Bjorken x
      auto & hardP = event[5];
      double kt2max_now = pT2min;
      double emin = .4;
      std::vector<Particle> all_extra;
      all_extra.clear();
      int Nhard_rad = 0;
      for (int i=0; i<event.size(); i++) {
          if (event[i].isFinal() && event[i].id()==21) Nhard_rad ++;
      }
      for (int i=0; i<event.size(); i++) {
          // find final-state quarks and gluons
          auto & p = event[i];
          if (!p.isFinal() ) continue;
          bool triggered =   (p.idAbs()==1) || (p.idAbs()==2)
                          || (p.idAbs()==3) || (p.idAbs()==21);
          if (!triggered) continue;
          if (p.e()<2*emin) continue;
          std::vector<double> qt2s = p.coll_qt2s(), ts = p.coll_ts(), phis = p.coll_phis();
          double vx = p.px()/p.e(),
                 vy = p.py()/p.e(),
                 vz = p.pz()/p.e();
          double L = eHIJING_Geometry.compute_L(event.Rx(), event.Ry(), event.Rz(),
                                            vx, vy, vz);
          double tau1 = 2.*p.e()/std::pow(p.scale(),2);
          //if (p.id()==21) fstat << 111 << " " << xB << " " << Q20 << " " << pGamma.e() << " " << tau1 << " " << std::pow(p.scale(),2) << " " << 0 <<" "<< 0 << " " << p.e() << " " << L << std::endl;
          int Ncolls = ts.size();
          if (Ncolls==0) {
              //if (p.id()==hardP.id())
              //    fstat << Ncolls << " " << Nhard_rad << " " << 0 << std::endl;
              continue;
          }
          double sumq2 = 0.;
          for (auto & q2 : qt2s) sumq2 += q2;
          if (mode==0 && sumq2<1e-9) continue;
          double taufmax = 2*p.e()/EHIJING::mu2;
          double tauf = 1./std::sqrt(Q20);
          double acceptance = 0;
          double z, kt2, phik, dphiqk;

          std::vector<Particle> frag_gluons, recoil_remnants;
          frag_gluons.clear();
          recoil_remnants.clear();

          while(tauf < taufmax && p.e()>2*emin){
              double zmin = std::min(emin / p.e(), .4);
              double zmax = 1.-zmin;
              if (zmax<zmin) break;
              double maxlogz =  std::log(zmax/zmin);
              double maxdiffz = 1./zmin - 1./zmax + 2.*maxlogz;
              // step1: next tauf
              double r = dist(gen);
              if (mode==1){
                  double invrpower = alphabar * maxlogz * 4. * Ncolls;
                  double step_factor = std::pow(1./r, 1./invrpower);
                  tauf = tauf * step_factor;
              }
              else {
                  double coeff = alphabar * maxdiffz * 4. * sumq2 / 2. / p.e();
                  tauf = tauf + std::log(1./r)/coeff;
              }

              if (tauf > taufmax || tauf<0.) break;
              acceptance = 0.;
              if (mode==1) {
                  for (int j=0; j<Ncolls; j++){
                      double phase = (1.-std::cos(ts[j]/tauf));
                      double z1mz = tauf * qt2s[j] / 2. / p.e();
                      if (z1mz>.25) acceptance += phase * maxlogz;
                      else {
                          double dz = std::sqrt(.25 - z1mz);
                          double z1 = .5 - dz;
                          double z2 = .5 + dz;
                          if (z1>zmin) acceptance += phase * std::log(z1/zmin);
                          if (z2<zmax) acceptance += phase * std::log(zmax/z2);
                      }
                  }
                  acceptance /= (maxlogz * 2. * Ncolls);
              }
              else{
                  for (int j=0; j<Ncolls; j++) acceptance += qt2s[j]*(1.-std::cos(ts[j]/tauf));
                  acceptance /= (2.*sumq2);
              }
              if (acceptance < dist(gen)) continue;
              // step 2: sample z, which also determines kt2
              acceptance = 0.;
              if (mode==0) {
                  double N1 = 2*(1./zmin-2.);
                  double N2 = -4*std::log(2.*(1.-zmax));
                  double Ntot = N1+N2;
                  double r0 = N1/Ntot;

                  double acceptance = 0.;
                  while(acceptance<dist(gen)){
                      double r = dist(gen);
                      if (r<r0){
                          z = zmin/(1. - zmin*r*Ntot/2.);
                          acceptance = .5/(1.-z);
                      } else {
                          z = 1. - std::exp(-(r*Ntot - N1)/4.)/2.;
                          acceptance = .25/z/z;
                      }
                  }
                  kt2 = 2*(1.-z)*z*p.e()/tauf;
                  // reject cases where qt2>kt2 for mode=0
                  double Num=0., Den = 0.;
                  for (int j=0; j<Ncolls; j++) {
                      double q2 = qt2s[j], t = ts[j];
                      if (kt2>q2) Num += q2*(1.-std::cos(t/tauf));
                      Den += q2*(1.-std::cos(t/tauf));
                  }
                  if (Num/Den < dist(gen)) continue;
              }
              else {
                  bool ok=false;
                  double minimum_q2 = 2*emin/tauf;
                  for (int j=0; j<Ncolls; j++){
                      if (qt2s[j]>minimum_q2) ok=true;
                  }
                  if (!ok) continue;
                  while(acceptance<dist(gen)){
                      z = zmin * std::pow(zmax/zmin, dist(gen));
                      kt2 = 2*(1.-z)*z*p.e()/tauf;
                      double num = 0.;
                      for (int j=0; j<Ncolls; j++)
                          if (kt2<qt2s[j])
                              num += 1.;
                      acceptance = num / Ncolls;
                  }
              }


              // correct for splitting function
              if (p.id()==21 && (1+std::pow(1.-z,3))/2.<dist(gen) ) continue;
              if (p.id()!=21 && (1+std::pow(1.-z,2))/2.<dist(gen) ) continue;

              double lt2;
              // finally, sample phikT2
              if (mode==0) {
                  phik = 2*M_PI*dist(gen);
                  lt2 = kt2;
              }
              else {
                  double Psum = 0.;
                  std::vector<double> dP;
                  dP.resize(Ncolls);
                  for (int j=0; j<Ncolls; j++){
                      if (kt2<qt2s[j])
                          Psum += (1.-std::cos(ts[j]/tauf));
                      dP[j] = Psum;
                  }
                  for (int j=0; j<Ncolls; j++) dP[j]/=Psum;
                  double rc = dist(gen);
                  int choice = -1;
                  for (int j=0; j<Ncolls; j++){
                      if ( rc<dP[j] ) {
                          choice = j;
                          break;
                      }
                  }
                  // sample phik ~ (1+delta cos) / (1+delta^2 + 2 delta cos)
                  double delta = std::sqrt(kt2/qt2s[choice]);
                  acceptance = 0.;
                  while(acceptance < dist(gen)){
                      r = dist(gen);
                      dphiqk = 2.*std::atan(std::tan(M_PI/2.*r) * (delta+1)/(delta-1));
                      acceptance = (1+delta*std::cos(dphiqk))/2.;
                  }
                  phik = phis[choice] + ( (dist(gen)>.5)? dphiqk : (-dphiqk) );
                  lt2 = kt2 + qt2s[choice] + 2.*std::sqrt(kt2*qt2s[choice])*std::cos(dphiqk);
              }

              if (mode==0){
                  if (kt2>kt2max_now) continue;
              }
              else {
                  if (lt2>kt2max_now) continue;
              }

              //if (EHIJING::alphas(std::max(4.*EHIJING::mu2, lt2))/alpha_fix < dist(gen)) continue;
              // Now, there is a radiation,
              // we do it in two steps:
              // first, parton goes to p -> p+k and k
              // recoil effect will goes to k and handled later
              double kt = std::sqrt(kt2), k0 = z*p.e();
              if (kt>k0) continue;
              double kz = std::sqrt(k0*k0-kt2);
              Vec4 kmu{kt*std::cos(phik), kt*std::sin(phik), kz, k0};
              kmu.rot(p.theta(), 0.);
              kmu.rot(0., p.phi());
              p.p(p.p()-kmu);
              p.e(std::sqrt(p.pAbs2()+p.m2()));
              kmu.e(std::sqrt(kmu.pAbs2()));

              taufmax = 2.*p.e()/EHIJING::mu2;
              //fstat << 222 << " " << xB << " " << Q20 << " " << pGamma.e() << " " << tauf << " " << lt2 << " " << kt2 << " " << z << " " << kmu.e() << " " << L << std::endl;
              // update the color if it is a hard gluon
              // first, the spliting process
              int k_col, k_acol;
              if (tauf<L) {
                Particle gluon = Particle(21, 201, i, 0, 0, 0,
                                  event.nextColTag(), event.nextColTag(),
                                  kmu, 0.0, 0);
                                  int qid, diqid;
                                  double mq, mdiq, mn;
                                  if (dist(gen) < ZoverA) { // diquark from a proton
                                      diqid = 2101; mdiq = 0.57933;
                                      qid = 2; mq = 0.33;
                                      mn = 0.93847;
                                      if (dist(gen) < 2./3.) { // take away a u
                                          qid = 2;
                                          if (dist(gen) < .75) diqid = 2101;
                                          else diqid = 2103;
                                      } else { // take away the d
                                          qid = 1;
                                          diqid = 2203;
                                      }
                                 }
                                 else { // diquark from a neutron
                                      diqid = 2101; mdiq = 0.57933;
                                      qid = 1; mq = 0.33;
                                      mn = 0.93957;
                                      if (dist(gen) < 2./3.) { // take away a d
                                          qid = 1;
                                          if (dist(gen) < .75) diqid = 2101;
                                          else diqid = 2103;
                                      } else { // take away the u
                                          qid = 2;
                                          diqid = 1103;
                                      }
                                 }
                                 double pabs = std::sqrt((mn*mn-std::pow(mq+mdiq,2))*(mn*mn-std::pow(mq-mdiq,2))) / (2.*mn);
                                 double costheta = dist(gen)*2.-1.;
                                 double sintheta = std::sqrt(std::max(1.-costheta*costheta,1e-9));
                                 double rphi = 2*M_PI*dist(gen);
                                 double Nqz = pabs*costheta,
                                        Nqx = pabs*sintheta*std::cos(rphi),
                                        Nqy = pabs*sintheta*std::sin(rphi);
                                 Vec4 pq  { Nqx,  Nqy,  Nqz, 0},
                                      pdiq{-Nqx, -Nqy,  -Nqz, 0};
                                 pq.e(std::sqrt(pq.pAbs2()+mq*mq));
                                 pdiq.e(std::sqrt(pdiq.pAbs2()+mdiq*mdiq));
                                 Particle recolQ = Particle(qid, 201, i, 0, 0, 0,
                                                   gluon.acol(), 0,
                                                   pq, mq, 0);
                                 Particle recoldiQ = Particle(diqid, 201, i, 0, 0, 0,
                                                   0, gluon.col(),
                                                   pdiq, mdiq, 0);
                                 recoil_remnants.push_back(gluon);
                                 recoil_remnants.push_back(recolQ);
                                 recoil_remnants.push_back(recoldiQ);
              } else{
                if (p.id()==21){
                    if (std::rand()%2==0){
                        k_col = p.col();
                        p.col( event.nextColTag() );
                        k_acol = p.col();
                    } else {
                        k_acol = p.acol();
                        p.acol( event.nextColTag() );
                        k_col = p.acol();
                    }
                } else if (p.id()>0){
                    k_col = p.col();
                    p.col(event.nextColTag() );
                    k_acol = p.col();
                } else {
                    k_acol = p.acol();
                    p.acol( event.nextColTag() );
                    k_col = p.acol();
                }
                // make the gluon with status 201
                Particle gluon = Particle(21, 201, i, 0, 0, 0,
                                  k_col, k_acol,
                                  kmu, 0.0, 0);
                frag_gluons.push_back(gluon);
              }

          }
          // Now handles recoil and remannts
          // if there are radiations, coil goes to radiations
          // else: goes to the hard quark
          int Nrad = frag_gluons.size();
          if (Nrad>0) {
              for (int j=0; j<Ncolls; j++){
                  // pick a gluon that suffers the recoil
                  int indexg = rand()%Nrad;
                  auto & fg = frag_gluons[indexg];
                  double qx, qy, qz;
                  double qT = std::sqrt(qt2s[j]), phiq = phis[j];
                         qx = qT*std::cos(phiq);
                         qy = qT*std::sin(phiq);
                         qz = qT*qT/4./p.e();

                  Vec4 qmu{qx, qy, qz, 0.};
                  qmu.rot(fg.theta(), 0.);
                  qmu.rot(0., fg.phi());
                  fg.p(fg.p()+qmu);
                  fg.e(fg.pAbs());
                  int q_col, q_acol;
                  // update color
                  if (std::rand()%2==0){
                      q_acol = fg.acol();
                      fg.acol(event.nextColTag());
                      q_col = fg.acol();
                  } else {
                      q_col = fg.col();
                      fg.col(event.nextColTag());
                      q_acol = fg.col();
                  }
                  // remnants
                  int qid, diqid;
                  double mq, mdiq, mn;
                  if (dist(gen) < ZoverA) { // diquark from a proton
                      diqid = 2101; mdiq = 0.57933;
                      qid = 2; mq = 0.33;
                      mn = 0.93847;
                      if (dist(gen) < 2./3.) { // take away a u
                          qid = 2;
                          if (dist(gen) < .75) diqid = 2101;
                          else diqid = 2103;
                      } else { // take away the d
                          qid = 1;
                          diqid = 2203;
                      }
                 }
                 else { // diquark from a neutron
                      diqid = 2101; mdiq = 0.57933;
                      qid = 1; mq = 0.33;
                      mn = 0.93957;
                      if (dist(gen) < 2./3.) { // take away a d
                          qid = 1;
                          if (dist(gen) < .75) diqid = 2101;
                          else diqid = 2103;
                      } else { // take away the u
                          qid = 2;
                          diqid = 1103;
                      }
                 }
                 double pabs = std::sqrt((mn*mn-std::pow(mq+mdiq,2))*(mn*mn-std::pow(mq-mdiq,2))) / (2.*mn);
                 double costheta = dist(gen)*2.-1.;
                 double sintheta = std::sqrt(std::max(1.-costheta*costheta,1e-9));
                 double rphi = 2*M_PI*dist(gen);
                 double Nqz = pabs*costheta,
                        Nqx = pabs*sintheta*std::cos(rphi),
                        Nqy = pabs*sintheta*std::sin(rphi);
                 Vec4 pq  { Nqx,  Nqy,  Nqz, 0},
                      pdiq{-Nqx, -Nqy,  -Nqz, 0};
                 // decide which object takes the recoil
                 if (std::rand()%2==0){
                     pq = pq - qmu;
                 } else {
                     pdiq = pdiq - qmu;
                 }
                 pq.e(std::sqrt(pq.pAbs2()+mq*mq));
                 pdiq.e(std::sqrt(pdiq.pAbs2()+mdiq*mdiq));
                 Particle recolQ = Particle(qid, 201, i, 0, 0, 0,
                                   q_col, 0,
                                   pq, mq, 0);
                 Particle recoldiQ = Particle(diqid, 201, i, 0, 0, 0,
                                   0, q_acol,
                                   pdiq, mdiq, 0);

                 recoil_remnants.push_back(recolQ);
                 recoil_remnants.push_back(recoldiQ);
             }
         }
           if (Nrad==0) {
               for (int j=0; j<Ncolls; j++){
                   double qT = std::sqrt(qt2s[j]), phiq = phis[j];
                   double qx = qT*std::cos(phiq);
                   double qy = qT*std::sin(phiq);
                   double qz = qT*qT/4./p.e();
                   Vec4 qmu{qx, qy, qz, 0};
                   qmu.rot(p.theta(), 0.);
                   qmu.rot(0., p.phi());
                   p.p(p.p()+qmu);
                   p.e(std::sqrt(p.pAbs2()+p.m2()));
                   int q_col, q_acol;
                   // update color
                   if (p.id()==21){
                       if (std::rand()%2==0){
                           q_acol = p.acol();
                           p.acol(event.nextColTag());
                           q_col = p.acol();
                       } else {
                           q_col = p.col();
                           p.col(event.nextColTag());
                           q_acol = p.col();
                       }
                   }
                   else if(p.id()>0){
                       q_col = p.col();
                       p.col(event.nextColTag());
                       q_acol = p.col();
                   }
                   else {
                       q_acol = p.acol();
                       p.acol(event.nextColTag());
                       q_col = p.acol();
                   }
                   int qid, diqid;
                   double mq, mdiq, mn;
                   if (dist(gen) < ZoverA) { // diquark from a proton
                       diqid = 2101; mdiq = 0.57933;
                       qid = 2; mq = 0.33;
                       mn = 0.93847;
                       if (dist(gen) < 2./3.) { // take away a u
                           qid = 2;
                           if (dist(gen) < .75) diqid = 2101;
                           else diqid = 2103;
                       } else { // take away the d
                           qid = 1;
                           diqid = 2203;
                       }
                  }
                  else { // diquark from a neutron
                       diqid = 2101; mdiq = 0.57933;
                       qid = 1; mq = 0.33;
                       mn = 0.93957;
                       if (dist(gen) < 2./3.) { // take away a d
                           qid = 1;
                           if (dist(gen) < .75) diqid = 2101;
                           else diqid = 2103;
                       } else { // take away the u
                           qid = 2;
                           diqid = 1103;
                       }
                  }
                  double pabs = std::sqrt((mn*mn-std::pow(mq+mdiq,2))*(mn*mn-std::pow(mq-mdiq,2))) / (2.*mn);
                  double costheta = dist(gen)*2.-1.;
                  double sintheta = std::sqrt(std::max(1.-costheta*costheta,1e-9));
                  double rphi = 2*M_PI*dist(gen);
                  double Nqz = pabs*costheta,
                         Nqx = pabs*sintheta*std::cos(rphi),
                         Nqy = pabs*sintheta*std::sin(rphi);
                  Vec4 pq  { Nqx,  Nqy,  Nqz, 0},
                       pdiq{-Nqx, -Nqy,  -Nqz, 0};
                  // decide which object takes the recoil
                  if (std::rand()%2==0){
                      pq = pq - qmu;
                  } else {
                      pdiq = pdiq - qmu;
                  }
                  pq.e(std::sqrt(pq.pAbs2()+mq*mq));
                  pdiq.e(std::sqrt(pdiq.pAbs2()+mdiq*mdiq));
                  Particle recolQ = Particle(qid, 201, i, 0, 0, 0,
                                    q_col, 0,
                                    pq, mq, 0);
                  Particle recoldiQ = Particle(diqid, 201, i, 0, 0, 0,
                                    0, q_acol,
                                    pdiq, mdiq, 0);

                  recoil_remnants.push_back(recolQ);
                  recoil_remnants.push_back(recoldiQ);
               }
           }
           for (auto & p : frag_gluons) all_extra.push_back(p);
           for (auto & p : recoil_remnants) all_extra.push_back(p);

           //if (p.id()==hardP.id())
           //    fstat << Ncolls << " " << Nhard_rad << " " << Nrad << std::endl;
        }



        for (auto & p : all_extra)
            event.append(p.id(), 201, p.col(), p.acol(),
                     p.px(), p.py(), p.pz(), p.e(), p.m());

        // put the parton level event into a separate hadronizer
        auto event2 = HZ.hadronize(pythia, Z, A);
        // output
        output<std::vector<Particle> >(pythia, event2, f);
    }
    std::cout << "TriggerRate = " << Ntriggered*1./count << std::endl;
    std::cout << "FailedRate = " << failed*1./count<< std::endl;
    // Done.
    return 0;
}
