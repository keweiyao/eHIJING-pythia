#include "Pythia8/Pythia.h"
using namespace Pythia8;
#include <fstream>
#include <random>
#include <sstream>
#include <algorithm>
std::ofstream fs("stat.dat");
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
    pythia.readString("StringFragmentation:stopMass = 0.");
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
    return (1.0<Q2) & (W2>10.) & (y<.85);
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

    for (int j=0; j<plist.size(); j++){
      auto p = plist[j];
      if (p.isFinal()){
         auto pbst = p.p();
         pbst.bstback(pCoM);
         double xF = dot3(pGamma2, pbst);
         if (xF<0) continue;
         double z = p.e()/nu;
         auto prot = p.p();
         prot.rot(0, phi);
         prot.rot(theta, 0);
         f << p.id() << " " << z << " " << prot.pT() << " " << nu << " " << Q2 << std::endl;
        }
    }
}

int main(int argc, char *argv[]) {
  // commandline args
  int nEvent = atoi(argv[1]);
  int Z = atoi(argv[2]);
  int A = atoi(argv[3]);
  double K = atof(argv[4]);
  double ZoverA = Z*1./A;
  auto header = std::string(argv[5]);
  int ishadow = 0;
  double pTmin = 0.5;
  double pT2min = std::pow(pTmin, 2);
  int inuclei = 100000000
          +   Z*10000
          +    A*10;
  hadronizer HZ;
  EHIJING::InMediumFragmentation gen2(1, K, 4, -0.5, 0.5);
  gen2.Tabulate("./Tables/");
  EHIJING::NuclearGeometry Geometry(A, Z);
  EHIJING::MultipleCollision Coll(K, 4, -0.5, 0.5);
  Coll.Tabulate("./Tables/");
  // Initialize
  Pythia pythia;        // Generator
  Event& event = pythia.event; // Event record
  pythia.readFile("pythia-ep-settings.txt"); // read settings
  add_arg<double>(pythia, "TimeShower:pTmin", pTmin);
  add_arg<int>(pythia, "PDF:nPDFSetA", A>2? 2:0);
  add_arg<int>(pythia, "PDF:nPDFBeamA", inuclei);
  add_arg<int>(pythia, "eHIJING:AtomicNumber", A);
  add_arg<int>(pythia, "eHIJING:ChargeNumber", Z);
  add_arg<double>(pythia, "eHIJING:Kfactor", K);

  pythia.init();
  double lambda2 = .04;
  // output
  std::stringstream  ss;
  ss << header << "/" << Z << "-" << A << "-cutRA.dat";
  std::ofstream f(ss.str());

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> dist(0.,1.);
  // Begin event loop.
  int Ntriggered = 0;
  int count = 0, failed=0;
  double qhat_avg = 0., qhat_w = 0.;
  while(Ntriggered<nEvent){
    count++;
    //std::cout << "1 ";
    if (!pythia.next()) {
        failed++;
        continue;
    }; // skip bad events
    //std::cout << "2 " << std::endl;
    if (!trigger(pythia)) continue; // only study triggered events
    Ntriggered ++;
    if (Ntriggered%1000==0)  std::cout << "# of trigged events: "
                             << Ntriggered << std::endl;

    std::vector<Particle> frag_gluons;
    frag_gluons.clear();
    Vec4 pProton = event[1].p(); // four-momentum of proton
    Vec4 peIn    = event[4].p(); // incoming electron
    Vec4 peOut   = event[6].p(); // outgoing electron
    Vec4 pGamma = peIn - peOut; // virtual boson photon/Z^0/W^+-
    double Q20 = - pGamma.m2Calc(); // hard scale square
    double xB  = Q20 / (2. * pProton * pGamma); // Bjorken x
    Vec4 hardP = event[5].p();


    // elastic broadening
    double L0 = Geometry.compute_L(event.Rx(), event.Ry(), event.Rz(),
		hardP.px()/hardP.e(), hardP.py()/hardP.e(), hardP.pz()/hardP.e());
    //std::vector<double> qt2s, ts;
    //Coll.sample_all_qt2(event[5].idAbs(), hardP.e(), L0, xB, Q20, qt2s, ts);

    // Medium modified Fragmentation
    for (int i=0; i<event.size(); i++) {
        // find final-state quarks and gluons
        auto & p = event[i];
        if (!p.isFinal() ) continue;
        bool triggered = (p.idAbs()==1) || (p.idAbs()==2) || (p.idAbs()==3) ||
                         (p.idAbs()==4) || (p.idAbs()==5) || (p.idAbs()==21);
        if (!triggered) continue;
        // path length
        double L = Geometry.compute_L(event.Rx(), event.Ry(), event.Rz(),
                                     p.px()/p.e(), p.py()/p.e(), p.pz()/p.e());
        // maximum scale for the medium-modfied hadronization
        double Qs2 = Coll.Qs2(xB, Q20, std::max(L,1.0)*EHIJING::rho0);
        qhat_avg += Coll.qhatA(xB, Q20, std::max(L,1.0)*EHIJING::rho0);
        qhat_w += 1.;
        double Qs = std::sqrt(Qs2);
        double pT2min_in_A = std::max(pT2min, Qs2);
        // From now on, make sure all the momentum is in the rest frame of A!
        // First, compute the maximum range of formation time varaible
        //     or, the minimum and maximum of the omegaL = L/tauf variable
        double omegaL_min = 2*lambda2*L/p.e();
        double omegaL = std::sqrt(Q20)*L;


        int Nrad = 0;
        while(omegaL > omegaL_min){
          double z;
          bool status = gen2.next_radiation(p.idAbs(), p.e(), L, xB, Q20,
                                pT2min_in_A, lambda2, omegaL, z);
          if (omegaL < omegaL_min) break;
          if (!status) continue;
          // momentum of the radiated gluon and transfer gluon in the +z frame of p
          double kt = std::sqrt(omegaL * 2*z*(1-z)* p.e() / L);
          double k0 = z*p.e();
          double phik = 2.*M_PI*dist(gen);
          double kz2 = k0*k0-kt*kt;
          if (kz2<0){
              std::cout << "warning! kz2<0"<< std::endl;
              continue;
          }

          Nrad ++;
          double kz = std::sqrt(kz2);
          Vec4 kmu{kt*std::cos(phik), kt*std::sin(phik), kz, k0};
          // sample the momentum transfer associated with the radiation
          // In the current model (GHT), this is the same as elastic collisions
          // but requiring |kT| < |qT|
          double qx, qy, xg, tq;
          Coll.sample_single_qt2(p.idAbs(), p.e(), L, xB, Q20,
                                 qx, qy, xg, tq, kt*kt, phik);
          Vec4 qmu{qx, qy, xg*pProton.e(), xg*pProton.e()};
          // back to lab frame:
          kmu.rot(p.theta(), 0.);
          kmu.rot(0., p.phi());
          qmu.rot(p.theta(), 0.);
          qmu.rot(0., p.phi());
          // Recoil of the hard dparton, and put it back on shell
          p.p(p.p()-kmu);
          kmu = kmu + qmu;
          p.e(std::sqrt(p.pAbs2()+p.m2()));
          // update the new minimum of omegaL
          omegaL_min = lambda2*L/(.5*p.e());
          int k_col, k_acol, q_col, q_acol;
          // update the color:
          // first, the spliting process
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
          // next, the radiated gluon change color due to the rescattering
          if (std::rand()%2==0){
            q_acol = k_acol;
            k_acol = event.nextColTag();
            q_col = k_acol;
          } else {
            q_col = k_col;
            k_col = event.nextColTag();
            q_acol = k_col;
          }
          // make the gluon with status 201
          Particle gluon = Particle(21, 201, i, 0, 0, 0,
                            k_col, k_acol,
                            kmu, 0.0, Qs);
          // The nucleon broken by the rescattering:
          int qid, diqid;
          double mq = 0.3, mdiq=0.6;
          double r1 = dist(gen), r2 = dist(gen);
          double pTstd = 0.2;
          double Nqx = pTstd * std::sqrt(-2*std::log(r1)) * std::cos(2.*M_PI*r2);
          double Nqy = pTstd * std::sqrt(-2*std::log(r1)) * std::sin(2.*M_PI*r2);
          Vec4 pq  { Nqx,  Nqy,  0, 0},
               pdiq{-Nqx, -Nqy,  0, 0};
          if (dist(gen) < ZoverA) { // diquark from a proton
            if (dist(gen) < 2./3.) { // take away a u
               qid = 2;
               if (dist(gen) < .75) diqid = 2101;
               else diqid = 2103;
            } else { // take away the d
               qid = 1;
               diqid = 2203;
            }
          } else { // diquark from a neutron
            if (dist(gen) < 2./3.) { // take away a d
                qid = 1;
                if (dist(gen) < .75) diqid = 2101;
                else diqid = 2103;
            } else { // take away the u
                qid = 2;
                diqid = 1103;
            }
          }
          if (std::rand()%2==0){
            pq = pq - qmu;
          } else {
            pdiq = pdiq - qmu;
          }
          pq.e(std::sqrt(pq.pAbs2()+mq*mq));
          pdiq.e(std::sqrt(pdiq.pAbs2()+mdiq*mdiq));
          Particle recolQ = Particle(qid, 201, i, 0, 0, 0,
                            q_col, 0,
                            pq, mq, Qs);
          Particle recoldiQ = Particle(diqid, 201, i, 0, 0, 0,
                            0, q_acol,
                            pdiq, mdiq, Qs);


          frag_gluons.push_back(gluon);
          frag_gluons.push_back(recolQ);
          frag_gluons.push_back(recoldiQ);
        }


        // pure elasitc collisions
        if (Nrad == 0) {
          std::vector<double> qt2s, ts;
          Coll.sample_all_qt2(p.idAbs(), p.e(), L, xB, Q20, qt2s, ts);

        for (auto & q2 : qt2s){
          double phi = 2*M_PI*dist(gen),
                 qT = std::sqrt(q2);
          Vec4 qmu{qT*std::cos(phi), qT*std::sin(phi),
                   q2/Q20*xB*pProton.e(), 0.};
          // back to lab frame:
          qmu.rot(p.theta(), 0.);
          qmu.rot(0., p.phi());
          // recoil the hard parton
          p.p(p.p()+qmu);
          p.e(std::sqrt(p.pAbs2()+p.m2()));

          // it also induces extra remnant
          int qid, diqid, q_col, q_acol;
          q_col = event.nextColTag();
          q_acol = q_col;
          double mq = 0.3, mdiq=0.6;
          double r1 = dist(gen), r2 = dist(gen);
          double pTstd = 0.2;
          double qx = pTstd * std::sqrt(-2*std::log(r1)) * std::cos(2.*M_PI*r2);
          double qy = pTstd * std::sqrt(-2*std::log(r1)) * std::sin(2.*M_PI*r2);
          Vec4 pq  { qx,  qy,  0, 0},
               pdiq{-qx, -qy,  0, 0};
          if (std::rand()%2==0){
            pq = pq - qmu;
          } else {
            pdiq = pdiq - qmu;
          }
          pq.e(std::sqrt(pq.pAbs2()+mq*mq));
          pdiq.e(std::sqrt(pdiq.pAbs2()+mdiq*mdiq));
          Particle recolQ = Particle(qid, 201, i, 0, 0, 0,
                            q_col, 0,
                            pq, mq, Qs);
          Particle recoldiQ = Particle(diqid, 201, i, 0, 0, 0,
                            0, q_acol,
                            pdiq, mdiq, Qs);
          frag_gluons.push_back(recolQ);
          frag_gluons.push_back(recoldiQ);
        }
      }
    }
    for (auto & p : frag_gluons) {
        event.append(p.id(), 201, p.col(), p.acol(),
                     p.px(), p.py(), p.pz(), p.e(), p.m());
    }


    // put the parton level event into a separate hadronizer
    auto event2 = HZ.hadronize(pythia, Z, A);
    // output
    output<std::vector<Particle> >(pythia, event2, f);
  }
  std::cout << "TriggerRate = " << Ntriggered*1./count << std::endl;
  std::cout << "FailedRate = " << failed*1./count<< std::endl;
  std::cout << "qhatg = " << qhat_avg / qhat_w *5.076<< std::endl;
  // Done.
  return 0;
}
