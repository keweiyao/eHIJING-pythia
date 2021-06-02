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
    pythia.readString("PDF:pSet = 8");
    pythia.readString("ProcessLevel:all = off");
    pythia.readString("Print:quiet = on"); 
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");
    pythia.readString("HadronLevel:all = on");
    pythia.readString("HadronLevel:Decay = on");
    pythia.readString("StringFragmentation:stopMass = 0.4");
    pythia.readString("StringZ:aLund = 0.7");
    pythia.readString("StringZ:bLund = 0.2");
    //pythia.readString("StringPT:sigma = 0.3");

    pythia.readString("111:mayDecay=off");
    pythia.readString("211:mayDecay=off");
    pythia.readString("311:mayDecay=off");
    pythia.readString("321:mayDecay=off");
    pythia.init();
   }
   std::vector<Particle> hadronize(Pythia & pythia, int Z, int A){
      auto & event_in = pythia.event;
      std::vector<Particle> plist; plist.clear();
      for (int i=0; i<event_in.size(); i++){   
        auto & p = event_in[i];
        if(p.isFinal() && p.isParton()){
            pythia.event.reset();
            pythia.event.append(p.id(), 23, p.col(), p.acol(), 
                              p.px(), p.py(), p.pz(), p.e(), p.m());
            pythia.event[1].scale(.4);

            double Mdiquark = 0.66;
            double Mquark = 0.33;
            int iddiq, idq;
            if (dist(gen)<Z*1./A) { 
                if (dist(gen)<2./3.) {
                    if(dist(gen)<.25) {iddiq = 2101; idq = 2;}
                    else {iddiq = 2103; idq = 2;}
                }
                else {iddiq = 2203; idq=1;}
            }
            else {
                if (dist(gen)<1./3.)  {iddiq = 1103; idq=2;}
                else {
                    if(dist(gen)<.25) {iddiq = 2101; idq=1;}
                    else { iddiq = 2103; idq=1;}
                }
            }

            if (p.id()==21){
                pythia.event.append(iddiq, 23, 0, p.col(), 
                            0.,0.,0., Mdiquark, Mdiquark);
                pythia.event[2].scale(.4);

                pythia.event.append(idq, 23, p.acol(), 0, 
                            0.,0.,0., Mquark, Mquark);
                pythia.event[3].scale(.4);

            }else{
                pythia.event.append(idq, 23, p.acol(), p.col(), 
                            0.,0.,0., Mdiquark, Mdiquark);
                pythia.event[2].scale(.4);

            }
             
            pythia.next();
            for (int j=0; j<pythia.event.size(); j++) 
                if (pythia.event[j].isFinal()) 
                    plist.push_back(pythia.event[j]);
            
           }
       }   
       return plist;
    }

   std::vector<Particle> hadronize_color_intact(Pythia & pythiaIn, int Z, int A){
      auto & event_in = pythiaIn.event; 
      std::vector<Particle> plist; plist.clear();
      pythia.event.reset();
      for (int i=0; i<event_in.size(); i++){   
        auto & p = event_in[i];
        if(p.isFinal() && p.isParton()){
            pythia.event.append(p.id(), 23, p.col(), p.acol(), 
                              p.px(), p.py(), p.pz(), p.e(), p.m());
        }
      }
      pythia.next();
      for (int j=0; j<pythia.event.size(); j++) 
        if (pythia.event[j].isFinal()) 
          plist.push_back(pythia.event[j]);
      return plist;
   }

   std::vector<Particle> hadronize_independent(Pythia & pythiaIn, int Z, int A){
      double ZoverA = Z*1./A;
      std::vector<Particle> Showers, Remnants, FinalParticles; 
      Showers.clear();
      Remnants.clear();
      FinalParticles.clear();
      int hardid = pythiaIn.event[5].id();
      // step 1: sort partons into hard parton showers, and remnantss
      for (int i=0; i<pythiaIn.event.size(); i++){  
          auto & p = pythiaIn.event[i]; 
          if (p.isFinal() && p.isParton()) {
              if (p.status()==63){
                  //fs << p.px() << " " << p.py() << " " << p.pz() << " " << p.m() << std::endl;
                  // This is a beam remanent
                  if (hardid==1 || hardid==2) {
                      // valence stuff, the remnants will contain the rest flavor compoennt.
                      // note that the hard quark has already been sampled accorrding to the
                      // the isospin content of the nuclear PDF; however, the remanent is generated
                      // assuming the rest stuff comes from a proton. Therefore, we need to resample 
                      // it according to the Z/A ratio this nuclei
                      // 1) decide wither it is from a neutron or proton
                      if (dist(gen) < ZoverA) { // From a proton 2212
                          if (hardid==1){ // produce 2203
                              p.id(2203);
                          } else { // produce 2101 and 2103 with ratio 3:1
                              if (dist(gen) < 0.75) p.id(2101);
                              else p.id(2103);
                          }
                      } else { // From a neutron 2112
                          if (hardid==1){ // produce 2101 and 2103 with ratio 3:1
                              if (dist(gen) < 0.75) p.id(2101);
                              else p.id(2103);
                          } else { // produce 1103
                              p.id(1103);
                          }
                      }
                  } else {
                      // sea stuff, should be an anti u,d or other flavors such as s,c,b
                      // the remnants should be a quark / anti-quark.
                      // However, the flavor selection of the remnants in Pythia sometimes puzzles me.
                      // I need to double check on this. But let's do nothing rightnow.
                      //p.id(-hardid);
                  }
                  Remnants.push_back(p);
              }
              else Showers.push_back(p);
          }
      }

      
      // Step 2: Find the first partons in the shower that are color connected to the remnants
      std::vector<Particle> StringA; StringA.clear();
      // 1) identitfy the net colors of the remnants
      std::vector<int> netCols, netAcols; netCols.clear(); netAcols.clear();
      for (auto & p : Remnants) {
          StringA.push_back(p);
          if (p.acol()!=0){
              auto it = std::find(netCols.begin(), netCols.end(), p.acol());
              if (it != netCols.end()) netCols.erase(it);
              else netAcols.push_back(p.acol());
          }
          if (p.col()!=0){  
              auto it = std::find(netAcols.begin(), netAcols.end(), p.col());
              if (it != netAcols.end()) netAcols.erase(it);
              else netCols.push_back(p.col());
          }
      }
      // 2) Find the immeidate connected shower partons to the net colors
      for (auto it=Showers.begin(); it!=Showers.end();){
          auto itc = std::find(netCols.begin(), netCols.end(), it->acol());
          auto itac = std::find(netAcols.begin(), netAcols.end(), it->col());
          bool connected_to_remnant_col = (itc != netCols.end());
          bool connected_to_remnant_acol = (itac != netAcols.end());
          bool connected_to_remnant = connected_to_remnant_col || connected_to_remnant_acol;
          if (connected_to_remnant) {
             double mq = 0.33, mdiq=0.66;
             double r1 = dist(gen), r2 = dist(gen);
             double qx = .2 * std::sqrt(-2*std::log(r1)) * std::cos(2.*M_PI*r2);
             double qy = .2 * std::sqrt(-2*std::log(r1)) * std::sin(2.*M_PI*r2);
             double qz = 0;
             Vec4 pq{qx,qy,qz,std::sqrt(qx*qx+qy*qy+qz*qz+mq*mq)}, 
                pdiq{-qx,-qy,-qz,std::sqrt(qx*qx+qy*qy+qz*qz+mdiq*mdiq)};

              StringA.push_back(*it);
              if ((!connected_to_remnant_col) && it->acol()!=0) { // end this color flow by sampling a new remnant quark
                  if (dist(gen) < (1.+ZoverA) / 3. ) {
                      Particle p(2, 63, 0, 0, 0, 0, it->acol(), 0, pq, mq);
                      StringA.push_back(p);
                  } else {
                      Particle p(1, 63, 0, 0, 0, 0, it->acol(), 0, pq, mq);
                      StringA.push_back(p);
                  }
              }
              if ((!connected_to_remnant_acol) && it->col()!=0) { // end this color flow by sampling a new remnant di-quark
                  if (dist(gen) < ZoverA) { // diquark from a proton
                      if (dist(gen) < 2./3.) { // take away a u
                          if (dist(gen) < .75) {
                              Particle p(2101, 63, 0, 0, 0, 0, 0, it->col(), pdiq, mdiq);
                              StringA.push_back(p);
                          } else {
                              Particle p(2103, 63, 0, 0, 0, 0, 0, it->col(), pdiq, mdiq);
                              StringA.push_back(p);
                          }
                      } else { // take away the d
                          Particle p(2203, 63, 0, 0, 0, 0, 0, it->col(), pdiq, mdiq);
                          StringA.push_back(p);
                      }
                  } else { // diquark from a neutron
                      if (dist(gen) < 2./3.) { // take away a d
                          if (dist(gen) < .75) {
                              Particle p(2101, 63, 0, 0, 0, 0, 0, it->col(), pdiq, mdiq);
                              StringA.push_back(p);
                          } else {
                              Particle p(2103, 63, 0, 0, 0, 0, 0, it->col(), pdiq, mdiq);
                              StringA.push_back(p);
                          }
                      } else { // take away the u
                          Particle p(1103, 63, 0, 0, 0, 0, 0, it->col(), pdiq, mdiq);
                          StringA.push_back(p);
                      }
                  }
              }
              it = Showers.erase(it);
          }
          else it++;
      }
      // 3) Hadronize this string that contains the original remnant
      pythia.event.reset(); int count=0;
      for (auto & p : StringA){
          count ++;
          pythia.event.append(p.id(), 23, p.col(), p.acol(), 
                              p.px(), p.py(), p.pz(), p.e(), p.m());
          pythia.event[count].scale(.4);
      }
      pythia.next();
      for (int i=0; i<pythia.event.size(); i++) {
          auto & p = pythia.event[i];
          if (p.isFinal()) FinalParticles.push_back(p);
      }

     // Step 3: connect all the rest shower partons to independent remanents
     
     for (auto & p : Showers) {
         // 1) perpare a quark + diquark
         int qid, diqid;
         double mq = 0.33, mdiq=0.66;
         double r1 = dist(gen), r2 = dist(gen);
         double qx = .2 * std::sqrt(-2*std::log(r1)) * std::cos(2.*M_PI*r2);
         double qy = .2 * std::sqrt(-2*std::log(r1)) * std::sin(2.*M_PI*r2);
         double qz = 0;
         Vec4 pq{ qx,  qy,  qz, std::sqrt(qx*qx+qy*qy+qz*qz+mq*mq)}, 
            pdiq{-qx, -qy, -qz, std::sqrt(qx*qx+qy*qy+qz*qz+mdiq*mdiq)};
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
         pythia.event.reset();
         if (p.id()==21) {
             pythia.event.append(p.id(), 23, p.col(), p.acol(), 
                                 p.px(), p.py(), p.pz(), p.e(), p.m());      
             pythia.event.append(qid, 23, p.acol(), 0, 
                                 pq.px(), pq.py(), pq.pz(), pq.e(), mq);      
             pythia.event.append(diqid, 23, 0, p.col(), 
                                 pdiq.px(), pdiq.py(), pdiq.pz(), pdiq.e(), mdiq);    
         } else if (p.id()<0) {
             pythia.event.append(p.id(), 23, p.col(), p.acol(), 
                                 p.px(), p.py(), p.pz(), p.e(), p.m());      
             pythia.event.append(qid, 23, p.acol(), 0, 
                                 pq.px(), pq.py(), pq.pz(), pq.e(), mq);      
         } else {
             pythia.event.append(p.id(), 23, p.col(), p.acol(), 
                                 p.px(), p.py(), p.pz(), p.e(), p.m());      
             pythia.event.append(diqid, 23, 0, p.col(), 
                                 pdiq.px(), pdiq.py(), pdiq.pz(), pdiq.e(), mdiq);      
         }
         pythia.next();  
         
         for (int i=0; i<pythia.event.size(); i++) {
             auto & p = pythia.event[i];
             if (p.isFinal()) FinalParticles.push_back(p);
         }      
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
    // Compute four-momenta of proton, electron, virtual 
    Vec4 pProton = pythia.event[1].p(); // four-momentum of proton
    Vec4 peIn    = pythia.event[4].p(); // incoming electron
    Vec4 peOut   = pythia.event[6].p(); // outgoing electron
    Vec4 pGamma = peIn - peOut; // virtual boson photon/Z^0/W^+-
    // Q2, W2, Bjorken x, y.
    double Q2 = - pGamma.m2Calc(); // hard scale square
    double Q  = std::sqrt(Q2);
    double W2 = (pProton + pGamma).m2Calc(); 
    double x  = Q2 / (2. * pProton * pGamma); // Bjorken x
    double nu = pGamma.e();
    // In Breit frame, where gamma ~ (0,0,0,-Q),
    double y     = (pProton * pGamma) / (pProton * peIn);

    return (1.0<Q2) & (W2>4) & (nu>6) & (y<.85) ;
}

void MomentumRescale(Pythia & pythia) {
    // Compute four-momenta of proton, electron, virtual 
    Vec4 pProton = pythia.event[1].p(); // four-momentum of proton
    Vec4 qIn     = pythia.event[3].p(); // incoming quark
    Vec4 peIn    = pythia.event[4].p(); // incoming electron
    Vec4 qOut    = pythia.event[5].p(); // outgoing quark
    Vec4 peOut   = pythia.event[6].p(); // outgoing electron
    Vec4 pGamma = peIn - peOut; // virtual boson photon/Z^0/W^+-
    // Q2, W2, Bjorken x, y.
    double Q2 = - pGamma.m2Calc(); // hard scale square
    double Q  = std::sqrt(Q2);
    double W2 = (pProton + pGamma).m2Calc(); 
    double x  = Q2 / (2. * pProton * pGamma); // Bjorken x
    double nu = pGamma.e();
    // Find the current CoM frame
    Vec4 pCoM{0,0,0,0};
    for(int i=0; i<pythia.event.size(); i++){
        auto & p = pythia.event[i];
        if (p.isFinal() && p.isParton()) pCoM += p.p();
    }
    double M2 = pCoM.m2Calc();
    double rescale_factor = std::sqrt(Q2/M2);
    // Boost everything to the CoM Frame
    for(int i=0; i<pythia.event.size(); i++){
        auto & p = pythia.event[i];
        if (p.isFinal() && p.isParton()) p.bstback(pCoM);
    }
    // Rescale momentum in the CoM frame to match Q2
    Vec4 pCoM2{0,0,0,0};   
    for(int i=0; i<pythia.event.size(); i++){
        auto & p = pythia.event[i];
        if (p.isFinal() && p.isParton()) {
            p.p(p.p()*rescale_factor); 
            p.e(std::sqrt(p.pAbs2()+p.m2())); 
            pCoM2 += p.p();
        }
    }
    double M2new = pCoM2.m2Calc();
    rescale_factor = std::sqrt(Q2/M2new);

    pCoM2 = 0.*pCoM2;
    for(int i=0; i<pythia.event.size(); i++){
        auto & p = pythia.event[i];
        if (p.isFinal() && p.isParton()) {
            p.p(p.p()*rescale_factor); 
            p.e(std::sqrt(p.pAbs2()+p.m2())); 
            pCoM2 += p.p();
        }
    }

    // std::cout<< "Q2 = "<< Q2 << ", M2 = " << M2new << std::endl;
    // Boost back to the original frame, but keep the three momentum of pCOM unchanged
    double newE = std::sqrt(pCoM.pAbs2() + M2new);
    double vx = pCoM.px()/newE, vy = pCoM.py()/newE, vz = pCoM.pz()/newE;
    Vec4 psum{0,0,0,0};   
    for(int i=0; i<pythia.event.size(); i++){
        auto & p = pythia.event[i];
        if (p.isFinal() && p.isParton()) {
          p.bst(vx, vy, vz);
          psum += p.p();
        }
    }
    //std::cout<< "Gamma "<< pGamma+qIn << "Q = "<< Q << std::endl << "Phadron " << psum << std::endl;
    return;
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
    double Q  = std::sqrt(Q2);
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
        if (p.isFinal() && p.isHadron()){
           
           auto pbst = p.p();
           pbst.bstback(pCoM);
           double xF = dot3(pGamma2, pbst);       
           if (xF<0) continue;
           auto prot = p.p();
           prot.rot(0, phi);
           prot.rot(theta, 0);
           
           f << p.id() << " " << p.e()/nu << " " << prot.pT() << " " << nu << " " << Q2 << std::endl;
          
        }
    }
}


int main(int argc, char *argv[]) {
  // commandline args
  int nEvent = atoi(argv[1]);
  int Z = atoi(argv[2]);
  int A = atoi(argv[3]);
  auto header = std::string(argv[4]);
  int ishadow = 0;
  double pTmin = 0.4;
  int inuclei = 100000000
              +   Z*10000
              +      A*10;
  hadronizer HZ;

  // Initialize
  Pythia pythia;            // Generator 
  Event& event = pythia.event; // Event record
  pythia.readFile("pythia-ep-settings.txt"); // read settings
  add_arg<double>(pythia, "TimeShower:pTmin", pTmin);
  add_arg<int>(pythia, "PDF:nPDFSetA", A>2? 2:0);
  add_arg<int>(pythia, "PDF:nPDFBeamA", inuclei);
  add_arg<int>(pythia, "eHIJING:AtomicNumber", A);
  add_arg<int>(pythia, "eHIJING:ChargeNumber", Z);
  add_arg<double>(pythia, "eHIJING:Kfactor", A>2? 3.0:0.0);

  pythia.init(); 

  // output
  std::stringstream  ss;
  ss << header << "/" << Z << "-" << A << "-cutRA.dat";
  std::ofstream f(ss.str());

  // Begin event loop.
  int Ntriggered = 0;
  while(Ntriggered<nEvent){
      if (!pythia.next()) continue; // skip bad events
      if (!trigger(pythia)) continue; // only study triggered events
      Ntriggered ++;
      if (Ntriggered%1000==0)  std::cout << "# of trigged events: " << Ntriggered << std::endl;

      MomentumRescale(pythia);
      // put the parton level event into a separate hadronizer
      auto event2 = HZ.hadronize_independent(pythia, Z, A);

      Vec4 pcom{0,0,0,0};
      for(auto&p:event2){
          if (p.isFinal()) pcom += p.p();
      }
      //std::cout <<pcom<<" "<<pcom.m2Calc()<<std::endl << "------------" <<std::endl;
      // output
      output<std::vector<Particle> >(pythia, event2, f);
  }

  // Done.
  return 0;
}
