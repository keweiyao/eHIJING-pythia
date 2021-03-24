#include "Pythia8/Pythia.h"
using namespace Pythia8;
#include <fstream>

// Rotate and boost the Lab frame
// to Breit frame for each event,
// where p_gamma' = (0,0,0,-Q)
//   and p_proton' = (E',0,0,P')
//   with Pz = Q/2x
Vec4 LabToBreit(Vec4 p, double theta, double phi, 
                        double vx, double vy, double vz){
  p.rot(0., phi);
  p.rot(theta, 0.);
  p.bst(0., 0., vz);
  p.bst(vx, vy, 0.);
  return p;
}

int main(int argc, char *argv[]) {
  int nEvent = 1000; // number of events
  if (argc == 2) nEvent = atoi(argv[1]);

  Pythia pythia;            // Generator 
  Event& event = pythia.event; // Event record
  pythia.readFile("pythia-ep-settings.txt"); // read settings
  pythia.init();  // Initialize.

  std::ofstream fi("InclusiveInfo.dat");
  std::ofstream fe("ExclusiveInfo.dat");
  fi << "# ievent\tQ2[GeV^2]\tW2[GeV^2]\tx\ty\n";
  fe << "## pid\tM[GeV]\tE[GeV]\tpx[GeV]\tpy[GeV]\tpz[GeV]\txz=2pz/Q\n";
  // Begin event loop.
  int count = 0;
  do {
      if (!pythia.next()) continue; // skip bad events
      // Compute four-momenta of proton, electron, virtual 
      Vec4 pProton = event[1].p(); // four-momentum of proton
      Vec4 peIn    = event[4].p(); // incoming electron
      Vec4 peOut   = event[6].p(); // outgoing electron
      Vec4 pGamma = peIn - peOut; // virtual boson photon/Z^0/W^+-

      // Q2, W2, Bjorken x, y.
      double Q2    = - pGamma.m2Calc(); // hard scale square
      double Q = std::sqrt(Q2);
      double W2    = (pProton + pGamma).m2Calc(); // center-of-mass energy 
                                                  // of gamma-p collision
      double x     = Q2 / (2. * pProton * pGamma); // Bjorken x
      double nu = pGamma.e();
      // In Breit frame, where gamma ~ (0,0,0,-Q),
      // x = Q*Q / (2*Pz*Q) = Q/2 / Pz, Parton energy ~ Q/2
      double y     = (pProton * pGamma) / (pProton * peIn);
      // y ~ s_{gamma-p} / s_{e-p}, inelasticity of the e-p collision
      bool trigger = (1.0<Q2) & (W2>4) & (nu>6.0) & (y<.85);
      //std::cout << Q2 << " ";
      if (!trigger) continue;
      count ++;

      // 1) boost into gamma+proton CoM frame
      Vec4 pCoM = pGamma + pProton;
      pGamma.bstback(pCoM);
      pProton.bstback(pCoM);
      // output inclusive event info
      fi << count << " " << Q2 << " " << W2 << " " 
                  << x << " " << y << std::endl;
      // output exclusive particle list in the Briet frame
      fe << "# event " << count << std::endl;
      if (count%1000==0) 
          std::cout << "# of trigged events: "<<count<<std::endl;
      for (int j=0; j<event.size(); j++){
        // transform final four momentum into Briet Frame
        auto P = event[j];
        if (P.isFinal()) {
            Vec4 p = P.p();
            p.bstback(pCoM);
            double pf = dot3(p, pGamma)/pGamma.pAbs();
            if (P.isParton() && (pf>0.)){
                 fe << P.id() << "\t" << P.e()/nu << std::endl; 
            }
        }
     }
  }while(count<nEvent);
  // Done.
  return 0;
}
