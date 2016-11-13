#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <chrono>
#include <random>
#include <numeric>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include "random/Bernoulli.h"
#include "random/Beta.h"
#include "random/Binomial.h"
#include "random/BooleanDistrib.h"
#include "random/Categorical.h"
#include "random/Exponential.h"
#include "random/Gaussian.h"
#include "random/Gamma.h"
#include "random/Geometric.h"
#include "random/Poisson.h"
#include "random/InvGamma.h"
#include "random/TruncatedGauss.h"
#include "random/UniformChoice.h"
#include "random/UniformInt.h"
#include "random/UniformReal.h"
#include "util/Hist.h"
#include "util/util.h"
#include "util/DynamicTable.h"


// MCMC Library included
#include "util/MCMC.h"
#include "util/util_MCMC.h"

// Matrix Library included
#include "armadillo"
#include "random/DiagGaussian.h"
#include "random/Dirichlet.h"
#include "random/Discrete.h"
#include "random/InvWishart.h"
#include "random/MultivarGaussian.h"
#include "random/MultivarGaussianIndep.h"
#include "random/Multinomial.h"
#include "random/UniformVector.h"
#include "random/PrecisionGaussian.h"
#include "util/Hist_matrix.h"
#include "util/util_matrix.h"
using namespace arma;

using namespace std;
using namespace swift::random;


int main();

namespace swift {

class _Var_beta1;
class _Var_beta2;
class _Var_logit;
class _Var_region_rate;

const vector<string> __vecstr_instance_Week = {"weeks[0]", "weeks[1]", "weeks[2]", "weeks[3]", "weeks[4]", "weeks[5]", "weeks[6]", "weeks[7]", "weeks[8]", "weeks[9]", "weeks[10]", "weeks[11]", "weeks[12]", "weeks[13]", "weeks[14]", "weeks[15]", "weeks[16]", "weeks[17]", "weeks[18]", "weeks[19]", "weeks[20]", "weeks[21]", "weeks[22]", "weeks[23]", "weeks[24]", "weeks[25]", "weeks[26]", "weeks[27]", "weeks[28]", "weeks[29]", "weeks[30]", "weeks[31]", "weeks[32]", "weeks[33]", "weeks[34]", "weeks[35]", "weeks[36]", "weeks[37]", "weeks[38]", "weeks[39]", "weeks[40]", "weeks[41]", "weeks[42]", "weeks[43]", "weeks[44]", "weeks[45]", "weeks[46]", "weeks[47]", "weeks[48]", "weeks[49]", "weeks[50]", "weeks[51]", "weeks[52]", "weeks[53]", "weeks[54]", "weeks[55]", "weeks[56]", "weeks[57]", "weeks[58]", "weeks[59]", "weeks[60]", "weeks[61]", "weeks[62]", "weeks[63]", "weeks[64]", "weeks[65]", "weeks[66]", "weeks[67]", "weeks[68]", "weeks[69]", "weeks[70]", "weeks[71]", "weeks[72]", "weeks[73]", "weeks[74]", "weeks[75]", "weeks[76]", "weeks[77]", "weeks[78]", "weeks[79]", "weeks[80]", "weeks[81]", "weeks[82]", "weeks[83]", "weeks[84]", "weeks[85]", "weeks[86]", "weeks[87]", "weeks[88]", "weeks[89]", "weeks[90]", "weeks[91]", "weeks[92]", "weeks[93]", "weeks[94]", "weeks[95]", "weeks[96]", "weeks[97]", "weeks[98]", "weeks[99]", "weeks[100]", "weeks[101]", "weeks[102]"};
const vector<string> __vecstr_instance_Region = {"regions[0]", "regions[1]", "regions[2]", "regions[3]", "regions[4]", "regions[5]", "regions[6]", "regions[7]", "regions[8]"};
const vector<string> __vecstr_instance_County = {"counties[0]", "counties[1]", "counties[2]", "counties[3]", "counties[4]", "counties[5]", "counties[6]", "counties[7]", "counties[8]", "counties[9]", "counties[10]", "counties[11]", "counties[12]", "counties[13]", "counties[14]", "counties[15]", "counties[16]", "counties[17]", "counties[18]", "counties[19]", "counties[20]", "counties[21]", "counties[22]", "counties[23]", "counties[24]", "counties[25]", "counties[26]", "counties[27]", "counties[28]", "counties[29]", "counties[30]", "counties[31]", "counties[32]", "counties[33]", "counties[34]", "counties[35]", "counties[36]", "counties[37]", "counties[38]", "counties[39]", "counties[40]", "counties[41]", "counties[42]", "counties[43]", "counties[44]", "counties[45]", "counties[46]", "counties[47]", "counties[48]", "counties[49]", "counties[50]", "counties[51]", "counties[52]", "counties[53]", "counties[54]", "counties[55]", "counties[56]", "counties[57]", "counties[58]", "counties[59]", "counties[60]", "counties[61]", "counties[62]", "counties[63]", "counties[64]", "counties[65]", "counties[66]", "counties[67]", "counties[68]", "counties[69]", "counties[70]", "counties[71]", "counties[72]", "counties[73]", "counties[74]", "counties[75]", "counties[76]", "counties[77]", "counties[78]", "counties[79]", "counties[80]", "counties[81]"};
void _eval_query();
void _init_storage();
void _init_world();
void _garbage_collection();
void _print_answer();
const int _TOT_LOOP = 100000;
const int _BURN_IN = 99900;
int _tot_round = -99900;
const mat __fixed_county_map = loadRealMatrix("data_processed/county_map.txt");
const mat __fixed_region_pop = loadRealMatrix("data_processed/region_pops.txt");
const mat __fixed_observations = loadRealMatrix("data_processed/obs.txt");
const mat __fixed_covariates1 = loadRealMatrix("data_processed/covariates1.txt");
const mat __fixed_covariates2 = loadRealMatrix("data_processed/covariates2.txt");
const mat __fixed_y = loadRealMatrix("data_processed/y.txt");
double __fixed_sigmoid(double);
class _Var_beta1: public BayesVar<double> {
public:
  _Var_beta1();
  string getname();
  double& getval();
  double& getcache();
  void clear();
  double getlikeli();
  double getcachelikeli();
  void sample();
  void sample_cache();
  void active_edge();
  void remove_edge();
  void mcmc_resample();
};
_Var_beta1* _mem_beta1;
class _Var_beta2: public BayesVar<double> {
public:
  _Var_beta2();
  string getname();
  double& getval();
  double& getcache();
  void clear();
  double getlikeli();
  double getcachelikeli();
  void sample();
  void sample_cache();
  void active_edge();
  void remove_edge();
  void mcmc_resample();
};
_Var_beta2* _mem_beta2;
class _Var_logit: public BayesVar<double> {
public:
  int c;
  int t;
  _Var_logit(int,int);
  string getname();
  double& getval();
  double& getcache();
  void clear();
  double getlikeli();
  double getcachelikeli();
  void sample();
  void sample_cache();
  void active_edge();
  void remove_edge();
  void mcmc_resample();
};
DynamicTable<_Var_logit*,2> _mem_logit;
class _Var_region_rate: public BayesVar<double> {
public:
  int r;
  int t;
  _Var_region_rate(int,int);
  string getname();
  double& getval();
  double& getcache();
  void clear();
  double getlikeli();
  double getcachelikeli();
  void sample();
  void sample_cache();
  void active_edge();
  void remove_edge();
  void mcmc_resample();
};
DynamicTable<_Var_region_rate*,2> _mem_region_rate;
Gaussian Gaussian140192602248224;
Gaussian Gaussian140192602248560;
Gaussian Gaussian140192602252544;
Gaussian Gaussian140192602259120;
Hist<double> _answer_0 = Hist<double>(false, 20);
Hist<double> _answer_1 = Hist<double>(false, 20);
DynamicTable<Hist<double>*,2> _answer_2;
void sample();

void _eval_query()
{
  _tot_round++;
  if (_tot_round<=0)
    return ;
  _answer_0.add(_mem_beta1->getval(),1);
  _answer_1.add(_mem_beta2->getval(),1);
  for (int c = 0;c<82;c++)
  for (int t = 0;t<103;t++)
  _answer_2[c][t]->add(_mem_logit[c][t]->getval(),1);


}
void _init_storage()
{
  _mem_beta1=new _Var_beta1();
  _mem_beta2=new _Var_beta2();
  _mem_logit.resize(0,82);
  _mem_logit.resize(1,103);
  for (int c = 0;c<82;c++)
  {
    for (int t = 0;t<103;t++)
    {
      _mem_logit[c][t]=new _Var_logit(c, t);
    }

  }

  _mem_region_rate.resize(0,9);
  _mem_region_rate.resize(1,103);
  for (int r = 0;r<9;r++)
  {
    for (int t = 0;t<103;t++)
    {
      _mem_region_rate[r][t]=new _Var_region_rate(r, t);
    }

  }

  Gaussian140192602248224.init(0.25000000,0.10000000);
  Gaussian140192602248560.init(0.50000000,0.10000000);
  _answer_2.resize(0,82);
  _answer_2.resize(1,103);
  for (int c = 0;c<82;c++)
  {
    for (int t = 0;t<103;t++)
    {
      _answer_2[c][t]=new Hist<double>(false, 20);
    }

  }

}
void _init_world()
{
  for (int r = 0;r<9;r++)
  for (int t = 0;t<103;t++)
  _util_set_evidence<double>(_mem_region_rate[r][t],__fixed_observations(t,r));


}
void _garbage_collection()
{
  _free_obj(_mem_beta1);
  _free_obj(_mem_beta2);
  _free_obj(_mem_logit);
  _free_obj(_mem_region_rate);
}
void _print_answer()
{
  _answer_0.print("beta1");
  _answer_1.print("beta2");
  char buffer2[256];
  for (int c = 0;c<82;c++)
  for (int t = 0;t<103;t++)
  {
    sprintf(buffer2,"logit(County[%d], Week[%d])\n",c,t);
    _answer_2[c][t]->print(buffer2);
  }


}
double __fixed_sigmoid(double value)
{
  return 1.00000000/(1.00000000+exp(-1.00000000*value));
}
_Var_beta1::_Var_beta1()
{}
string _Var_beta1::getname()
{
  return "beta1";
}
double& _Var_beta1::getval()
{
  return getval_arg(this);
}
double& _Var_beta1::getcache()
{
  return getcache_arg(this);
}
void _Var_beta1::clear()
{
  return clear_arg(this);
}
double _Var_beta1::getlikeli()
{
  return Gaussian140192602248224.loglikeli(val);
}
double _Var_beta1::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian140192602248224.loglikeli(_t_val);
}
void _Var_beta1::sample()
{
  val=Gaussian140192602248224.gen();
}
void _Var_beta1::sample_cache()
{
  cache_val=Gaussian140192602248224.gen();
}
void _Var_beta1::active_edge()
{}
void _Var_beta1::remove_edge()
{}
void _Var_beta1::mcmc_resample()
{
  mh_parent_resample_arg(this);
}
_Var_beta2::_Var_beta2()
{}
string _Var_beta2::getname()
{
  return "beta2";
}
double& _Var_beta2::getval()
{
  return getval_arg(this);
}
double& _Var_beta2::getcache()
{
  return getcache_arg(this);
}
void _Var_beta2::clear()
{
  return clear_arg(this);
}
double _Var_beta2::getlikeli()
{
  return Gaussian140192602248560.loglikeli(val);
}
double _Var_beta2::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian140192602248560.loglikeli(_t_val);
}
void _Var_beta2::sample()
{
  val=Gaussian140192602248560.gen();
}
void _Var_beta2::sample_cache()
{
  cache_val=Gaussian140192602248560.gen();
}
void _Var_beta2::active_edge()
{}
void _Var_beta2::remove_edge()
{}
void _Var_beta2::mcmc_resample()
{
  mh_parent_resample_arg(this);
}
_Var_logit::_Var_logit(int _c, int _t):c(_c),t(_t)
{}
string _Var_logit::getname()
{
  return "logit";
}
double& _Var_logit::getval()
{
  return getval_arg(this);
}
double& _Var_logit::getcache()
{
  return getcache_arg(this);
}
void _Var_logit::clear()
{
  return clear_arg(this);
}
double _Var_logit::getlikeli()
{
  return Gaussian140192602252544.init(_mem_beta1->getval()*__fixed_covariates1(c,t)+_mem_beta2->getval()*__fixed_covariates2(c,t)+__fixed_y(c,t),0.01000000),Gaussian140192602252544.loglikeli(val);
}
double _Var_logit::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian140192602252544.init(_mem_beta1->getcache()*__fixed_covariates1(c,t)+_mem_beta2->getcache()*__fixed_covariates2(c,t)+__fixed_y(c,t),0.01000000),Gaussian140192602252544.loglikeli(_t_val);
}
void _Var_logit::sample()
{
  val=(Gaussian140192602252544.init(_mem_beta1->getval()*__fixed_covariates1(c,t)+_mem_beta2->getval()*__fixed_covariates2(c,t)+__fixed_y(c,t),0.01000000),Gaussian140192602252544.gen());
}
void _Var_logit::sample_cache()
{
  cache_val=(Gaussian140192602252544.init(_mem_beta1->getcache()*__fixed_covariates1(c,t)+_mem_beta2->getcache()*__fixed_covariates2(c,t)+__fixed_y(c,t),0.01000000),Gaussian140192602252544.gen());
}
void _Var_logit::active_edge()
{
  _mem_beta1->add_contig(this);
  _mem_beta2->add_contig(this);
  _mem_beta1->add_child(this);
  _mem_beta2->add_child(this);
}
void _Var_logit::remove_edge()
{
  _mem_beta1->erase_contig(this);
  _mem_beta2->erase_contig(this);
  _mem_beta1->erase_child(this);
  _mem_beta2->erase_child(this);
}
void _Var_logit::mcmc_resample()
{
  mh_parent_resample_arg(this);
}
_Var_region_rate::_Var_region_rate(int _r, int _t):r(_r),t(_t)
{}
string _Var_region_rate::getname()
{
  return "region_rate";
}
double& _Var_region_rate::getval()
{
  return getval_arg(this);
}
double& _Var_region_rate::getcache()
{
  return getcache_arg(this);
}
void _Var_region_rate::clear()
{
  return clear_arg(this);
}
double _Var_region_rate::getlikeli()
{
  return Gaussian140192602259120.init(accu(__fixed_county_map[r]*vstack({__fixed_sigmoid(_mem_logit[0][t]->getval()), __fixed_sigmoid(_mem_logit[1][t]->getval()), __fixed_sigmoid(_mem_logit[2][t]->getval()), __fixed_sigmoid(_mem_logit[3][t]->getval()), __fixed_sigmoid(_mem_logit[4][t]->getval()), __fixed_sigmoid(_mem_logit[5][t]->getval()), __fixed_sigmoid(_mem_logit[6][t]->getval()), __fixed_sigmoid(_mem_logit[7][t]->getval()), __fixed_sigmoid(_mem_logit[8][t]->getval()), __fixed_sigmoid(_mem_logit[9][t]->getval()), __fixed_sigmoid(_mem_logit[10][t]->getval()), __fixed_sigmoid(_mem_logit[11][t]->getval()), __fixed_sigmoid(_mem_logit[12][t]->getval()), __fixed_sigmoid(_mem_logit[13][t]->getval()), __fixed_sigmoid(_mem_logit[14][t]->getval()), __fixed_sigmoid(_mem_logit[15][t]->getval()), __fixed_sigmoid(_mem_logit[16][t]->getval()), __fixed_sigmoid(_mem_logit[17][t]->getval()), __fixed_sigmoid(_mem_logit[18][t]->getval()), __fixed_sigmoid(_mem_logit[19][t]->getval()), __fixed_sigmoid(_mem_logit[20][t]->getval()), __fixed_sigmoid(_mem_logit[21][t]->getval()), __fixed_sigmoid(_mem_logit[22][t]->getval()), __fixed_sigmoid(_mem_logit[23][t]->getval()), __fixed_sigmoid(_mem_logit[24][t]->getval()), __fixed_sigmoid(_mem_logit[25][t]->getval()), __fixed_sigmoid(_mem_logit[26][t]->getval()), __fixed_sigmoid(_mem_logit[27][t]->getval()), __fixed_sigmoid(_mem_logit[28][t]->getval()), __fixed_sigmoid(_mem_logit[29][t]->getval()), __fixed_sigmoid(_mem_logit[30][t]->getval()), __fixed_sigmoid(_mem_logit[31][t]->getval()), __fixed_sigmoid(_mem_logit[32][t]->getval()), __fixed_sigmoid(_mem_logit[33][t]->getval()), __fixed_sigmoid(_mem_logit[34][t]->getval()), __fixed_sigmoid(_mem_logit[35][t]->getval()), __fixed_sigmoid(_mem_logit[36][t]->getval()), __fixed_sigmoid(_mem_logit[37][t]->getval()), __fixed_sigmoid(_mem_logit[38][t]->getval()), __fixed_sigmoid(_mem_logit[39][t]->getval()), __fixed_sigmoid(_mem_logit[40][t]->getval()), __fixed_sigmoid(_mem_logit[41][t]->getval()), __fixed_sigmoid(_mem_logit[42][t]->getval()), __fixed_sigmoid(_mem_logit[43][t]->getval()), __fixed_sigmoid(_mem_logit[44][t]->getval()), __fixed_sigmoid(_mem_logit[45][t]->getval()), __fixed_sigmoid(_mem_logit[46][t]->getval()), __fixed_sigmoid(_mem_logit[47][t]->getval()), __fixed_sigmoid(_mem_logit[48][t]->getval()), __fixed_sigmoid(_mem_logit[49][t]->getval()), __fixed_sigmoid(_mem_logit[50][t]->getval()), __fixed_sigmoid(_mem_logit[51][t]->getval()), __fixed_sigmoid(_mem_logit[52][t]->getval()), __fixed_sigmoid(_mem_logit[53][t]->getval()), __fixed_sigmoid(_mem_logit[54][t]->getval()), __fixed_sigmoid(_mem_logit[55][t]->getval()), __fixed_sigmoid(_mem_logit[56][t]->getval()), __fixed_sigmoid(_mem_logit[57][t]->getval()), __fixed_sigmoid(_mem_logit[58][t]->getval()), __fixed_sigmoid(_mem_logit[59][t]->getval()), __fixed_sigmoid(_mem_logit[60][t]->getval()), __fixed_sigmoid(_mem_logit[61][t]->getval()), __fixed_sigmoid(_mem_logit[62][t]->getval()), __fixed_sigmoid(_mem_logit[63][t]->getval()), __fixed_sigmoid(_mem_logit[64][t]->getval()), __fixed_sigmoid(_mem_logit[65][t]->getval()), __fixed_sigmoid(_mem_logit[66][t]->getval()), __fixed_sigmoid(_mem_logit[67][t]->getval()), __fixed_sigmoid(_mem_logit[68][t]->getval()), __fixed_sigmoid(_mem_logit[69][t]->getval()), __fixed_sigmoid(_mem_logit[70][t]->getval()), __fixed_sigmoid(_mem_logit[71][t]->getval()), __fixed_sigmoid(_mem_logit[72][t]->getval()), __fixed_sigmoid(_mem_logit[73][t]->getval()), __fixed_sigmoid(_mem_logit[74][t]->getval()), __fixed_sigmoid(_mem_logit[75][t]->getval()), __fixed_sigmoid(_mem_logit[76][t]->getval()), __fixed_sigmoid(_mem_logit[77][t]->getval()), __fixed_sigmoid(_mem_logit[78][t]->getval()), __fixed_sigmoid(_mem_logit[79][t]->getval()), __fixed_sigmoid(_mem_logit[80][t]->getval()), __fixed_sigmoid(_mem_logit[81][t]->getval())}))/__fixed_region_pop[r],0.05000000),Gaussian140192602259120.loglikeli(val);
}
double _Var_region_rate::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian140192602259120.init(accu(__fixed_county_map[r]*vstack({__fixed_sigmoid(_mem_logit[0][t]->getcache()), __fixed_sigmoid(_mem_logit[1][t]->getcache()), __fixed_sigmoid(_mem_logit[2][t]->getcache()), __fixed_sigmoid(_mem_logit[3][t]->getcache()), __fixed_sigmoid(_mem_logit[4][t]->getcache()), __fixed_sigmoid(_mem_logit[5][t]->getcache()), __fixed_sigmoid(_mem_logit[6][t]->getcache()), __fixed_sigmoid(_mem_logit[7][t]->getcache()), __fixed_sigmoid(_mem_logit[8][t]->getcache()), __fixed_sigmoid(_mem_logit[9][t]->getcache()), __fixed_sigmoid(_mem_logit[10][t]->getcache()), __fixed_sigmoid(_mem_logit[11][t]->getcache()), __fixed_sigmoid(_mem_logit[12][t]->getcache()), __fixed_sigmoid(_mem_logit[13][t]->getcache()), __fixed_sigmoid(_mem_logit[14][t]->getcache()), __fixed_sigmoid(_mem_logit[15][t]->getcache()), __fixed_sigmoid(_mem_logit[16][t]->getcache()), __fixed_sigmoid(_mem_logit[17][t]->getcache()), __fixed_sigmoid(_mem_logit[18][t]->getcache()), __fixed_sigmoid(_mem_logit[19][t]->getcache()), __fixed_sigmoid(_mem_logit[20][t]->getcache()), __fixed_sigmoid(_mem_logit[21][t]->getcache()), __fixed_sigmoid(_mem_logit[22][t]->getcache()), __fixed_sigmoid(_mem_logit[23][t]->getcache()), __fixed_sigmoid(_mem_logit[24][t]->getcache()), __fixed_sigmoid(_mem_logit[25][t]->getcache()), __fixed_sigmoid(_mem_logit[26][t]->getcache()), __fixed_sigmoid(_mem_logit[27][t]->getcache()), __fixed_sigmoid(_mem_logit[28][t]->getcache()), __fixed_sigmoid(_mem_logit[29][t]->getcache()), __fixed_sigmoid(_mem_logit[30][t]->getcache()), __fixed_sigmoid(_mem_logit[31][t]->getcache()), __fixed_sigmoid(_mem_logit[32][t]->getcache()), __fixed_sigmoid(_mem_logit[33][t]->getcache()), __fixed_sigmoid(_mem_logit[34][t]->getcache()), __fixed_sigmoid(_mem_logit[35][t]->getcache()), __fixed_sigmoid(_mem_logit[36][t]->getcache()), __fixed_sigmoid(_mem_logit[37][t]->getcache()), __fixed_sigmoid(_mem_logit[38][t]->getcache()), __fixed_sigmoid(_mem_logit[39][t]->getcache()), __fixed_sigmoid(_mem_logit[40][t]->getcache()), __fixed_sigmoid(_mem_logit[41][t]->getcache()), __fixed_sigmoid(_mem_logit[42][t]->getcache()), __fixed_sigmoid(_mem_logit[43][t]->getcache()), __fixed_sigmoid(_mem_logit[44][t]->getcache()), __fixed_sigmoid(_mem_logit[45][t]->getcache()), __fixed_sigmoid(_mem_logit[46][t]->getcache()), __fixed_sigmoid(_mem_logit[47][t]->getcache()), __fixed_sigmoid(_mem_logit[48][t]->getcache()), __fixed_sigmoid(_mem_logit[49][t]->getcache()), __fixed_sigmoid(_mem_logit[50][t]->getcache()), __fixed_sigmoid(_mem_logit[51][t]->getcache()), __fixed_sigmoid(_mem_logit[52][t]->getcache()), __fixed_sigmoid(_mem_logit[53][t]->getcache()), __fixed_sigmoid(_mem_logit[54][t]->getcache()), __fixed_sigmoid(_mem_logit[55][t]->getcache()), __fixed_sigmoid(_mem_logit[56][t]->getcache()), __fixed_sigmoid(_mem_logit[57][t]->getcache()), __fixed_sigmoid(_mem_logit[58][t]->getcache()), __fixed_sigmoid(_mem_logit[59][t]->getcache()), __fixed_sigmoid(_mem_logit[60][t]->getcache()), __fixed_sigmoid(_mem_logit[61][t]->getcache()), __fixed_sigmoid(_mem_logit[62][t]->getcache()), __fixed_sigmoid(_mem_logit[63][t]->getcache()), __fixed_sigmoid(_mem_logit[64][t]->getcache()), __fixed_sigmoid(_mem_logit[65][t]->getcache()), __fixed_sigmoid(_mem_logit[66][t]->getcache()), __fixed_sigmoid(_mem_logit[67][t]->getcache()), __fixed_sigmoid(_mem_logit[68][t]->getcache()), __fixed_sigmoid(_mem_logit[69][t]->getcache()), __fixed_sigmoid(_mem_logit[70][t]->getcache()), __fixed_sigmoid(_mem_logit[71][t]->getcache()), __fixed_sigmoid(_mem_logit[72][t]->getcache()), __fixed_sigmoid(_mem_logit[73][t]->getcache()), __fixed_sigmoid(_mem_logit[74][t]->getcache()), __fixed_sigmoid(_mem_logit[75][t]->getcache()), __fixed_sigmoid(_mem_logit[76][t]->getcache()), __fixed_sigmoid(_mem_logit[77][t]->getcache()), __fixed_sigmoid(_mem_logit[78][t]->getcache()), __fixed_sigmoid(_mem_logit[79][t]->getcache()), __fixed_sigmoid(_mem_logit[80][t]->getcache()), __fixed_sigmoid(_mem_logit[81][t]->getcache())}))/__fixed_region_pop[r],0.05000000),Gaussian140192602259120.loglikeli(_t_val);
}
void _Var_region_rate::sample()
{
  val=(Gaussian140192602259120.init(accu(__fixed_county_map[r]*vstack({__fixed_sigmoid(_mem_logit[0][t]->getval()), __fixed_sigmoid(_mem_logit[1][t]->getval()), __fixed_sigmoid(_mem_logit[2][t]->getval()), __fixed_sigmoid(_mem_logit[3][t]->getval()), __fixed_sigmoid(_mem_logit[4][t]->getval()), __fixed_sigmoid(_mem_logit[5][t]->getval()), __fixed_sigmoid(_mem_logit[6][t]->getval()), __fixed_sigmoid(_mem_logit[7][t]->getval()), __fixed_sigmoid(_mem_logit[8][t]->getval()), __fixed_sigmoid(_mem_logit[9][t]->getval()), __fixed_sigmoid(_mem_logit[10][t]->getval()), __fixed_sigmoid(_mem_logit[11][t]->getval()), __fixed_sigmoid(_mem_logit[12][t]->getval()), __fixed_sigmoid(_mem_logit[13][t]->getval()), __fixed_sigmoid(_mem_logit[14][t]->getval()), __fixed_sigmoid(_mem_logit[15][t]->getval()), __fixed_sigmoid(_mem_logit[16][t]->getval()), __fixed_sigmoid(_mem_logit[17][t]->getval()), __fixed_sigmoid(_mem_logit[18][t]->getval()), __fixed_sigmoid(_mem_logit[19][t]->getval()), __fixed_sigmoid(_mem_logit[20][t]->getval()), __fixed_sigmoid(_mem_logit[21][t]->getval()), __fixed_sigmoid(_mem_logit[22][t]->getval()), __fixed_sigmoid(_mem_logit[23][t]->getval()), __fixed_sigmoid(_mem_logit[24][t]->getval()), __fixed_sigmoid(_mem_logit[25][t]->getval()), __fixed_sigmoid(_mem_logit[26][t]->getval()), __fixed_sigmoid(_mem_logit[27][t]->getval()), __fixed_sigmoid(_mem_logit[28][t]->getval()), __fixed_sigmoid(_mem_logit[29][t]->getval()), __fixed_sigmoid(_mem_logit[30][t]->getval()), __fixed_sigmoid(_mem_logit[31][t]->getval()), __fixed_sigmoid(_mem_logit[32][t]->getval()), __fixed_sigmoid(_mem_logit[33][t]->getval()), __fixed_sigmoid(_mem_logit[34][t]->getval()), __fixed_sigmoid(_mem_logit[35][t]->getval()), __fixed_sigmoid(_mem_logit[36][t]->getval()), __fixed_sigmoid(_mem_logit[37][t]->getval()), __fixed_sigmoid(_mem_logit[38][t]->getval()), __fixed_sigmoid(_mem_logit[39][t]->getval()), __fixed_sigmoid(_mem_logit[40][t]->getval()), __fixed_sigmoid(_mem_logit[41][t]->getval()), __fixed_sigmoid(_mem_logit[42][t]->getval()), __fixed_sigmoid(_mem_logit[43][t]->getval()), __fixed_sigmoid(_mem_logit[44][t]->getval()), __fixed_sigmoid(_mem_logit[45][t]->getval()), __fixed_sigmoid(_mem_logit[46][t]->getval()), __fixed_sigmoid(_mem_logit[47][t]->getval()), __fixed_sigmoid(_mem_logit[48][t]->getval()), __fixed_sigmoid(_mem_logit[49][t]->getval()), __fixed_sigmoid(_mem_logit[50][t]->getval()), __fixed_sigmoid(_mem_logit[51][t]->getval()), __fixed_sigmoid(_mem_logit[52][t]->getval()), __fixed_sigmoid(_mem_logit[53][t]->getval()), __fixed_sigmoid(_mem_logit[54][t]->getval()), __fixed_sigmoid(_mem_logit[55][t]->getval()), __fixed_sigmoid(_mem_logit[56][t]->getval()), __fixed_sigmoid(_mem_logit[57][t]->getval()), __fixed_sigmoid(_mem_logit[58][t]->getval()), __fixed_sigmoid(_mem_logit[59][t]->getval()), __fixed_sigmoid(_mem_logit[60][t]->getval()), __fixed_sigmoid(_mem_logit[61][t]->getval()), __fixed_sigmoid(_mem_logit[62][t]->getval()), __fixed_sigmoid(_mem_logit[63][t]->getval()), __fixed_sigmoid(_mem_logit[64][t]->getval()), __fixed_sigmoid(_mem_logit[65][t]->getval()), __fixed_sigmoid(_mem_logit[66][t]->getval()), __fixed_sigmoid(_mem_logit[67][t]->getval()), __fixed_sigmoid(_mem_logit[68][t]->getval()), __fixed_sigmoid(_mem_logit[69][t]->getval()), __fixed_sigmoid(_mem_logit[70][t]->getval()), __fixed_sigmoid(_mem_logit[71][t]->getval()), __fixed_sigmoid(_mem_logit[72][t]->getval()), __fixed_sigmoid(_mem_logit[73][t]->getval()), __fixed_sigmoid(_mem_logit[74][t]->getval()), __fixed_sigmoid(_mem_logit[75][t]->getval()), __fixed_sigmoid(_mem_logit[76][t]->getval()), __fixed_sigmoid(_mem_logit[77][t]->getval()), __fixed_sigmoid(_mem_logit[78][t]->getval()), __fixed_sigmoid(_mem_logit[79][t]->getval()), __fixed_sigmoid(_mem_logit[80][t]->getval()), __fixed_sigmoid(_mem_logit[81][t]->getval())}))/__fixed_region_pop[r],0.05000000),Gaussian140192602259120.gen());
}
void _Var_region_rate::sample_cache()
{
  cache_val=(Gaussian140192602259120.init(accu(__fixed_county_map[r]*vstack({__fixed_sigmoid(_mem_logit[0][t]->getcache()), __fixed_sigmoid(_mem_logit[1][t]->getcache()), __fixed_sigmoid(_mem_logit[2][t]->getcache()), __fixed_sigmoid(_mem_logit[3][t]->getcache()), __fixed_sigmoid(_mem_logit[4][t]->getcache()), __fixed_sigmoid(_mem_logit[5][t]->getcache()), __fixed_sigmoid(_mem_logit[6][t]->getcache()), __fixed_sigmoid(_mem_logit[7][t]->getcache()), __fixed_sigmoid(_mem_logit[8][t]->getcache()), __fixed_sigmoid(_mem_logit[9][t]->getcache()), __fixed_sigmoid(_mem_logit[10][t]->getcache()), __fixed_sigmoid(_mem_logit[11][t]->getcache()), __fixed_sigmoid(_mem_logit[12][t]->getcache()), __fixed_sigmoid(_mem_logit[13][t]->getcache()), __fixed_sigmoid(_mem_logit[14][t]->getcache()), __fixed_sigmoid(_mem_logit[15][t]->getcache()), __fixed_sigmoid(_mem_logit[16][t]->getcache()), __fixed_sigmoid(_mem_logit[17][t]->getcache()), __fixed_sigmoid(_mem_logit[18][t]->getcache()), __fixed_sigmoid(_mem_logit[19][t]->getcache()), __fixed_sigmoid(_mem_logit[20][t]->getcache()), __fixed_sigmoid(_mem_logit[21][t]->getcache()), __fixed_sigmoid(_mem_logit[22][t]->getcache()), __fixed_sigmoid(_mem_logit[23][t]->getcache()), __fixed_sigmoid(_mem_logit[24][t]->getcache()), __fixed_sigmoid(_mem_logit[25][t]->getcache()), __fixed_sigmoid(_mem_logit[26][t]->getcache()), __fixed_sigmoid(_mem_logit[27][t]->getcache()), __fixed_sigmoid(_mem_logit[28][t]->getcache()), __fixed_sigmoid(_mem_logit[29][t]->getcache()), __fixed_sigmoid(_mem_logit[30][t]->getcache()), __fixed_sigmoid(_mem_logit[31][t]->getcache()), __fixed_sigmoid(_mem_logit[32][t]->getcache()), __fixed_sigmoid(_mem_logit[33][t]->getcache()), __fixed_sigmoid(_mem_logit[34][t]->getcache()), __fixed_sigmoid(_mem_logit[35][t]->getcache()), __fixed_sigmoid(_mem_logit[36][t]->getcache()), __fixed_sigmoid(_mem_logit[37][t]->getcache()), __fixed_sigmoid(_mem_logit[38][t]->getcache()), __fixed_sigmoid(_mem_logit[39][t]->getcache()), __fixed_sigmoid(_mem_logit[40][t]->getcache()), __fixed_sigmoid(_mem_logit[41][t]->getcache()), __fixed_sigmoid(_mem_logit[42][t]->getcache()), __fixed_sigmoid(_mem_logit[43][t]->getcache()), __fixed_sigmoid(_mem_logit[44][t]->getcache()), __fixed_sigmoid(_mem_logit[45][t]->getcache()), __fixed_sigmoid(_mem_logit[46][t]->getcache()), __fixed_sigmoid(_mem_logit[47][t]->getcache()), __fixed_sigmoid(_mem_logit[48][t]->getcache()), __fixed_sigmoid(_mem_logit[49][t]->getcache()), __fixed_sigmoid(_mem_logit[50][t]->getcache()), __fixed_sigmoid(_mem_logit[51][t]->getcache()), __fixed_sigmoid(_mem_logit[52][t]->getcache()), __fixed_sigmoid(_mem_logit[53][t]->getcache()), __fixed_sigmoid(_mem_logit[54][t]->getcache()), __fixed_sigmoid(_mem_logit[55][t]->getcache()), __fixed_sigmoid(_mem_logit[56][t]->getcache()), __fixed_sigmoid(_mem_logit[57][t]->getcache()), __fixed_sigmoid(_mem_logit[58][t]->getcache()), __fixed_sigmoid(_mem_logit[59][t]->getcache()), __fixed_sigmoid(_mem_logit[60][t]->getcache()), __fixed_sigmoid(_mem_logit[61][t]->getcache()), __fixed_sigmoid(_mem_logit[62][t]->getcache()), __fixed_sigmoid(_mem_logit[63][t]->getcache()), __fixed_sigmoid(_mem_logit[64][t]->getcache()), __fixed_sigmoid(_mem_logit[65][t]->getcache()), __fixed_sigmoid(_mem_logit[66][t]->getcache()), __fixed_sigmoid(_mem_logit[67][t]->getcache()), __fixed_sigmoid(_mem_logit[68][t]->getcache()), __fixed_sigmoid(_mem_logit[69][t]->getcache()), __fixed_sigmoid(_mem_logit[70][t]->getcache()), __fixed_sigmoid(_mem_logit[71][t]->getcache()), __fixed_sigmoid(_mem_logit[72][t]->getcache()), __fixed_sigmoid(_mem_logit[73][t]->getcache()), __fixed_sigmoid(_mem_logit[74][t]->getcache()), __fixed_sigmoid(_mem_logit[75][t]->getcache()), __fixed_sigmoid(_mem_logit[76][t]->getcache()), __fixed_sigmoid(_mem_logit[77][t]->getcache()), __fixed_sigmoid(_mem_logit[78][t]->getcache()), __fixed_sigmoid(_mem_logit[79][t]->getcache()), __fixed_sigmoid(_mem_logit[80][t]->getcache()), __fixed_sigmoid(_mem_logit[81][t]->getcache())}))/__fixed_region_pop[r],0.05000000),Gaussian140192602259120.gen());
}
void _Var_region_rate::active_edge()
{
  _mem_logit[0][t]->add_contig(this);
  _mem_logit[10][t]->add_contig(this);
  _mem_logit[11][t]->add_contig(this);
  _mem_logit[12][t]->add_contig(this);
  _mem_logit[13][t]->add_contig(this);
  _mem_logit[14][t]->add_contig(this);
  _mem_logit[15][t]->add_contig(this);
  _mem_logit[16][t]->add_contig(this);
  _mem_logit[17][t]->add_contig(this);
  _mem_logit[18][t]->add_contig(this);
  _mem_logit[19][t]->add_contig(this);
  _mem_logit[1][t]->add_contig(this);
  _mem_logit[20][t]->add_contig(this);
  _mem_logit[21][t]->add_contig(this);
  _mem_logit[22][t]->add_contig(this);
  _mem_logit[23][t]->add_contig(this);
  _mem_logit[24][t]->add_contig(this);
  _mem_logit[25][t]->add_contig(this);
  _mem_logit[26][t]->add_contig(this);
  _mem_logit[27][t]->add_contig(this);
  _mem_logit[28][t]->add_contig(this);
  _mem_logit[29][t]->add_contig(this);
  _mem_logit[2][t]->add_contig(this);
  _mem_logit[30][t]->add_contig(this);
  _mem_logit[31][t]->add_contig(this);
  _mem_logit[32][t]->add_contig(this);
  _mem_logit[33][t]->add_contig(this);
  _mem_logit[34][t]->add_contig(this);
  _mem_logit[35][t]->add_contig(this);
  _mem_logit[36][t]->add_contig(this);
  _mem_logit[37][t]->add_contig(this);
  _mem_logit[38][t]->add_contig(this);
  _mem_logit[39][t]->add_contig(this);
  _mem_logit[3][t]->add_contig(this);
  _mem_logit[40][t]->add_contig(this);
  _mem_logit[41][t]->add_contig(this);
  _mem_logit[42][t]->add_contig(this);
  _mem_logit[43][t]->add_contig(this);
  _mem_logit[44][t]->add_contig(this);
  _mem_logit[45][t]->add_contig(this);
  _mem_logit[46][t]->add_contig(this);
  _mem_logit[47][t]->add_contig(this);
  _mem_logit[48][t]->add_contig(this);
  _mem_logit[49][t]->add_contig(this);
  _mem_logit[4][t]->add_contig(this);
  _mem_logit[50][t]->add_contig(this);
  _mem_logit[51][t]->add_contig(this);
  _mem_logit[52][t]->add_contig(this);
  _mem_logit[53][t]->add_contig(this);
  _mem_logit[54][t]->add_contig(this);
  _mem_logit[55][t]->add_contig(this);
  _mem_logit[56][t]->add_contig(this);
  _mem_logit[57][t]->add_contig(this);
  _mem_logit[58][t]->add_contig(this);
  _mem_logit[59][t]->add_contig(this);
  _mem_logit[5][t]->add_contig(this);
  _mem_logit[60][t]->add_contig(this);
  _mem_logit[61][t]->add_contig(this);
  _mem_logit[62][t]->add_contig(this);
  _mem_logit[63][t]->add_contig(this);
  _mem_logit[64][t]->add_contig(this);
  _mem_logit[65][t]->add_contig(this);
  _mem_logit[66][t]->add_contig(this);
  _mem_logit[67][t]->add_contig(this);
  _mem_logit[68][t]->add_contig(this);
  _mem_logit[69][t]->add_contig(this);
  _mem_logit[6][t]->add_contig(this);
  _mem_logit[70][t]->add_contig(this);
  _mem_logit[71][t]->add_contig(this);
  _mem_logit[72][t]->add_contig(this);
  _mem_logit[73][t]->add_contig(this);
  _mem_logit[74][t]->add_contig(this);
  _mem_logit[75][t]->add_contig(this);
  _mem_logit[76][t]->add_contig(this);
  _mem_logit[77][t]->add_contig(this);
  _mem_logit[78][t]->add_contig(this);
  _mem_logit[79][t]->add_contig(this);
  _mem_logit[7][t]->add_contig(this);
  _mem_logit[80][t]->add_contig(this);
  _mem_logit[81][t]->add_contig(this);
  _mem_logit[8][t]->add_contig(this);
  _mem_logit[9][t]->add_contig(this);
  _mem_logit[0][t]->add_child(this);
  _mem_logit[10][t]->add_child(this);
  _mem_logit[11][t]->add_child(this);
  _mem_logit[12][t]->add_child(this);
  _mem_logit[13][t]->add_child(this);
  _mem_logit[14][t]->add_child(this);
  _mem_logit[15][t]->add_child(this);
  _mem_logit[16][t]->add_child(this);
  _mem_logit[17][t]->add_child(this);
  _mem_logit[18][t]->add_child(this);
  _mem_logit[19][t]->add_child(this);
  _mem_logit[1][t]->add_child(this);
  _mem_logit[20][t]->add_child(this);
  _mem_logit[21][t]->add_child(this);
  _mem_logit[22][t]->add_child(this);
  _mem_logit[23][t]->add_child(this);
  _mem_logit[24][t]->add_child(this);
  _mem_logit[25][t]->add_child(this);
  _mem_logit[26][t]->add_child(this);
  _mem_logit[27][t]->add_child(this);
  _mem_logit[28][t]->add_child(this);
  _mem_logit[29][t]->add_child(this);
  _mem_logit[2][t]->add_child(this);
  _mem_logit[30][t]->add_child(this);
  _mem_logit[31][t]->add_child(this);
  _mem_logit[32][t]->add_child(this);
  _mem_logit[33][t]->add_child(this);
  _mem_logit[34][t]->add_child(this);
  _mem_logit[35][t]->add_child(this);
  _mem_logit[36][t]->add_child(this);
  _mem_logit[37][t]->add_child(this);
  _mem_logit[38][t]->add_child(this);
  _mem_logit[39][t]->add_child(this);
  _mem_logit[3][t]->add_child(this);
  _mem_logit[40][t]->add_child(this);
  _mem_logit[41][t]->add_child(this);
  _mem_logit[42][t]->add_child(this);
  _mem_logit[43][t]->add_child(this);
  _mem_logit[44][t]->add_child(this);
  _mem_logit[45][t]->add_child(this);
  _mem_logit[46][t]->add_child(this);
  _mem_logit[47][t]->add_child(this);
  _mem_logit[48][t]->add_child(this);
  _mem_logit[49][t]->add_child(this);
  _mem_logit[4][t]->add_child(this);
  _mem_logit[50][t]->add_child(this);
  _mem_logit[51][t]->add_child(this);
  _mem_logit[52][t]->add_child(this);
  _mem_logit[53][t]->add_child(this);
  _mem_logit[54][t]->add_child(this);
  _mem_logit[55][t]->add_child(this);
  _mem_logit[56][t]->add_child(this);
  _mem_logit[57][t]->add_child(this);
  _mem_logit[58][t]->add_child(this);
  _mem_logit[59][t]->add_child(this);
  _mem_logit[5][t]->add_child(this);
  _mem_logit[60][t]->add_child(this);
  _mem_logit[61][t]->add_child(this);
  _mem_logit[62][t]->add_child(this);
  _mem_logit[63][t]->add_child(this);
  _mem_logit[64][t]->add_child(this);
  _mem_logit[65][t]->add_child(this);
  _mem_logit[66][t]->add_child(this);
  _mem_logit[67][t]->add_child(this);
  _mem_logit[68][t]->add_child(this);
  _mem_logit[69][t]->add_child(this);
  _mem_logit[6][t]->add_child(this);
  _mem_logit[70][t]->add_child(this);
  _mem_logit[71][t]->add_child(this);
  _mem_logit[72][t]->add_child(this);
  _mem_logit[73][t]->add_child(this);
  _mem_logit[74][t]->add_child(this);
  _mem_logit[75][t]->add_child(this);
  _mem_logit[76][t]->add_child(this);
  _mem_logit[77][t]->add_child(this);
  _mem_logit[78][t]->add_child(this);
  _mem_logit[79][t]->add_child(this);
  _mem_logit[7][t]->add_child(this);
  _mem_logit[80][t]->add_child(this);
  _mem_logit[81][t]->add_child(this);
  _mem_logit[8][t]->add_child(this);
  _mem_logit[9][t]->add_child(this);
}
void _Var_region_rate::remove_edge()
{
  _mem_logit[0][t]->erase_contig(this);
  _mem_logit[10][t]->erase_contig(this);
  _mem_logit[11][t]->erase_contig(this);
  _mem_logit[12][t]->erase_contig(this);
  _mem_logit[13][t]->erase_contig(this);
  _mem_logit[14][t]->erase_contig(this);
  _mem_logit[15][t]->erase_contig(this);
  _mem_logit[16][t]->erase_contig(this);
  _mem_logit[17][t]->erase_contig(this);
  _mem_logit[18][t]->erase_contig(this);
  _mem_logit[19][t]->erase_contig(this);
  _mem_logit[1][t]->erase_contig(this);
  _mem_logit[20][t]->erase_contig(this);
  _mem_logit[21][t]->erase_contig(this);
  _mem_logit[22][t]->erase_contig(this);
  _mem_logit[23][t]->erase_contig(this);
  _mem_logit[24][t]->erase_contig(this);
  _mem_logit[25][t]->erase_contig(this);
  _mem_logit[26][t]->erase_contig(this);
  _mem_logit[27][t]->erase_contig(this);
  _mem_logit[28][t]->erase_contig(this);
  _mem_logit[29][t]->erase_contig(this);
  _mem_logit[2][t]->erase_contig(this);
  _mem_logit[30][t]->erase_contig(this);
  _mem_logit[31][t]->erase_contig(this);
  _mem_logit[32][t]->erase_contig(this);
  _mem_logit[33][t]->erase_contig(this);
  _mem_logit[34][t]->erase_contig(this);
  _mem_logit[35][t]->erase_contig(this);
  _mem_logit[36][t]->erase_contig(this);
  _mem_logit[37][t]->erase_contig(this);
  _mem_logit[38][t]->erase_contig(this);
  _mem_logit[39][t]->erase_contig(this);
  _mem_logit[3][t]->erase_contig(this);
  _mem_logit[40][t]->erase_contig(this);
  _mem_logit[41][t]->erase_contig(this);
  _mem_logit[42][t]->erase_contig(this);
  _mem_logit[43][t]->erase_contig(this);
  _mem_logit[44][t]->erase_contig(this);
  _mem_logit[45][t]->erase_contig(this);
  _mem_logit[46][t]->erase_contig(this);
  _mem_logit[47][t]->erase_contig(this);
  _mem_logit[48][t]->erase_contig(this);
  _mem_logit[49][t]->erase_contig(this);
  _mem_logit[4][t]->erase_contig(this);
  _mem_logit[50][t]->erase_contig(this);
  _mem_logit[51][t]->erase_contig(this);
  _mem_logit[52][t]->erase_contig(this);
  _mem_logit[53][t]->erase_contig(this);
  _mem_logit[54][t]->erase_contig(this);
  _mem_logit[55][t]->erase_contig(this);
  _mem_logit[56][t]->erase_contig(this);
  _mem_logit[57][t]->erase_contig(this);
  _mem_logit[58][t]->erase_contig(this);
  _mem_logit[59][t]->erase_contig(this);
  _mem_logit[5][t]->erase_contig(this);
  _mem_logit[60][t]->erase_contig(this);
  _mem_logit[61][t]->erase_contig(this);
  _mem_logit[62][t]->erase_contig(this);
  _mem_logit[63][t]->erase_contig(this);
  _mem_logit[64][t]->erase_contig(this);
  _mem_logit[65][t]->erase_contig(this);
  _mem_logit[66][t]->erase_contig(this);
  _mem_logit[67][t]->erase_contig(this);
  _mem_logit[68][t]->erase_contig(this);
  _mem_logit[69][t]->erase_contig(this);
  _mem_logit[6][t]->erase_contig(this);
  _mem_logit[70][t]->erase_contig(this);
  _mem_logit[71][t]->erase_contig(this);
  _mem_logit[72][t]->erase_contig(this);
  _mem_logit[73][t]->erase_contig(this);
  _mem_logit[74][t]->erase_contig(this);
  _mem_logit[75][t]->erase_contig(this);
  _mem_logit[76][t]->erase_contig(this);
  _mem_logit[77][t]->erase_contig(this);
  _mem_logit[78][t]->erase_contig(this);
  _mem_logit[79][t]->erase_contig(this);
  _mem_logit[7][t]->erase_contig(this);
  _mem_logit[80][t]->erase_contig(this);
  _mem_logit[81][t]->erase_contig(this);
  _mem_logit[8][t]->erase_contig(this);
  _mem_logit[9][t]->erase_contig(this);
  _mem_logit[0][t]->erase_child(this);
  _mem_logit[10][t]->erase_child(this);
  _mem_logit[11][t]->erase_child(this);
  _mem_logit[12][t]->erase_child(this);
  _mem_logit[13][t]->erase_child(this);
  _mem_logit[14][t]->erase_child(this);
  _mem_logit[15][t]->erase_child(this);
  _mem_logit[16][t]->erase_child(this);
  _mem_logit[17][t]->erase_child(this);
  _mem_logit[18][t]->erase_child(this);
  _mem_logit[19][t]->erase_child(this);
  _mem_logit[1][t]->erase_child(this);
  _mem_logit[20][t]->erase_child(this);
  _mem_logit[21][t]->erase_child(this);
  _mem_logit[22][t]->erase_child(this);
  _mem_logit[23][t]->erase_child(this);
  _mem_logit[24][t]->erase_child(this);
  _mem_logit[25][t]->erase_child(this);
  _mem_logit[26][t]->erase_child(this);
  _mem_logit[27][t]->erase_child(this);
  _mem_logit[28][t]->erase_child(this);
  _mem_logit[29][t]->erase_child(this);
  _mem_logit[2][t]->erase_child(this);
  _mem_logit[30][t]->erase_child(this);
  _mem_logit[31][t]->erase_child(this);
  _mem_logit[32][t]->erase_child(this);
  _mem_logit[33][t]->erase_child(this);
  _mem_logit[34][t]->erase_child(this);
  _mem_logit[35][t]->erase_child(this);
  _mem_logit[36][t]->erase_child(this);
  _mem_logit[37][t]->erase_child(this);
  _mem_logit[38][t]->erase_child(this);
  _mem_logit[39][t]->erase_child(this);
  _mem_logit[3][t]->erase_child(this);
  _mem_logit[40][t]->erase_child(this);
  _mem_logit[41][t]->erase_child(this);
  _mem_logit[42][t]->erase_child(this);
  _mem_logit[43][t]->erase_child(this);
  _mem_logit[44][t]->erase_child(this);
  _mem_logit[45][t]->erase_child(this);
  _mem_logit[46][t]->erase_child(this);
  _mem_logit[47][t]->erase_child(this);
  _mem_logit[48][t]->erase_child(this);
  _mem_logit[49][t]->erase_child(this);
  _mem_logit[4][t]->erase_child(this);
  _mem_logit[50][t]->erase_child(this);
  _mem_logit[51][t]->erase_child(this);
  _mem_logit[52][t]->erase_child(this);
  _mem_logit[53][t]->erase_child(this);
  _mem_logit[54][t]->erase_child(this);
  _mem_logit[55][t]->erase_child(this);
  _mem_logit[56][t]->erase_child(this);
  _mem_logit[57][t]->erase_child(this);
  _mem_logit[58][t]->erase_child(this);
  _mem_logit[59][t]->erase_child(this);
  _mem_logit[5][t]->erase_child(this);
  _mem_logit[60][t]->erase_child(this);
  _mem_logit[61][t]->erase_child(this);
  _mem_logit[62][t]->erase_child(this);
  _mem_logit[63][t]->erase_child(this);
  _mem_logit[64][t]->erase_child(this);
  _mem_logit[65][t]->erase_child(this);
  _mem_logit[66][t]->erase_child(this);
  _mem_logit[67][t]->erase_child(this);
  _mem_logit[68][t]->erase_child(this);
  _mem_logit[69][t]->erase_child(this);
  _mem_logit[6][t]->erase_child(this);
  _mem_logit[70][t]->erase_child(this);
  _mem_logit[71][t]->erase_child(this);
  _mem_logit[72][t]->erase_child(this);
  _mem_logit[73][t]->erase_child(this);
  _mem_logit[74][t]->erase_child(this);
  _mem_logit[75][t]->erase_child(this);
  _mem_logit[76][t]->erase_child(this);
  _mem_logit[77][t]->erase_child(this);
  _mem_logit[78][t]->erase_child(this);
  _mem_logit[79][t]->erase_child(this);
  _mem_logit[7][t]->erase_child(this);
  _mem_logit[80][t]->erase_child(this);
  _mem_logit[81][t]->erase_child(this);
  _mem_logit[8][t]->erase_child(this);
  _mem_logit[9][t]->erase_child(this);
}
void _Var_region_rate::mcmc_resample()
{
  mh_parent_resample_arg(this);
}
void sample()
{
  for (_cur_loop=1;_cur_loop<=_TOT_LOOP;_cur_loop++)
  {
    mcmc_sample_single_iter();
    _eval_query();
  }

}

}
int main()
{
  std::chrono::time_point<std::chrono::system_clock> __start_time = std::chrono::system_clock::now();
  swift::_init_storage();
  swift::_init_world();
  std::chrono::duration<double> __elapsed_seconds = std::chrono::system_clock::now()-__start_time;
  printf("\ninit time: %fs\n",__elapsed_seconds.count());
  __start_time=std::chrono::system_clock::now();
  swift::sample();
  __elapsed_seconds=std::chrono::system_clock::now()-__start_time;
  printf("\nsample time: %fs (#iter = %d)\n",__elapsed_seconds.count(),100000);
  swift::_print_answer();
  swift::_garbage_collection();
}
