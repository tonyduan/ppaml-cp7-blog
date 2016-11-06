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

class _Var_rho;
class _Var_tau1;
class _Var_tau2;
class _Var_beta1;
class _Var_beta2;
class _Var_y;
class _Var_temporal_edge;
class _Var_spatial_edge;
class _Var_noise;
class _Var_logit;
class _Var_county_rate;
class _Var_region_rate;

const vector<string> __vecstr_instance_Week = {"weeks[0]", "weeks[1]", "weeks[2]", "weeks[3]", "weeks[4]", "weeks[5]", "weeks[6]", "weeks[7]", "weeks[8]", "weeks[9]", "weeks[10]", "weeks[11]", "weeks[12]", "weeks[13]", "weeks[14]", "weeks[15]", "weeks[16]", "weeks[17]", "weeks[18]", "weeks[19]", "weeks[20]", "weeks[21]", "weeks[22]", "weeks[23]", "weeks[24]", "weeks[25]", "weeks[26]", "weeks[27]", "weeks[28]", "weeks[29]", "weeks[30]", "weeks[31]", "weeks[32]", "weeks[33]", "weeks[34]", "weeks[35]", "weeks[36]", "weeks[37]", "weeks[38]", "weeks[39]", "weeks[40]", "weeks[41]", "weeks[42]", "weeks[43]", "weeks[44]", "weeks[45]", "weeks[46]", "weeks[47]", "weeks[48]", "weeks[49]", "weeks[50]", "weeks[51]", "weeks[52]", "weeks[53]", "weeks[54]", "weeks[55]", "weeks[56]", "weeks[57]", "weeks[58]", "weeks[59]", "weeks[60]", "weeks[61]", "weeks[62]", "weeks[63]", "weeks[64]", "weeks[65]", "weeks[66]", "weeks[67]", "weeks[68]", "weeks[69]", "weeks[70]", "weeks[71]", "weeks[72]", "weeks[73]", "weeks[74]", "weeks[75]", "weeks[76]", "weeks[77]", "weeks[78]", "weeks[79]", "weeks[80]", "weeks[81]", "weeks[82]", "weeks[83]", "weeks[84]", "weeks[85]", "weeks[86]", "weeks[87]", "weeks[88]", "weeks[89]", "weeks[90]", "weeks[91]", "weeks[92]", "weeks[93]", "weeks[94]", "weeks[95]", "weeks[96]", "weeks[97]", "weeks[98]", "weeks[99]", "weeks[100]", "weeks[101]", "weeks[102]"};
const vector<string> __vecstr_instance_Region = {"regions[0]", "regions[1]", "regions[2]", "regions[3]", "regions[4]", "regions[5]", "regions[6]", "regions[7]", "regions[8]", "regions[9]", "regions[10]", "regions[11]", "regions[12]", "regions[13]", "regions[14]", "regions[15]", "regions[16]", "regions[17]", "regions[18]", "regions[19]", "regions[20]", "regions[21]", "regions[22]", "regions[23]", "regions[24]"};
const vector<string> __vecstr_instance_County = {"counties[0]", "counties[1]", "counties[2]", "counties[3]", "counties[4]", "counties[5]", "counties[6]", "counties[7]", "counties[8]", "counties[9]", "counties[10]", "counties[11]", "counties[12]", "counties[13]", "counties[14]", "counties[15]", "counties[16]", "counties[17]", "counties[18]", "counties[19]", "counties[20]", "counties[21]", "counties[22]", "counties[23]", "counties[24]", "counties[25]", "counties[26]", "counties[27]", "counties[28]", "counties[29]", "counties[30]", "counties[31]", "counties[32]", "counties[33]", "counties[34]", "counties[35]", "counties[36]", "counties[37]", "counties[38]", "counties[39]", "counties[40]", "counties[41]", "counties[42]", "counties[43]", "counties[44]", "counties[45]", "counties[46]", "counties[47]", "counties[48]", "counties[49]", "counties[50]", "counties[51]", "counties[52]", "counties[53]", "counties[54]", "counties[55]", "counties[56]", "counties[57]", "counties[58]", "counties[59]", "counties[60]", "counties[61]", "counties[62]", "counties[63]", "counties[64]", "counties[65]", "counties[66]", "counties[67]", "counties[68]", "counties[69]", "counties[70]", "counties[71]", "counties[72]", "counties[73]", "counties[74]", "counties[75]", "counties[76]", "counties[77]", "counties[78]", "counties[79]", "counties[80]", "counties[81]", "counties[82]", "counties[83]", "counties[84]", "counties[85]", "counties[86]", "counties[87]", "counties[88]", "counties[89]", "counties[90]", "counties[91]", "counties[92]", "counties[93]", "counties[94]", "counties[95]", "counties[96]", "counties[97]", "counties[98]", "counties[99]", "counties[100]", "counties[101]", "counties[102]", "counties[103]", "counties[104]", "counties[105]", "counties[106]", "counties[107]", "counties[108]", "counties[109]", "counties[110]", "counties[111]", "counties[112]", "counties[113]", "counties[114]", "counties[115]", "counties[116]", "counties[117]", "counties[118]", "counties[119]", "counties[120]", "counties[121]", "counties[122]", "counties[123]", "counties[124]", "counties[125]", "counties[126]", "counties[127]", "counties[128]", "counties[129]", "counties[130]", "counties[131]", "counties[132]", "counties[133]", "counties[134]", "counties[135]", "counties[136]", "counties[137]", "counties[138]", "counties[139]", "counties[140]", "counties[141]", "counties[142]", "counties[143]", "counties[144]", "counties[145]", "counties[146]", "counties[147]", "counties[148]", "counties[149]", "counties[150]", "counties[151]", "counties[152]", "counties[153]", "counties[154]", "counties[155]", "counties[156]", "counties[157]", "counties[158]", "counties[159]", "counties[160]", "counties[161]", "counties[162]", "counties[163]", "counties[164]", "counties[165]", "counties[166]", "counties[167]", "counties[168]", "counties[169]", "counties[170]", "counties[171]", "counties[172]", "counties[173]", "counties[174]", "counties[175]", "counties[176]", "counties[177]", "counties[178]", "counties[179]", "counties[180]", "counties[181]", "counties[182]", "counties[183]", "counties[184]", "counties[185]", "counties[186]", "counties[187]", "counties[188]", "counties[189]", "counties[190]", "counties[191]", "counties[192]", "counties[193]", "counties[194]", "counties[195]", "counties[196]", "counties[197]", "counties[198]", "counties[199]", "counties[200]", "counties[201]", "counties[202]", "counties[203]", "counties[204]", "counties[205]", "counties[206]", "counties[207]", "counties[208]", "counties[209]", "counties[210]", "counties[211]", "counties[212]", "counties[213]", "counties[214]", "counties[215]", "counties[216]", "counties[217]", "counties[218]", "counties[219]", "counties[220]", "counties[221]", "counties[222]", "counties[223]", "counties[224]", "counties[225]", "counties[226]", "counties[227]", "counties[228]", "counties[229]", "counties[230]", "counties[231]", "counties[232]", "counties[233]", "counties[234]", "counties[235]", "counties[236]", "counties[237]", "counties[238]", "counties[239]", "counties[240]", "counties[241]", "counties[242]", "counties[243]", "counties[244]", "counties[245]", "counties[246]", "counties[247]", "counties[248]", "counties[249]", "counties[250]", "counties[251]", "counties[252]", "counties[253]", "counties[254]", "counties[255]", "counties[256]", "counties[257]", "counties[258]", "counties[259]", "counties[260]", "counties[261]", "counties[262]", "counties[263]", "counties[264]", "counties[265]", "counties[266]", "counties[267]", "counties[268]", "counties[269]", "counties[270]", "counties[271]", "counties[272]", "counties[273]", "counties[274]", "counties[275]", "counties[276]"};
void _eval_query();
void _init_storage();
void _init_world();
void _garbage_collection();
void _print_answer();
const int _TOT_LOOP = 10000000;
const int _BURN_IN = 8000000;
int _tot_round = -8000000;
const mat __fixed_county_map = loadRealMatrix("data_processed/county_map.txt");
const mat __fixed_region_pop = loadRealMatrix("data_processed/region_pops.txt");
const mat __fixed_covariates1 = loadRealMatrix("data_processed/covariates1.txt");
const mat __fixed_covariates2 = loadRealMatrix("data_processed/covariates2.txt");
const mat __fixed_D = loadRealMatrix("data_processed/D.txt");
const mat __fixed_W = loadRealMatrix("data_processed/W.txt");
const mat __fixed_observations = loadRealMatrix("data_processed/obs.txt");
int __fixed_toWeek(int);
class _Var_rho: public BayesVar<double> {
public:
  _Var_rho();
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
  void conjugacy_analysis(double&);
};
_Var_rho* _mem_rho;
class _Var_tau1: public BayesVar<double> {
public:
  _Var_tau1();
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
  void conjugacy_analysis(double&);
};
_Var_tau1* _mem_tau1;
class _Var_tau2: public BayesVar<double> {
public:
  _Var_tau2();
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
  void conjugacy_analysis(double&);
};
_Var_tau2* _mem_tau2;
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
  void conjugacy_analysis(double&);
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
  void conjugacy_analysis(double&);
};
_Var_beta2* _mem_beta2;
class _Var_y: public BayesVar<double> {
public:
  int c;
  int t;
  _Var_y(int,int);
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
  void conjugacy_analysis(double&);
};
DynamicTable<_Var_y*,2> _mem_y;
class _Var_temporal_edge: public BayesVar<char> {
public:
  int t;
  int c;
  _Var_temporal_edge(int,int);
  string getname();
  char& getval();
  char& getcache();
  void clear();
  double getlikeli();
  double getcachelikeli();
  void sample();
  void sample_cache();
  void active_edge();
  void remove_edge();
  void mcmc_resample();
  void conjugacy_analysis(char&);
};
DynamicTable<_Var_temporal_edge*,2> _mem_temporal_edge;
class _Var_spatial_edge: public BayesVar<char> {
public:
  int c1;
  int c2;
  int t;
  _Var_spatial_edge(int,int,int);
  string getname();
  char& getval();
  char& getcache();
  void clear();
  double getlikeli();
  double getcachelikeli();
  void sample();
  void sample_cache();
  void active_edge();
  void remove_edge();
  void mcmc_resample();
  void conjugacy_analysis(char&);
};
DynamicTable<_Var_spatial_edge*,3> _mem_spatial_edge;
class _Var_noise: public BayesVar<double> {
public:
  int c;
  int t;
  _Var_noise(int,int);
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
  void conjugacy_analysis(double&);
};
DynamicTable<_Var_noise*,2> _mem_noise;
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
  void conjugacy_analysis(double&);
};
DynamicTable<_Var_logit*,2> _mem_logit;
class _Var_county_rate: public BayesVar<double> {
public:
  int c;
  int t;
  _Var_county_rate(int,int);
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
  void conjugacy_analysis(double&);
};
DynamicTable<_Var_county_rate*,2> _mem_county_rate;
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
  void conjugacy_analysis(double&);
};
DynamicTable<_Var_region_rate*,2> _mem_region_rate;
Gamma Gamma35772256;
Gamma Gamma35772624;
Gamma Gamma35772944;
Gaussian Gaussian35773328;
Gaussian Gaussian35773712;
Gaussian Gaussian35778448;
BooleanDistrib BooleanDistrib35781104;
BooleanDistrib BooleanDistrib35783616;
Gaussian Gaussian35786016;
Gaussian Gaussian35791840;
DynamicTable<Hist<double>*,2> _answer_0;
void sample();

void _eval_query()
{
  _tot_round++;
  if (_tot_round<=0)
    return ;
  for (int c = 0;c<277;c++)
  for (int t = 0;t<103;t++)
  _answer_0[c][t]->add(_mem_county_rate[c][t]->getval(),1);


}
void _init_storage()
{
  _mem_rho=new _Var_rho();
  _mem_tau1=new _Var_tau1();
  _mem_tau2=new _Var_tau2();
  _mem_beta1=new _Var_beta1();
  _mem_beta2=new _Var_beta2();
  _mem_y.resize(0,277);
  _mem_y.resize(1,103);
  for (int c = 0;c<277;c++)
  {
    for (int t = 0;t<103;t++)
    {
      _mem_y[c][t]=new _Var_y(c, t);
    }

  }

  _mem_temporal_edge.resize(0,103);
  _mem_temporal_edge.resize(1,277);
  for (int t = 0;t<103;t++)
  {
    for (int c = 0;c<277;c++)
    {
      _mem_temporal_edge[t][c]=new _Var_temporal_edge(t, c);
    }

  }

  _mem_spatial_edge.resize(0,277);
  _mem_spatial_edge.resize(1,277);
  _mem_spatial_edge.resize(2,103);
  for (int c1 = 0;c1<277;c1++)
  {
    for (int c2 = 0;c2<277;c2++)
    {
      for (int t = 0;t<103;t++)
      {
        _mem_spatial_edge[c1][c2][t]=new _Var_spatial_edge(c1, c2, t);
      }

    }

  }

  _mem_noise.resize(0,277);
  _mem_noise.resize(1,103);
  for (int c = 0;c<277;c++)
  {
    for (int t = 0;t<103;t++)
    {
      _mem_noise[c][t]=new _Var_noise(c, t);
    }

  }

  _mem_logit.resize(0,277);
  _mem_logit.resize(1,103);
  for (int c = 0;c<277;c++)
  {
    for (int t = 0;t<103;t++)
    {
      _mem_logit[c][t]=new _Var_logit(c, t);
    }

  }

  _mem_county_rate.resize(0,277);
  _mem_county_rate.resize(1,103);
  for (int c = 0;c<277;c++)
  {
    for (int t = 0;t<103;t++)
    {
      _mem_county_rate[c][t]=new _Var_county_rate(c, t);
    }

  }

  _mem_region_rate.resize(0,25);
  _mem_region_rate.resize(1,103);
  for (int r = 0;r<25;r++)
  {
    for (int t = 0;t<103;t++)
    {
      _mem_region_rate[r][t]=new _Var_region_rate(r, t);
    }

  }

  Gamma35772256.init(0.50000000,0.10000000);
  Gamma35772624.init(3.00000000,0.10000000);
  Gamma35772944.init(10.00000000,0.10000000);
  Gaussian35773328.init(0.000000,10.00000000);
  Gaussian35773712.init(0.000000,10.00000000);
  _answer_0.resize(0,277);
  _answer_0.resize(1,103);
  for (int c = 0;c<277;c++)
  {
    for (int t = 0;t<103;t++)
    {
      _answer_0[c][t]=new Hist<double>(false, 20);
    }

  }

}
void _init_world()
{
  for (int t = 0;t<103;t++)
  for (int c = 0;c<277;c++)
  if (t>0)
    _util_set_evidence<char>(_mem_temporal_edge[t][c],1);


  for (int c1 = 0;c1<277;c1++)
  for (int c2 = 0;c2<277;c2++)
  for (int t = 0;t<103;t++)
  if (__fixed_W(c1,c2)==1.00000000)
    _util_set_evidence<char>(_mem_spatial_edge[c1][c2][t],1);



  for (int r = 0;r<25;r++)
  for (int t = 0;t<103;t++)
  if (__fixed_observations(t,r)!=0.000000)
    _util_set_evidence<double>(_mem_region_rate[r][t],__fixed_observations(t,r));


}
void _garbage_collection()
{
  _free_obj(_mem_rho);
  _free_obj(_mem_tau1);
  _free_obj(_mem_tau2);
  _free_obj(_mem_beta1);
  _free_obj(_mem_beta2);
  _free_obj(_mem_y);
  _free_obj(_mem_temporal_edge);
  _free_obj(_mem_spatial_edge);
  _free_obj(_mem_noise);
  _free_obj(_mem_logit);
  _free_obj(_mem_county_rate);
  _free_obj(_mem_region_rate);
}
void _print_answer()
{
  char buffer0[256];
  for (int c = 0;c<277;c++)
  for (int t = 0;t<103;t++)
  {
    sprintf(buffer0,"county_rate(County[%d], Week[%d])\n",c,t);
    _answer_0[c][t]->print(buffer0);
  }


}
int __fixed_toWeek(int i)
{
  return i;
}
_Var_rho::_Var_rho()
{}
string _Var_rho::getname()
{
  return "rho";
}
double& _Var_rho::getval()
{
  return getval_arg(this);
}
double& _Var_rho::getcache()
{
  return getcache_arg(this);
}
void _Var_rho::clear()
{
  return clear_arg(this);
}
double _Var_rho::getlikeli()
{
  return Gamma35772256.loglikeli(val);
}
double _Var_rho::getcachelikeli()
{
  auto _t_val = getcache();
  return Gamma35772256.loglikeli(_t_val);
}
void _Var_rho::sample()
{
  val=Gamma35772256.gen();
}
void _Var_rho::sample_cache()
{
  cache_val=Gamma35772256.gen();
}
void _Var_rho::active_edge()
{}
void _Var_rho::remove_edge()
{}
void _Var_rho::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_rho::conjugacy_analysis(double& _nxt_val)
{}
_Var_tau1::_Var_tau1()
{}
string _Var_tau1::getname()
{
  return "tau1";
}
double& _Var_tau1::getval()
{
  return getval_arg(this);
}
double& _Var_tau1::getcache()
{
  return getcache_arg(this);
}
void _Var_tau1::clear()
{
  return clear_arg(this);
}
double _Var_tau1::getlikeli()
{
  return Gamma35772624.loglikeli(val);
}
double _Var_tau1::getcachelikeli()
{
  auto _t_val = getcache();
  return Gamma35772624.loglikeli(_t_val);
}
void _Var_tau1::sample()
{
  val=Gamma35772624.gen();
}
void _Var_tau1::sample_cache()
{
  cache_val=Gamma35772624.gen();
}
void _Var_tau1::active_edge()
{}
void _Var_tau1::remove_edge()
{}
void _Var_tau1::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_tau1::conjugacy_analysis(double& _nxt_val)
{}
_Var_tau2::_Var_tau2()
{}
string _Var_tau2::getname()
{
  return "tau2";
}
double& _Var_tau2::getval()
{
  return getval_arg(this);
}
double& _Var_tau2::getcache()
{
  return getcache_arg(this);
}
void _Var_tau2::clear()
{
  return clear_arg(this);
}
double _Var_tau2::getlikeli()
{
  return Gamma35772944.loglikeli(val);
}
double _Var_tau2::getcachelikeli()
{
  auto _t_val = getcache();
  return Gamma35772944.loglikeli(_t_val);
}
void _Var_tau2::sample()
{
  val=Gamma35772944.gen();
}
void _Var_tau2::sample_cache()
{
  cache_val=Gamma35772944.gen();
}
void _Var_tau2::active_edge()
{}
void _Var_tau2::remove_edge()
{}
void _Var_tau2::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_tau2::conjugacy_analysis(double& _nxt_val)
{}
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
  return Gaussian35773328.loglikeli(val);
}
double _Var_beta1::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian35773328.loglikeli(_t_val);
}
void _Var_beta1::sample()
{
  val=Gaussian35773328.gen();
}
void _Var_beta1::sample_cache()
{
  cache_val=Gaussian35773328.gen();
}
void _Var_beta1::active_edge()
{}
void _Var_beta1::remove_edge()
{}
void _Var_beta1::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_beta1::conjugacy_analysis(double& _nxt_val)
{}
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
  return Gaussian35773712.loglikeli(val);
}
double _Var_beta2::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian35773712.loglikeli(_t_val);
}
void _Var_beta2::sample()
{
  val=Gaussian35773712.gen();
}
void _Var_beta2::sample_cache()
{
  cache_val=Gaussian35773712.gen();
}
void _Var_beta2::active_edge()
{}
void _Var_beta2::remove_edge()
{}
void _Var_beta2::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_beta2::conjugacy_analysis(double& _nxt_val)
{}
_Var_y::_Var_y(int _c, int _t):c(_c),t(_t)
{}
string _Var_y::getname()
{
  return "y";
}
double& _Var_y::getval()
{
  return getval_arg(this);
}
double& _Var_y::getcache()
{
  return getcache_arg(this);
}
void _Var_y::clear()
{
  return clear_arg(this);
}
double _Var_y::getlikeli()
{
  return Gaussian35778448.init(0.000000,__fixed_D[c]),Gaussian35778448.loglikeli(val);
}
double _Var_y::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian35778448.init(0.000000,__fixed_D[c]),Gaussian35778448.loglikeli(_t_val);
}
void _Var_y::sample()
{
  val=(Gaussian35778448.init(0.000000,__fixed_D[c]),Gaussian35778448.gen());
}
void _Var_y::sample_cache()
{
  cache_val=(Gaussian35778448.init(0.000000,__fixed_D[c]),Gaussian35778448.gen());
}
void _Var_y::active_edge()
{}
void _Var_y::remove_edge()
{}
void _Var_y::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_y::conjugacy_analysis(double& _nxt_val)
{}
_Var_temporal_edge::_Var_temporal_edge(int _t, int _c):t(_t),c(_c)
{}
string _Var_temporal_edge::getname()
{
  return "temporal_edge";
}
char& _Var_temporal_edge::getval()
{
  return getval_arg(this);
}
char& _Var_temporal_edge::getcache()
{
  return getcache_arg(this);
}
void _Var_temporal_edge::clear()
{
  return clear_arg(this);
}
double _Var_temporal_edge::getlikeli()
{
  return BooleanDistrib35781104.init(exp(_mem_tau1->getval()*_mem_y[c][t]->getval()*_mem_y[c][__fixed_toWeek(t-1)]->getval())),BooleanDistrib35781104.loglikeli(val);
}
double _Var_temporal_edge::getcachelikeli()
{
  auto _t_val = getcache();
  return BooleanDistrib35781104.init(exp(_mem_tau1->getcache()*_mem_y[c][t]->getcache()*_mem_y[c][__fixed_toWeek(t-1)]->getcache())),BooleanDistrib35781104.loglikeli(_t_val);
}
void _Var_temporal_edge::sample()
{
  val=(BooleanDistrib35781104.init(exp(_mem_tau1->getval()*_mem_y[c][t]->getval()*_mem_y[c][__fixed_toWeek(t-1)]->getval())),BooleanDistrib35781104.gen());
}
void _Var_temporal_edge::sample_cache()
{
  cache_val=(BooleanDistrib35781104.init(exp(_mem_tau1->getcache()*_mem_y[c][t]->getcache()*_mem_y[c][__fixed_toWeek(t-1)]->getcache())),BooleanDistrib35781104.gen());
}
void _Var_temporal_edge::active_edge()
{
  _mem_tau1->add_contig(this);
  _mem_y[c][t]->add_contig(this);
  _mem_y[c][__fixed_toWeek(t-1)]->add_contig(this);
  _mem_tau1->add_child(this);
  _mem_y[c][t]->add_child(this);
  _mem_y[c][__fixed_toWeek(t-1)]->add_child(this);
}
void _Var_temporal_edge::remove_edge()
{
  _mem_tau1->erase_contig(this);
  _mem_y[c][t]->erase_contig(this);
  _mem_y[c][__fixed_toWeek(t-1)]->erase_contig(this);
  _mem_tau1->erase_child(this);
  _mem_y[c][t]->erase_child(this);
  _mem_y[c][__fixed_toWeek(t-1)]->erase_child(this);
}
void _Var_temporal_edge::mcmc_resample()
{
  mh_parent_resample_arg(this);
}
void _Var_temporal_edge::conjugacy_analysis(char& _nxt_val)
{}
_Var_spatial_edge::_Var_spatial_edge(int _c1, int _c2, int _t):c1(_c1),c2(_c2),t(_t)
{}
string _Var_spatial_edge::getname()
{
  return "spatial_edge";
}
char& _Var_spatial_edge::getval()
{
  return getval_arg(this);
}
char& _Var_spatial_edge::getcache()
{
  return getcache_arg(this);
}
void _Var_spatial_edge::clear()
{
  return clear_arg(this);
}
double _Var_spatial_edge::getlikeli()
{
  return BooleanDistrib35783616.init(exp(_mem_tau1->getval()*_mem_rho->getval()*_mem_y[c1][t]->getval()*_mem_y[c2][t]->getval())),BooleanDistrib35783616.loglikeli(val);
}
double _Var_spatial_edge::getcachelikeli()
{
  auto _t_val = getcache();
  return BooleanDistrib35783616.init(exp(_mem_tau1->getcache()*_mem_rho->getcache()*_mem_y[c1][t]->getcache()*_mem_y[c2][t]->getcache())),BooleanDistrib35783616.loglikeli(_t_val);
}
void _Var_spatial_edge::sample()
{
  val=(BooleanDistrib35783616.init(exp(_mem_tau1->getval()*_mem_rho->getval()*_mem_y[c1][t]->getval()*_mem_y[c2][t]->getval())),BooleanDistrib35783616.gen());
}
void _Var_spatial_edge::sample_cache()
{
  cache_val=(BooleanDistrib35783616.init(exp(_mem_tau1->getcache()*_mem_rho->getcache()*_mem_y[c1][t]->getcache()*_mem_y[c2][t]->getcache())),BooleanDistrib35783616.gen());
}
void _Var_spatial_edge::active_edge()
{
  _mem_rho->add_contig(this);
  _mem_tau1->add_contig(this);
  _mem_y[c1][t]->add_contig(this);
  _mem_y[c2][t]->add_contig(this);
  _mem_rho->add_child(this);
  _mem_tau1->add_child(this);
  _mem_y[c1][t]->add_child(this);
  _mem_y[c2][t]->add_child(this);
}
void _Var_spatial_edge::remove_edge()
{
  _mem_rho->erase_contig(this);
  _mem_tau1->erase_contig(this);
  _mem_y[c1][t]->erase_contig(this);
  _mem_y[c2][t]->erase_contig(this);
  _mem_rho->erase_child(this);
  _mem_tau1->erase_child(this);
  _mem_y[c1][t]->erase_child(this);
  _mem_y[c2][t]->erase_child(this);
}
void _Var_spatial_edge::mcmc_resample()
{
  mh_parent_resample_arg(this);
}
void _Var_spatial_edge::conjugacy_analysis(char& _nxt_val)
{}
_Var_noise::_Var_noise(int _c, int _t):c(_c),t(_t)
{}
string _Var_noise::getname()
{
  return "noise";
}
double& _Var_noise::getval()
{
  return getval_arg(this);
}
double& _Var_noise::getcache()
{
  return getcache_arg(this);
}
void _Var_noise::clear()
{
  return clear_arg(this);
}
double _Var_noise::getlikeli()
{
  return Gaussian35786016.init(0.000000,1.00000000/_mem_tau2->getval()),Gaussian35786016.loglikeli(val);
}
double _Var_noise::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian35786016.init(0.000000,1.00000000/_mem_tau2->getcache()),Gaussian35786016.loglikeli(_t_val);
}
void _Var_noise::sample()
{
  val=(Gaussian35786016.init(0.000000,1.00000000/_mem_tau2->getval()),Gaussian35786016.gen());
}
void _Var_noise::sample_cache()
{
  cache_val=(Gaussian35786016.init(0.000000,1.00000000/_mem_tau2->getcache()),Gaussian35786016.gen());
}
void _Var_noise::active_edge()
{
  _mem_tau2->add_contig(this);
  _mem_tau2->add_child(this);
}
void _Var_noise::remove_edge()
{
  _mem_tau2->erase_contig(this);
  _mem_tau2->erase_child(this);
}
void _Var_noise::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_noise::conjugacy_analysis(double& _nxt_val)
{}
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
  return logEqual(fabs(val-(_mem_beta1->getval()*__fixed_covariates1(c,t)+_mem_beta2->getval()*__fixed_covariates2(c,t)+_mem_y[c][t]->getval()))<=0.000000,true);
}
double _Var_logit::getcachelikeli()
{
  auto _t_val = getcache();
  return logEqual(fabs(_t_val-(_mem_beta1->getcache()*__fixed_covariates1(c,t)+_mem_beta2->getcache()*__fixed_covariates2(c,t)+_mem_y[c][t]->getcache()))<=0.000000,true);
}
void _Var_logit::sample()
{
  val=_mem_beta1->getval()*__fixed_covariates1(c,t)+_mem_beta2->getval()*__fixed_covariates2(c,t)+_mem_y[c][t]->getval();
}
void _Var_logit::sample_cache()
{
  cache_val=_mem_beta1->getcache()*__fixed_covariates1(c,t)+_mem_beta2->getcache()*__fixed_covariates2(c,t)+_mem_y[c][t]->getcache();
}
void _Var_logit::active_edge()
{
  _mem_beta1->add_contig(this);
  _mem_beta2->add_contig(this);
  _mem_y[c][t]->add_contig(this);
  _mem_beta1->add_child(this);
  _mem_beta2->add_child(this);
  _mem_y[c][t]->add_child(this);
}
void _Var_logit::remove_edge()
{
  _mem_beta1->erase_contig(this);
  _mem_beta2->erase_contig(this);
  _mem_y[c][t]->erase_contig(this);
  _mem_beta1->erase_child(this);
  _mem_beta2->erase_child(this);
  _mem_y[c][t]->erase_child(this);
}
void _Var_logit::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_logit::conjugacy_analysis(double& _nxt_val)
{}
_Var_county_rate::_Var_county_rate(int _c, int _t):c(_c),t(_t)
{}
string _Var_county_rate::getname()
{
  return "county_rate";
}
double& _Var_county_rate::getval()
{
  return getval_arg(this);
}
double& _Var_county_rate::getcache()
{
  return getcache_arg(this);
}
void _Var_county_rate::clear()
{
  return clear_arg(this);
}
double _Var_county_rate::getlikeli()
{
  return logEqual(fabs(val-exp(_mem_logit[c][t]->getval())/(1.00000000+exp(_mem_logit[c][t]->getval())))<=0.000000,true);
}
double _Var_county_rate::getcachelikeli()
{
  auto _t_val = getcache();
  return logEqual(fabs(_t_val-exp(_mem_logit[c][t]->getcache())/(1.00000000+exp(_mem_logit[c][t]->getcache())))<=0.000000,true);
}
void _Var_county_rate::sample()
{
  val=exp(_mem_logit[c][t]->getval())/(1.00000000+exp(_mem_logit[c][t]->getval()));
}
void _Var_county_rate::sample_cache()
{
  cache_val=exp(_mem_logit[c][t]->getcache())/(1.00000000+exp(_mem_logit[c][t]->getcache()));
}
void _Var_county_rate::active_edge()
{
  _mem_logit[c][t]->add_contig(this);
  _mem_logit[c][t]->add_child(this);
}
void _Var_county_rate::remove_edge()
{
  _mem_logit[c][t]->erase_contig(this);
  _mem_logit[c][t]->erase_child(this);
}
void _Var_county_rate::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_county_rate::conjugacy_analysis(double& _nxt_val)
{}
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
  return Gaussian35791840.init(dot(__fixed_county_map.row(r),vstack({_mem_county_rate[0][t]->getval(), _mem_county_rate[1][t]->getval(), _mem_county_rate[2][t]->getval(), _mem_county_rate[3][t]->getval(), _mem_county_rate[4][t]->getval(), _mem_county_rate[5][t]->getval(), _mem_county_rate[6][t]->getval(), _mem_county_rate[7][t]->getval(), _mem_county_rate[8][t]->getval(), _mem_county_rate[9][t]->getval(), _mem_county_rate[10][t]->getval(), _mem_county_rate[11][t]->getval(), _mem_county_rate[12][t]->getval(), _mem_county_rate[13][t]->getval(), _mem_county_rate[14][t]->getval(), _mem_county_rate[15][t]->getval(), _mem_county_rate[16][t]->getval(), _mem_county_rate[17][t]->getval(), _mem_county_rate[18][t]->getval(), _mem_county_rate[19][t]->getval(), _mem_county_rate[20][t]->getval(), _mem_county_rate[21][t]->getval(), _mem_county_rate[22][t]->getval(), _mem_county_rate[23][t]->getval(), _mem_county_rate[24][t]->getval(), _mem_county_rate[25][t]->getval(), _mem_county_rate[26][t]->getval(), _mem_county_rate[27][t]->getval(), _mem_county_rate[28][t]->getval(), _mem_county_rate[29][t]->getval(), _mem_county_rate[30][t]->getval(), _mem_county_rate[31][t]->getval(), _mem_county_rate[32][t]->getval(), _mem_county_rate[33][t]->getval(), _mem_county_rate[34][t]->getval(), _mem_county_rate[35][t]->getval(), _mem_county_rate[36][t]->getval(), _mem_county_rate[37][t]->getval(), _mem_county_rate[38][t]->getval(), _mem_county_rate[39][t]->getval(), _mem_county_rate[40][t]->getval(), _mem_county_rate[41][t]->getval(), _mem_county_rate[42][t]->getval(), _mem_county_rate[43][t]->getval(), _mem_county_rate[44][t]->getval(), _mem_county_rate[45][t]->getval(), _mem_county_rate[46][t]->getval(), _mem_county_rate[47][t]->getval(), _mem_county_rate[48][t]->getval(), _mem_county_rate[49][t]->getval(), _mem_county_rate[50][t]->getval(), _mem_county_rate[51][t]->getval(), _mem_county_rate[52][t]->getval(), _mem_county_rate[53][t]->getval(), _mem_county_rate[54][t]->getval(), _mem_county_rate[55][t]->getval(), _mem_county_rate[56][t]->getval(), _mem_county_rate[57][t]->getval(), _mem_county_rate[58][t]->getval(), _mem_county_rate[59][t]->getval(), _mem_county_rate[60][t]->getval(), _mem_county_rate[61][t]->getval(), _mem_county_rate[62][t]->getval(), _mem_county_rate[63][t]->getval(), _mem_county_rate[64][t]->getval(), _mem_county_rate[65][t]->getval(), _mem_county_rate[66][t]->getval(), _mem_county_rate[67][t]->getval(), _mem_county_rate[68][t]->getval(), _mem_county_rate[69][t]->getval(), _mem_county_rate[70][t]->getval(), _mem_county_rate[71][t]->getval(), _mem_county_rate[72][t]->getval(), _mem_county_rate[73][t]->getval(), _mem_county_rate[74][t]->getval(), _mem_county_rate[75][t]->getval(), _mem_county_rate[76][t]->getval(), _mem_county_rate[77][t]->getval(), _mem_county_rate[78][t]->getval(), _mem_county_rate[79][t]->getval(), _mem_county_rate[80][t]->getval(), _mem_county_rate[81][t]->getval(), _mem_county_rate[82][t]->getval(), _mem_county_rate[83][t]->getval(), _mem_county_rate[84][t]->getval(), _mem_county_rate[85][t]->getval(), _mem_county_rate[86][t]->getval(), _mem_county_rate[87][t]->getval(), _mem_county_rate[88][t]->getval(), _mem_county_rate[89][t]->getval(), _mem_county_rate[90][t]->getval(), _mem_county_rate[91][t]->getval(), _mem_county_rate[92][t]->getval(), _mem_county_rate[93][t]->getval(), _mem_county_rate[94][t]->getval(), _mem_county_rate[95][t]->getval(), _mem_county_rate[96][t]->getval(), _mem_county_rate[97][t]->getval(), _mem_county_rate[98][t]->getval(), _mem_county_rate[99][t]->getval(), _mem_county_rate[100][t]->getval(), _mem_county_rate[101][t]->getval(), _mem_county_rate[102][t]->getval(), _mem_county_rate[103][t]->getval(), _mem_county_rate[104][t]->getval(), _mem_county_rate[105][t]->getval(), _mem_county_rate[106][t]->getval(), _mem_county_rate[107][t]->getval(), _mem_county_rate[108][t]->getval(), _mem_county_rate[109][t]->getval(), _mem_county_rate[110][t]->getval(), _mem_county_rate[111][t]->getval(), _mem_county_rate[112][t]->getval(), _mem_county_rate[113][t]->getval(), _mem_county_rate[114][t]->getval(), _mem_county_rate[115][t]->getval(), _mem_county_rate[116][t]->getval(), _mem_county_rate[117][t]->getval(), _mem_county_rate[118][t]->getval(), _mem_county_rate[119][t]->getval(), _mem_county_rate[120][t]->getval(), _mem_county_rate[121][t]->getval(), _mem_county_rate[122][t]->getval(), _mem_county_rate[123][t]->getval(), _mem_county_rate[124][t]->getval(), _mem_county_rate[125][t]->getval(), _mem_county_rate[126][t]->getval(), _mem_county_rate[127][t]->getval(), _mem_county_rate[128][t]->getval(), _mem_county_rate[129][t]->getval(), _mem_county_rate[130][t]->getval(), _mem_county_rate[131][t]->getval(), _mem_county_rate[132][t]->getval(), _mem_county_rate[133][t]->getval(), _mem_county_rate[134][t]->getval(), _mem_county_rate[135][t]->getval(), _mem_county_rate[136][t]->getval(), _mem_county_rate[137][t]->getval(), _mem_county_rate[138][t]->getval(), _mem_county_rate[139][t]->getval(), _mem_county_rate[140][t]->getval(), _mem_county_rate[141][t]->getval(), _mem_county_rate[142][t]->getval(), _mem_county_rate[143][t]->getval(), _mem_county_rate[144][t]->getval(), _mem_county_rate[145][t]->getval(), _mem_county_rate[146][t]->getval(), _mem_county_rate[147][t]->getval(), _mem_county_rate[148][t]->getval(), _mem_county_rate[149][t]->getval(), _mem_county_rate[150][t]->getval(), _mem_county_rate[151][t]->getval(), _mem_county_rate[152][t]->getval(), _mem_county_rate[153][t]->getval(), _mem_county_rate[154][t]->getval(), _mem_county_rate[155][t]->getval(), _mem_county_rate[156][t]->getval(), _mem_county_rate[157][t]->getval(), _mem_county_rate[158][t]->getval(), _mem_county_rate[159][t]->getval(), _mem_county_rate[160][t]->getval(), _mem_county_rate[161][t]->getval(), _mem_county_rate[162][t]->getval(), _mem_county_rate[163][t]->getval(), _mem_county_rate[164][t]->getval(), _mem_county_rate[165][t]->getval(), _mem_county_rate[166][t]->getval(), _mem_county_rate[167][t]->getval(), _mem_county_rate[168][t]->getval(), _mem_county_rate[169][t]->getval(), _mem_county_rate[170][t]->getval(), _mem_county_rate[171][t]->getval(), _mem_county_rate[172][t]->getval(), _mem_county_rate[173][t]->getval(), _mem_county_rate[174][t]->getval(), _mem_county_rate[175][t]->getval(), _mem_county_rate[176][t]->getval(), _mem_county_rate[177][t]->getval(), _mem_county_rate[178][t]->getval(), _mem_county_rate[179][t]->getval(), _mem_county_rate[180][t]->getval(), _mem_county_rate[181][t]->getval(), _mem_county_rate[182][t]->getval(), _mem_county_rate[183][t]->getval(), _mem_county_rate[184][t]->getval(), _mem_county_rate[185][t]->getval(), _mem_county_rate[186][t]->getval(), _mem_county_rate[187][t]->getval(), _mem_county_rate[188][t]->getval(), _mem_county_rate[189][t]->getval(), _mem_county_rate[190][t]->getval(), _mem_county_rate[191][t]->getval(), _mem_county_rate[192][t]->getval(), _mem_county_rate[193][t]->getval(), _mem_county_rate[194][t]->getval(), _mem_county_rate[195][t]->getval(), _mem_county_rate[196][t]->getval(), _mem_county_rate[197][t]->getval(), _mem_county_rate[198][t]->getval(), _mem_county_rate[199][t]->getval(), _mem_county_rate[200][t]->getval(), _mem_county_rate[201][t]->getval(), _mem_county_rate[202][t]->getval(), _mem_county_rate[203][t]->getval(), _mem_county_rate[204][t]->getval(), _mem_county_rate[205][t]->getval(), _mem_county_rate[206][t]->getval(), _mem_county_rate[207][t]->getval(), _mem_county_rate[208][t]->getval(), _mem_county_rate[209][t]->getval(), _mem_county_rate[210][t]->getval(), _mem_county_rate[211][t]->getval(), _mem_county_rate[212][t]->getval(), _mem_county_rate[213][t]->getval(), _mem_county_rate[214][t]->getval(), _mem_county_rate[215][t]->getval(), _mem_county_rate[216][t]->getval(), _mem_county_rate[217][t]->getval(), _mem_county_rate[218][t]->getval(), _mem_county_rate[219][t]->getval(), _mem_county_rate[220][t]->getval(), _mem_county_rate[221][t]->getval(), _mem_county_rate[222][t]->getval(), _mem_county_rate[223][t]->getval(), _mem_county_rate[224][t]->getval(), _mem_county_rate[225][t]->getval(), _mem_county_rate[226][t]->getval(), _mem_county_rate[227][t]->getval(), _mem_county_rate[228][t]->getval(), _mem_county_rate[229][t]->getval(), _mem_county_rate[230][t]->getval(), _mem_county_rate[231][t]->getval(), _mem_county_rate[232][t]->getval(), _mem_county_rate[233][t]->getval(), _mem_county_rate[234][t]->getval(), _mem_county_rate[235][t]->getval(), _mem_county_rate[236][t]->getval(), _mem_county_rate[237][t]->getval(), _mem_county_rate[238][t]->getval(), _mem_county_rate[239][t]->getval(), _mem_county_rate[240][t]->getval(), _mem_county_rate[241][t]->getval(), _mem_county_rate[242][t]->getval(), _mem_county_rate[243][t]->getval(), _mem_county_rate[244][t]->getval(), _mem_county_rate[245][t]->getval(), _mem_county_rate[246][t]->getval(), _mem_county_rate[247][t]->getval(), _mem_county_rate[248][t]->getval(), _mem_county_rate[249][t]->getval(), _mem_county_rate[250][t]->getval(), _mem_county_rate[251][t]->getval(), _mem_county_rate[252][t]->getval(), _mem_county_rate[253][t]->getval(), _mem_county_rate[254][t]->getval(), _mem_county_rate[255][t]->getval(), _mem_county_rate[256][t]->getval(), _mem_county_rate[257][t]->getval(), _mem_county_rate[258][t]->getval(), _mem_county_rate[259][t]->getval(), _mem_county_rate[260][t]->getval(), _mem_county_rate[261][t]->getval(), _mem_county_rate[262][t]->getval(), _mem_county_rate[263][t]->getval(), _mem_county_rate[264][t]->getval(), _mem_county_rate[265][t]->getval(), _mem_county_rate[266][t]->getval(), _mem_county_rate[267][t]->getval(), _mem_county_rate[268][t]->getval(), _mem_county_rate[269][t]->getval(), _mem_county_rate[270][t]->getval(), _mem_county_rate[271][t]->getval(), _mem_county_rate[272][t]->getval(), _mem_county_rate[273][t]->getval(), _mem_county_rate[274][t]->getval(), _mem_county_rate[275][t]->getval(), _mem_county_rate[276][t]->getval()}))/__fixed_region_pop[r],0.05000000),Gaussian35791840.loglikeli(val);
}
double _Var_region_rate::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian35791840.init(dot(__fixed_county_map.row(r),vstack({_mem_county_rate[0][t]->getcache(), _mem_county_rate[1][t]->getcache(), _mem_county_rate[2][t]->getcache(), _mem_county_rate[3][t]->getcache(), _mem_county_rate[4][t]->getcache(), _mem_county_rate[5][t]->getcache(), _mem_county_rate[6][t]->getcache(), _mem_county_rate[7][t]->getcache(), _mem_county_rate[8][t]->getcache(), _mem_county_rate[9][t]->getcache(), _mem_county_rate[10][t]->getcache(), _mem_county_rate[11][t]->getcache(), _mem_county_rate[12][t]->getcache(), _mem_county_rate[13][t]->getcache(), _mem_county_rate[14][t]->getcache(), _mem_county_rate[15][t]->getcache(), _mem_county_rate[16][t]->getcache(), _mem_county_rate[17][t]->getcache(), _mem_county_rate[18][t]->getcache(), _mem_county_rate[19][t]->getcache(), _mem_county_rate[20][t]->getcache(), _mem_county_rate[21][t]->getcache(), _mem_county_rate[22][t]->getcache(), _mem_county_rate[23][t]->getcache(), _mem_county_rate[24][t]->getcache(), _mem_county_rate[25][t]->getcache(), _mem_county_rate[26][t]->getcache(), _mem_county_rate[27][t]->getcache(), _mem_county_rate[28][t]->getcache(), _mem_county_rate[29][t]->getcache(), _mem_county_rate[30][t]->getcache(), _mem_county_rate[31][t]->getcache(), _mem_county_rate[32][t]->getcache(), _mem_county_rate[33][t]->getcache(), _mem_county_rate[34][t]->getcache(), _mem_county_rate[35][t]->getcache(), _mem_county_rate[36][t]->getcache(), _mem_county_rate[37][t]->getcache(), _mem_county_rate[38][t]->getcache(), _mem_county_rate[39][t]->getcache(), _mem_county_rate[40][t]->getcache(), _mem_county_rate[41][t]->getcache(), _mem_county_rate[42][t]->getcache(), _mem_county_rate[43][t]->getcache(), _mem_county_rate[44][t]->getcache(), _mem_county_rate[45][t]->getcache(), _mem_county_rate[46][t]->getcache(), _mem_county_rate[47][t]->getcache(), _mem_county_rate[48][t]->getcache(), _mem_county_rate[49][t]->getcache(), _mem_county_rate[50][t]->getcache(), _mem_county_rate[51][t]->getcache(), _mem_county_rate[52][t]->getcache(), _mem_county_rate[53][t]->getcache(), _mem_county_rate[54][t]->getcache(), _mem_county_rate[55][t]->getcache(), _mem_county_rate[56][t]->getcache(), _mem_county_rate[57][t]->getcache(), _mem_county_rate[58][t]->getcache(), _mem_county_rate[59][t]->getcache(), _mem_county_rate[60][t]->getcache(), _mem_county_rate[61][t]->getcache(), _mem_county_rate[62][t]->getcache(), _mem_county_rate[63][t]->getcache(), _mem_county_rate[64][t]->getcache(), _mem_county_rate[65][t]->getcache(), _mem_county_rate[66][t]->getcache(), _mem_county_rate[67][t]->getcache(), _mem_county_rate[68][t]->getcache(), _mem_county_rate[69][t]->getcache(), _mem_county_rate[70][t]->getcache(), _mem_county_rate[71][t]->getcache(), _mem_county_rate[72][t]->getcache(), _mem_county_rate[73][t]->getcache(), _mem_county_rate[74][t]->getcache(), _mem_county_rate[75][t]->getcache(), _mem_county_rate[76][t]->getcache(), _mem_county_rate[77][t]->getcache(), _mem_county_rate[78][t]->getcache(), _mem_county_rate[79][t]->getcache(), _mem_county_rate[80][t]->getcache(), _mem_county_rate[81][t]->getcache(), _mem_county_rate[82][t]->getcache(), _mem_county_rate[83][t]->getcache(), _mem_county_rate[84][t]->getcache(), _mem_county_rate[85][t]->getcache(), _mem_county_rate[86][t]->getcache(), _mem_county_rate[87][t]->getcache(), _mem_county_rate[88][t]->getcache(), _mem_county_rate[89][t]->getcache(), _mem_county_rate[90][t]->getcache(), _mem_county_rate[91][t]->getcache(), _mem_county_rate[92][t]->getcache(), _mem_county_rate[93][t]->getcache(), _mem_county_rate[94][t]->getcache(), _mem_county_rate[95][t]->getcache(), _mem_county_rate[96][t]->getcache(), _mem_county_rate[97][t]->getcache(), _mem_county_rate[98][t]->getcache(), _mem_county_rate[99][t]->getcache(), _mem_county_rate[100][t]->getcache(), _mem_county_rate[101][t]->getcache(), _mem_county_rate[102][t]->getcache(), _mem_county_rate[103][t]->getcache(), _mem_county_rate[104][t]->getcache(), _mem_county_rate[105][t]->getcache(), _mem_county_rate[106][t]->getcache(), _mem_county_rate[107][t]->getcache(), _mem_county_rate[108][t]->getcache(), _mem_county_rate[109][t]->getcache(), _mem_county_rate[110][t]->getcache(), _mem_county_rate[111][t]->getcache(), _mem_county_rate[112][t]->getcache(), _mem_county_rate[113][t]->getcache(), _mem_county_rate[114][t]->getcache(), _mem_county_rate[115][t]->getcache(), _mem_county_rate[116][t]->getcache(), _mem_county_rate[117][t]->getcache(), _mem_county_rate[118][t]->getcache(), _mem_county_rate[119][t]->getcache(), _mem_county_rate[120][t]->getcache(), _mem_county_rate[121][t]->getcache(), _mem_county_rate[122][t]->getcache(), _mem_county_rate[123][t]->getcache(), _mem_county_rate[124][t]->getcache(), _mem_county_rate[125][t]->getcache(), _mem_county_rate[126][t]->getcache(), _mem_county_rate[127][t]->getcache(), _mem_county_rate[128][t]->getcache(), _mem_county_rate[129][t]->getcache(), _mem_county_rate[130][t]->getcache(), _mem_county_rate[131][t]->getcache(), _mem_county_rate[132][t]->getcache(), _mem_county_rate[133][t]->getcache(), _mem_county_rate[134][t]->getcache(), _mem_county_rate[135][t]->getcache(), _mem_county_rate[136][t]->getcache(), _mem_county_rate[137][t]->getcache(), _mem_county_rate[138][t]->getcache(), _mem_county_rate[139][t]->getcache(), _mem_county_rate[140][t]->getcache(), _mem_county_rate[141][t]->getcache(), _mem_county_rate[142][t]->getcache(), _mem_county_rate[143][t]->getcache(), _mem_county_rate[144][t]->getcache(), _mem_county_rate[145][t]->getcache(), _mem_county_rate[146][t]->getcache(), _mem_county_rate[147][t]->getcache(), _mem_county_rate[148][t]->getcache(), _mem_county_rate[149][t]->getcache(), _mem_county_rate[150][t]->getcache(), _mem_county_rate[151][t]->getcache(), _mem_county_rate[152][t]->getcache(), _mem_county_rate[153][t]->getcache(), _mem_county_rate[154][t]->getcache(), _mem_county_rate[155][t]->getcache(), _mem_county_rate[156][t]->getcache(), _mem_county_rate[157][t]->getcache(), _mem_county_rate[158][t]->getcache(), _mem_county_rate[159][t]->getcache(), _mem_county_rate[160][t]->getcache(), _mem_county_rate[161][t]->getcache(), _mem_county_rate[162][t]->getcache(), _mem_county_rate[163][t]->getcache(), _mem_county_rate[164][t]->getcache(), _mem_county_rate[165][t]->getcache(), _mem_county_rate[166][t]->getcache(), _mem_county_rate[167][t]->getcache(), _mem_county_rate[168][t]->getcache(), _mem_county_rate[169][t]->getcache(), _mem_county_rate[170][t]->getcache(), _mem_county_rate[171][t]->getcache(), _mem_county_rate[172][t]->getcache(), _mem_county_rate[173][t]->getcache(), _mem_county_rate[174][t]->getcache(), _mem_county_rate[175][t]->getcache(), _mem_county_rate[176][t]->getcache(), _mem_county_rate[177][t]->getcache(), _mem_county_rate[178][t]->getcache(), _mem_county_rate[179][t]->getcache(), _mem_county_rate[180][t]->getcache(), _mem_county_rate[181][t]->getcache(), _mem_county_rate[182][t]->getcache(), _mem_county_rate[183][t]->getcache(), _mem_county_rate[184][t]->getcache(), _mem_county_rate[185][t]->getcache(), _mem_county_rate[186][t]->getcache(), _mem_county_rate[187][t]->getcache(), _mem_county_rate[188][t]->getcache(), _mem_county_rate[189][t]->getcache(), _mem_county_rate[190][t]->getcache(), _mem_county_rate[191][t]->getcache(), _mem_county_rate[192][t]->getcache(), _mem_county_rate[193][t]->getcache(), _mem_county_rate[194][t]->getcache(), _mem_county_rate[195][t]->getcache(), _mem_county_rate[196][t]->getcache(), _mem_county_rate[197][t]->getcache(), _mem_county_rate[198][t]->getcache(), _mem_county_rate[199][t]->getcache(), _mem_county_rate[200][t]->getcache(), _mem_county_rate[201][t]->getcache(), _mem_county_rate[202][t]->getcache(), _mem_county_rate[203][t]->getcache(), _mem_county_rate[204][t]->getcache(), _mem_county_rate[205][t]->getcache(), _mem_county_rate[206][t]->getcache(), _mem_county_rate[207][t]->getcache(), _mem_county_rate[208][t]->getcache(), _mem_county_rate[209][t]->getcache(), _mem_county_rate[210][t]->getcache(), _mem_county_rate[211][t]->getcache(), _mem_county_rate[212][t]->getcache(), _mem_county_rate[213][t]->getcache(), _mem_county_rate[214][t]->getcache(), _mem_county_rate[215][t]->getcache(), _mem_county_rate[216][t]->getcache(), _mem_county_rate[217][t]->getcache(), _mem_county_rate[218][t]->getcache(), _mem_county_rate[219][t]->getcache(), _mem_county_rate[220][t]->getcache(), _mem_county_rate[221][t]->getcache(), _mem_county_rate[222][t]->getcache(), _mem_county_rate[223][t]->getcache(), _mem_county_rate[224][t]->getcache(), _mem_county_rate[225][t]->getcache(), _mem_county_rate[226][t]->getcache(), _mem_county_rate[227][t]->getcache(), _mem_county_rate[228][t]->getcache(), _mem_county_rate[229][t]->getcache(), _mem_county_rate[230][t]->getcache(), _mem_county_rate[231][t]->getcache(), _mem_county_rate[232][t]->getcache(), _mem_county_rate[233][t]->getcache(), _mem_county_rate[234][t]->getcache(), _mem_county_rate[235][t]->getcache(), _mem_county_rate[236][t]->getcache(), _mem_county_rate[237][t]->getcache(), _mem_county_rate[238][t]->getcache(), _mem_county_rate[239][t]->getcache(), _mem_county_rate[240][t]->getcache(), _mem_county_rate[241][t]->getcache(), _mem_county_rate[242][t]->getcache(), _mem_county_rate[243][t]->getcache(), _mem_county_rate[244][t]->getcache(), _mem_county_rate[245][t]->getcache(), _mem_county_rate[246][t]->getcache(), _mem_county_rate[247][t]->getcache(), _mem_county_rate[248][t]->getcache(), _mem_county_rate[249][t]->getcache(), _mem_county_rate[250][t]->getcache(), _mem_county_rate[251][t]->getcache(), _mem_county_rate[252][t]->getcache(), _mem_county_rate[253][t]->getcache(), _mem_county_rate[254][t]->getcache(), _mem_county_rate[255][t]->getcache(), _mem_county_rate[256][t]->getcache(), _mem_county_rate[257][t]->getcache(), _mem_county_rate[258][t]->getcache(), _mem_county_rate[259][t]->getcache(), _mem_county_rate[260][t]->getcache(), _mem_county_rate[261][t]->getcache(), _mem_county_rate[262][t]->getcache(), _mem_county_rate[263][t]->getcache(), _mem_county_rate[264][t]->getcache(), _mem_county_rate[265][t]->getcache(), _mem_county_rate[266][t]->getcache(), _mem_county_rate[267][t]->getcache(), _mem_county_rate[268][t]->getcache(), _mem_county_rate[269][t]->getcache(), _mem_county_rate[270][t]->getcache(), _mem_county_rate[271][t]->getcache(), _mem_county_rate[272][t]->getcache(), _mem_county_rate[273][t]->getcache(), _mem_county_rate[274][t]->getcache(), _mem_county_rate[275][t]->getcache(), _mem_county_rate[276][t]->getcache()}))/__fixed_region_pop[r],0.05000000),Gaussian35791840.loglikeli(_t_val);
}
void _Var_region_rate::sample()
{
  val=(Gaussian35791840.init(dot(__fixed_county_map.row(r),vstack({_mem_county_rate[0][t]->getval(), _mem_county_rate[1][t]->getval(), _mem_county_rate[2][t]->getval(), _mem_county_rate[3][t]->getval(), _mem_county_rate[4][t]->getval(), _mem_county_rate[5][t]->getval(), _mem_county_rate[6][t]->getval(), _mem_county_rate[7][t]->getval(), _mem_county_rate[8][t]->getval(), _mem_county_rate[9][t]->getval(), _mem_county_rate[10][t]->getval(), _mem_county_rate[11][t]->getval(), _mem_county_rate[12][t]->getval(), _mem_county_rate[13][t]->getval(), _mem_county_rate[14][t]->getval(), _mem_county_rate[15][t]->getval(), _mem_county_rate[16][t]->getval(), _mem_county_rate[17][t]->getval(), _mem_county_rate[18][t]->getval(), _mem_county_rate[19][t]->getval(), _mem_county_rate[20][t]->getval(), _mem_county_rate[21][t]->getval(), _mem_county_rate[22][t]->getval(), _mem_county_rate[23][t]->getval(), _mem_county_rate[24][t]->getval(), _mem_county_rate[25][t]->getval(), _mem_county_rate[26][t]->getval(), _mem_county_rate[27][t]->getval(), _mem_county_rate[28][t]->getval(), _mem_county_rate[29][t]->getval(), _mem_county_rate[30][t]->getval(), _mem_county_rate[31][t]->getval(), _mem_county_rate[32][t]->getval(), _mem_county_rate[33][t]->getval(), _mem_county_rate[34][t]->getval(), _mem_county_rate[35][t]->getval(), _mem_county_rate[36][t]->getval(), _mem_county_rate[37][t]->getval(), _mem_county_rate[38][t]->getval(), _mem_county_rate[39][t]->getval(), _mem_county_rate[40][t]->getval(), _mem_county_rate[41][t]->getval(), _mem_county_rate[42][t]->getval(), _mem_county_rate[43][t]->getval(), _mem_county_rate[44][t]->getval(), _mem_county_rate[45][t]->getval(), _mem_county_rate[46][t]->getval(), _mem_county_rate[47][t]->getval(), _mem_county_rate[48][t]->getval(), _mem_county_rate[49][t]->getval(), _mem_county_rate[50][t]->getval(), _mem_county_rate[51][t]->getval(), _mem_county_rate[52][t]->getval(), _mem_county_rate[53][t]->getval(), _mem_county_rate[54][t]->getval(), _mem_county_rate[55][t]->getval(), _mem_county_rate[56][t]->getval(), _mem_county_rate[57][t]->getval(), _mem_county_rate[58][t]->getval(), _mem_county_rate[59][t]->getval(), _mem_county_rate[60][t]->getval(), _mem_county_rate[61][t]->getval(), _mem_county_rate[62][t]->getval(), _mem_county_rate[63][t]->getval(), _mem_county_rate[64][t]->getval(), _mem_county_rate[65][t]->getval(), _mem_county_rate[66][t]->getval(), _mem_county_rate[67][t]->getval(), _mem_county_rate[68][t]->getval(), _mem_county_rate[69][t]->getval(), _mem_county_rate[70][t]->getval(), _mem_county_rate[71][t]->getval(), _mem_county_rate[72][t]->getval(), _mem_county_rate[73][t]->getval(), _mem_county_rate[74][t]->getval(), _mem_county_rate[75][t]->getval(), _mem_county_rate[76][t]->getval(), _mem_county_rate[77][t]->getval(), _mem_county_rate[78][t]->getval(), _mem_county_rate[79][t]->getval(), _mem_county_rate[80][t]->getval(), _mem_county_rate[81][t]->getval(), _mem_county_rate[82][t]->getval(), _mem_county_rate[83][t]->getval(), _mem_county_rate[84][t]->getval(), _mem_county_rate[85][t]->getval(), _mem_county_rate[86][t]->getval(), _mem_county_rate[87][t]->getval(), _mem_county_rate[88][t]->getval(), _mem_county_rate[89][t]->getval(), _mem_county_rate[90][t]->getval(), _mem_county_rate[91][t]->getval(), _mem_county_rate[92][t]->getval(), _mem_county_rate[93][t]->getval(), _mem_county_rate[94][t]->getval(), _mem_county_rate[95][t]->getval(), _mem_county_rate[96][t]->getval(), _mem_county_rate[97][t]->getval(), _mem_county_rate[98][t]->getval(), _mem_county_rate[99][t]->getval(), _mem_county_rate[100][t]->getval(), _mem_county_rate[101][t]->getval(), _mem_county_rate[102][t]->getval(), _mem_county_rate[103][t]->getval(), _mem_county_rate[104][t]->getval(), _mem_county_rate[105][t]->getval(), _mem_county_rate[106][t]->getval(), _mem_county_rate[107][t]->getval(), _mem_county_rate[108][t]->getval(), _mem_county_rate[109][t]->getval(), _mem_county_rate[110][t]->getval(), _mem_county_rate[111][t]->getval(), _mem_county_rate[112][t]->getval(), _mem_county_rate[113][t]->getval(), _mem_county_rate[114][t]->getval(), _mem_county_rate[115][t]->getval(), _mem_county_rate[116][t]->getval(), _mem_county_rate[117][t]->getval(), _mem_county_rate[118][t]->getval(), _mem_county_rate[119][t]->getval(), _mem_county_rate[120][t]->getval(), _mem_county_rate[121][t]->getval(), _mem_county_rate[122][t]->getval(), _mem_county_rate[123][t]->getval(), _mem_county_rate[124][t]->getval(), _mem_county_rate[125][t]->getval(), _mem_county_rate[126][t]->getval(), _mem_county_rate[127][t]->getval(), _mem_county_rate[128][t]->getval(), _mem_county_rate[129][t]->getval(), _mem_county_rate[130][t]->getval(), _mem_county_rate[131][t]->getval(), _mem_county_rate[132][t]->getval(), _mem_county_rate[133][t]->getval(), _mem_county_rate[134][t]->getval(), _mem_county_rate[135][t]->getval(), _mem_county_rate[136][t]->getval(), _mem_county_rate[137][t]->getval(), _mem_county_rate[138][t]->getval(), _mem_county_rate[139][t]->getval(), _mem_county_rate[140][t]->getval(), _mem_county_rate[141][t]->getval(), _mem_county_rate[142][t]->getval(), _mem_county_rate[143][t]->getval(), _mem_county_rate[144][t]->getval(), _mem_county_rate[145][t]->getval(), _mem_county_rate[146][t]->getval(), _mem_county_rate[147][t]->getval(), _mem_county_rate[148][t]->getval(), _mem_county_rate[149][t]->getval(), _mem_county_rate[150][t]->getval(), _mem_county_rate[151][t]->getval(), _mem_county_rate[152][t]->getval(), _mem_county_rate[153][t]->getval(), _mem_county_rate[154][t]->getval(), _mem_county_rate[155][t]->getval(), _mem_county_rate[156][t]->getval(), _mem_county_rate[157][t]->getval(), _mem_county_rate[158][t]->getval(), _mem_county_rate[159][t]->getval(), _mem_county_rate[160][t]->getval(), _mem_county_rate[161][t]->getval(), _mem_county_rate[162][t]->getval(), _mem_county_rate[163][t]->getval(), _mem_county_rate[164][t]->getval(), _mem_county_rate[165][t]->getval(), _mem_county_rate[166][t]->getval(), _mem_county_rate[167][t]->getval(), _mem_county_rate[168][t]->getval(), _mem_county_rate[169][t]->getval(), _mem_county_rate[170][t]->getval(), _mem_county_rate[171][t]->getval(), _mem_county_rate[172][t]->getval(), _mem_county_rate[173][t]->getval(), _mem_county_rate[174][t]->getval(), _mem_county_rate[175][t]->getval(), _mem_county_rate[176][t]->getval(), _mem_county_rate[177][t]->getval(), _mem_county_rate[178][t]->getval(), _mem_county_rate[179][t]->getval(), _mem_county_rate[180][t]->getval(), _mem_county_rate[181][t]->getval(), _mem_county_rate[182][t]->getval(), _mem_county_rate[183][t]->getval(), _mem_county_rate[184][t]->getval(), _mem_county_rate[185][t]->getval(), _mem_county_rate[186][t]->getval(), _mem_county_rate[187][t]->getval(), _mem_county_rate[188][t]->getval(), _mem_county_rate[189][t]->getval(), _mem_county_rate[190][t]->getval(), _mem_county_rate[191][t]->getval(), _mem_county_rate[192][t]->getval(), _mem_county_rate[193][t]->getval(), _mem_county_rate[194][t]->getval(), _mem_county_rate[195][t]->getval(), _mem_county_rate[196][t]->getval(), _mem_county_rate[197][t]->getval(), _mem_county_rate[198][t]->getval(), _mem_county_rate[199][t]->getval(), _mem_county_rate[200][t]->getval(), _mem_county_rate[201][t]->getval(), _mem_county_rate[202][t]->getval(), _mem_county_rate[203][t]->getval(), _mem_county_rate[204][t]->getval(), _mem_county_rate[205][t]->getval(), _mem_county_rate[206][t]->getval(), _mem_county_rate[207][t]->getval(), _mem_county_rate[208][t]->getval(), _mem_county_rate[209][t]->getval(), _mem_county_rate[210][t]->getval(), _mem_county_rate[211][t]->getval(), _mem_county_rate[212][t]->getval(), _mem_county_rate[213][t]->getval(), _mem_county_rate[214][t]->getval(), _mem_county_rate[215][t]->getval(), _mem_county_rate[216][t]->getval(), _mem_county_rate[217][t]->getval(), _mem_county_rate[218][t]->getval(), _mem_county_rate[219][t]->getval(), _mem_county_rate[220][t]->getval(), _mem_county_rate[221][t]->getval(), _mem_county_rate[222][t]->getval(), _mem_county_rate[223][t]->getval(), _mem_county_rate[224][t]->getval(), _mem_county_rate[225][t]->getval(), _mem_county_rate[226][t]->getval(), _mem_county_rate[227][t]->getval(), _mem_county_rate[228][t]->getval(), _mem_county_rate[229][t]->getval(), _mem_county_rate[230][t]->getval(), _mem_county_rate[231][t]->getval(), _mem_county_rate[232][t]->getval(), _mem_county_rate[233][t]->getval(), _mem_county_rate[234][t]->getval(), _mem_county_rate[235][t]->getval(), _mem_county_rate[236][t]->getval(), _mem_county_rate[237][t]->getval(), _mem_county_rate[238][t]->getval(), _mem_county_rate[239][t]->getval(), _mem_county_rate[240][t]->getval(), _mem_county_rate[241][t]->getval(), _mem_county_rate[242][t]->getval(), _mem_county_rate[243][t]->getval(), _mem_county_rate[244][t]->getval(), _mem_county_rate[245][t]->getval(), _mem_county_rate[246][t]->getval(), _mem_county_rate[247][t]->getval(), _mem_county_rate[248][t]->getval(), _mem_county_rate[249][t]->getval(), _mem_county_rate[250][t]->getval(), _mem_county_rate[251][t]->getval(), _mem_county_rate[252][t]->getval(), _mem_county_rate[253][t]->getval(), _mem_county_rate[254][t]->getval(), _mem_county_rate[255][t]->getval(), _mem_county_rate[256][t]->getval(), _mem_county_rate[257][t]->getval(), _mem_county_rate[258][t]->getval(), _mem_county_rate[259][t]->getval(), _mem_county_rate[260][t]->getval(), _mem_county_rate[261][t]->getval(), _mem_county_rate[262][t]->getval(), _mem_county_rate[263][t]->getval(), _mem_county_rate[264][t]->getval(), _mem_county_rate[265][t]->getval(), _mem_county_rate[266][t]->getval(), _mem_county_rate[267][t]->getval(), _mem_county_rate[268][t]->getval(), _mem_county_rate[269][t]->getval(), _mem_county_rate[270][t]->getval(), _mem_county_rate[271][t]->getval(), _mem_county_rate[272][t]->getval(), _mem_county_rate[273][t]->getval(), _mem_county_rate[274][t]->getval(), _mem_county_rate[275][t]->getval(), _mem_county_rate[276][t]->getval()}))/__fixed_region_pop[r],0.05000000),Gaussian35791840.gen());
}
void _Var_region_rate::sample_cache()
{
  cache_val=(Gaussian35791840.init(dot(__fixed_county_map.row(r),vstack({_mem_county_rate[0][t]->getcache(), _mem_county_rate[1][t]->getcache(), _mem_county_rate[2][t]->getcache(), _mem_county_rate[3][t]->getcache(), _mem_county_rate[4][t]->getcache(), _mem_county_rate[5][t]->getcache(), _mem_county_rate[6][t]->getcache(), _mem_county_rate[7][t]->getcache(), _mem_county_rate[8][t]->getcache(), _mem_county_rate[9][t]->getcache(), _mem_county_rate[10][t]->getcache(), _mem_county_rate[11][t]->getcache(), _mem_county_rate[12][t]->getcache(), _mem_county_rate[13][t]->getcache(), _mem_county_rate[14][t]->getcache(), _mem_county_rate[15][t]->getcache(), _mem_county_rate[16][t]->getcache(), _mem_county_rate[17][t]->getcache(), _mem_county_rate[18][t]->getcache(), _mem_county_rate[19][t]->getcache(), _mem_county_rate[20][t]->getcache(), _mem_county_rate[21][t]->getcache(), _mem_county_rate[22][t]->getcache(), _mem_county_rate[23][t]->getcache(), _mem_county_rate[24][t]->getcache(), _mem_county_rate[25][t]->getcache(), _mem_county_rate[26][t]->getcache(), _mem_county_rate[27][t]->getcache(), _mem_county_rate[28][t]->getcache(), _mem_county_rate[29][t]->getcache(), _mem_county_rate[30][t]->getcache(), _mem_county_rate[31][t]->getcache(), _mem_county_rate[32][t]->getcache(), _mem_county_rate[33][t]->getcache(), _mem_county_rate[34][t]->getcache(), _mem_county_rate[35][t]->getcache(), _mem_county_rate[36][t]->getcache(), _mem_county_rate[37][t]->getcache(), _mem_county_rate[38][t]->getcache(), _mem_county_rate[39][t]->getcache(), _mem_county_rate[40][t]->getcache(), _mem_county_rate[41][t]->getcache(), _mem_county_rate[42][t]->getcache(), _mem_county_rate[43][t]->getcache(), _mem_county_rate[44][t]->getcache(), _mem_county_rate[45][t]->getcache(), _mem_county_rate[46][t]->getcache(), _mem_county_rate[47][t]->getcache(), _mem_county_rate[48][t]->getcache(), _mem_county_rate[49][t]->getcache(), _mem_county_rate[50][t]->getcache(), _mem_county_rate[51][t]->getcache(), _mem_county_rate[52][t]->getcache(), _mem_county_rate[53][t]->getcache(), _mem_county_rate[54][t]->getcache(), _mem_county_rate[55][t]->getcache(), _mem_county_rate[56][t]->getcache(), _mem_county_rate[57][t]->getcache(), _mem_county_rate[58][t]->getcache(), _mem_county_rate[59][t]->getcache(), _mem_county_rate[60][t]->getcache(), _mem_county_rate[61][t]->getcache(), _mem_county_rate[62][t]->getcache(), _mem_county_rate[63][t]->getcache(), _mem_county_rate[64][t]->getcache(), _mem_county_rate[65][t]->getcache(), _mem_county_rate[66][t]->getcache(), _mem_county_rate[67][t]->getcache(), _mem_county_rate[68][t]->getcache(), _mem_county_rate[69][t]->getcache(), _mem_county_rate[70][t]->getcache(), _mem_county_rate[71][t]->getcache(), _mem_county_rate[72][t]->getcache(), _mem_county_rate[73][t]->getcache(), _mem_county_rate[74][t]->getcache(), _mem_county_rate[75][t]->getcache(), _mem_county_rate[76][t]->getcache(), _mem_county_rate[77][t]->getcache(), _mem_county_rate[78][t]->getcache(), _mem_county_rate[79][t]->getcache(), _mem_county_rate[80][t]->getcache(), _mem_county_rate[81][t]->getcache(), _mem_county_rate[82][t]->getcache(), _mem_county_rate[83][t]->getcache(), _mem_county_rate[84][t]->getcache(), _mem_county_rate[85][t]->getcache(), _mem_county_rate[86][t]->getcache(), _mem_county_rate[87][t]->getcache(), _mem_county_rate[88][t]->getcache(), _mem_county_rate[89][t]->getcache(), _mem_county_rate[90][t]->getcache(), _mem_county_rate[91][t]->getcache(), _mem_county_rate[92][t]->getcache(), _mem_county_rate[93][t]->getcache(), _mem_county_rate[94][t]->getcache(), _mem_county_rate[95][t]->getcache(), _mem_county_rate[96][t]->getcache(), _mem_county_rate[97][t]->getcache(), _mem_county_rate[98][t]->getcache(), _mem_county_rate[99][t]->getcache(), _mem_county_rate[100][t]->getcache(), _mem_county_rate[101][t]->getcache(), _mem_county_rate[102][t]->getcache(), _mem_county_rate[103][t]->getcache(), _mem_county_rate[104][t]->getcache(), _mem_county_rate[105][t]->getcache(), _mem_county_rate[106][t]->getcache(), _mem_county_rate[107][t]->getcache(), _mem_county_rate[108][t]->getcache(), _mem_county_rate[109][t]->getcache(), _mem_county_rate[110][t]->getcache(), _mem_county_rate[111][t]->getcache(), _mem_county_rate[112][t]->getcache(), _mem_county_rate[113][t]->getcache(), _mem_county_rate[114][t]->getcache(), _mem_county_rate[115][t]->getcache(), _mem_county_rate[116][t]->getcache(), _mem_county_rate[117][t]->getcache(), _mem_county_rate[118][t]->getcache(), _mem_county_rate[119][t]->getcache(), _mem_county_rate[120][t]->getcache(), _mem_county_rate[121][t]->getcache(), _mem_county_rate[122][t]->getcache(), _mem_county_rate[123][t]->getcache(), _mem_county_rate[124][t]->getcache(), _mem_county_rate[125][t]->getcache(), _mem_county_rate[126][t]->getcache(), _mem_county_rate[127][t]->getcache(), _mem_county_rate[128][t]->getcache(), _mem_county_rate[129][t]->getcache(), _mem_county_rate[130][t]->getcache(), _mem_county_rate[131][t]->getcache(), _mem_county_rate[132][t]->getcache(), _mem_county_rate[133][t]->getcache(), _mem_county_rate[134][t]->getcache(), _mem_county_rate[135][t]->getcache(), _mem_county_rate[136][t]->getcache(), _mem_county_rate[137][t]->getcache(), _mem_county_rate[138][t]->getcache(), _mem_county_rate[139][t]->getcache(), _mem_county_rate[140][t]->getcache(), _mem_county_rate[141][t]->getcache(), _mem_county_rate[142][t]->getcache(), _mem_county_rate[143][t]->getcache(), _mem_county_rate[144][t]->getcache(), _mem_county_rate[145][t]->getcache(), _mem_county_rate[146][t]->getcache(), _mem_county_rate[147][t]->getcache(), _mem_county_rate[148][t]->getcache(), _mem_county_rate[149][t]->getcache(), _mem_county_rate[150][t]->getcache(), _mem_county_rate[151][t]->getcache(), _mem_county_rate[152][t]->getcache(), _mem_county_rate[153][t]->getcache(), _mem_county_rate[154][t]->getcache(), _mem_county_rate[155][t]->getcache(), _mem_county_rate[156][t]->getcache(), _mem_county_rate[157][t]->getcache(), _mem_county_rate[158][t]->getcache(), _mem_county_rate[159][t]->getcache(), _mem_county_rate[160][t]->getcache(), _mem_county_rate[161][t]->getcache(), _mem_county_rate[162][t]->getcache(), _mem_county_rate[163][t]->getcache(), _mem_county_rate[164][t]->getcache(), _mem_county_rate[165][t]->getcache(), _mem_county_rate[166][t]->getcache(), _mem_county_rate[167][t]->getcache(), _mem_county_rate[168][t]->getcache(), _mem_county_rate[169][t]->getcache(), _mem_county_rate[170][t]->getcache(), _mem_county_rate[171][t]->getcache(), _mem_county_rate[172][t]->getcache(), _mem_county_rate[173][t]->getcache(), _mem_county_rate[174][t]->getcache(), _mem_county_rate[175][t]->getcache(), _mem_county_rate[176][t]->getcache(), _mem_county_rate[177][t]->getcache(), _mem_county_rate[178][t]->getcache(), _mem_county_rate[179][t]->getcache(), _mem_county_rate[180][t]->getcache(), _mem_county_rate[181][t]->getcache(), _mem_county_rate[182][t]->getcache(), _mem_county_rate[183][t]->getcache(), _mem_county_rate[184][t]->getcache(), _mem_county_rate[185][t]->getcache(), _mem_county_rate[186][t]->getcache(), _mem_county_rate[187][t]->getcache(), _mem_county_rate[188][t]->getcache(), _mem_county_rate[189][t]->getcache(), _mem_county_rate[190][t]->getcache(), _mem_county_rate[191][t]->getcache(), _mem_county_rate[192][t]->getcache(), _mem_county_rate[193][t]->getcache(), _mem_county_rate[194][t]->getcache(), _mem_county_rate[195][t]->getcache(), _mem_county_rate[196][t]->getcache(), _mem_county_rate[197][t]->getcache(), _mem_county_rate[198][t]->getcache(), _mem_county_rate[199][t]->getcache(), _mem_county_rate[200][t]->getcache(), _mem_county_rate[201][t]->getcache(), _mem_county_rate[202][t]->getcache(), _mem_county_rate[203][t]->getcache(), _mem_county_rate[204][t]->getcache(), _mem_county_rate[205][t]->getcache(), _mem_county_rate[206][t]->getcache(), _mem_county_rate[207][t]->getcache(), _mem_county_rate[208][t]->getcache(), _mem_county_rate[209][t]->getcache(), _mem_county_rate[210][t]->getcache(), _mem_county_rate[211][t]->getcache(), _mem_county_rate[212][t]->getcache(), _mem_county_rate[213][t]->getcache(), _mem_county_rate[214][t]->getcache(), _mem_county_rate[215][t]->getcache(), _mem_county_rate[216][t]->getcache(), _mem_county_rate[217][t]->getcache(), _mem_county_rate[218][t]->getcache(), _mem_county_rate[219][t]->getcache(), _mem_county_rate[220][t]->getcache(), _mem_county_rate[221][t]->getcache(), _mem_county_rate[222][t]->getcache(), _mem_county_rate[223][t]->getcache(), _mem_county_rate[224][t]->getcache(), _mem_county_rate[225][t]->getcache(), _mem_county_rate[226][t]->getcache(), _mem_county_rate[227][t]->getcache(), _mem_county_rate[228][t]->getcache(), _mem_county_rate[229][t]->getcache(), _mem_county_rate[230][t]->getcache(), _mem_county_rate[231][t]->getcache(), _mem_county_rate[232][t]->getcache(), _mem_county_rate[233][t]->getcache(), _mem_county_rate[234][t]->getcache(), _mem_county_rate[235][t]->getcache(), _mem_county_rate[236][t]->getcache(), _mem_county_rate[237][t]->getcache(), _mem_county_rate[238][t]->getcache(), _mem_county_rate[239][t]->getcache(), _mem_county_rate[240][t]->getcache(), _mem_county_rate[241][t]->getcache(), _mem_county_rate[242][t]->getcache(), _mem_county_rate[243][t]->getcache(), _mem_county_rate[244][t]->getcache(), _mem_county_rate[245][t]->getcache(), _mem_county_rate[246][t]->getcache(), _mem_county_rate[247][t]->getcache(), _mem_county_rate[248][t]->getcache(), _mem_county_rate[249][t]->getcache(), _mem_county_rate[250][t]->getcache(), _mem_county_rate[251][t]->getcache(), _mem_county_rate[252][t]->getcache(), _mem_county_rate[253][t]->getcache(), _mem_county_rate[254][t]->getcache(), _mem_county_rate[255][t]->getcache(), _mem_county_rate[256][t]->getcache(), _mem_county_rate[257][t]->getcache(), _mem_county_rate[258][t]->getcache(), _mem_county_rate[259][t]->getcache(), _mem_county_rate[260][t]->getcache(), _mem_county_rate[261][t]->getcache(), _mem_county_rate[262][t]->getcache(), _mem_county_rate[263][t]->getcache(), _mem_county_rate[264][t]->getcache(), _mem_county_rate[265][t]->getcache(), _mem_county_rate[266][t]->getcache(), _mem_county_rate[267][t]->getcache(), _mem_county_rate[268][t]->getcache(), _mem_county_rate[269][t]->getcache(), _mem_county_rate[270][t]->getcache(), _mem_county_rate[271][t]->getcache(), _mem_county_rate[272][t]->getcache(), _mem_county_rate[273][t]->getcache(), _mem_county_rate[274][t]->getcache(), _mem_county_rate[275][t]->getcache(), _mem_county_rate[276][t]->getcache()}))/__fixed_region_pop[r],0.05000000),Gaussian35791840.gen());
}
void _Var_region_rate::active_edge()
{
  _mem_county_rate[0][t]->add_contig(this);
  _mem_county_rate[100][t]->add_contig(this);
  _mem_county_rate[101][t]->add_contig(this);
  _mem_county_rate[102][t]->add_contig(this);
  _mem_county_rate[103][t]->add_contig(this);
  _mem_county_rate[104][t]->add_contig(this);
  _mem_county_rate[105][t]->add_contig(this);
  _mem_county_rate[106][t]->add_contig(this);
  _mem_county_rate[107][t]->add_contig(this);
  _mem_county_rate[108][t]->add_contig(this);
  _mem_county_rate[109][t]->add_contig(this);
  _mem_county_rate[10][t]->add_contig(this);
  _mem_county_rate[110][t]->add_contig(this);
  _mem_county_rate[111][t]->add_contig(this);
  _mem_county_rate[112][t]->add_contig(this);
  _mem_county_rate[113][t]->add_contig(this);
  _mem_county_rate[114][t]->add_contig(this);
  _mem_county_rate[115][t]->add_contig(this);
  _mem_county_rate[116][t]->add_contig(this);
  _mem_county_rate[117][t]->add_contig(this);
  _mem_county_rate[118][t]->add_contig(this);
  _mem_county_rate[119][t]->add_contig(this);
  _mem_county_rate[11][t]->add_contig(this);
  _mem_county_rate[120][t]->add_contig(this);
  _mem_county_rate[121][t]->add_contig(this);
  _mem_county_rate[122][t]->add_contig(this);
  _mem_county_rate[123][t]->add_contig(this);
  _mem_county_rate[124][t]->add_contig(this);
  _mem_county_rate[125][t]->add_contig(this);
  _mem_county_rate[126][t]->add_contig(this);
  _mem_county_rate[127][t]->add_contig(this);
  _mem_county_rate[128][t]->add_contig(this);
  _mem_county_rate[129][t]->add_contig(this);
  _mem_county_rate[12][t]->add_contig(this);
  _mem_county_rate[130][t]->add_contig(this);
  _mem_county_rate[131][t]->add_contig(this);
  _mem_county_rate[132][t]->add_contig(this);
  _mem_county_rate[133][t]->add_contig(this);
  _mem_county_rate[134][t]->add_contig(this);
  _mem_county_rate[135][t]->add_contig(this);
  _mem_county_rate[136][t]->add_contig(this);
  _mem_county_rate[137][t]->add_contig(this);
  _mem_county_rate[138][t]->add_contig(this);
  _mem_county_rate[139][t]->add_contig(this);
  _mem_county_rate[13][t]->add_contig(this);
  _mem_county_rate[140][t]->add_contig(this);
  _mem_county_rate[141][t]->add_contig(this);
  _mem_county_rate[142][t]->add_contig(this);
  _mem_county_rate[143][t]->add_contig(this);
  _mem_county_rate[144][t]->add_contig(this);
  _mem_county_rate[145][t]->add_contig(this);
  _mem_county_rate[146][t]->add_contig(this);
  _mem_county_rate[147][t]->add_contig(this);
  _mem_county_rate[148][t]->add_contig(this);
  _mem_county_rate[149][t]->add_contig(this);
  _mem_county_rate[14][t]->add_contig(this);
  _mem_county_rate[150][t]->add_contig(this);
  _mem_county_rate[151][t]->add_contig(this);
  _mem_county_rate[152][t]->add_contig(this);
  _mem_county_rate[153][t]->add_contig(this);
  _mem_county_rate[154][t]->add_contig(this);
  _mem_county_rate[155][t]->add_contig(this);
  _mem_county_rate[156][t]->add_contig(this);
  _mem_county_rate[157][t]->add_contig(this);
  _mem_county_rate[158][t]->add_contig(this);
  _mem_county_rate[159][t]->add_contig(this);
  _mem_county_rate[15][t]->add_contig(this);
  _mem_county_rate[160][t]->add_contig(this);
  _mem_county_rate[161][t]->add_contig(this);
  _mem_county_rate[162][t]->add_contig(this);
  _mem_county_rate[163][t]->add_contig(this);
  _mem_county_rate[164][t]->add_contig(this);
  _mem_county_rate[165][t]->add_contig(this);
  _mem_county_rate[166][t]->add_contig(this);
  _mem_county_rate[167][t]->add_contig(this);
  _mem_county_rate[168][t]->add_contig(this);
  _mem_county_rate[169][t]->add_contig(this);
  _mem_county_rate[16][t]->add_contig(this);
  _mem_county_rate[170][t]->add_contig(this);
  _mem_county_rate[171][t]->add_contig(this);
  _mem_county_rate[172][t]->add_contig(this);
  _mem_county_rate[173][t]->add_contig(this);
  _mem_county_rate[174][t]->add_contig(this);
  _mem_county_rate[175][t]->add_contig(this);
  _mem_county_rate[176][t]->add_contig(this);
  _mem_county_rate[177][t]->add_contig(this);
  _mem_county_rate[178][t]->add_contig(this);
  _mem_county_rate[179][t]->add_contig(this);
  _mem_county_rate[17][t]->add_contig(this);
  _mem_county_rate[180][t]->add_contig(this);
  _mem_county_rate[181][t]->add_contig(this);
  _mem_county_rate[182][t]->add_contig(this);
  _mem_county_rate[183][t]->add_contig(this);
  _mem_county_rate[184][t]->add_contig(this);
  _mem_county_rate[185][t]->add_contig(this);
  _mem_county_rate[186][t]->add_contig(this);
  _mem_county_rate[187][t]->add_contig(this);
  _mem_county_rate[188][t]->add_contig(this);
  _mem_county_rate[189][t]->add_contig(this);
  _mem_county_rate[18][t]->add_contig(this);
  _mem_county_rate[190][t]->add_contig(this);
  _mem_county_rate[191][t]->add_contig(this);
  _mem_county_rate[192][t]->add_contig(this);
  _mem_county_rate[193][t]->add_contig(this);
  _mem_county_rate[194][t]->add_contig(this);
  _mem_county_rate[195][t]->add_contig(this);
  _mem_county_rate[196][t]->add_contig(this);
  _mem_county_rate[197][t]->add_contig(this);
  _mem_county_rate[198][t]->add_contig(this);
  _mem_county_rate[199][t]->add_contig(this);
  _mem_county_rate[19][t]->add_contig(this);
  _mem_county_rate[1][t]->add_contig(this);
  _mem_county_rate[200][t]->add_contig(this);
  _mem_county_rate[201][t]->add_contig(this);
  _mem_county_rate[202][t]->add_contig(this);
  _mem_county_rate[203][t]->add_contig(this);
  _mem_county_rate[204][t]->add_contig(this);
  _mem_county_rate[205][t]->add_contig(this);
  _mem_county_rate[206][t]->add_contig(this);
  _mem_county_rate[207][t]->add_contig(this);
  _mem_county_rate[208][t]->add_contig(this);
  _mem_county_rate[209][t]->add_contig(this);
  _mem_county_rate[20][t]->add_contig(this);
  _mem_county_rate[210][t]->add_contig(this);
  _mem_county_rate[211][t]->add_contig(this);
  _mem_county_rate[212][t]->add_contig(this);
  _mem_county_rate[213][t]->add_contig(this);
  _mem_county_rate[214][t]->add_contig(this);
  _mem_county_rate[215][t]->add_contig(this);
  _mem_county_rate[216][t]->add_contig(this);
  _mem_county_rate[217][t]->add_contig(this);
  _mem_county_rate[218][t]->add_contig(this);
  _mem_county_rate[219][t]->add_contig(this);
  _mem_county_rate[21][t]->add_contig(this);
  _mem_county_rate[220][t]->add_contig(this);
  _mem_county_rate[221][t]->add_contig(this);
  _mem_county_rate[222][t]->add_contig(this);
  _mem_county_rate[223][t]->add_contig(this);
  _mem_county_rate[224][t]->add_contig(this);
  _mem_county_rate[225][t]->add_contig(this);
  _mem_county_rate[226][t]->add_contig(this);
  _mem_county_rate[227][t]->add_contig(this);
  _mem_county_rate[228][t]->add_contig(this);
  _mem_county_rate[229][t]->add_contig(this);
  _mem_county_rate[22][t]->add_contig(this);
  _mem_county_rate[230][t]->add_contig(this);
  _mem_county_rate[231][t]->add_contig(this);
  _mem_county_rate[232][t]->add_contig(this);
  _mem_county_rate[233][t]->add_contig(this);
  _mem_county_rate[234][t]->add_contig(this);
  _mem_county_rate[235][t]->add_contig(this);
  _mem_county_rate[236][t]->add_contig(this);
  _mem_county_rate[237][t]->add_contig(this);
  _mem_county_rate[238][t]->add_contig(this);
  _mem_county_rate[239][t]->add_contig(this);
  _mem_county_rate[23][t]->add_contig(this);
  _mem_county_rate[240][t]->add_contig(this);
  _mem_county_rate[241][t]->add_contig(this);
  _mem_county_rate[242][t]->add_contig(this);
  _mem_county_rate[243][t]->add_contig(this);
  _mem_county_rate[244][t]->add_contig(this);
  _mem_county_rate[245][t]->add_contig(this);
  _mem_county_rate[246][t]->add_contig(this);
  _mem_county_rate[247][t]->add_contig(this);
  _mem_county_rate[248][t]->add_contig(this);
  _mem_county_rate[249][t]->add_contig(this);
  _mem_county_rate[24][t]->add_contig(this);
  _mem_county_rate[250][t]->add_contig(this);
  _mem_county_rate[251][t]->add_contig(this);
  _mem_county_rate[252][t]->add_contig(this);
  _mem_county_rate[253][t]->add_contig(this);
  _mem_county_rate[254][t]->add_contig(this);
  _mem_county_rate[255][t]->add_contig(this);
  _mem_county_rate[256][t]->add_contig(this);
  _mem_county_rate[257][t]->add_contig(this);
  _mem_county_rate[258][t]->add_contig(this);
  _mem_county_rate[259][t]->add_contig(this);
  _mem_county_rate[25][t]->add_contig(this);
  _mem_county_rate[260][t]->add_contig(this);
  _mem_county_rate[261][t]->add_contig(this);
  _mem_county_rate[262][t]->add_contig(this);
  _mem_county_rate[263][t]->add_contig(this);
  _mem_county_rate[264][t]->add_contig(this);
  _mem_county_rate[265][t]->add_contig(this);
  _mem_county_rate[266][t]->add_contig(this);
  _mem_county_rate[267][t]->add_contig(this);
  _mem_county_rate[268][t]->add_contig(this);
  _mem_county_rate[269][t]->add_contig(this);
  _mem_county_rate[26][t]->add_contig(this);
  _mem_county_rate[270][t]->add_contig(this);
  _mem_county_rate[271][t]->add_contig(this);
  _mem_county_rate[272][t]->add_contig(this);
  _mem_county_rate[273][t]->add_contig(this);
  _mem_county_rate[274][t]->add_contig(this);
  _mem_county_rate[275][t]->add_contig(this);
  _mem_county_rate[276][t]->add_contig(this);
  _mem_county_rate[27][t]->add_contig(this);
  _mem_county_rate[28][t]->add_contig(this);
  _mem_county_rate[29][t]->add_contig(this);
  _mem_county_rate[2][t]->add_contig(this);
  _mem_county_rate[30][t]->add_contig(this);
  _mem_county_rate[31][t]->add_contig(this);
  _mem_county_rate[32][t]->add_contig(this);
  _mem_county_rate[33][t]->add_contig(this);
  _mem_county_rate[34][t]->add_contig(this);
  _mem_county_rate[35][t]->add_contig(this);
  _mem_county_rate[36][t]->add_contig(this);
  _mem_county_rate[37][t]->add_contig(this);
  _mem_county_rate[38][t]->add_contig(this);
  _mem_county_rate[39][t]->add_contig(this);
  _mem_county_rate[3][t]->add_contig(this);
  _mem_county_rate[40][t]->add_contig(this);
  _mem_county_rate[41][t]->add_contig(this);
  _mem_county_rate[42][t]->add_contig(this);
  _mem_county_rate[43][t]->add_contig(this);
  _mem_county_rate[44][t]->add_contig(this);
  _mem_county_rate[45][t]->add_contig(this);
  _mem_county_rate[46][t]->add_contig(this);
  _mem_county_rate[47][t]->add_contig(this);
  _mem_county_rate[48][t]->add_contig(this);
  _mem_county_rate[49][t]->add_contig(this);
  _mem_county_rate[4][t]->add_contig(this);
  _mem_county_rate[50][t]->add_contig(this);
  _mem_county_rate[51][t]->add_contig(this);
  _mem_county_rate[52][t]->add_contig(this);
  _mem_county_rate[53][t]->add_contig(this);
  _mem_county_rate[54][t]->add_contig(this);
  _mem_county_rate[55][t]->add_contig(this);
  _mem_county_rate[56][t]->add_contig(this);
  _mem_county_rate[57][t]->add_contig(this);
  _mem_county_rate[58][t]->add_contig(this);
  _mem_county_rate[59][t]->add_contig(this);
  _mem_county_rate[5][t]->add_contig(this);
  _mem_county_rate[60][t]->add_contig(this);
  _mem_county_rate[61][t]->add_contig(this);
  _mem_county_rate[62][t]->add_contig(this);
  _mem_county_rate[63][t]->add_contig(this);
  _mem_county_rate[64][t]->add_contig(this);
  _mem_county_rate[65][t]->add_contig(this);
  _mem_county_rate[66][t]->add_contig(this);
  _mem_county_rate[67][t]->add_contig(this);
  _mem_county_rate[68][t]->add_contig(this);
  _mem_county_rate[69][t]->add_contig(this);
  _mem_county_rate[6][t]->add_contig(this);
  _mem_county_rate[70][t]->add_contig(this);
  _mem_county_rate[71][t]->add_contig(this);
  _mem_county_rate[72][t]->add_contig(this);
  _mem_county_rate[73][t]->add_contig(this);
  _mem_county_rate[74][t]->add_contig(this);
  _mem_county_rate[75][t]->add_contig(this);
  _mem_county_rate[76][t]->add_contig(this);
  _mem_county_rate[77][t]->add_contig(this);
  _mem_county_rate[78][t]->add_contig(this);
  _mem_county_rate[79][t]->add_contig(this);
  _mem_county_rate[7][t]->add_contig(this);
  _mem_county_rate[80][t]->add_contig(this);
  _mem_county_rate[81][t]->add_contig(this);
  _mem_county_rate[82][t]->add_contig(this);
  _mem_county_rate[83][t]->add_contig(this);
  _mem_county_rate[84][t]->add_contig(this);
  _mem_county_rate[85][t]->add_contig(this);
  _mem_county_rate[86][t]->add_contig(this);
  _mem_county_rate[87][t]->add_contig(this);
  _mem_county_rate[88][t]->add_contig(this);
  _mem_county_rate[89][t]->add_contig(this);
  _mem_county_rate[8][t]->add_contig(this);
  _mem_county_rate[90][t]->add_contig(this);
  _mem_county_rate[91][t]->add_contig(this);
  _mem_county_rate[92][t]->add_contig(this);
  _mem_county_rate[93][t]->add_contig(this);
  _mem_county_rate[94][t]->add_contig(this);
  _mem_county_rate[95][t]->add_contig(this);
  _mem_county_rate[96][t]->add_contig(this);
  _mem_county_rate[97][t]->add_contig(this);
  _mem_county_rate[98][t]->add_contig(this);
  _mem_county_rate[99][t]->add_contig(this);
  _mem_county_rate[9][t]->add_contig(this);
  _mem_county_rate[0][t]->add_child(this);
  _mem_county_rate[100][t]->add_child(this);
  _mem_county_rate[101][t]->add_child(this);
  _mem_county_rate[102][t]->add_child(this);
  _mem_county_rate[103][t]->add_child(this);
  _mem_county_rate[104][t]->add_child(this);
  _mem_county_rate[105][t]->add_child(this);
  _mem_county_rate[106][t]->add_child(this);
  _mem_county_rate[107][t]->add_child(this);
  _mem_county_rate[108][t]->add_child(this);
  _mem_county_rate[109][t]->add_child(this);
  _mem_county_rate[10][t]->add_child(this);
  _mem_county_rate[110][t]->add_child(this);
  _mem_county_rate[111][t]->add_child(this);
  _mem_county_rate[112][t]->add_child(this);
  _mem_county_rate[113][t]->add_child(this);
  _mem_county_rate[114][t]->add_child(this);
  _mem_county_rate[115][t]->add_child(this);
  _mem_county_rate[116][t]->add_child(this);
  _mem_county_rate[117][t]->add_child(this);
  _mem_county_rate[118][t]->add_child(this);
  _mem_county_rate[119][t]->add_child(this);
  _mem_county_rate[11][t]->add_child(this);
  _mem_county_rate[120][t]->add_child(this);
  _mem_county_rate[121][t]->add_child(this);
  _mem_county_rate[122][t]->add_child(this);
  _mem_county_rate[123][t]->add_child(this);
  _mem_county_rate[124][t]->add_child(this);
  _mem_county_rate[125][t]->add_child(this);
  _mem_county_rate[126][t]->add_child(this);
  _mem_county_rate[127][t]->add_child(this);
  _mem_county_rate[128][t]->add_child(this);
  _mem_county_rate[129][t]->add_child(this);
  _mem_county_rate[12][t]->add_child(this);
  _mem_county_rate[130][t]->add_child(this);
  _mem_county_rate[131][t]->add_child(this);
  _mem_county_rate[132][t]->add_child(this);
  _mem_county_rate[133][t]->add_child(this);
  _mem_county_rate[134][t]->add_child(this);
  _mem_county_rate[135][t]->add_child(this);
  _mem_county_rate[136][t]->add_child(this);
  _mem_county_rate[137][t]->add_child(this);
  _mem_county_rate[138][t]->add_child(this);
  _mem_county_rate[139][t]->add_child(this);
  _mem_county_rate[13][t]->add_child(this);
  _mem_county_rate[140][t]->add_child(this);
  _mem_county_rate[141][t]->add_child(this);
  _mem_county_rate[142][t]->add_child(this);
  _mem_county_rate[143][t]->add_child(this);
  _mem_county_rate[144][t]->add_child(this);
  _mem_county_rate[145][t]->add_child(this);
  _mem_county_rate[146][t]->add_child(this);
  _mem_county_rate[147][t]->add_child(this);
  _mem_county_rate[148][t]->add_child(this);
  _mem_county_rate[149][t]->add_child(this);
  _mem_county_rate[14][t]->add_child(this);
  _mem_county_rate[150][t]->add_child(this);
  _mem_county_rate[151][t]->add_child(this);
  _mem_county_rate[152][t]->add_child(this);
  _mem_county_rate[153][t]->add_child(this);
  _mem_county_rate[154][t]->add_child(this);
  _mem_county_rate[155][t]->add_child(this);
  _mem_county_rate[156][t]->add_child(this);
  _mem_county_rate[157][t]->add_child(this);
  _mem_county_rate[158][t]->add_child(this);
  _mem_county_rate[159][t]->add_child(this);
  _mem_county_rate[15][t]->add_child(this);
  _mem_county_rate[160][t]->add_child(this);
  _mem_county_rate[161][t]->add_child(this);
  _mem_county_rate[162][t]->add_child(this);
  _mem_county_rate[163][t]->add_child(this);
  _mem_county_rate[164][t]->add_child(this);
  _mem_county_rate[165][t]->add_child(this);
  _mem_county_rate[166][t]->add_child(this);
  _mem_county_rate[167][t]->add_child(this);
  _mem_county_rate[168][t]->add_child(this);
  _mem_county_rate[169][t]->add_child(this);
  _mem_county_rate[16][t]->add_child(this);
  _mem_county_rate[170][t]->add_child(this);
  _mem_county_rate[171][t]->add_child(this);
  _mem_county_rate[172][t]->add_child(this);
  _mem_county_rate[173][t]->add_child(this);
  _mem_county_rate[174][t]->add_child(this);
  _mem_county_rate[175][t]->add_child(this);
  _mem_county_rate[176][t]->add_child(this);
  _mem_county_rate[177][t]->add_child(this);
  _mem_county_rate[178][t]->add_child(this);
  _mem_county_rate[179][t]->add_child(this);
  _mem_county_rate[17][t]->add_child(this);
  _mem_county_rate[180][t]->add_child(this);
  _mem_county_rate[181][t]->add_child(this);
  _mem_county_rate[182][t]->add_child(this);
  _mem_county_rate[183][t]->add_child(this);
  _mem_county_rate[184][t]->add_child(this);
  _mem_county_rate[185][t]->add_child(this);
  _mem_county_rate[186][t]->add_child(this);
  _mem_county_rate[187][t]->add_child(this);
  _mem_county_rate[188][t]->add_child(this);
  _mem_county_rate[189][t]->add_child(this);
  _mem_county_rate[18][t]->add_child(this);
  _mem_county_rate[190][t]->add_child(this);
  _mem_county_rate[191][t]->add_child(this);
  _mem_county_rate[192][t]->add_child(this);
  _mem_county_rate[193][t]->add_child(this);
  _mem_county_rate[194][t]->add_child(this);
  _mem_county_rate[195][t]->add_child(this);
  _mem_county_rate[196][t]->add_child(this);
  _mem_county_rate[197][t]->add_child(this);
  _mem_county_rate[198][t]->add_child(this);
  _mem_county_rate[199][t]->add_child(this);
  _mem_county_rate[19][t]->add_child(this);
  _mem_county_rate[1][t]->add_child(this);
  _mem_county_rate[200][t]->add_child(this);
  _mem_county_rate[201][t]->add_child(this);
  _mem_county_rate[202][t]->add_child(this);
  _mem_county_rate[203][t]->add_child(this);
  _mem_county_rate[204][t]->add_child(this);
  _mem_county_rate[205][t]->add_child(this);
  _mem_county_rate[206][t]->add_child(this);
  _mem_county_rate[207][t]->add_child(this);
  _mem_county_rate[208][t]->add_child(this);
  _mem_county_rate[209][t]->add_child(this);
  _mem_county_rate[20][t]->add_child(this);
  _mem_county_rate[210][t]->add_child(this);
  _mem_county_rate[211][t]->add_child(this);
  _mem_county_rate[212][t]->add_child(this);
  _mem_county_rate[213][t]->add_child(this);
  _mem_county_rate[214][t]->add_child(this);
  _mem_county_rate[215][t]->add_child(this);
  _mem_county_rate[216][t]->add_child(this);
  _mem_county_rate[217][t]->add_child(this);
  _mem_county_rate[218][t]->add_child(this);
  _mem_county_rate[219][t]->add_child(this);
  _mem_county_rate[21][t]->add_child(this);
  _mem_county_rate[220][t]->add_child(this);
  _mem_county_rate[221][t]->add_child(this);
  _mem_county_rate[222][t]->add_child(this);
  _mem_county_rate[223][t]->add_child(this);
  _mem_county_rate[224][t]->add_child(this);
  _mem_county_rate[225][t]->add_child(this);
  _mem_county_rate[226][t]->add_child(this);
  _mem_county_rate[227][t]->add_child(this);
  _mem_county_rate[228][t]->add_child(this);
  _mem_county_rate[229][t]->add_child(this);
  _mem_county_rate[22][t]->add_child(this);
  _mem_county_rate[230][t]->add_child(this);
  _mem_county_rate[231][t]->add_child(this);
  _mem_county_rate[232][t]->add_child(this);
  _mem_county_rate[233][t]->add_child(this);
  _mem_county_rate[234][t]->add_child(this);
  _mem_county_rate[235][t]->add_child(this);
  _mem_county_rate[236][t]->add_child(this);
  _mem_county_rate[237][t]->add_child(this);
  _mem_county_rate[238][t]->add_child(this);
  _mem_county_rate[239][t]->add_child(this);
  _mem_county_rate[23][t]->add_child(this);
  _mem_county_rate[240][t]->add_child(this);
  _mem_county_rate[241][t]->add_child(this);
  _mem_county_rate[242][t]->add_child(this);
  _mem_county_rate[243][t]->add_child(this);
  _mem_county_rate[244][t]->add_child(this);
  _mem_county_rate[245][t]->add_child(this);
  _mem_county_rate[246][t]->add_child(this);
  _mem_county_rate[247][t]->add_child(this);
  _mem_county_rate[248][t]->add_child(this);
  _mem_county_rate[249][t]->add_child(this);
  _mem_county_rate[24][t]->add_child(this);
  _mem_county_rate[250][t]->add_child(this);
  _mem_county_rate[251][t]->add_child(this);
  _mem_county_rate[252][t]->add_child(this);
  _mem_county_rate[253][t]->add_child(this);
  _mem_county_rate[254][t]->add_child(this);
  _mem_county_rate[255][t]->add_child(this);
  _mem_county_rate[256][t]->add_child(this);
  _mem_county_rate[257][t]->add_child(this);
  _mem_county_rate[258][t]->add_child(this);
  _mem_county_rate[259][t]->add_child(this);
  _mem_county_rate[25][t]->add_child(this);
  _mem_county_rate[260][t]->add_child(this);
  _mem_county_rate[261][t]->add_child(this);
  _mem_county_rate[262][t]->add_child(this);
  _mem_county_rate[263][t]->add_child(this);
  _mem_county_rate[264][t]->add_child(this);
  _mem_county_rate[265][t]->add_child(this);
  _mem_county_rate[266][t]->add_child(this);
  _mem_county_rate[267][t]->add_child(this);
  _mem_county_rate[268][t]->add_child(this);
  _mem_county_rate[269][t]->add_child(this);
  _mem_county_rate[26][t]->add_child(this);
  _mem_county_rate[270][t]->add_child(this);
  _mem_county_rate[271][t]->add_child(this);
  _mem_county_rate[272][t]->add_child(this);
  _mem_county_rate[273][t]->add_child(this);
  _mem_county_rate[274][t]->add_child(this);
  _mem_county_rate[275][t]->add_child(this);
  _mem_county_rate[276][t]->add_child(this);
  _mem_county_rate[27][t]->add_child(this);
  _mem_county_rate[28][t]->add_child(this);
  _mem_county_rate[29][t]->add_child(this);
  _mem_county_rate[2][t]->add_child(this);
  _mem_county_rate[30][t]->add_child(this);
  _mem_county_rate[31][t]->add_child(this);
  _mem_county_rate[32][t]->add_child(this);
  _mem_county_rate[33][t]->add_child(this);
  _mem_county_rate[34][t]->add_child(this);
  _mem_county_rate[35][t]->add_child(this);
  _mem_county_rate[36][t]->add_child(this);
  _mem_county_rate[37][t]->add_child(this);
  _mem_county_rate[38][t]->add_child(this);
  _mem_county_rate[39][t]->add_child(this);
  _mem_county_rate[3][t]->add_child(this);
  _mem_county_rate[40][t]->add_child(this);
  _mem_county_rate[41][t]->add_child(this);
  _mem_county_rate[42][t]->add_child(this);
  _mem_county_rate[43][t]->add_child(this);
  _mem_county_rate[44][t]->add_child(this);
  _mem_county_rate[45][t]->add_child(this);
  _mem_county_rate[46][t]->add_child(this);
  _mem_county_rate[47][t]->add_child(this);
  _mem_county_rate[48][t]->add_child(this);
  _mem_county_rate[49][t]->add_child(this);
  _mem_county_rate[4][t]->add_child(this);
  _mem_county_rate[50][t]->add_child(this);
  _mem_county_rate[51][t]->add_child(this);
  _mem_county_rate[52][t]->add_child(this);
  _mem_county_rate[53][t]->add_child(this);
  _mem_county_rate[54][t]->add_child(this);
  _mem_county_rate[55][t]->add_child(this);
  _mem_county_rate[56][t]->add_child(this);
  _mem_county_rate[57][t]->add_child(this);
  _mem_county_rate[58][t]->add_child(this);
  _mem_county_rate[59][t]->add_child(this);
  _mem_county_rate[5][t]->add_child(this);
  _mem_county_rate[60][t]->add_child(this);
  _mem_county_rate[61][t]->add_child(this);
  _mem_county_rate[62][t]->add_child(this);
  _mem_county_rate[63][t]->add_child(this);
  _mem_county_rate[64][t]->add_child(this);
  _mem_county_rate[65][t]->add_child(this);
  _mem_county_rate[66][t]->add_child(this);
  _mem_county_rate[67][t]->add_child(this);
  _mem_county_rate[68][t]->add_child(this);
  _mem_county_rate[69][t]->add_child(this);
  _mem_county_rate[6][t]->add_child(this);
  _mem_county_rate[70][t]->add_child(this);
  _mem_county_rate[71][t]->add_child(this);
  _mem_county_rate[72][t]->add_child(this);
  _mem_county_rate[73][t]->add_child(this);
  _mem_county_rate[74][t]->add_child(this);
  _mem_county_rate[75][t]->add_child(this);
  _mem_county_rate[76][t]->add_child(this);
  _mem_county_rate[77][t]->add_child(this);
  _mem_county_rate[78][t]->add_child(this);
  _mem_county_rate[79][t]->add_child(this);
  _mem_county_rate[7][t]->add_child(this);
  _mem_county_rate[80][t]->add_child(this);
  _mem_county_rate[81][t]->add_child(this);
  _mem_county_rate[82][t]->add_child(this);
  _mem_county_rate[83][t]->add_child(this);
  _mem_county_rate[84][t]->add_child(this);
  _mem_county_rate[85][t]->add_child(this);
  _mem_county_rate[86][t]->add_child(this);
  _mem_county_rate[87][t]->add_child(this);
  _mem_county_rate[88][t]->add_child(this);
  _mem_county_rate[89][t]->add_child(this);
  _mem_county_rate[8][t]->add_child(this);
  _mem_county_rate[90][t]->add_child(this);
  _mem_county_rate[91][t]->add_child(this);
  _mem_county_rate[92][t]->add_child(this);
  _mem_county_rate[93][t]->add_child(this);
  _mem_county_rate[94][t]->add_child(this);
  _mem_county_rate[95][t]->add_child(this);
  _mem_county_rate[96][t]->add_child(this);
  _mem_county_rate[97][t]->add_child(this);
  _mem_county_rate[98][t]->add_child(this);
  _mem_county_rate[99][t]->add_child(this);
  _mem_county_rate[9][t]->add_child(this);
}
void _Var_region_rate::remove_edge()
{
  _mem_county_rate[0][t]->erase_contig(this);
  _mem_county_rate[100][t]->erase_contig(this);
  _mem_county_rate[101][t]->erase_contig(this);
  _mem_county_rate[102][t]->erase_contig(this);
  _mem_county_rate[103][t]->erase_contig(this);
  _mem_county_rate[104][t]->erase_contig(this);
  _mem_county_rate[105][t]->erase_contig(this);
  _mem_county_rate[106][t]->erase_contig(this);
  _mem_county_rate[107][t]->erase_contig(this);
  _mem_county_rate[108][t]->erase_contig(this);
  _mem_county_rate[109][t]->erase_contig(this);
  _mem_county_rate[10][t]->erase_contig(this);
  _mem_county_rate[110][t]->erase_contig(this);
  _mem_county_rate[111][t]->erase_contig(this);
  _mem_county_rate[112][t]->erase_contig(this);
  _mem_county_rate[113][t]->erase_contig(this);
  _mem_county_rate[114][t]->erase_contig(this);
  _mem_county_rate[115][t]->erase_contig(this);
  _mem_county_rate[116][t]->erase_contig(this);
  _mem_county_rate[117][t]->erase_contig(this);
  _mem_county_rate[118][t]->erase_contig(this);
  _mem_county_rate[119][t]->erase_contig(this);
  _mem_county_rate[11][t]->erase_contig(this);
  _mem_county_rate[120][t]->erase_contig(this);
  _mem_county_rate[121][t]->erase_contig(this);
  _mem_county_rate[122][t]->erase_contig(this);
  _mem_county_rate[123][t]->erase_contig(this);
  _mem_county_rate[124][t]->erase_contig(this);
  _mem_county_rate[125][t]->erase_contig(this);
  _mem_county_rate[126][t]->erase_contig(this);
  _mem_county_rate[127][t]->erase_contig(this);
  _mem_county_rate[128][t]->erase_contig(this);
  _mem_county_rate[129][t]->erase_contig(this);
  _mem_county_rate[12][t]->erase_contig(this);
  _mem_county_rate[130][t]->erase_contig(this);
  _mem_county_rate[131][t]->erase_contig(this);
  _mem_county_rate[132][t]->erase_contig(this);
  _mem_county_rate[133][t]->erase_contig(this);
  _mem_county_rate[134][t]->erase_contig(this);
  _mem_county_rate[135][t]->erase_contig(this);
  _mem_county_rate[136][t]->erase_contig(this);
  _mem_county_rate[137][t]->erase_contig(this);
  _mem_county_rate[138][t]->erase_contig(this);
  _mem_county_rate[139][t]->erase_contig(this);
  _mem_county_rate[13][t]->erase_contig(this);
  _mem_county_rate[140][t]->erase_contig(this);
  _mem_county_rate[141][t]->erase_contig(this);
  _mem_county_rate[142][t]->erase_contig(this);
  _mem_county_rate[143][t]->erase_contig(this);
  _mem_county_rate[144][t]->erase_contig(this);
  _mem_county_rate[145][t]->erase_contig(this);
  _mem_county_rate[146][t]->erase_contig(this);
  _mem_county_rate[147][t]->erase_contig(this);
  _mem_county_rate[148][t]->erase_contig(this);
  _mem_county_rate[149][t]->erase_contig(this);
  _mem_county_rate[14][t]->erase_contig(this);
  _mem_county_rate[150][t]->erase_contig(this);
  _mem_county_rate[151][t]->erase_contig(this);
  _mem_county_rate[152][t]->erase_contig(this);
  _mem_county_rate[153][t]->erase_contig(this);
  _mem_county_rate[154][t]->erase_contig(this);
  _mem_county_rate[155][t]->erase_contig(this);
  _mem_county_rate[156][t]->erase_contig(this);
  _mem_county_rate[157][t]->erase_contig(this);
  _mem_county_rate[158][t]->erase_contig(this);
  _mem_county_rate[159][t]->erase_contig(this);
  _mem_county_rate[15][t]->erase_contig(this);
  _mem_county_rate[160][t]->erase_contig(this);
  _mem_county_rate[161][t]->erase_contig(this);
  _mem_county_rate[162][t]->erase_contig(this);
  _mem_county_rate[163][t]->erase_contig(this);
  _mem_county_rate[164][t]->erase_contig(this);
  _mem_county_rate[165][t]->erase_contig(this);
  _mem_county_rate[166][t]->erase_contig(this);
  _mem_county_rate[167][t]->erase_contig(this);
  _mem_county_rate[168][t]->erase_contig(this);
  _mem_county_rate[169][t]->erase_contig(this);
  _mem_county_rate[16][t]->erase_contig(this);
  _mem_county_rate[170][t]->erase_contig(this);
  _mem_county_rate[171][t]->erase_contig(this);
  _mem_county_rate[172][t]->erase_contig(this);
  _mem_county_rate[173][t]->erase_contig(this);
  _mem_county_rate[174][t]->erase_contig(this);
  _mem_county_rate[175][t]->erase_contig(this);
  _mem_county_rate[176][t]->erase_contig(this);
  _mem_county_rate[177][t]->erase_contig(this);
  _mem_county_rate[178][t]->erase_contig(this);
  _mem_county_rate[179][t]->erase_contig(this);
  _mem_county_rate[17][t]->erase_contig(this);
  _mem_county_rate[180][t]->erase_contig(this);
  _mem_county_rate[181][t]->erase_contig(this);
  _mem_county_rate[182][t]->erase_contig(this);
  _mem_county_rate[183][t]->erase_contig(this);
  _mem_county_rate[184][t]->erase_contig(this);
  _mem_county_rate[185][t]->erase_contig(this);
  _mem_county_rate[186][t]->erase_contig(this);
  _mem_county_rate[187][t]->erase_contig(this);
  _mem_county_rate[188][t]->erase_contig(this);
  _mem_county_rate[189][t]->erase_contig(this);
  _mem_county_rate[18][t]->erase_contig(this);
  _mem_county_rate[190][t]->erase_contig(this);
  _mem_county_rate[191][t]->erase_contig(this);
  _mem_county_rate[192][t]->erase_contig(this);
  _mem_county_rate[193][t]->erase_contig(this);
  _mem_county_rate[194][t]->erase_contig(this);
  _mem_county_rate[195][t]->erase_contig(this);
  _mem_county_rate[196][t]->erase_contig(this);
  _mem_county_rate[197][t]->erase_contig(this);
  _mem_county_rate[198][t]->erase_contig(this);
  _mem_county_rate[199][t]->erase_contig(this);
  _mem_county_rate[19][t]->erase_contig(this);
  _mem_county_rate[1][t]->erase_contig(this);
  _mem_county_rate[200][t]->erase_contig(this);
  _mem_county_rate[201][t]->erase_contig(this);
  _mem_county_rate[202][t]->erase_contig(this);
  _mem_county_rate[203][t]->erase_contig(this);
  _mem_county_rate[204][t]->erase_contig(this);
  _mem_county_rate[205][t]->erase_contig(this);
  _mem_county_rate[206][t]->erase_contig(this);
  _mem_county_rate[207][t]->erase_contig(this);
  _mem_county_rate[208][t]->erase_contig(this);
  _mem_county_rate[209][t]->erase_contig(this);
  _mem_county_rate[20][t]->erase_contig(this);
  _mem_county_rate[210][t]->erase_contig(this);
  _mem_county_rate[211][t]->erase_contig(this);
  _mem_county_rate[212][t]->erase_contig(this);
  _mem_county_rate[213][t]->erase_contig(this);
  _mem_county_rate[214][t]->erase_contig(this);
  _mem_county_rate[215][t]->erase_contig(this);
  _mem_county_rate[216][t]->erase_contig(this);
  _mem_county_rate[217][t]->erase_contig(this);
  _mem_county_rate[218][t]->erase_contig(this);
  _mem_county_rate[219][t]->erase_contig(this);
  _mem_county_rate[21][t]->erase_contig(this);
  _mem_county_rate[220][t]->erase_contig(this);
  _mem_county_rate[221][t]->erase_contig(this);
  _mem_county_rate[222][t]->erase_contig(this);
  _mem_county_rate[223][t]->erase_contig(this);
  _mem_county_rate[224][t]->erase_contig(this);
  _mem_county_rate[225][t]->erase_contig(this);
  _mem_county_rate[226][t]->erase_contig(this);
  _mem_county_rate[227][t]->erase_contig(this);
  _mem_county_rate[228][t]->erase_contig(this);
  _mem_county_rate[229][t]->erase_contig(this);
  _mem_county_rate[22][t]->erase_contig(this);
  _mem_county_rate[230][t]->erase_contig(this);
  _mem_county_rate[231][t]->erase_contig(this);
  _mem_county_rate[232][t]->erase_contig(this);
  _mem_county_rate[233][t]->erase_contig(this);
  _mem_county_rate[234][t]->erase_contig(this);
  _mem_county_rate[235][t]->erase_contig(this);
  _mem_county_rate[236][t]->erase_contig(this);
  _mem_county_rate[237][t]->erase_contig(this);
  _mem_county_rate[238][t]->erase_contig(this);
  _mem_county_rate[239][t]->erase_contig(this);
  _mem_county_rate[23][t]->erase_contig(this);
  _mem_county_rate[240][t]->erase_contig(this);
  _mem_county_rate[241][t]->erase_contig(this);
  _mem_county_rate[242][t]->erase_contig(this);
  _mem_county_rate[243][t]->erase_contig(this);
  _mem_county_rate[244][t]->erase_contig(this);
  _mem_county_rate[245][t]->erase_contig(this);
  _mem_county_rate[246][t]->erase_contig(this);
  _mem_county_rate[247][t]->erase_contig(this);
  _mem_county_rate[248][t]->erase_contig(this);
  _mem_county_rate[249][t]->erase_contig(this);
  _mem_county_rate[24][t]->erase_contig(this);
  _mem_county_rate[250][t]->erase_contig(this);
  _mem_county_rate[251][t]->erase_contig(this);
  _mem_county_rate[252][t]->erase_contig(this);
  _mem_county_rate[253][t]->erase_contig(this);
  _mem_county_rate[254][t]->erase_contig(this);
  _mem_county_rate[255][t]->erase_contig(this);
  _mem_county_rate[256][t]->erase_contig(this);
  _mem_county_rate[257][t]->erase_contig(this);
  _mem_county_rate[258][t]->erase_contig(this);
  _mem_county_rate[259][t]->erase_contig(this);
  _mem_county_rate[25][t]->erase_contig(this);
  _mem_county_rate[260][t]->erase_contig(this);
  _mem_county_rate[261][t]->erase_contig(this);
  _mem_county_rate[262][t]->erase_contig(this);
  _mem_county_rate[263][t]->erase_contig(this);
  _mem_county_rate[264][t]->erase_contig(this);
  _mem_county_rate[265][t]->erase_contig(this);
  _mem_county_rate[266][t]->erase_contig(this);
  _mem_county_rate[267][t]->erase_contig(this);
  _mem_county_rate[268][t]->erase_contig(this);
  _mem_county_rate[269][t]->erase_contig(this);
  _mem_county_rate[26][t]->erase_contig(this);
  _mem_county_rate[270][t]->erase_contig(this);
  _mem_county_rate[271][t]->erase_contig(this);
  _mem_county_rate[272][t]->erase_contig(this);
  _mem_county_rate[273][t]->erase_contig(this);
  _mem_county_rate[274][t]->erase_contig(this);
  _mem_county_rate[275][t]->erase_contig(this);
  _mem_county_rate[276][t]->erase_contig(this);
  _mem_county_rate[27][t]->erase_contig(this);
  _mem_county_rate[28][t]->erase_contig(this);
  _mem_county_rate[29][t]->erase_contig(this);
  _mem_county_rate[2][t]->erase_contig(this);
  _mem_county_rate[30][t]->erase_contig(this);
  _mem_county_rate[31][t]->erase_contig(this);
  _mem_county_rate[32][t]->erase_contig(this);
  _mem_county_rate[33][t]->erase_contig(this);
  _mem_county_rate[34][t]->erase_contig(this);
  _mem_county_rate[35][t]->erase_contig(this);
  _mem_county_rate[36][t]->erase_contig(this);
  _mem_county_rate[37][t]->erase_contig(this);
  _mem_county_rate[38][t]->erase_contig(this);
  _mem_county_rate[39][t]->erase_contig(this);
  _mem_county_rate[3][t]->erase_contig(this);
  _mem_county_rate[40][t]->erase_contig(this);
  _mem_county_rate[41][t]->erase_contig(this);
  _mem_county_rate[42][t]->erase_contig(this);
  _mem_county_rate[43][t]->erase_contig(this);
  _mem_county_rate[44][t]->erase_contig(this);
  _mem_county_rate[45][t]->erase_contig(this);
  _mem_county_rate[46][t]->erase_contig(this);
  _mem_county_rate[47][t]->erase_contig(this);
  _mem_county_rate[48][t]->erase_contig(this);
  _mem_county_rate[49][t]->erase_contig(this);
  _mem_county_rate[4][t]->erase_contig(this);
  _mem_county_rate[50][t]->erase_contig(this);
  _mem_county_rate[51][t]->erase_contig(this);
  _mem_county_rate[52][t]->erase_contig(this);
  _mem_county_rate[53][t]->erase_contig(this);
  _mem_county_rate[54][t]->erase_contig(this);
  _mem_county_rate[55][t]->erase_contig(this);
  _mem_county_rate[56][t]->erase_contig(this);
  _mem_county_rate[57][t]->erase_contig(this);
  _mem_county_rate[58][t]->erase_contig(this);
  _mem_county_rate[59][t]->erase_contig(this);
  _mem_county_rate[5][t]->erase_contig(this);
  _mem_county_rate[60][t]->erase_contig(this);
  _mem_county_rate[61][t]->erase_contig(this);
  _mem_county_rate[62][t]->erase_contig(this);
  _mem_county_rate[63][t]->erase_contig(this);
  _mem_county_rate[64][t]->erase_contig(this);
  _mem_county_rate[65][t]->erase_contig(this);
  _mem_county_rate[66][t]->erase_contig(this);
  _mem_county_rate[67][t]->erase_contig(this);
  _mem_county_rate[68][t]->erase_contig(this);
  _mem_county_rate[69][t]->erase_contig(this);
  _mem_county_rate[6][t]->erase_contig(this);
  _mem_county_rate[70][t]->erase_contig(this);
  _mem_county_rate[71][t]->erase_contig(this);
  _mem_county_rate[72][t]->erase_contig(this);
  _mem_county_rate[73][t]->erase_contig(this);
  _mem_county_rate[74][t]->erase_contig(this);
  _mem_county_rate[75][t]->erase_contig(this);
  _mem_county_rate[76][t]->erase_contig(this);
  _mem_county_rate[77][t]->erase_contig(this);
  _mem_county_rate[78][t]->erase_contig(this);
  _mem_county_rate[79][t]->erase_contig(this);
  _mem_county_rate[7][t]->erase_contig(this);
  _mem_county_rate[80][t]->erase_contig(this);
  _mem_county_rate[81][t]->erase_contig(this);
  _mem_county_rate[82][t]->erase_contig(this);
  _mem_county_rate[83][t]->erase_contig(this);
  _mem_county_rate[84][t]->erase_contig(this);
  _mem_county_rate[85][t]->erase_contig(this);
  _mem_county_rate[86][t]->erase_contig(this);
  _mem_county_rate[87][t]->erase_contig(this);
  _mem_county_rate[88][t]->erase_contig(this);
  _mem_county_rate[89][t]->erase_contig(this);
  _mem_county_rate[8][t]->erase_contig(this);
  _mem_county_rate[90][t]->erase_contig(this);
  _mem_county_rate[91][t]->erase_contig(this);
  _mem_county_rate[92][t]->erase_contig(this);
  _mem_county_rate[93][t]->erase_contig(this);
  _mem_county_rate[94][t]->erase_contig(this);
  _mem_county_rate[95][t]->erase_contig(this);
  _mem_county_rate[96][t]->erase_contig(this);
  _mem_county_rate[97][t]->erase_contig(this);
  _mem_county_rate[98][t]->erase_contig(this);
  _mem_county_rate[99][t]->erase_contig(this);
  _mem_county_rate[9][t]->erase_contig(this);
  _mem_county_rate[0][t]->erase_child(this);
  _mem_county_rate[100][t]->erase_child(this);
  _mem_county_rate[101][t]->erase_child(this);
  _mem_county_rate[102][t]->erase_child(this);
  _mem_county_rate[103][t]->erase_child(this);
  _mem_county_rate[104][t]->erase_child(this);
  _mem_county_rate[105][t]->erase_child(this);
  _mem_county_rate[106][t]->erase_child(this);
  _mem_county_rate[107][t]->erase_child(this);
  _mem_county_rate[108][t]->erase_child(this);
  _mem_county_rate[109][t]->erase_child(this);
  _mem_county_rate[10][t]->erase_child(this);
  _mem_county_rate[110][t]->erase_child(this);
  _mem_county_rate[111][t]->erase_child(this);
  _mem_county_rate[112][t]->erase_child(this);
  _mem_county_rate[113][t]->erase_child(this);
  _mem_county_rate[114][t]->erase_child(this);
  _mem_county_rate[115][t]->erase_child(this);
  _mem_county_rate[116][t]->erase_child(this);
  _mem_county_rate[117][t]->erase_child(this);
  _mem_county_rate[118][t]->erase_child(this);
  _mem_county_rate[119][t]->erase_child(this);
  _mem_county_rate[11][t]->erase_child(this);
  _mem_county_rate[120][t]->erase_child(this);
  _mem_county_rate[121][t]->erase_child(this);
  _mem_county_rate[122][t]->erase_child(this);
  _mem_county_rate[123][t]->erase_child(this);
  _mem_county_rate[124][t]->erase_child(this);
  _mem_county_rate[125][t]->erase_child(this);
  _mem_county_rate[126][t]->erase_child(this);
  _mem_county_rate[127][t]->erase_child(this);
  _mem_county_rate[128][t]->erase_child(this);
  _mem_county_rate[129][t]->erase_child(this);
  _mem_county_rate[12][t]->erase_child(this);
  _mem_county_rate[130][t]->erase_child(this);
  _mem_county_rate[131][t]->erase_child(this);
  _mem_county_rate[132][t]->erase_child(this);
  _mem_county_rate[133][t]->erase_child(this);
  _mem_county_rate[134][t]->erase_child(this);
  _mem_county_rate[135][t]->erase_child(this);
  _mem_county_rate[136][t]->erase_child(this);
  _mem_county_rate[137][t]->erase_child(this);
  _mem_county_rate[138][t]->erase_child(this);
  _mem_county_rate[139][t]->erase_child(this);
  _mem_county_rate[13][t]->erase_child(this);
  _mem_county_rate[140][t]->erase_child(this);
  _mem_county_rate[141][t]->erase_child(this);
  _mem_county_rate[142][t]->erase_child(this);
  _mem_county_rate[143][t]->erase_child(this);
  _mem_county_rate[144][t]->erase_child(this);
  _mem_county_rate[145][t]->erase_child(this);
  _mem_county_rate[146][t]->erase_child(this);
  _mem_county_rate[147][t]->erase_child(this);
  _mem_county_rate[148][t]->erase_child(this);
  _mem_county_rate[149][t]->erase_child(this);
  _mem_county_rate[14][t]->erase_child(this);
  _mem_county_rate[150][t]->erase_child(this);
  _mem_county_rate[151][t]->erase_child(this);
  _mem_county_rate[152][t]->erase_child(this);
  _mem_county_rate[153][t]->erase_child(this);
  _mem_county_rate[154][t]->erase_child(this);
  _mem_county_rate[155][t]->erase_child(this);
  _mem_county_rate[156][t]->erase_child(this);
  _mem_county_rate[157][t]->erase_child(this);
  _mem_county_rate[158][t]->erase_child(this);
  _mem_county_rate[159][t]->erase_child(this);
  _mem_county_rate[15][t]->erase_child(this);
  _mem_county_rate[160][t]->erase_child(this);
  _mem_county_rate[161][t]->erase_child(this);
  _mem_county_rate[162][t]->erase_child(this);
  _mem_county_rate[163][t]->erase_child(this);
  _mem_county_rate[164][t]->erase_child(this);
  _mem_county_rate[165][t]->erase_child(this);
  _mem_county_rate[166][t]->erase_child(this);
  _mem_county_rate[167][t]->erase_child(this);
  _mem_county_rate[168][t]->erase_child(this);
  _mem_county_rate[169][t]->erase_child(this);
  _mem_county_rate[16][t]->erase_child(this);
  _mem_county_rate[170][t]->erase_child(this);
  _mem_county_rate[171][t]->erase_child(this);
  _mem_county_rate[172][t]->erase_child(this);
  _mem_county_rate[173][t]->erase_child(this);
  _mem_county_rate[174][t]->erase_child(this);
  _mem_county_rate[175][t]->erase_child(this);
  _mem_county_rate[176][t]->erase_child(this);
  _mem_county_rate[177][t]->erase_child(this);
  _mem_county_rate[178][t]->erase_child(this);
  _mem_county_rate[179][t]->erase_child(this);
  _mem_county_rate[17][t]->erase_child(this);
  _mem_county_rate[180][t]->erase_child(this);
  _mem_county_rate[181][t]->erase_child(this);
  _mem_county_rate[182][t]->erase_child(this);
  _mem_county_rate[183][t]->erase_child(this);
  _mem_county_rate[184][t]->erase_child(this);
  _mem_county_rate[185][t]->erase_child(this);
  _mem_county_rate[186][t]->erase_child(this);
  _mem_county_rate[187][t]->erase_child(this);
  _mem_county_rate[188][t]->erase_child(this);
  _mem_county_rate[189][t]->erase_child(this);
  _mem_county_rate[18][t]->erase_child(this);
  _mem_county_rate[190][t]->erase_child(this);
  _mem_county_rate[191][t]->erase_child(this);
  _mem_county_rate[192][t]->erase_child(this);
  _mem_county_rate[193][t]->erase_child(this);
  _mem_county_rate[194][t]->erase_child(this);
  _mem_county_rate[195][t]->erase_child(this);
  _mem_county_rate[196][t]->erase_child(this);
  _mem_county_rate[197][t]->erase_child(this);
  _mem_county_rate[198][t]->erase_child(this);
  _mem_county_rate[199][t]->erase_child(this);
  _mem_county_rate[19][t]->erase_child(this);
  _mem_county_rate[1][t]->erase_child(this);
  _mem_county_rate[200][t]->erase_child(this);
  _mem_county_rate[201][t]->erase_child(this);
  _mem_county_rate[202][t]->erase_child(this);
  _mem_county_rate[203][t]->erase_child(this);
  _mem_county_rate[204][t]->erase_child(this);
  _mem_county_rate[205][t]->erase_child(this);
  _mem_county_rate[206][t]->erase_child(this);
  _mem_county_rate[207][t]->erase_child(this);
  _mem_county_rate[208][t]->erase_child(this);
  _mem_county_rate[209][t]->erase_child(this);
  _mem_county_rate[20][t]->erase_child(this);
  _mem_county_rate[210][t]->erase_child(this);
  _mem_county_rate[211][t]->erase_child(this);
  _mem_county_rate[212][t]->erase_child(this);
  _mem_county_rate[213][t]->erase_child(this);
  _mem_county_rate[214][t]->erase_child(this);
  _mem_county_rate[215][t]->erase_child(this);
  _mem_county_rate[216][t]->erase_child(this);
  _mem_county_rate[217][t]->erase_child(this);
  _mem_county_rate[218][t]->erase_child(this);
  _mem_county_rate[219][t]->erase_child(this);
  _mem_county_rate[21][t]->erase_child(this);
  _mem_county_rate[220][t]->erase_child(this);
  _mem_county_rate[221][t]->erase_child(this);
  _mem_county_rate[222][t]->erase_child(this);
  _mem_county_rate[223][t]->erase_child(this);
  _mem_county_rate[224][t]->erase_child(this);
  _mem_county_rate[225][t]->erase_child(this);
  _mem_county_rate[226][t]->erase_child(this);
  _mem_county_rate[227][t]->erase_child(this);
  _mem_county_rate[228][t]->erase_child(this);
  _mem_county_rate[229][t]->erase_child(this);
  _mem_county_rate[22][t]->erase_child(this);
  _mem_county_rate[230][t]->erase_child(this);
  _mem_county_rate[231][t]->erase_child(this);
  _mem_county_rate[232][t]->erase_child(this);
  _mem_county_rate[233][t]->erase_child(this);
  _mem_county_rate[234][t]->erase_child(this);
  _mem_county_rate[235][t]->erase_child(this);
  _mem_county_rate[236][t]->erase_child(this);
  _mem_county_rate[237][t]->erase_child(this);
  _mem_county_rate[238][t]->erase_child(this);
  _mem_county_rate[239][t]->erase_child(this);
  _mem_county_rate[23][t]->erase_child(this);
  _mem_county_rate[240][t]->erase_child(this);
  _mem_county_rate[241][t]->erase_child(this);
  _mem_county_rate[242][t]->erase_child(this);
  _mem_county_rate[243][t]->erase_child(this);
  _mem_county_rate[244][t]->erase_child(this);
  _mem_county_rate[245][t]->erase_child(this);
  _mem_county_rate[246][t]->erase_child(this);
  _mem_county_rate[247][t]->erase_child(this);
  _mem_county_rate[248][t]->erase_child(this);
  _mem_county_rate[249][t]->erase_child(this);
  _mem_county_rate[24][t]->erase_child(this);
  _mem_county_rate[250][t]->erase_child(this);
  _mem_county_rate[251][t]->erase_child(this);
  _mem_county_rate[252][t]->erase_child(this);
  _mem_county_rate[253][t]->erase_child(this);
  _mem_county_rate[254][t]->erase_child(this);
  _mem_county_rate[255][t]->erase_child(this);
  _mem_county_rate[256][t]->erase_child(this);
  _mem_county_rate[257][t]->erase_child(this);
  _mem_county_rate[258][t]->erase_child(this);
  _mem_county_rate[259][t]->erase_child(this);
  _mem_county_rate[25][t]->erase_child(this);
  _mem_county_rate[260][t]->erase_child(this);
  _mem_county_rate[261][t]->erase_child(this);
  _mem_county_rate[262][t]->erase_child(this);
  _mem_county_rate[263][t]->erase_child(this);
  _mem_county_rate[264][t]->erase_child(this);
  _mem_county_rate[265][t]->erase_child(this);
  _mem_county_rate[266][t]->erase_child(this);
  _mem_county_rate[267][t]->erase_child(this);
  _mem_county_rate[268][t]->erase_child(this);
  _mem_county_rate[269][t]->erase_child(this);
  _mem_county_rate[26][t]->erase_child(this);
  _mem_county_rate[270][t]->erase_child(this);
  _mem_county_rate[271][t]->erase_child(this);
  _mem_county_rate[272][t]->erase_child(this);
  _mem_county_rate[273][t]->erase_child(this);
  _mem_county_rate[274][t]->erase_child(this);
  _mem_county_rate[275][t]->erase_child(this);
  _mem_county_rate[276][t]->erase_child(this);
  _mem_county_rate[27][t]->erase_child(this);
  _mem_county_rate[28][t]->erase_child(this);
  _mem_county_rate[29][t]->erase_child(this);
  _mem_county_rate[2][t]->erase_child(this);
  _mem_county_rate[30][t]->erase_child(this);
  _mem_county_rate[31][t]->erase_child(this);
  _mem_county_rate[32][t]->erase_child(this);
  _mem_county_rate[33][t]->erase_child(this);
  _mem_county_rate[34][t]->erase_child(this);
  _mem_county_rate[35][t]->erase_child(this);
  _mem_county_rate[36][t]->erase_child(this);
  _mem_county_rate[37][t]->erase_child(this);
  _mem_county_rate[38][t]->erase_child(this);
  _mem_county_rate[39][t]->erase_child(this);
  _mem_county_rate[3][t]->erase_child(this);
  _mem_county_rate[40][t]->erase_child(this);
  _mem_county_rate[41][t]->erase_child(this);
  _mem_county_rate[42][t]->erase_child(this);
  _mem_county_rate[43][t]->erase_child(this);
  _mem_county_rate[44][t]->erase_child(this);
  _mem_county_rate[45][t]->erase_child(this);
  _mem_county_rate[46][t]->erase_child(this);
  _mem_county_rate[47][t]->erase_child(this);
  _mem_county_rate[48][t]->erase_child(this);
  _mem_county_rate[49][t]->erase_child(this);
  _mem_county_rate[4][t]->erase_child(this);
  _mem_county_rate[50][t]->erase_child(this);
  _mem_county_rate[51][t]->erase_child(this);
  _mem_county_rate[52][t]->erase_child(this);
  _mem_county_rate[53][t]->erase_child(this);
  _mem_county_rate[54][t]->erase_child(this);
  _mem_county_rate[55][t]->erase_child(this);
  _mem_county_rate[56][t]->erase_child(this);
  _mem_county_rate[57][t]->erase_child(this);
  _mem_county_rate[58][t]->erase_child(this);
  _mem_county_rate[59][t]->erase_child(this);
  _mem_county_rate[5][t]->erase_child(this);
  _mem_county_rate[60][t]->erase_child(this);
  _mem_county_rate[61][t]->erase_child(this);
  _mem_county_rate[62][t]->erase_child(this);
  _mem_county_rate[63][t]->erase_child(this);
  _mem_county_rate[64][t]->erase_child(this);
  _mem_county_rate[65][t]->erase_child(this);
  _mem_county_rate[66][t]->erase_child(this);
  _mem_county_rate[67][t]->erase_child(this);
  _mem_county_rate[68][t]->erase_child(this);
  _mem_county_rate[69][t]->erase_child(this);
  _mem_county_rate[6][t]->erase_child(this);
  _mem_county_rate[70][t]->erase_child(this);
  _mem_county_rate[71][t]->erase_child(this);
  _mem_county_rate[72][t]->erase_child(this);
  _mem_county_rate[73][t]->erase_child(this);
  _mem_county_rate[74][t]->erase_child(this);
  _mem_county_rate[75][t]->erase_child(this);
  _mem_county_rate[76][t]->erase_child(this);
  _mem_county_rate[77][t]->erase_child(this);
  _mem_county_rate[78][t]->erase_child(this);
  _mem_county_rate[79][t]->erase_child(this);
  _mem_county_rate[7][t]->erase_child(this);
  _mem_county_rate[80][t]->erase_child(this);
  _mem_county_rate[81][t]->erase_child(this);
  _mem_county_rate[82][t]->erase_child(this);
  _mem_county_rate[83][t]->erase_child(this);
  _mem_county_rate[84][t]->erase_child(this);
  _mem_county_rate[85][t]->erase_child(this);
  _mem_county_rate[86][t]->erase_child(this);
  _mem_county_rate[87][t]->erase_child(this);
  _mem_county_rate[88][t]->erase_child(this);
  _mem_county_rate[89][t]->erase_child(this);
  _mem_county_rate[8][t]->erase_child(this);
  _mem_county_rate[90][t]->erase_child(this);
  _mem_county_rate[91][t]->erase_child(this);
  _mem_county_rate[92][t]->erase_child(this);
  _mem_county_rate[93][t]->erase_child(this);
  _mem_county_rate[94][t]->erase_child(this);
  _mem_county_rate[95][t]->erase_child(this);
  _mem_county_rate[96][t]->erase_child(this);
  _mem_county_rate[97][t]->erase_child(this);
  _mem_county_rate[98][t]->erase_child(this);
  _mem_county_rate[99][t]->erase_child(this);
  _mem_county_rate[9][t]->erase_child(this);
}
void _Var_region_rate::mcmc_resample()
{
  mh_symmetric_resample_arg(this,_gaussian_prop);
}
void _Var_region_rate::conjugacy_analysis(double& _nxt_val)
{}
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
  printf("\nsample time: %fs (#iter = %d)\n",__elapsed_seconds.count(),10000000);
  swift::_print_answer();
  swift::_garbage_collection();
}
