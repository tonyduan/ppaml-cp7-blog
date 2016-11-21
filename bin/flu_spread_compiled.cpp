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
class _Var_bias;
class _Var_temporal_edge;
class _Var_spatial_edge;
class _Var_logit;
class _Var_region_rate;

const vector<string> __vecstr_instance_Week = {"weeks[0]", "weeks[1]", "weeks[2]", "weeks[3]", "weeks[4]", "weeks[5]", "weeks[6]", "weeks[7]", "weeks[8]", "weeks[9]", "weeks[10]", "weeks[11]", "weeks[12]", "weeks[13]", "weeks[14]", "weeks[15]", "weeks[16]", "weeks[17]", "weeks[18]", "weeks[19]", "weeks[20]", "weeks[21]", "weeks[22]", "weeks[23]", "weeks[24]", "weeks[25]", "weeks[26]", "weeks[27]", "weeks[28]", "weeks[29]", "weeks[30]", "weeks[31]", "weeks[32]", "weeks[33]", "weeks[34]", "weeks[35]", "weeks[36]", "weeks[37]", "weeks[38]", "weeks[39]", "weeks[40]", "weeks[41]", "weeks[42]", "weeks[43]", "weeks[44]", "weeks[45]", "weeks[46]", "weeks[47]", "weeks[48]", "weeks[49]", "weeks[50]", "weeks[51]", "weeks[52]", "weeks[53]", "weeks[54]", "weeks[55]", "weeks[56]", "weeks[57]", "weeks[58]", "weeks[59]", "weeks[60]", "weeks[61]", "weeks[62]", "weeks[63]", "weeks[64]", "weeks[65]", "weeks[66]", "weeks[67]", "weeks[68]", "weeks[69]", "weeks[70]", "weeks[71]", "weeks[72]", "weeks[73]", "weeks[74]", "weeks[75]", "weeks[76]", "weeks[77]", "weeks[78]", "weeks[79]", "weeks[80]", "weeks[81]", "weeks[82]", "weeks[83]", "weeks[84]", "weeks[85]", "weeks[86]", "weeks[87]", "weeks[88]", "weeks[89]", "weeks[90]", "weeks[91]", "weeks[92]", "weeks[93]", "weeks[94]", "weeks[95]", "weeks[96]", "weeks[97]", "weeks[98]", "weeks[99]", "weeks[100]", "weeks[101]", "weeks[102]"};
const vector<string> __vecstr_instance_TemporalPair = {"temporal_pairs[0]", "temporal_pairs[1]", "temporal_pairs[2]", "temporal_pairs[3]", "temporal_pairs[4]", "temporal_pairs[5]", "temporal_pairs[6]", "temporal_pairs[7]", "temporal_pairs[8]", "temporal_pairs[9]", "temporal_pairs[10]", "temporal_pairs[11]", "temporal_pairs[12]", "temporal_pairs[13]", "temporal_pairs[14]", "temporal_pairs[15]", "temporal_pairs[16]", "temporal_pairs[17]", "temporal_pairs[18]", "temporal_pairs[19]", "temporal_pairs[20]", "temporal_pairs[21]", "temporal_pairs[22]", "temporal_pairs[23]", "temporal_pairs[24]", "temporal_pairs[25]", "temporal_pairs[26]", "temporal_pairs[27]", "temporal_pairs[28]", "temporal_pairs[29]", "temporal_pairs[30]", "temporal_pairs[31]", "temporal_pairs[32]", "temporal_pairs[33]", "temporal_pairs[34]", "temporal_pairs[35]", "temporal_pairs[36]", "temporal_pairs[37]", "temporal_pairs[38]", "temporal_pairs[39]", "temporal_pairs[40]", "temporal_pairs[41]", "temporal_pairs[42]", "temporal_pairs[43]", "temporal_pairs[44]", "temporal_pairs[45]", "temporal_pairs[46]", "temporal_pairs[47]", "temporal_pairs[48]", "temporal_pairs[49]", "temporal_pairs[50]", "temporal_pairs[51]", "temporal_pairs[52]", "temporal_pairs[53]", "temporal_pairs[54]", "temporal_pairs[55]", "temporal_pairs[56]", "temporal_pairs[57]", "temporal_pairs[58]", "temporal_pairs[59]", "temporal_pairs[60]", "temporal_pairs[61]", "temporal_pairs[62]", "temporal_pairs[63]", "temporal_pairs[64]", "temporal_pairs[65]", "temporal_pairs[66]", "temporal_pairs[67]", "temporal_pairs[68]", "temporal_pairs[69]", "temporal_pairs[70]", "temporal_pairs[71]", "temporal_pairs[72]", "temporal_pairs[73]", "temporal_pairs[74]", "temporal_pairs[75]", "temporal_pairs[76]", "temporal_pairs[77]", "temporal_pairs[78]", "temporal_pairs[79]", "temporal_pairs[80]", "temporal_pairs[81]", "temporal_pairs[82]", "temporal_pairs[83]", "temporal_pairs[84]", "temporal_pairs[85]", "temporal_pairs[86]", "temporal_pairs[87]", "temporal_pairs[88]", "temporal_pairs[89]", "temporal_pairs[90]", "temporal_pairs[91]", "temporal_pairs[92]", "temporal_pairs[93]", "temporal_pairs[94]", "temporal_pairs[95]", "temporal_pairs[96]", "temporal_pairs[97]", "temporal_pairs[98]", "temporal_pairs[99]", "temporal_pairs[100]", "temporal_pairs[101]"};
const vector<string> __vecstr_instance_SpatialPair = {"spatial_pairs[0]", "spatial_pairs[1]", "spatial_pairs[2]", "spatial_pairs[3]", "spatial_pairs[4]", "spatial_pairs[5]", "spatial_pairs[6]", "spatial_pairs[7]", "spatial_pairs[8]", "spatial_pairs[9]", "spatial_pairs[10]", "spatial_pairs[11]", "spatial_pairs[12]", "spatial_pairs[13]", "spatial_pairs[14]", "spatial_pairs[15]", "spatial_pairs[16]", "spatial_pairs[17]", "spatial_pairs[18]", "spatial_pairs[19]", "spatial_pairs[20]", "spatial_pairs[21]", "spatial_pairs[22]", "spatial_pairs[23]", "spatial_pairs[24]", "spatial_pairs[25]", "spatial_pairs[26]", "spatial_pairs[27]", "spatial_pairs[28]", "spatial_pairs[29]", "spatial_pairs[30]", "spatial_pairs[31]", "spatial_pairs[32]", "spatial_pairs[33]", "spatial_pairs[34]", "spatial_pairs[35]", "spatial_pairs[36]", "spatial_pairs[37]", "spatial_pairs[38]", "spatial_pairs[39]", "spatial_pairs[40]", "spatial_pairs[41]", "spatial_pairs[42]", "spatial_pairs[43]", "spatial_pairs[44]", "spatial_pairs[45]", "spatial_pairs[46]", "spatial_pairs[47]", "spatial_pairs[48]", "spatial_pairs[49]", "spatial_pairs[50]", "spatial_pairs[51]", "spatial_pairs[52]", "spatial_pairs[53]", "spatial_pairs[54]", "spatial_pairs[55]", "spatial_pairs[56]", "spatial_pairs[57]", "spatial_pairs[58]", "spatial_pairs[59]", "spatial_pairs[60]", "spatial_pairs[61]", "spatial_pairs[62]", "spatial_pairs[63]", "spatial_pairs[64]", "spatial_pairs[65]", "spatial_pairs[66]", "spatial_pairs[67]", "spatial_pairs[68]", "spatial_pairs[69]", "spatial_pairs[70]", "spatial_pairs[71]", "spatial_pairs[72]", "spatial_pairs[73]", "spatial_pairs[74]", "spatial_pairs[75]", "spatial_pairs[76]", "spatial_pairs[77]", "spatial_pairs[78]", "spatial_pairs[79]", "spatial_pairs[80]", "spatial_pairs[81]", "spatial_pairs[82]", "spatial_pairs[83]", "spatial_pairs[84]", "spatial_pairs[85]", "spatial_pairs[86]", "spatial_pairs[87]", "spatial_pairs[88]", "spatial_pairs[89]", "spatial_pairs[90]", "spatial_pairs[91]", "spatial_pairs[92]", "spatial_pairs[93]", "spatial_pairs[94]", "spatial_pairs[95]", "spatial_pairs[96]", "spatial_pairs[97]", "spatial_pairs[98]", "spatial_pairs[99]", "spatial_pairs[100]", "spatial_pairs[101]", "spatial_pairs[102]", "spatial_pairs[103]", "spatial_pairs[104]", "spatial_pairs[105]", "spatial_pairs[106]", "spatial_pairs[107]", "spatial_pairs[108]", "spatial_pairs[109]", "spatial_pairs[110]", "spatial_pairs[111]", "spatial_pairs[112]", "spatial_pairs[113]", "spatial_pairs[114]", "spatial_pairs[115]", "spatial_pairs[116]", "spatial_pairs[117]", "spatial_pairs[118]", "spatial_pairs[119]", "spatial_pairs[120]", "spatial_pairs[121]", "spatial_pairs[122]", "spatial_pairs[123]", "spatial_pairs[124]", "spatial_pairs[125]", "spatial_pairs[126]", "spatial_pairs[127]", "spatial_pairs[128]", "spatial_pairs[129]", "spatial_pairs[130]", "spatial_pairs[131]", "spatial_pairs[132]", "spatial_pairs[133]", "spatial_pairs[134]", "spatial_pairs[135]", "spatial_pairs[136]", "spatial_pairs[137]", "spatial_pairs[138]", "spatial_pairs[139]", "spatial_pairs[140]", "spatial_pairs[141]", "spatial_pairs[142]", "spatial_pairs[143]", "spatial_pairs[144]", "spatial_pairs[145]", "spatial_pairs[146]", "spatial_pairs[147]", "spatial_pairs[148]", "spatial_pairs[149]", "spatial_pairs[150]", "spatial_pairs[151]", "spatial_pairs[152]", "spatial_pairs[153]", "spatial_pairs[154]", "spatial_pairs[155]", "spatial_pairs[156]", "spatial_pairs[157]", "spatial_pairs[158]", "spatial_pairs[159]", "spatial_pairs[160]", "spatial_pairs[161]", "spatial_pairs[162]", "spatial_pairs[163]", "spatial_pairs[164]", "spatial_pairs[165]", "spatial_pairs[166]", "spatial_pairs[167]", "spatial_pairs[168]", "spatial_pairs[169]", "spatial_pairs[170]", "spatial_pairs[171]", "spatial_pairs[172]", "spatial_pairs[173]", "spatial_pairs[174]", "spatial_pairs[175]", "spatial_pairs[176]", "spatial_pairs[177]", "spatial_pairs[178]", "spatial_pairs[179]", "spatial_pairs[180]", "spatial_pairs[181]", "spatial_pairs[182]", "spatial_pairs[183]", "spatial_pairs[184]", "spatial_pairs[185]", "spatial_pairs[186]", "spatial_pairs[187]", "spatial_pairs[188]", "spatial_pairs[189]", "spatial_pairs[190]", "spatial_pairs[191]", "spatial_pairs[192]", "spatial_pairs[193]", "spatial_pairs[194]", "spatial_pairs[195]", "spatial_pairs[196]", "spatial_pairs[197]", "spatial_pairs[198]", "spatial_pairs[199]", "spatial_pairs[200]", "spatial_pairs[201]", "spatial_pairs[202]", "spatial_pairs[203]", "spatial_pairs[204]", "spatial_pairs[205]", "spatial_pairs[206]", "spatial_pairs[207]", "spatial_pairs[208]", "spatial_pairs[209]", "spatial_pairs[210]", "spatial_pairs[211]", "spatial_pairs[212]", "spatial_pairs[213]", "spatial_pairs[214]", "spatial_pairs[215]", "spatial_pairs[216]", "spatial_pairs[217]", "spatial_pairs[218]", "spatial_pairs[219]", "spatial_pairs[220]", "spatial_pairs[221]", "spatial_pairs[222]", "spatial_pairs[223]", "spatial_pairs[224]", "spatial_pairs[225]", "spatial_pairs[226]", "spatial_pairs[227]", "spatial_pairs[228]", "spatial_pairs[229]", "spatial_pairs[230]", "spatial_pairs[231]", "spatial_pairs[232]", "spatial_pairs[233]", "spatial_pairs[234]", "spatial_pairs[235]", "spatial_pairs[236]", "spatial_pairs[237]", "spatial_pairs[238]", "spatial_pairs[239]", "spatial_pairs[240]", "spatial_pairs[241]", "spatial_pairs[242]", "spatial_pairs[243]", "spatial_pairs[244]", "spatial_pairs[245]", "spatial_pairs[246]", "spatial_pairs[247]", "spatial_pairs[248]", "spatial_pairs[249]", "spatial_pairs[250]", "spatial_pairs[251]", "spatial_pairs[252]", "spatial_pairs[253]", "spatial_pairs[254]", "spatial_pairs[255]", "spatial_pairs[256]", "spatial_pairs[257]", "spatial_pairs[258]", "spatial_pairs[259]", "spatial_pairs[260]", "spatial_pairs[261]", "spatial_pairs[262]", "spatial_pairs[263]", "spatial_pairs[264]", "spatial_pairs[265]", "spatial_pairs[266]", "spatial_pairs[267]", "spatial_pairs[268]", "spatial_pairs[269]", "spatial_pairs[270]", "spatial_pairs[271]", "spatial_pairs[272]", "spatial_pairs[273]", "spatial_pairs[274]", "spatial_pairs[275]", "spatial_pairs[276]", "spatial_pairs[277]", "spatial_pairs[278]", "spatial_pairs[279]", "spatial_pairs[280]", "spatial_pairs[281]", "spatial_pairs[282]", "spatial_pairs[283]", "spatial_pairs[284]", "spatial_pairs[285]", "spatial_pairs[286]", "spatial_pairs[287]", "spatial_pairs[288]", "spatial_pairs[289]", "spatial_pairs[290]", "spatial_pairs[291]", "spatial_pairs[292]", "spatial_pairs[293]", "spatial_pairs[294]", "spatial_pairs[295]", "spatial_pairs[296]", "spatial_pairs[297]", "spatial_pairs[298]", "spatial_pairs[299]", "spatial_pairs[300]", "spatial_pairs[301]", "spatial_pairs[302]", "spatial_pairs[303]", "spatial_pairs[304]", "spatial_pairs[305]", "spatial_pairs[306]", "spatial_pairs[307]", "spatial_pairs[308]", "spatial_pairs[309]", "spatial_pairs[310]", "spatial_pairs[311]", "spatial_pairs[312]", "spatial_pairs[313]", "spatial_pairs[314]", "spatial_pairs[315]", "spatial_pairs[316]", "spatial_pairs[317]", "spatial_pairs[318]", "spatial_pairs[319]", "spatial_pairs[320]", "spatial_pairs[321]", "spatial_pairs[322]", "spatial_pairs[323]", "spatial_pairs[324]", "spatial_pairs[325]", "spatial_pairs[326]", "spatial_pairs[327]", "spatial_pairs[328]", "spatial_pairs[329]", "spatial_pairs[330]", "spatial_pairs[331]", "spatial_pairs[332]", "spatial_pairs[333]", "spatial_pairs[334]", "spatial_pairs[335]", "spatial_pairs[336]", "spatial_pairs[337]", "spatial_pairs[338]", "spatial_pairs[339]", "spatial_pairs[340]", "spatial_pairs[341]", "spatial_pairs[342]", "spatial_pairs[343]", "spatial_pairs[344]", "spatial_pairs[345]", "spatial_pairs[346]", "spatial_pairs[347]", "spatial_pairs[348]", "spatial_pairs[349]", "spatial_pairs[350]", "spatial_pairs[351]", "spatial_pairs[352]", "spatial_pairs[353]", "spatial_pairs[354]", "spatial_pairs[355]", "spatial_pairs[356]", "spatial_pairs[357]", "spatial_pairs[358]", "spatial_pairs[359]", "spatial_pairs[360]", "spatial_pairs[361]", "spatial_pairs[362]", "spatial_pairs[363]", "spatial_pairs[364]", "spatial_pairs[365]", "spatial_pairs[366]", "spatial_pairs[367]", "spatial_pairs[368]", "spatial_pairs[369]", "spatial_pairs[370]", "spatial_pairs[371]", "spatial_pairs[372]", "spatial_pairs[373]", "spatial_pairs[374]", "spatial_pairs[375]", "spatial_pairs[376]", "spatial_pairs[377]", "spatial_pairs[378]", "spatial_pairs[379]", "spatial_pairs[380]", "spatial_pairs[381]", "spatial_pairs[382]", "spatial_pairs[383]", "spatial_pairs[384]", "spatial_pairs[385]", "spatial_pairs[386]", "spatial_pairs[387]", "spatial_pairs[388]", "spatial_pairs[389]", "spatial_pairs[390]", "spatial_pairs[391]", "spatial_pairs[392]", "spatial_pairs[393]", "spatial_pairs[394]", "spatial_pairs[395]", "spatial_pairs[396]", "spatial_pairs[397]", "spatial_pairs[398]", "spatial_pairs[399]", "spatial_pairs[400]", "spatial_pairs[401]", "spatial_pairs[402]", "spatial_pairs[403]", "spatial_pairs[404]", "spatial_pairs[405]", "spatial_pairs[406]", "spatial_pairs[407]", "spatial_pairs[408]", "spatial_pairs[409]", "spatial_pairs[410]", "spatial_pairs[411]", "spatial_pairs[412]", "spatial_pairs[413]", "spatial_pairs[414]", "spatial_pairs[415]", "spatial_pairs[416]", "spatial_pairs[417]", "spatial_pairs[418]", "spatial_pairs[419]", "spatial_pairs[420]", "spatial_pairs[421]", "spatial_pairs[422]", "spatial_pairs[423]", "spatial_pairs[424]", "spatial_pairs[425]", "spatial_pairs[426]", "spatial_pairs[427]", "spatial_pairs[428]", "spatial_pairs[429]", "spatial_pairs[430]", "spatial_pairs[431]"};
const vector<string> __vecstr_instance_Region = {"regions[0]", "regions[1]", "regions[2]", "regions[3]", "regions[4]", "regions[5]", "regions[6]", "regions[7]", "regions[8]"};
const vector<string> __vecstr_instance_County = {"counties[0]", "counties[1]", "counties[2]", "counties[3]", "counties[4]", "counties[5]", "counties[6]", "counties[7]", "counties[8]", "counties[9]", "counties[10]", "counties[11]", "counties[12]", "counties[13]", "counties[14]", "counties[15]", "counties[16]", "counties[17]", "counties[18]", "counties[19]", "counties[20]", "counties[21]", "counties[22]", "counties[23]", "counties[24]", "counties[25]", "counties[26]", "counties[27]", "counties[28]", "counties[29]", "counties[30]", "counties[31]", "counties[32]", "counties[33]", "counties[34]", "counties[35]", "counties[36]", "counties[37]", "counties[38]", "counties[39]", "counties[40]", "counties[41]", "counties[42]", "counties[43]", "counties[44]", "counties[45]", "counties[46]", "counties[47]", "counties[48]", "counties[49]", "counties[50]", "counties[51]", "counties[52]", "counties[53]", "counties[54]", "counties[55]", "counties[56]", "counties[57]", "counties[58]", "counties[59]", "counties[60]", "counties[61]", "counties[62]", "counties[63]", "counties[64]", "counties[65]", "counties[66]", "counties[67]", "counties[68]", "counties[69]", "counties[70]", "counties[71]", "counties[72]", "counties[73]", "counties[74]", "counties[75]", "counties[76]", "counties[77]", "counties[78]", "counties[79]", "counties[80]", "counties[81]"};
void _eval_query();
void _init_storage();
void _init_world();
void _garbage_collection();
void _print_answer();
const int _TOT_LOOP = 20000000;
const int _BURN_IN = 19999900;
int _tot_round = -19999900;
const double __fixed_rho = 0.50000000;
const double __fixed_tau1 = 3.00000000;
const mat __fixed_county_map = loadRealMatrix("data_processed/county_map.txt");
const mat __fixed_region_pop = loadRealMatrix("data_processed/region_pops.txt");
const mat __fixed_covariates1 = loadRealMatrix("data_processed/covariates1.txt");
const mat __fixed_covariates2 = loadRealMatrix("data_processed/covariates2.txt");
const mat __fixed_priors = loadRealMatrix("data_processed/priors.txt");
const mat __fixed_D = loadRealMatrix("data_processed/D.txt");
const mat __fixed_W = loadRealMatrix("data_processed/W.txt");
const mat __fixed_observations = loadRealMatrix("data_processed/obs.txt");
const mat __fixed_spatial_obs = loadRealMatrix("data_processed/spatial_obs.txt");
const mat __fixed_temporal_obs = loadRealMatrix("data_processed/temporal_obs.txt");
int __fixed_toWeek(int);
int __fixed_toCounty(int);
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
class _Var_bias: public BayesVar<double> {
public:
  _Var_bias();
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
_Var_bias* _mem_bias;
class _Var_temporal_edge: public BayesVar<char> {
public:
  int c;
  int t;
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
};
DynamicTable<_Var_temporal_edge*,2> _mem_temporal_edge;
class _Var_spatial_edge: public BayesVar<char> {
public:
  int t;
  int s;
  _Var_spatial_edge(int,int);
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
};
DynamicTable<_Var_spatial_edge*,2> _mem_spatial_edge;
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
TruncatedGauss TruncatedGauss26331968;
TruncatedGauss TruncatedGauss26332480;
Gaussian Gaussian26332832;
BooleanDistrib BooleanDistrib26344160;
BooleanDistrib BooleanDistrib26348160;
Gaussian Gaussian26352432;
Gaussian Gaussian26410224;
Hist<double> _answer_0 = Hist<double>(false, 20);
Hist<double> _answer_1 = Hist<double>(false, 20);
Hist<double> _answer_2 = Hist<double>(false, 20);
Hist<double> _answer_3 = Hist<double>(false, 20);
DynamicTable<Hist<double>*,2> _answer_4;
void sample();

void _eval_query()
{
  _tot_round++;
  if (_tot_round<=0)
    return ;
  _answer_0.add(__fixed_tau1,1);
  _answer_1.add(__fixed_rho,1);
  _answer_2.add(_mem_beta1->getval(),1);
  _answer_3.add(_mem_beta2->getval(),1);
  for (int c = 0;c<82;c++)
  for (int t = 0;t<103;t++)
  _answer_4[c][t]->add(_mem_logit[c][t]->getval(),1);


}
void _init_storage()
{
  _mem_beta1=new _Var_beta1();
  _mem_beta2=new _Var_beta2();
  _mem_bias=new _Var_bias();
  _mem_temporal_edge.resize(0,82);
  _mem_temporal_edge.resize(1,102);
  for (int c = 0;c<82;c++)
  {
    for (int t = 0;t<102;t++)
    {
      _mem_temporal_edge[c][t]=new _Var_temporal_edge(c, t);
    }

  }

  _mem_spatial_edge.resize(0,103);
  _mem_spatial_edge.resize(1,432);
  for (int t = 0;t<103;t++)
  {
    for (int s = 0;s<432;s++)
    {
      _mem_spatial_edge[t][s]=new _Var_spatial_edge(t, s);
    }

  }

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

  TruncatedGauss26331968.init(0.75000000,0.50000000,0.000000,2.00000000);
  TruncatedGauss26332480.init(0.75000000,0.50000000,0.000000,2.00000000);
  Gaussian26332832.init(-4.00000000,1.00000000);
  _answer_4.resize(0,82);
  _answer_4.resize(1,103);
  for (int c = 0;c<82;c++)
  {
    for (int t = 0;t<103;t++)
    {
      _answer_4[c][t]=new Hist<double>(false, 20);
    }

  }

}
void _init_world()
{
  for (int c = 0;c<82;c++)
  for (int t = 0;t<102;t++)
  _util_set_evidence<char>(_mem_temporal_edge[c][t],1);


  for (int t = 0;t<103;t++)
  for (int s = 0;s<432;s++)
  _util_set_evidence<char>(_mem_spatial_edge[t][s],1);


  for (int r = 0;r<9;r++)
  for (int t = 0;t<103;t++)
  if (__fixed_observations(t,r)>-1.00000000)
    _util_set_evidence<double>(_mem_region_rate[r][t],__fixed_observations(t,r));


}
void _garbage_collection()
{
  _free_obj(_mem_beta1);
  _free_obj(_mem_beta2);
  _free_obj(_mem_bias);
  _free_obj(_mem_temporal_edge);
  _free_obj(_mem_spatial_edge);
  _free_obj(_mem_logit);
  _free_obj(_mem_region_rate);
}
void _print_answer()
{
  _answer_0.print("tau1");
  _answer_1.print("rho");
  _answer_2.print("beta1");
  _answer_3.print("beta2");
  char buffer4[256];
  for (int c = 0;c<82;c++)
  for (int t = 0;t<103;t++)
  {
    sprintf(buffer4,"logit(County[%d], Week[%d])\n",c,t);
    _answer_4[c][t]->print(buffer4);
  }


}
int __fixed_toWeek(int i)
{
  return i;
}
int __fixed_toCounty(int i)
{
  return i;
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
  return TruncatedGauss26331968.loglikeli(val);
}
double _Var_beta1::getcachelikeli()
{
  auto _t_val = getcache();
  return TruncatedGauss26331968.loglikeli(_t_val);
}
void _Var_beta1::sample()
{
  val=TruncatedGauss26331968.gen();
}
void _Var_beta1::sample_cache()
{
  cache_val=TruncatedGauss26331968.gen();
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
  return TruncatedGauss26332480.loglikeli(val);
}
double _Var_beta2::getcachelikeli()
{
  auto _t_val = getcache();
  return TruncatedGauss26332480.loglikeli(_t_val);
}
void _Var_beta2::sample()
{
  val=TruncatedGauss26332480.gen();
}
void _Var_beta2::sample_cache()
{
  cache_val=TruncatedGauss26332480.gen();
}
void _Var_beta2::active_edge()
{}
void _Var_beta2::remove_edge()
{}
void _Var_beta2::mcmc_resample()
{
  mh_parent_resample_arg(this);
}
_Var_bias::_Var_bias()
{}
string _Var_bias::getname()
{
  return "bias";
}
double& _Var_bias::getval()
{
  return getval_arg(this);
}
double& _Var_bias::getcache()
{
  return getcache_arg(this);
}
void _Var_bias::clear()
{
  return clear_arg(this);
}
double _Var_bias::getlikeli()
{
  return Gaussian26332832.loglikeli(val);
}
double _Var_bias::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian26332832.loglikeli(_t_val);
}
void _Var_bias::sample()
{
  val=Gaussian26332832.gen();
}
void _Var_bias::sample_cache()
{
  cache_val=Gaussian26332832.gen();
}
void _Var_bias::active_edge()
{}
void _Var_bias::remove_edge()
{}
void _Var_bias::mcmc_resample()
{
  mh_parent_resample_arg(this);
}
_Var_temporal_edge::_Var_temporal_edge(int _c, int _t):c(_c),t(_t)
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
  return BooleanDistrib26344160.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getval()-_mem_bias->getval())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getval()-_mem_bias->getval()))),BooleanDistrib26344160.loglikeli(val);
}
double _Var_temporal_edge::getcachelikeli()
{
  auto _t_val = getcache();
  return BooleanDistrib26344160.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getcache()-_mem_bias->getcache())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getcache()-_mem_bias->getcache()))),BooleanDistrib26344160.loglikeli(_t_val);
}
void _Var_temporal_edge::sample()
{
  val=(BooleanDistrib26344160.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getval()-_mem_bias->getval())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getval()-_mem_bias->getval()))),BooleanDistrib26344160.gen());
}
void _Var_temporal_edge::sample_cache()
{
  cache_val=(BooleanDistrib26344160.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getcache()-_mem_bias->getcache())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getcache()-_mem_bias->getcache()))),BooleanDistrib26344160.gen());
}
void _Var_temporal_edge::active_edge()
{
  _mem_bias->add_contig(this);
  _mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->add_contig(this);
  _mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->add_contig(this);
  _mem_bias->add_child(this);
  _mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->add_child(this);
  _mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->add_child(this);
}
void _Var_temporal_edge::remove_edge()
{
  _mem_bias->erase_contig(this);
  _mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->erase_contig(this);
  _mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->erase_contig(this);
  _mem_bias->erase_child(this);
  _mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->erase_child(this);
  _mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->erase_child(this);
}
void _Var_temporal_edge::mcmc_resample()
{
  mh_parent_resample_arg(this);
}
_Var_spatial_edge::_Var_spatial_edge(int _t, int _s):t(_t),s(_s)
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
  return BooleanDistrib26348160.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getval()-_mem_bias->getval())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getval()-_mem_bias->getval()))),BooleanDistrib26348160.loglikeli(val);
}
double _Var_spatial_edge::getcachelikeli()
{
  auto _t_val = getcache();
  return BooleanDistrib26348160.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getcache()-_mem_bias->getcache())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getcache()-_mem_bias->getcache()))),BooleanDistrib26348160.loglikeli(_t_val);
}
void _Var_spatial_edge::sample()
{
  val=(BooleanDistrib26348160.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getval()-_mem_bias->getval())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getval()-_mem_bias->getval()))),BooleanDistrib26348160.gen());
}
void _Var_spatial_edge::sample_cache()
{
  cache_val=(BooleanDistrib26348160.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getcache()-_mem_bias->getcache())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getcache()-_mem_bias->getcache()))),BooleanDistrib26348160.gen());
}
void _Var_spatial_edge::active_edge()
{
  _mem_bias->add_contig(this);
  _mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->add_contig(this);
  _mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->add_contig(this);
  _mem_bias->add_child(this);
  _mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->add_child(this);
  _mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->add_child(this);
}
void _Var_spatial_edge::remove_edge()
{
  _mem_bias->erase_contig(this);
  _mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->erase_contig(this);
  _mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->erase_contig(this);
  _mem_bias->erase_child(this);
  _mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->erase_child(this);
  _mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->erase_child(this);
}
void _Var_spatial_edge::mcmc_resample()
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
  return Gaussian26352432.init(_mem_bias->getval()+_mem_beta1->getval()*__fixed_covariates1(c,t)+_mem_beta2->getval()*__fixed_covariates2(c,t),__fixed_D[c]/9.00000000),Gaussian26352432.loglikeli(val);
}
double _Var_logit::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian26352432.init(_mem_bias->getcache()+_mem_beta1->getcache()*__fixed_covariates1(c,t)+_mem_beta2->getcache()*__fixed_covariates2(c,t),__fixed_D[c]/9.00000000),Gaussian26352432.loglikeli(_t_val);
}
void _Var_logit::sample()
{
  val=(Gaussian26352432.init(_mem_bias->getval()+_mem_beta1->getval()*__fixed_covariates1(c,t)+_mem_beta2->getval()*__fixed_covariates2(c,t),__fixed_D[c]/9.00000000),Gaussian26352432.gen());
}
void _Var_logit::sample_cache()
{
  cache_val=(Gaussian26352432.init(_mem_bias->getcache()+_mem_beta1->getcache()*__fixed_covariates1(c,t)+_mem_beta2->getcache()*__fixed_covariates2(c,t),__fixed_D[c]/9.00000000),Gaussian26352432.gen());
}
void _Var_logit::active_edge()
{
  _mem_beta1->add_contig(this);
  _mem_beta2->add_contig(this);
  _mem_bias->add_contig(this);
  _mem_beta1->add_child(this);
  _mem_beta2->add_child(this);
  _mem_bias->add_child(this);
}
void _Var_logit::remove_edge()
{
  _mem_beta1->erase_contig(this);
  _mem_beta2->erase_contig(this);
  _mem_bias->erase_contig(this);
  _mem_beta1->erase_child(this);
  _mem_beta2->erase_child(this);
  _mem_bias->erase_child(this);
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
  return Gaussian26410224.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getval()), __fixed_sigmoid(_mem_logit[1][t]->getval()), __fixed_sigmoid(_mem_logit[2][t]->getval()), __fixed_sigmoid(_mem_logit[3][t]->getval()), __fixed_sigmoid(_mem_logit[4][t]->getval()), __fixed_sigmoid(_mem_logit[5][t]->getval()), __fixed_sigmoid(_mem_logit[6][t]->getval()), __fixed_sigmoid(_mem_logit[7][t]->getval()), __fixed_sigmoid(_mem_logit[8][t]->getval()), __fixed_sigmoid(_mem_logit[9][t]->getval()), __fixed_sigmoid(_mem_logit[10][t]->getval()), __fixed_sigmoid(_mem_logit[11][t]->getval()), __fixed_sigmoid(_mem_logit[12][t]->getval()), __fixed_sigmoid(_mem_logit[13][t]->getval()), __fixed_sigmoid(_mem_logit[14][t]->getval()), __fixed_sigmoid(_mem_logit[15][t]->getval()), __fixed_sigmoid(_mem_logit[16][t]->getval()), __fixed_sigmoid(_mem_logit[17][t]->getval()), __fixed_sigmoid(_mem_logit[18][t]->getval()), __fixed_sigmoid(_mem_logit[19][t]->getval()), __fixed_sigmoid(_mem_logit[20][t]->getval()), __fixed_sigmoid(_mem_logit[21][t]->getval()), __fixed_sigmoid(_mem_logit[22][t]->getval()), __fixed_sigmoid(_mem_logit[23][t]->getval()), __fixed_sigmoid(_mem_logit[24][t]->getval()), __fixed_sigmoid(_mem_logit[25][t]->getval()), __fixed_sigmoid(_mem_logit[26][t]->getval()), __fixed_sigmoid(_mem_logit[27][t]->getval()), __fixed_sigmoid(_mem_logit[28][t]->getval()), __fixed_sigmoid(_mem_logit[29][t]->getval()), __fixed_sigmoid(_mem_logit[30][t]->getval()), __fixed_sigmoid(_mem_logit[31][t]->getval()), __fixed_sigmoid(_mem_logit[32][t]->getval()), __fixed_sigmoid(_mem_logit[33][t]->getval()), __fixed_sigmoid(_mem_logit[34][t]->getval()), __fixed_sigmoid(_mem_logit[35][t]->getval()), __fixed_sigmoid(_mem_logit[36][t]->getval()), __fixed_sigmoid(_mem_logit[37][t]->getval()), __fixed_sigmoid(_mem_logit[38][t]->getval()), __fixed_sigmoid(_mem_logit[39][t]->getval()), __fixed_sigmoid(_mem_logit[40][t]->getval()), __fixed_sigmoid(_mem_logit[41][t]->getval()), __fixed_sigmoid(_mem_logit[42][t]->getval()), __fixed_sigmoid(_mem_logit[43][t]->getval()), __fixed_sigmoid(_mem_logit[44][t]->getval()), __fixed_sigmoid(_mem_logit[45][t]->getval()), __fixed_sigmoid(_mem_logit[46][t]->getval()), __fixed_sigmoid(_mem_logit[47][t]->getval()), __fixed_sigmoid(_mem_logit[48][t]->getval()), __fixed_sigmoid(_mem_logit[49][t]->getval()), __fixed_sigmoid(_mem_logit[50][t]->getval()), __fixed_sigmoid(_mem_logit[51][t]->getval()), __fixed_sigmoid(_mem_logit[52][t]->getval()), __fixed_sigmoid(_mem_logit[53][t]->getval()), __fixed_sigmoid(_mem_logit[54][t]->getval()), __fixed_sigmoid(_mem_logit[55][t]->getval()), __fixed_sigmoid(_mem_logit[56][t]->getval()), __fixed_sigmoid(_mem_logit[57][t]->getval()), __fixed_sigmoid(_mem_logit[58][t]->getval()), __fixed_sigmoid(_mem_logit[59][t]->getval()), __fixed_sigmoid(_mem_logit[60][t]->getval()), __fixed_sigmoid(_mem_logit[61][t]->getval()), __fixed_sigmoid(_mem_logit[62][t]->getval()), __fixed_sigmoid(_mem_logit[63][t]->getval()), __fixed_sigmoid(_mem_logit[64][t]->getval()), __fixed_sigmoid(_mem_logit[65][t]->getval()), __fixed_sigmoid(_mem_logit[66][t]->getval()), __fixed_sigmoid(_mem_logit[67][t]->getval()), __fixed_sigmoid(_mem_logit[68][t]->getval()), __fixed_sigmoid(_mem_logit[69][t]->getval()), __fixed_sigmoid(_mem_logit[70][t]->getval()), __fixed_sigmoid(_mem_logit[71][t]->getval()), __fixed_sigmoid(_mem_logit[72][t]->getval()), __fixed_sigmoid(_mem_logit[73][t]->getval()), __fixed_sigmoid(_mem_logit[74][t]->getval()), __fixed_sigmoid(_mem_logit[75][t]->getval()), __fixed_sigmoid(_mem_logit[76][t]->getval()), __fixed_sigmoid(_mem_logit[77][t]->getval()), __fixed_sigmoid(_mem_logit[78][t]->getval()), __fixed_sigmoid(_mem_logit[79][t]->getval()), __fixed_sigmoid(_mem_logit[80][t]->getval()), __fixed_sigmoid(_mem_logit[81][t]->getval())}))/__fixed_region_pop[r],0.00500000),Gaussian26410224.loglikeli(val);
}
double _Var_region_rate::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian26410224.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getcache()), __fixed_sigmoid(_mem_logit[1][t]->getcache()), __fixed_sigmoid(_mem_logit[2][t]->getcache()), __fixed_sigmoid(_mem_logit[3][t]->getcache()), __fixed_sigmoid(_mem_logit[4][t]->getcache()), __fixed_sigmoid(_mem_logit[5][t]->getcache()), __fixed_sigmoid(_mem_logit[6][t]->getcache()), __fixed_sigmoid(_mem_logit[7][t]->getcache()), __fixed_sigmoid(_mem_logit[8][t]->getcache()), __fixed_sigmoid(_mem_logit[9][t]->getcache()), __fixed_sigmoid(_mem_logit[10][t]->getcache()), __fixed_sigmoid(_mem_logit[11][t]->getcache()), __fixed_sigmoid(_mem_logit[12][t]->getcache()), __fixed_sigmoid(_mem_logit[13][t]->getcache()), __fixed_sigmoid(_mem_logit[14][t]->getcache()), __fixed_sigmoid(_mem_logit[15][t]->getcache()), __fixed_sigmoid(_mem_logit[16][t]->getcache()), __fixed_sigmoid(_mem_logit[17][t]->getcache()), __fixed_sigmoid(_mem_logit[18][t]->getcache()), __fixed_sigmoid(_mem_logit[19][t]->getcache()), __fixed_sigmoid(_mem_logit[20][t]->getcache()), __fixed_sigmoid(_mem_logit[21][t]->getcache()), __fixed_sigmoid(_mem_logit[22][t]->getcache()), __fixed_sigmoid(_mem_logit[23][t]->getcache()), __fixed_sigmoid(_mem_logit[24][t]->getcache()), __fixed_sigmoid(_mem_logit[25][t]->getcache()), __fixed_sigmoid(_mem_logit[26][t]->getcache()), __fixed_sigmoid(_mem_logit[27][t]->getcache()), __fixed_sigmoid(_mem_logit[28][t]->getcache()), __fixed_sigmoid(_mem_logit[29][t]->getcache()), __fixed_sigmoid(_mem_logit[30][t]->getcache()), __fixed_sigmoid(_mem_logit[31][t]->getcache()), __fixed_sigmoid(_mem_logit[32][t]->getcache()), __fixed_sigmoid(_mem_logit[33][t]->getcache()), __fixed_sigmoid(_mem_logit[34][t]->getcache()), __fixed_sigmoid(_mem_logit[35][t]->getcache()), __fixed_sigmoid(_mem_logit[36][t]->getcache()), __fixed_sigmoid(_mem_logit[37][t]->getcache()), __fixed_sigmoid(_mem_logit[38][t]->getcache()), __fixed_sigmoid(_mem_logit[39][t]->getcache()), __fixed_sigmoid(_mem_logit[40][t]->getcache()), __fixed_sigmoid(_mem_logit[41][t]->getcache()), __fixed_sigmoid(_mem_logit[42][t]->getcache()), __fixed_sigmoid(_mem_logit[43][t]->getcache()), __fixed_sigmoid(_mem_logit[44][t]->getcache()), __fixed_sigmoid(_mem_logit[45][t]->getcache()), __fixed_sigmoid(_mem_logit[46][t]->getcache()), __fixed_sigmoid(_mem_logit[47][t]->getcache()), __fixed_sigmoid(_mem_logit[48][t]->getcache()), __fixed_sigmoid(_mem_logit[49][t]->getcache()), __fixed_sigmoid(_mem_logit[50][t]->getcache()), __fixed_sigmoid(_mem_logit[51][t]->getcache()), __fixed_sigmoid(_mem_logit[52][t]->getcache()), __fixed_sigmoid(_mem_logit[53][t]->getcache()), __fixed_sigmoid(_mem_logit[54][t]->getcache()), __fixed_sigmoid(_mem_logit[55][t]->getcache()), __fixed_sigmoid(_mem_logit[56][t]->getcache()), __fixed_sigmoid(_mem_logit[57][t]->getcache()), __fixed_sigmoid(_mem_logit[58][t]->getcache()), __fixed_sigmoid(_mem_logit[59][t]->getcache()), __fixed_sigmoid(_mem_logit[60][t]->getcache()), __fixed_sigmoid(_mem_logit[61][t]->getcache()), __fixed_sigmoid(_mem_logit[62][t]->getcache()), __fixed_sigmoid(_mem_logit[63][t]->getcache()), __fixed_sigmoid(_mem_logit[64][t]->getcache()), __fixed_sigmoid(_mem_logit[65][t]->getcache()), __fixed_sigmoid(_mem_logit[66][t]->getcache()), __fixed_sigmoid(_mem_logit[67][t]->getcache()), __fixed_sigmoid(_mem_logit[68][t]->getcache()), __fixed_sigmoid(_mem_logit[69][t]->getcache()), __fixed_sigmoid(_mem_logit[70][t]->getcache()), __fixed_sigmoid(_mem_logit[71][t]->getcache()), __fixed_sigmoid(_mem_logit[72][t]->getcache()), __fixed_sigmoid(_mem_logit[73][t]->getcache()), __fixed_sigmoid(_mem_logit[74][t]->getcache()), __fixed_sigmoid(_mem_logit[75][t]->getcache()), __fixed_sigmoid(_mem_logit[76][t]->getcache()), __fixed_sigmoid(_mem_logit[77][t]->getcache()), __fixed_sigmoid(_mem_logit[78][t]->getcache()), __fixed_sigmoid(_mem_logit[79][t]->getcache()), __fixed_sigmoid(_mem_logit[80][t]->getcache()), __fixed_sigmoid(_mem_logit[81][t]->getcache())}))/__fixed_region_pop[r],0.00500000),Gaussian26410224.loglikeli(_t_val);
}
void _Var_region_rate::sample()
{
  val=(Gaussian26410224.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getval()), __fixed_sigmoid(_mem_logit[1][t]->getval()), __fixed_sigmoid(_mem_logit[2][t]->getval()), __fixed_sigmoid(_mem_logit[3][t]->getval()), __fixed_sigmoid(_mem_logit[4][t]->getval()), __fixed_sigmoid(_mem_logit[5][t]->getval()), __fixed_sigmoid(_mem_logit[6][t]->getval()), __fixed_sigmoid(_mem_logit[7][t]->getval()), __fixed_sigmoid(_mem_logit[8][t]->getval()), __fixed_sigmoid(_mem_logit[9][t]->getval()), __fixed_sigmoid(_mem_logit[10][t]->getval()), __fixed_sigmoid(_mem_logit[11][t]->getval()), __fixed_sigmoid(_mem_logit[12][t]->getval()), __fixed_sigmoid(_mem_logit[13][t]->getval()), __fixed_sigmoid(_mem_logit[14][t]->getval()), __fixed_sigmoid(_mem_logit[15][t]->getval()), __fixed_sigmoid(_mem_logit[16][t]->getval()), __fixed_sigmoid(_mem_logit[17][t]->getval()), __fixed_sigmoid(_mem_logit[18][t]->getval()), __fixed_sigmoid(_mem_logit[19][t]->getval()), __fixed_sigmoid(_mem_logit[20][t]->getval()), __fixed_sigmoid(_mem_logit[21][t]->getval()), __fixed_sigmoid(_mem_logit[22][t]->getval()), __fixed_sigmoid(_mem_logit[23][t]->getval()), __fixed_sigmoid(_mem_logit[24][t]->getval()), __fixed_sigmoid(_mem_logit[25][t]->getval()), __fixed_sigmoid(_mem_logit[26][t]->getval()), __fixed_sigmoid(_mem_logit[27][t]->getval()), __fixed_sigmoid(_mem_logit[28][t]->getval()), __fixed_sigmoid(_mem_logit[29][t]->getval()), __fixed_sigmoid(_mem_logit[30][t]->getval()), __fixed_sigmoid(_mem_logit[31][t]->getval()), __fixed_sigmoid(_mem_logit[32][t]->getval()), __fixed_sigmoid(_mem_logit[33][t]->getval()), __fixed_sigmoid(_mem_logit[34][t]->getval()), __fixed_sigmoid(_mem_logit[35][t]->getval()), __fixed_sigmoid(_mem_logit[36][t]->getval()), __fixed_sigmoid(_mem_logit[37][t]->getval()), __fixed_sigmoid(_mem_logit[38][t]->getval()), __fixed_sigmoid(_mem_logit[39][t]->getval()), __fixed_sigmoid(_mem_logit[40][t]->getval()), __fixed_sigmoid(_mem_logit[41][t]->getval()), __fixed_sigmoid(_mem_logit[42][t]->getval()), __fixed_sigmoid(_mem_logit[43][t]->getval()), __fixed_sigmoid(_mem_logit[44][t]->getval()), __fixed_sigmoid(_mem_logit[45][t]->getval()), __fixed_sigmoid(_mem_logit[46][t]->getval()), __fixed_sigmoid(_mem_logit[47][t]->getval()), __fixed_sigmoid(_mem_logit[48][t]->getval()), __fixed_sigmoid(_mem_logit[49][t]->getval()), __fixed_sigmoid(_mem_logit[50][t]->getval()), __fixed_sigmoid(_mem_logit[51][t]->getval()), __fixed_sigmoid(_mem_logit[52][t]->getval()), __fixed_sigmoid(_mem_logit[53][t]->getval()), __fixed_sigmoid(_mem_logit[54][t]->getval()), __fixed_sigmoid(_mem_logit[55][t]->getval()), __fixed_sigmoid(_mem_logit[56][t]->getval()), __fixed_sigmoid(_mem_logit[57][t]->getval()), __fixed_sigmoid(_mem_logit[58][t]->getval()), __fixed_sigmoid(_mem_logit[59][t]->getval()), __fixed_sigmoid(_mem_logit[60][t]->getval()), __fixed_sigmoid(_mem_logit[61][t]->getval()), __fixed_sigmoid(_mem_logit[62][t]->getval()), __fixed_sigmoid(_mem_logit[63][t]->getval()), __fixed_sigmoid(_mem_logit[64][t]->getval()), __fixed_sigmoid(_mem_logit[65][t]->getval()), __fixed_sigmoid(_mem_logit[66][t]->getval()), __fixed_sigmoid(_mem_logit[67][t]->getval()), __fixed_sigmoid(_mem_logit[68][t]->getval()), __fixed_sigmoid(_mem_logit[69][t]->getval()), __fixed_sigmoid(_mem_logit[70][t]->getval()), __fixed_sigmoid(_mem_logit[71][t]->getval()), __fixed_sigmoid(_mem_logit[72][t]->getval()), __fixed_sigmoid(_mem_logit[73][t]->getval()), __fixed_sigmoid(_mem_logit[74][t]->getval()), __fixed_sigmoid(_mem_logit[75][t]->getval()), __fixed_sigmoid(_mem_logit[76][t]->getval()), __fixed_sigmoid(_mem_logit[77][t]->getval()), __fixed_sigmoid(_mem_logit[78][t]->getval()), __fixed_sigmoid(_mem_logit[79][t]->getval()), __fixed_sigmoid(_mem_logit[80][t]->getval()), __fixed_sigmoid(_mem_logit[81][t]->getval())}))/__fixed_region_pop[r],0.00500000),Gaussian26410224.gen());
}
void _Var_region_rate::sample_cache()
{
  cache_val=(Gaussian26410224.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getcache()), __fixed_sigmoid(_mem_logit[1][t]->getcache()), __fixed_sigmoid(_mem_logit[2][t]->getcache()), __fixed_sigmoid(_mem_logit[3][t]->getcache()), __fixed_sigmoid(_mem_logit[4][t]->getcache()), __fixed_sigmoid(_mem_logit[5][t]->getcache()), __fixed_sigmoid(_mem_logit[6][t]->getcache()), __fixed_sigmoid(_mem_logit[7][t]->getcache()), __fixed_sigmoid(_mem_logit[8][t]->getcache()), __fixed_sigmoid(_mem_logit[9][t]->getcache()), __fixed_sigmoid(_mem_logit[10][t]->getcache()), __fixed_sigmoid(_mem_logit[11][t]->getcache()), __fixed_sigmoid(_mem_logit[12][t]->getcache()), __fixed_sigmoid(_mem_logit[13][t]->getcache()), __fixed_sigmoid(_mem_logit[14][t]->getcache()), __fixed_sigmoid(_mem_logit[15][t]->getcache()), __fixed_sigmoid(_mem_logit[16][t]->getcache()), __fixed_sigmoid(_mem_logit[17][t]->getcache()), __fixed_sigmoid(_mem_logit[18][t]->getcache()), __fixed_sigmoid(_mem_logit[19][t]->getcache()), __fixed_sigmoid(_mem_logit[20][t]->getcache()), __fixed_sigmoid(_mem_logit[21][t]->getcache()), __fixed_sigmoid(_mem_logit[22][t]->getcache()), __fixed_sigmoid(_mem_logit[23][t]->getcache()), __fixed_sigmoid(_mem_logit[24][t]->getcache()), __fixed_sigmoid(_mem_logit[25][t]->getcache()), __fixed_sigmoid(_mem_logit[26][t]->getcache()), __fixed_sigmoid(_mem_logit[27][t]->getcache()), __fixed_sigmoid(_mem_logit[28][t]->getcache()), __fixed_sigmoid(_mem_logit[29][t]->getcache()), __fixed_sigmoid(_mem_logit[30][t]->getcache()), __fixed_sigmoid(_mem_logit[31][t]->getcache()), __fixed_sigmoid(_mem_logit[32][t]->getcache()), __fixed_sigmoid(_mem_logit[33][t]->getcache()), __fixed_sigmoid(_mem_logit[34][t]->getcache()), __fixed_sigmoid(_mem_logit[35][t]->getcache()), __fixed_sigmoid(_mem_logit[36][t]->getcache()), __fixed_sigmoid(_mem_logit[37][t]->getcache()), __fixed_sigmoid(_mem_logit[38][t]->getcache()), __fixed_sigmoid(_mem_logit[39][t]->getcache()), __fixed_sigmoid(_mem_logit[40][t]->getcache()), __fixed_sigmoid(_mem_logit[41][t]->getcache()), __fixed_sigmoid(_mem_logit[42][t]->getcache()), __fixed_sigmoid(_mem_logit[43][t]->getcache()), __fixed_sigmoid(_mem_logit[44][t]->getcache()), __fixed_sigmoid(_mem_logit[45][t]->getcache()), __fixed_sigmoid(_mem_logit[46][t]->getcache()), __fixed_sigmoid(_mem_logit[47][t]->getcache()), __fixed_sigmoid(_mem_logit[48][t]->getcache()), __fixed_sigmoid(_mem_logit[49][t]->getcache()), __fixed_sigmoid(_mem_logit[50][t]->getcache()), __fixed_sigmoid(_mem_logit[51][t]->getcache()), __fixed_sigmoid(_mem_logit[52][t]->getcache()), __fixed_sigmoid(_mem_logit[53][t]->getcache()), __fixed_sigmoid(_mem_logit[54][t]->getcache()), __fixed_sigmoid(_mem_logit[55][t]->getcache()), __fixed_sigmoid(_mem_logit[56][t]->getcache()), __fixed_sigmoid(_mem_logit[57][t]->getcache()), __fixed_sigmoid(_mem_logit[58][t]->getcache()), __fixed_sigmoid(_mem_logit[59][t]->getcache()), __fixed_sigmoid(_mem_logit[60][t]->getcache()), __fixed_sigmoid(_mem_logit[61][t]->getcache()), __fixed_sigmoid(_mem_logit[62][t]->getcache()), __fixed_sigmoid(_mem_logit[63][t]->getcache()), __fixed_sigmoid(_mem_logit[64][t]->getcache()), __fixed_sigmoid(_mem_logit[65][t]->getcache()), __fixed_sigmoid(_mem_logit[66][t]->getcache()), __fixed_sigmoid(_mem_logit[67][t]->getcache()), __fixed_sigmoid(_mem_logit[68][t]->getcache()), __fixed_sigmoid(_mem_logit[69][t]->getcache()), __fixed_sigmoid(_mem_logit[70][t]->getcache()), __fixed_sigmoid(_mem_logit[71][t]->getcache()), __fixed_sigmoid(_mem_logit[72][t]->getcache()), __fixed_sigmoid(_mem_logit[73][t]->getcache()), __fixed_sigmoid(_mem_logit[74][t]->getcache()), __fixed_sigmoid(_mem_logit[75][t]->getcache()), __fixed_sigmoid(_mem_logit[76][t]->getcache()), __fixed_sigmoid(_mem_logit[77][t]->getcache()), __fixed_sigmoid(_mem_logit[78][t]->getcache()), __fixed_sigmoid(_mem_logit[79][t]->getcache()), __fixed_sigmoid(_mem_logit[80][t]->getcache()), __fixed_sigmoid(_mem_logit[81][t]->getcache())}))/__fixed_region_pop[r],0.00500000),Gaussian26410224.gen());
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
  printf("\nsample time: %fs (#iter = %d)\n",__elapsed_seconds.count(),20000000);
  swift::_print_answer();
  swift::_garbage_collection();
}
