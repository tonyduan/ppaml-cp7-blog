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

class _Var_bias;
class _Var_temporal_edge;
class _Var_spatial_edge;
class _Var_logit;
class _Var_region_rate;

const vector<string> __vecstr_instance_Week = {"weeks[0]", "weeks[1]", "weeks[2]", "weeks[3]", "weeks[4]", "weeks[5]", "weeks[6]", "weeks[7]", "weeks[8]", "weeks[9]", "weeks[10]", "weeks[11]", "weeks[12]", "weeks[13]", "weeks[14]", "weeks[15]", "weeks[16]", "weeks[17]", "weeks[18]", "weeks[19]", "weeks[20]", "weeks[21]", "weeks[22]", "weeks[23]", "weeks[24]", "weeks[25]", "weeks[26]", "weeks[27]", "weeks[28]", "weeks[29]", "weeks[30]", "weeks[31]", "weeks[32]", "weeks[33]", "weeks[34]", "weeks[35]", "weeks[36]", "weeks[37]", "weeks[38]", "weeks[39]", "weeks[40]", "weeks[41]", "weeks[42]", "weeks[43]", "weeks[44]", "weeks[45]", "weeks[46]", "weeks[47]", "weeks[48]", "weeks[49]", "weeks[50]", "weeks[51]", "weeks[52]", "weeks[53]", "weeks[54]", "weeks[55]", "weeks[56]", "weeks[57]", "weeks[58]", "weeks[59]", "weeks[60]", "weeks[61]", "weeks[62]", "weeks[63]", "weeks[64]", "weeks[65]", "weeks[66]", "weeks[67]", "weeks[68]", "weeks[69]", "weeks[70]", "weeks[71]", "weeks[72]", "weeks[73]", "weeks[74]", "weeks[75]", "weeks[76]", "weeks[77]", "weeks[78]", "weeks[79]", "weeks[80]", "weeks[81]", "weeks[82]", "weeks[83]", "weeks[84]", "weeks[85]", "weeks[86]", "weeks[87]", "weeks[88]", "weeks[89]", "weeks[90]", "weeks[91]", "weeks[92]", "weeks[93]", "weeks[94]", "weeks[95]", "weeks[96]", "weeks[97]", "weeks[98]", "weeks[99]", "weeks[100]", "weeks[101]", "weeks[102]"};
const vector<string> __vecstr_instance_TemporalTriple = {"temporal_triples[0]", "temporal_triples[1]", "temporal_triples[2]", "temporal_triples[3]", "temporal_triples[4]", "temporal_triples[5]", "temporal_triples[6]", "temporal_triples[7]", "temporal_triples[8]", "temporal_triples[9]", "temporal_triples[10]", "temporal_triples[11]", "temporal_triples[12]", "temporal_triples[13]", "temporal_triples[14]", "temporal_triples[15]", "temporal_triples[16]", "temporal_triples[17]", "temporal_triples[18]", "temporal_triples[19]", "temporal_triples[20]", "temporal_triples[21]", "temporal_triples[22]", "temporal_triples[23]", "temporal_triples[24]", "temporal_triples[25]", "temporal_triples[26]", "temporal_triples[27]", "temporal_triples[28]", "temporal_triples[29]", "temporal_triples[30]", "temporal_triples[31]", "temporal_triples[32]", "temporal_triples[33]", "temporal_triples[34]", "temporal_triples[35]", "temporal_triples[36]", "temporal_triples[37]", "temporal_triples[38]", "temporal_triples[39]", "temporal_triples[40]", "temporal_triples[41]", "temporal_triples[42]", "temporal_triples[43]", "temporal_triples[44]", "temporal_triples[45]", "temporal_triples[46]", "temporal_triples[47]", "temporal_triples[48]", "temporal_triples[49]", "temporal_triples[50]", "temporal_triples[51]", "temporal_triples[52]", "temporal_triples[53]", "temporal_triples[54]", "temporal_triples[55]", "temporal_triples[56]", "temporal_triples[57]", "temporal_triples[58]", "temporal_triples[59]", "temporal_triples[60]", "temporal_triples[61]", "temporal_triples[62]", "temporal_triples[63]", "temporal_triples[64]", "temporal_triples[65]", "temporal_triples[66]", "temporal_triples[67]", "temporal_triples[68]", "temporal_triples[69]", "temporal_triples[70]", "temporal_triples[71]", "temporal_triples[72]", "temporal_triples[73]", "temporal_triples[74]", "temporal_triples[75]", "temporal_triples[76]", "temporal_triples[77]", "temporal_triples[78]", "temporal_triples[79]", "temporal_triples[80]", "temporal_triples[81]", "temporal_triples[82]", "temporal_triples[83]", "temporal_triples[84]", "temporal_triples[85]", "temporal_triples[86]", "temporal_triples[87]", "temporal_triples[88]", "temporal_triples[89]", "temporal_triples[90]", "temporal_triples[91]", "temporal_triples[92]", "temporal_triples[93]", "temporal_triples[94]", "temporal_triples[95]", "temporal_triples[96]", "temporal_triples[97]", "temporal_triples[98]", "temporal_triples[99]", "temporal_triples[100]", "temporal_triples[101]"};
const vector<string> __vecstr_instance_SpatialTriple = {"spatial_triples[0]", "spatial_triples[1]", "spatial_triples[2]", "spatial_triples[3]", "spatial_triples[4]", "spatial_triples[5]", "spatial_triples[6]", "spatial_triples[7]", "spatial_triples[8]", "spatial_triples[9]", "spatial_triples[10]", "spatial_triples[11]", "spatial_triples[12]", "spatial_triples[13]", "spatial_triples[14]", "spatial_triples[15]", "spatial_triples[16]", "spatial_triples[17]", "spatial_triples[18]", "spatial_triples[19]", "spatial_triples[20]", "spatial_triples[21]", "spatial_triples[22]", "spatial_triples[23]", "spatial_triples[24]", "spatial_triples[25]", "spatial_triples[26]", "spatial_triples[27]", "spatial_triples[28]", "spatial_triples[29]", "spatial_triples[30]", "spatial_triples[31]", "spatial_triples[32]", "spatial_triples[33]", "spatial_triples[34]", "spatial_triples[35]", "spatial_triples[36]", "spatial_triples[37]", "spatial_triples[38]", "spatial_triples[39]", "spatial_triples[40]", "spatial_triples[41]", "spatial_triples[42]", "spatial_triples[43]", "spatial_triples[44]", "spatial_triples[45]", "spatial_triples[46]", "spatial_triples[47]", "spatial_triples[48]", "spatial_triples[49]", "spatial_triples[50]", "spatial_triples[51]", "spatial_triples[52]", "spatial_triples[53]", "spatial_triples[54]", "spatial_triples[55]", "spatial_triples[56]", "spatial_triples[57]", "spatial_triples[58]", "spatial_triples[59]", "spatial_triples[60]", "spatial_triples[61]", "spatial_triples[62]", "spatial_triples[63]", "spatial_triples[64]", "spatial_triples[65]", "spatial_triples[66]", "spatial_triples[67]", "spatial_triples[68]", "spatial_triples[69]", "spatial_triples[70]", "spatial_triples[71]", "spatial_triples[72]", "spatial_triples[73]", "spatial_triples[74]", "spatial_triples[75]", "spatial_triples[76]", "spatial_triples[77]", "spatial_triples[78]", "spatial_triples[79]", "spatial_triples[80]", "spatial_triples[81]", "spatial_triples[82]", "spatial_triples[83]", "spatial_triples[84]", "spatial_triples[85]", "spatial_triples[86]", "spatial_triples[87]", "spatial_triples[88]", "spatial_triples[89]", "spatial_triples[90]", "spatial_triples[91]", "spatial_triples[92]", "spatial_triples[93]", "spatial_triples[94]", "spatial_triples[95]", "spatial_triples[96]", "spatial_triples[97]", "spatial_triples[98]", "spatial_triples[99]", "spatial_triples[100]", "spatial_triples[101]", "spatial_triples[102]", "spatial_triples[103]", "spatial_triples[104]", "spatial_triples[105]", "spatial_triples[106]", "spatial_triples[107]", "spatial_triples[108]", "spatial_triples[109]", "spatial_triples[110]", "spatial_triples[111]", "spatial_triples[112]", "spatial_triples[113]", "spatial_triples[114]", "spatial_triples[115]", "spatial_triples[116]", "spatial_triples[117]", "spatial_triples[118]", "spatial_triples[119]", "spatial_triples[120]", "spatial_triples[121]", "spatial_triples[122]", "spatial_triples[123]", "spatial_triples[124]", "spatial_triples[125]", "spatial_triples[126]", "spatial_triples[127]", "spatial_triples[128]", "spatial_triples[129]", "spatial_triples[130]", "spatial_triples[131]", "spatial_triples[132]", "spatial_triples[133]", "spatial_triples[134]", "spatial_triples[135]", "spatial_triples[136]", "spatial_triples[137]", "spatial_triples[138]", "spatial_triples[139]", "spatial_triples[140]", "spatial_triples[141]", "spatial_triples[142]", "spatial_triples[143]", "spatial_triples[144]", "spatial_triples[145]", "spatial_triples[146]", "spatial_triples[147]", "spatial_triples[148]", "spatial_triples[149]", "spatial_triples[150]", "spatial_triples[151]", "spatial_triples[152]", "spatial_triples[153]", "spatial_triples[154]", "spatial_triples[155]", "spatial_triples[156]", "spatial_triples[157]", "spatial_triples[158]", "spatial_triples[159]", "spatial_triples[160]", "spatial_triples[161]", "spatial_triples[162]", "spatial_triples[163]", "spatial_triples[164]", "spatial_triples[165]", "spatial_triples[166]", "spatial_triples[167]", "spatial_triples[168]", "spatial_triples[169]", "spatial_triples[170]", "spatial_triples[171]", "spatial_triples[172]", "spatial_triples[173]", "spatial_triples[174]", "spatial_triples[175]", "spatial_triples[176]", "spatial_triples[177]", "spatial_triples[178]", "spatial_triples[179]", "spatial_triples[180]", "spatial_triples[181]", "spatial_triples[182]", "spatial_triples[183]", "spatial_triples[184]", "spatial_triples[185]", "spatial_triples[186]", "spatial_triples[187]", "spatial_triples[188]", "spatial_triples[189]", "spatial_triples[190]", "spatial_triples[191]", "spatial_triples[192]", "spatial_triples[193]", "spatial_triples[194]", "spatial_triples[195]", "spatial_triples[196]", "spatial_triples[197]", "spatial_triples[198]", "spatial_triples[199]", "spatial_triples[200]", "spatial_triples[201]", "spatial_triples[202]", "spatial_triples[203]", "spatial_triples[204]", "spatial_triples[205]", "spatial_triples[206]", "spatial_triples[207]", "spatial_triples[208]", "spatial_triples[209]", "spatial_triples[210]", "spatial_triples[211]", "spatial_triples[212]", "spatial_triples[213]", "spatial_triples[214]", "spatial_triples[215]", "spatial_triples[216]", "spatial_triples[217]", "spatial_triples[218]", "spatial_triples[219]", "spatial_triples[220]", "spatial_triples[221]", "spatial_triples[222]", "spatial_triples[223]", "spatial_triples[224]", "spatial_triples[225]", "spatial_triples[226]", "spatial_triples[227]", "spatial_triples[228]", "spatial_triples[229]", "spatial_triples[230]", "spatial_triples[231]", "spatial_triples[232]", "spatial_triples[233]", "spatial_triples[234]", "spatial_triples[235]", "spatial_triples[236]", "spatial_triples[237]", "spatial_triples[238]", "spatial_triples[239]", "spatial_triples[240]", "spatial_triples[241]", "spatial_triples[242]", "spatial_triples[243]", "spatial_triples[244]", "spatial_triples[245]", "spatial_triples[246]", "spatial_triples[247]", "spatial_triples[248]", "spatial_triples[249]", "spatial_triples[250]", "spatial_triples[251]", "spatial_triples[252]", "spatial_triples[253]", "spatial_triples[254]", "spatial_triples[255]", "spatial_triples[256]", "spatial_triples[257]", "spatial_triples[258]", "spatial_triples[259]", "spatial_triples[260]", "spatial_triples[261]", "spatial_triples[262]", "spatial_triples[263]", "spatial_triples[264]", "spatial_triples[265]", "spatial_triples[266]", "spatial_triples[267]", "spatial_triples[268]", "spatial_triples[269]", "spatial_triples[270]", "spatial_triples[271]", "spatial_triples[272]", "spatial_triples[273]", "spatial_triples[274]", "spatial_triples[275]", "spatial_triples[276]", "spatial_triples[277]", "spatial_triples[278]", "spatial_triples[279]", "spatial_triples[280]", "spatial_triples[281]", "spatial_triples[282]", "spatial_triples[283]", "spatial_triples[284]", "spatial_triples[285]", "spatial_triples[286]", "spatial_triples[287]", "spatial_triples[288]", "spatial_triples[289]", "spatial_triples[290]", "spatial_triples[291]", "spatial_triples[292]", "spatial_triples[293]", "spatial_triples[294]", "spatial_triples[295]", "spatial_triples[296]", "spatial_triples[297]", "spatial_triples[298]", "spatial_triples[299]", "spatial_triples[300]", "spatial_triples[301]", "spatial_triples[302]", "spatial_triples[303]", "spatial_triples[304]", "spatial_triples[305]", "spatial_triples[306]", "spatial_triples[307]", "spatial_triples[308]", "spatial_triples[309]", "spatial_triples[310]", "spatial_triples[311]", "spatial_triples[312]", "spatial_triples[313]", "spatial_triples[314]", "spatial_triples[315]", "spatial_triples[316]", "spatial_triples[317]", "spatial_triples[318]", "spatial_triples[319]", "spatial_triples[320]", "spatial_triples[321]", "spatial_triples[322]", "spatial_triples[323]", "spatial_triples[324]", "spatial_triples[325]", "spatial_triples[326]", "spatial_triples[327]", "spatial_triples[328]", "spatial_triples[329]", "spatial_triples[330]", "spatial_triples[331]", "spatial_triples[332]", "spatial_triples[333]", "spatial_triples[334]", "spatial_triples[335]", "spatial_triples[336]", "spatial_triples[337]", "spatial_triples[338]", "spatial_triples[339]", "spatial_triples[340]", "spatial_triples[341]", "spatial_triples[342]", "spatial_triples[343]", "spatial_triples[344]", "spatial_triples[345]", "spatial_triples[346]", "spatial_triples[347]", "spatial_triples[348]", "spatial_triples[349]", "spatial_triples[350]", "spatial_triples[351]", "spatial_triples[352]", "spatial_triples[353]", "spatial_triples[354]", "spatial_triples[355]", "spatial_triples[356]", "spatial_triples[357]", "spatial_triples[358]", "spatial_triples[359]", "spatial_triples[360]", "spatial_triples[361]", "spatial_triples[362]", "spatial_triples[363]", "spatial_triples[364]", "spatial_triples[365]", "spatial_triples[366]", "spatial_triples[367]", "spatial_triples[368]", "spatial_triples[369]", "spatial_triples[370]", "spatial_triples[371]", "spatial_triples[372]", "spatial_triples[373]", "spatial_triples[374]", "spatial_triples[375]", "spatial_triples[376]", "spatial_triples[377]", "spatial_triples[378]", "spatial_triples[379]", "spatial_triples[380]", "spatial_triples[381]", "spatial_triples[382]", "spatial_triples[383]", "spatial_triples[384]", "spatial_triples[385]", "spatial_triples[386]", "spatial_triples[387]", "spatial_triples[388]", "spatial_triples[389]", "spatial_triples[390]", "spatial_triples[391]", "spatial_triples[392]", "spatial_triples[393]", "spatial_triples[394]", "spatial_triples[395]", "spatial_triples[396]", "spatial_triples[397]", "spatial_triples[398]", "spatial_triples[399]", "spatial_triples[400]", "spatial_triples[401]", "spatial_triples[402]", "spatial_triples[403]", "spatial_triples[404]", "spatial_triples[405]", "spatial_triples[406]", "spatial_triples[407]", "spatial_triples[408]", "spatial_triples[409]", "spatial_triples[410]", "spatial_triples[411]", "spatial_triples[412]", "spatial_triples[413]", "spatial_triples[414]", "spatial_triples[415]", "spatial_triples[416]", "spatial_triples[417]", "spatial_triples[418]", "spatial_triples[419]", "spatial_triples[420]", "spatial_triples[421]", "spatial_triples[422]", "spatial_triples[423]", "spatial_triples[424]", "spatial_triples[425]", "spatial_triples[426]", "spatial_triples[427]", "spatial_triples[428]", "spatial_triples[429]", "spatial_triples[430]", "spatial_triples[431]"};
const vector<string> __vecstr_instance_Region = {"regions[0]", "regions[1]", "regions[2]", "regions[3]", "regions[4]", "regions[5]", "regions[6]", "regions[7]", "regions[8]"};
const vector<string> __vecstr_instance_County = {"counties[0]", "counties[1]", "counties[2]", "counties[3]", "counties[4]", "counties[5]", "counties[6]", "counties[7]", "counties[8]", "counties[9]", "counties[10]", "counties[11]", "counties[12]", "counties[13]", "counties[14]", "counties[15]", "counties[16]", "counties[17]", "counties[18]", "counties[19]", "counties[20]", "counties[21]", "counties[22]", "counties[23]", "counties[24]", "counties[25]", "counties[26]", "counties[27]", "counties[28]", "counties[29]", "counties[30]", "counties[31]", "counties[32]", "counties[33]", "counties[34]", "counties[35]", "counties[36]", "counties[37]", "counties[38]", "counties[39]", "counties[40]", "counties[41]", "counties[42]", "counties[43]", "counties[44]", "counties[45]", "counties[46]", "counties[47]", "counties[48]", "counties[49]", "counties[50]", "counties[51]", "counties[52]", "counties[53]", "counties[54]", "counties[55]", "counties[56]", "counties[57]", "counties[58]", "counties[59]", "counties[60]", "counties[61]", "counties[62]", "counties[63]", "counties[64]", "counties[65]", "counties[66]", "counties[67]", "counties[68]", "counties[69]", "counties[70]", "counties[71]", "counties[72]", "counties[73]", "counties[74]", "counties[75]", "counties[76]", "counties[77]", "counties[78]", "counties[79]", "counties[80]", "counties[81]"};
void _eval_query();
void _init_storage();
void _init_world();
void _garbage_collection();
void _print_answer();
const int _TOT_LOOP = 500000000;
const int _BURN_IN = 499999990;
int _tot_round = -499999990;
const double __fixed_rho = 0.50000000;
const double __fixed_tau1 = 0.50000000;
const double __fixed_beta1 = 0;
const double __fixed_beta2 = 0;
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
Gaussian Gaussian14699744;
BooleanDistrib BooleanDistrib14711216;
BooleanDistrib BooleanDistrib14715216;
UniformReal UniformReal14717168;
Gaussian Gaussian14775056;
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
  _answer_2.add(__fixed_beta1,1);
  _answer_3.add(__fixed_beta2,1);
  for (int c = 0;c<82;c++)
  for (int t = 0;t<103;t++)
  _answer_4[c][t]->add(_mem_logit[c][t]->getval(),1);


}
void _init_storage()
{
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

  Gaussian14699744.init(-4.00000000,0.01000000);
  UniformReal14717168.init(-7.00000000,-0.25000000);
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
  if (__fixed_observations(t,r)>0.000000)
    _util_set_evidence<double>(_mem_region_rate[r][t],__fixed_observations(t,r));


}
void _garbage_collection()
{
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
  return Gaussian14699744.loglikeli(val);
}
double _Var_bias::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian14699744.loglikeli(_t_val);
}
void _Var_bias::sample()
{
  val=Gaussian14699744.gen();
}
void _Var_bias::sample_cache()
{
  cache_val=Gaussian14699744.gen();
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
  return BooleanDistrib14711216.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getval()-_mem_bias->getval())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getval()-_mem_bias->getval()))),BooleanDistrib14711216.loglikeli(val);
}
double _Var_temporal_edge::getcachelikeli()
{
  auto _t_val = getcache();
  return BooleanDistrib14711216.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getcache()-_mem_bias->getcache())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getcache()-_mem_bias->getcache()))),BooleanDistrib14711216.loglikeli(_t_val);
}
void _Var_temporal_edge::sample()
{
  val=(BooleanDistrib14711216.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getval()-_mem_bias->getval())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getval()-_mem_bias->getval()))),BooleanDistrib14711216.gen());
}
void _Var_temporal_edge::sample_cache()
{
  cache_val=(BooleanDistrib14711216.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getcache()-_mem_bias->getcache())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getcache()-_mem_bias->getcache()))),BooleanDistrib14711216.gen());
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
  return BooleanDistrib14715216.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getval()-_mem_bias->getval())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getval()-_mem_bias->getval()))),BooleanDistrib14715216.loglikeli(val);
}
double _Var_spatial_edge::getcachelikeli()
{
  auto _t_val = getcache();
  return BooleanDistrib14715216.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getcache()-_mem_bias->getcache())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getcache()-_mem_bias->getcache()))),BooleanDistrib14715216.loglikeli(_t_val);
}
void _Var_spatial_edge::sample()
{
  val=(BooleanDistrib14715216.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getval()-_mem_bias->getval())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getval()-_mem_bias->getval()))),BooleanDistrib14715216.gen());
}
void _Var_spatial_edge::sample_cache()
{
  cache_val=(BooleanDistrib14715216.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getcache()-_mem_bias->getcache())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getcache()-_mem_bias->getcache()))),BooleanDistrib14715216.gen());
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
  return UniformReal14717168.loglikeli(val);
}
double _Var_logit::getcachelikeli()
{
  auto _t_val = getcache();
  return UniformReal14717168.loglikeli(_t_val);
}
void _Var_logit::sample()
{
  val=UniformReal14717168.gen();
}
void _Var_logit::sample_cache()
{
  cache_val=UniformReal14717168.gen();
}
void _Var_logit::active_edge()
{}
void _Var_logit::remove_edge()
{}
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
  return Gaussian14775056.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getval()), __fixed_sigmoid(_mem_logit[1][t]->getval()), __fixed_sigmoid(_mem_logit[2][t]->getval()), __fixed_sigmoid(_mem_logit[3][t]->getval()), __fixed_sigmoid(_mem_logit[4][t]->getval()), __fixed_sigmoid(_mem_logit[5][t]->getval()), __fixed_sigmoid(_mem_logit[6][t]->getval()), __fixed_sigmoid(_mem_logit[7][t]->getval()), __fixed_sigmoid(_mem_logit[8][t]->getval()), __fixed_sigmoid(_mem_logit[9][t]->getval()), __fixed_sigmoid(_mem_logit[10][t]->getval()), __fixed_sigmoid(_mem_logit[11][t]->getval()), __fixed_sigmoid(_mem_logit[12][t]->getval()), __fixed_sigmoid(_mem_logit[13][t]->getval()), __fixed_sigmoid(_mem_logit[14][t]->getval()), __fixed_sigmoid(_mem_logit[15][t]->getval()), __fixed_sigmoid(_mem_logit[16][t]->getval()), __fixed_sigmoid(_mem_logit[17][t]->getval()), __fixed_sigmoid(_mem_logit[18][t]->getval()), __fixed_sigmoid(_mem_logit[19][t]->getval()), __fixed_sigmoid(_mem_logit[20][t]->getval()), __fixed_sigmoid(_mem_logit[21][t]->getval()), __fixed_sigmoid(_mem_logit[22][t]->getval()), __fixed_sigmoid(_mem_logit[23][t]->getval()), __fixed_sigmoid(_mem_logit[24][t]->getval()), __fixed_sigmoid(_mem_logit[25][t]->getval()), __fixed_sigmoid(_mem_logit[26][t]->getval()), __fixed_sigmoid(_mem_logit[27][t]->getval()), __fixed_sigmoid(_mem_logit[28][t]->getval()), __fixed_sigmoid(_mem_logit[29][t]->getval()), __fixed_sigmoid(_mem_logit[30][t]->getval()), __fixed_sigmoid(_mem_logit[31][t]->getval()), __fixed_sigmoid(_mem_logit[32][t]->getval()), __fixed_sigmoid(_mem_logit[33][t]->getval()), __fixed_sigmoid(_mem_logit[34][t]->getval()), __fixed_sigmoid(_mem_logit[35][t]->getval()), __fixed_sigmoid(_mem_logit[36][t]->getval()), __fixed_sigmoid(_mem_logit[37][t]->getval()), __fixed_sigmoid(_mem_logit[38][t]->getval()), __fixed_sigmoid(_mem_logit[39][t]->getval()), __fixed_sigmoid(_mem_logit[40][t]->getval()), __fixed_sigmoid(_mem_logit[41][t]->getval()), __fixed_sigmoid(_mem_logit[42][t]->getval()), __fixed_sigmoid(_mem_logit[43][t]->getval()), __fixed_sigmoid(_mem_logit[44][t]->getval()), __fixed_sigmoid(_mem_logit[45][t]->getval()), __fixed_sigmoid(_mem_logit[46][t]->getval()), __fixed_sigmoid(_mem_logit[47][t]->getval()), __fixed_sigmoid(_mem_logit[48][t]->getval()), __fixed_sigmoid(_mem_logit[49][t]->getval()), __fixed_sigmoid(_mem_logit[50][t]->getval()), __fixed_sigmoid(_mem_logit[51][t]->getval()), __fixed_sigmoid(_mem_logit[52][t]->getval()), __fixed_sigmoid(_mem_logit[53][t]->getval()), __fixed_sigmoid(_mem_logit[54][t]->getval()), __fixed_sigmoid(_mem_logit[55][t]->getval()), __fixed_sigmoid(_mem_logit[56][t]->getval()), __fixed_sigmoid(_mem_logit[57][t]->getval()), __fixed_sigmoid(_mem_logit[58][t]->getval()), __fixed_sigmoid(_mem_logit[59][t]->getval()), __fixed_sigmoid(_mem_logit[60][t]->getval()), __fixed_sigmoid(_mem_logit[61][t]->getval()), __fixed_sigmoid(_mem_logit[62][t]->getval()), __fixed_sigmoid(_mem_logit[63][t]->getval()), __fixed_sigmoid(_mem_logit[64][t]->getval()), __fixed_sigmoid(_mem_logit[65][t]->getval()), __fixed_sigmoid(_mem_logit[66][t]->getval()), __fixed_sigmoid(_mem_logit[67][t]->getval()), __fixed_sigmoid(_mem_logit[68][t]->getval()), __fixed_sigmoid(_mem_logit[69][t]->getval()), __fixed_sigmoid(_mem_logit[70][t]->getval()), __fixed_sigmoid(_mem_logit[71][t]->getval()), __fixed_sigmoid(_mem_logit[72][t]->getval()), __fixed_sigmoid(_mem_logit[73][t]->getval()), __fixed_sigmoid(_mem_logit[74][t]->getval()), __fixed_sigmoid(_mem_logit[75][t]->getval()), __fixed_sigmoid(_mem_logit[76][t]->getval()), __fixed_sigmoid(_mem_logit[77][t]->getval()), __fixed_sigmoid(_mem_logit[78][t]->getval()), __fixed_sigmoid(_mem_logit[79][t]->getval()), __fixed_sigmoid(_mem_logit[80][t]->getval()), __fixed_sigmoid(_mem_logit[81][t]->getval())}))/__fixed_region_pop[r],0.10000000),Gaussian14775056.loglikeli(val);
}
double _Var_region_rate::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian14775056.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getcache()), __fixed_sigmoid(_mem_logit[1][t]->getcache()), __fixed_sigmoid(_mem_logit[2][t]->getcache()), __fixed_sigmoid(_mem_logit[3][t]->getcache()), __fixed_sigmoid(_mem_logit[4][t]->getcache()), __fixed_sigmoid(_mem_logit[5][t]->getcache()), __fixed_sigmoid(_mem_logit[6][t]->getcache()), __fixed_sigmoid(_mem_logit[7][t]->getcache()), __fixed_sigmoid(_mem_logit[8][t]->getcache()), __fixed_sigmoid(_mem_logit[9][t]->getcache()), __fixed_sigmoid(_mem_logit[10][t]->getcache()), __fixed_sigmoid(_mem_logit[11][t]->getcache()), __fixed_sigmoid(_mem_logit[12][t]->getcache()), __fixed_sigmoid(_mem_logit[13][t]->getcache()), __fixed_sigmoid(_mem_logit[14][t]->getcache()), __fixed_sigmoid(_mem_logit[15][t]->getcache()), __fixed_sigmoid(_mem_logit[16][t]->getcache()), __fixed_sigmoid(_mem_logit[17][t]->getcache()), __fixed_sigmoid(_mem_logit[18][t]->getcache()), __fixed_sigmoid(_mem_logit[19][t]->getcache()), __fixed_sigmoid(_mem_logit[20][t]->getcache()), __fixed_sigmoid(_mem_logit[21][t]->getcache()), __fixed_sigmoid(_mem_logit[22][t]->getcache()), __fixed_sigmoid(_mem_logit[23][t]->getcache()), __fixed_sigmoid(_mem_logit[24][t]->getcache()), __fixed_sigmoid(_mem_logit[25][t]->getcache()), __fixed_sigmoid(_mem_logit[26][t]->getcache()), __fixed_sigmoid(_mem_logit[27][t]->getcache()), __fixed_sigmoid(_mem_logit[28][t]->getcache()), __fixed_sigmoid(_mem_logit[29][t]->getcache()), __fixed_sigmoid(_mem_logit[30][t]->getcache()), __fixed_sigmoid(_mem_logit[31][t]->getcache()), __fixed_sigmoid(_mem_logit[32][t]->getcache()), __fixed_sigmoid(_mem_logit[33][t]->getcache()), __fixed_sigmoid(_mem_logit[34][t]->getcache()), __fixed_sigmoid(_mem_logit[35][t]->getcache()), __fixed_sigmoid(_mem_logit[36][t]->getcache()), __fixed_sigmoid(_mem_logit[37][t]->getcache()), __fixed_sigmoid(_mem_logit[38][t]->getcache()), __fixed_sigmoid(_mem_logit[39][t]->getcache()), __fixed_sigmoid(_mem_logit[40][t]->getcache()), __fixed_sigmoid(_mem_logit[41][t]->getcache()), __fixed_sigmoid(_mem_logit[42][t]->getcache()), __fixed_sigmoid(_mem_logit[43][t]->getcache()), __fixed_sigmoid(_mem_logit[44][t]->getcache()), __fixed_sigmoid(_mem_logit[45][t]->getcache()), __fixed_sigmoid(_mem_logit[46][t]->getcache()), __fixed_sigmoid(_mem_logit[47][t]->getcache()), __fixed_sigmoid(_mem_logit[48][t]->getcache()), __fixed_sigmoid(_mem_logit[49][t]->getcache()), __fixed_sigmoid(_mem_logit[50][t]->getcache()), __fixed_sigmoid(_mem_logit[51][t]->getcache()), __fixed_sigmoid(_mem_logit[52][t]->getcache()), __fixed_sigmoid(_mem_logit[53][t]->getcache()), __fixed_sigmoid(_mem_logit[54][t]->getcache()), __fixed_sigmoid(_mem_logit[55][t]->getcache()), __fixed_sigmoid(_mem_logit[56][t]->getcache()), __fixed_sigmoid(_mem_logit[57][t]->getcache()), __fixed_sigmoid(_mem_logit[58][t]->getcache()), __fixed_sigmoid(_mem_logit[59][t]->getcache()), __fixed_sigmoid(_mem_logit[60][t]->getcache()), __fixed_sigmoid(_mem_logit[61][t]->getcache()), __fixed_sigmoid(_mem_logit[62][t]->getcache()), __fixed_sigmoid(_mem_logit[63][t]->getcache()), __fixed_sigmoid(_mem_logit[64][t]->getcache()), __fixed_sigmoid(_mem_logit[65][t]->getcache()), __fixed_sigmoid(_mem_logit[66][t]->getcache()), __fixed_sigmoid(_mem_logit[67][t]->getcache()), __fixed_sigmoid(_mem_logit[68][t]->getcache()), __fixed_sigmoid(_mem_logit[69][t]->getcache()), __fixed_sigmoid(_mem_logit[70][t]->getcache()), __fixed_sigmoid(_mem_logit[71][t]->getcache()), __fixed_sigmoid(_mem_logit[72][t]->getcache()), __fixed_sigmoid(_mem_logit[73][t]->getcache()), __fixed_sigmoid(_mem_logit[74][t]->getcache()), __fixed_sigmoid(_mem_logit[75][t]->getcache()), __fixed_sigmoid(_mem_logit[76][t]->getcache()), __fixed_sigmoid(_mem_logit[77][t]->getcache()), __fixed_sigmoid(_mem_logit[78][t]->getcache()), __fixed_sigmoid(_mem_logit[79][t]->getcache()), __fixed_sigmoid(_mem_logit[80][t]->getcache()), __fixed_sigmoid(_mem_logit[81][t]->getcache())}))/__fixed_region_pop[r],0.10000000),Gaussian14775056.loglikeli(_t_val);
}
void _Var_region_rate::sample()
{
  val=(Gaussian14775056.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getval()), __fixed_sigmoid(_mem_logit[1][t]->getval()), __fixed_sigmoid(_mem_logit[2][t]->getval()), __fixed_sigmoid(_mem_logit[3][t]->getval()), __fixed_sigmoid(_mem_logit[4][t]->getval()), __fixed_sigmoid(_mem_logit[5][t]->getval()), __fixed_sigmoid(_mem_logit[6][t]->getval()), __fixed_sigmoid(_mem_logit[7][t]->getval()), __fixed_sigmoid(_mem_logit[8][t]->getval()), __fixed_sigmoid(_mem_logit[9][t]->getval()), __fixed_sigmoid(_mem_logit[10][t]->getval()), __fixed_sigmoid(_mem_logit[11][t]->getval()), __fixed_sigmoid(_mem_logit[12][t]->getval()), __fixed_sigmoid(_mem_logit[13][t]->getval()), __fixed_sigmoid(_mem_logit[14][t]->getval()), __fixed_sigmoid(_mem_logit[15][t]->getval()), __fixed_sigmoid(_mem_logit[16][t]->getval()), __fixed_sigmoid(_mem_logit[17][t]->getval()), __fixed_sigmoid(_mem_logit[18][t]->getval()), __fixed_sigmoid(_mem_logit[19][t]->getval()), __fixed_sigmoid(_mem_logit[20][t]->getval()), __fixed_sigmoid(_mem_logit[21][t]->getval()), __fixed_sigmoid(_mem_logit[22][t]->getval()), __fixed_sigmoid(_mem_logit[23][t]->getval()), __fixed_sigmoid(_mem_logit[24][t]->getval()), __fixed_sigmoid(_mem_logit[25][t]->getval()), __fixed_sigmoid(_mem_logit[26][t]->getval()), __fixed_sigmoid(_mem_logit[27][t]->getval()), __fixed_sigmoid(_mem_logit[28][t]->getval()), __fixed_sigmoid(_mem_logit[29][t]->getval()), __fixed_sigmoid(_mem_logit[30][t]->getval()), __fixed_sigmoid(_mem_logit[31][t]->getval()), __fixed_sigmoid(_mem_logit[32][t]->getval()), __fixed_sigmoid(_mem_logit[33][t]->getval()), __fixed_sigmoid(_mem_logit[34][t]->getval()), __fixed_sigmoid(_mem_logit[35][t]->getval()), __fixed_sigmoid(_mem_logit[36][t]->getval()), __fixed_sigmoid(_mem_logit[37][t]->getval()), __fixed_sigmoid(_mem_logit[38][t]->getval()), __fixed_sigmoid(_mem_logit[39][t]->getval()), __fixed_sigmoid(_mem_logit[40][t]->getval()), __fixed_sigmoid(_mem_logit[41][t]->getval()), __fixed_sigmoid(_mem_logit[42][t]->getval()), __fixed_sigmoid(_mem_logit[43][t]->getval()), __fixed_sigmoid(_mem_logit[44][t]->getval()), __fixed_sigmoid(_mem_logit[45][t]->getval()), __fixed_sigmoid(_mem_logit[46][t]->getval()), __fixed_sigmoid(_mem_logit[47][t]->getval()), __fixed_sigmoid(_mem_logit[48][t]->getval()), __fixed_sigmoid(_mem_logit[49][t]->getval()), __fixed_sigmoid(_mem_logit[50][t]->getval()), __fixed_sigmoid(_mem_logit[51][t]->getval()), __fixed_sigmoid(_mem_logit[52][t]->getval()), __fixed_sigmoid(_mem_logit[53][t]->getval()), __fixed_sigmoid(_mem_logit[54][t]->getval()), __fixed_sigmoid(_mem_logit[55][t]->getval()), __fixed_sigmoid(_mem_logit[56][t]->getval()), __fixed_sigmoid(_mem_logit[57][t]->getval()), __fixed_sigmoid(_mem_logit[58][t]->getval()), __fixed_sigmoid(_mem_logit[59][t]->getval()), __fixed_sigmoid(_mem_logit[60][t]->getval()), __fixed_sigmoid(_mem_logit[61][t]->getval()), __fixed_sigmoid(_mem_logit[62][t]->getval()), __fixed_sigmoid(_mem_logit[63][t]->getval()), __fixed_sigmoid(_mem_logit[64][t]->getval()), __fixed_sigmoid(_mem_logit[65][t]->getval()), __fixed_sigmoid(_mem_logit[66][t]->getval()), __fixed_sigmoid(_mem_logit[67][t]->getval()), __fixed_sigmoid(_mem_logit[68][t]->getval()), __fixed_sigmoid(_mem_logit[69][t]->getval()), __fixed_sigmoid(_mem_logit[70][t]->getval()), __fixed_sigmoid(_mem_logit[71][t]->getval()), __fixed_sigmoid(_mem_logit[72][t]->getval()), __fixed_sigmoid(_mem_logit[73][t]->getval()), __fixed_sigmoid(_mem_logit[74][t]->getval()), __fixed_sigmoid(_mem_logit[75][t]->getval()), __fixed_sigmoid(_mem_logit[76][t]->getval()), __fixed_sigmoid(_mem_logit[77][t]->getval()), __fixed_sigmoid(_mem_logit[78][t]->getval()), __fixed_sigmoid(_mem_logit[79][t]->getval()), __fixed_sigmoid(_mem_logit[80][t]->getval()), __fixed_sigmoid(_mem_logit[81][t]->getval())}))/__fixed_region_pop[r],0.10000000),Gaussian14775056.gen());
}
void _Var_region_rate::sample_cache()
{
  cache_val=(Gaussian14775056.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getcache()), __fixed_sigmoid(_mem_logit[1][t]->getcache()), __fixed_sigmoid(_mem_logit[2][t]->getcache()), __fixed_sigmoid(_mem_logit[3][t]->getcache()), __fixed_sigmoid(_mem_logit[4][t]->getcache()), __fixed_sigmoid(_mem_logit[5][t]->getcache()), __fixed_sigmoid(_mem_logit[6][t]->getcache()), __fixed_sigmoid(_mem_logit[7][t]->getcache()), __fixed_sigmoid(_mem_logit[8][t]->getcache()), __fixed_sigmoid(_mem_logit[9][t]->getcache()), __fixed_sigmoid(_mem_logit[10][t]->getcache()), __fixed_sigmoid(_mem_logit[11][t]->getcache()), __fixed_sigmoid(_mem_logit[12][t]->getcache()), __fixed_sigmoid(_mem_logit[13][t]->getcache()), __fixed_sigmoid(_mem_logit[14][t]->getcache()), __fixed_sigmoid(_mem_logit[15][t]->getcache()), __fixed_sigmoid(_mem_logit[16][t]->getcache()), __fixed_sigmoid(_mem_logit[17][t]->getcache()), __fixed_sigmoid(_mem_logit[18][t]->getcache()), __fixed_sigmoid(_mem_logit[19][t]->getcache()), __fixed_sigmoid(_mem_logit[20][t]->getcache()), __fixed_sigmoid(_mem_logit[21][t]->getcache()), __fixed_sigmoid(_mem_logit[22][t]->getcache()), __fixed_sigmoid(_mem_logit[23][t]->getcache()), __fixed_sigmoid(_mem_logit[24][t]->getcache()), __fixed_sigmoid(_mem_logit[25][t]->getcache()), __fixed_sigmoid(_mem_logit[26][t]->getcache()), __fixed_sigmoid(_mem_logit[27][t]->getcache()), __fixed_sigmoid(_mem_logit[28][t]->getcache()), __fixed_sigmoid(_mem_logit[29][t]->getcache()), __fixed_sigmoid(_mem_logit[30][t]->getcache()), __fixed_sigmoid(_mem_logit[31][t]->getcache()), __fixed_sigmoid(_mem_logit[32][t]->getcache()), __fixed_sigmoid(_mem_logit[33][t]->getcache()), __fixed_sigmoid(_mem_logit[34][t]->getcache()), __fixed_sigmoid(_mem_logit[35][t]->getcache()), __fixed_sigmoid(_mem_logit[36][t]->getcache()), __fixed_sigmoid(_mem_logit[37][t]->getcache()), __fixed_sigmoid(_mem_logit[38][t]->getcache()), __fixed_sigmoid(_mem_logit[39][t]->getcache()), __fixed_sigmoid(_mem_logit[40][t]->getcache()), __fixed_sigmoid(_mem_logit[41][t]->getcache()), __fixed_sigmoid(_mem_logit[42][t]->getcache()), __fixed_sigmoid(_mem_logit[43][t]->getcache()), __fixed_sigmoid(_mem_logit[44][t]->getcache()), __fixed_sigmoid(_mem_logit[45][t]->getcache()), __fixed_sigmoid(_mem_logit[46][t]->getcache()), __fixed_sigmoid(_mem_logit[47][t]->getcache()), __fixed_sigmoid(_mem_logit[48][t]->getcache()), __fixed_sigmoid(_mem_logit[49][t]->getcache()), __fixed_sigmoid(_mem_logit[50][t]->getcache()), __fixed_sigmoid(_mem_logit[51][t]->getcache()), __fixed_sigmoid(_mem_logit[52][t]->getcache()), __fixed_sigmoid(_mem_logit[53][t]->getcache()), __fixed_sigmoid(_mem_logit[54][t]->getcache()), __fixed_sigmoid(_mem_logit[55][t]->getcache()), __fixed_sigmoid(_mem_logit[56][t]->getcache()), __fixed_sigmoid(_mem_logit[57][t]->getcache()), __fixed_sigmoid(_mem_logit[58][t]->getcache()), __fixed_sigmoid(_mem_logit[59][t]->getcache()), __fixed_sigmoid(_mem_logit[60][t]->getcache()), __fixed_sigmoid(_mem_logit[61][t]->getcache()), __fixed_sigmoid(_mem_logit[62][t]->getcache()), __fixed_sigmoid(_mem_logit[63][t]->getcache()), __fixed_sigmoid(_mem_logit[64][t]->getcache()), __fixed_sigmoid(_mem_logit[65][t]->getcache()), __fixed_sigmoid(_mem_logit[66][t]->getcache()), __fixed_sigmoid(_mem_logit[67][t]->getcache()), __fixed_sigmoid(_mem_logit[68][t]->getcache()), __fixed_sigmoid(_mem_logit[69][t]->getcache()), __fixed_sigmoid(_mem_logit[70][t]->getcache()), __fixed_sigmoid(_mem_logit[71][t]->getcache()), __fixed_sigmoid(_mem_logit[72][t]->getcache()), __fixed_sigmoid(_mem_logit[73][t]->getcache()), __fixed_sigmoid(_mem_logit[74][t]->getcache()), __fixed_sigmoid(_mem_logit[75][t]->getcache()), __fixed_sigmoid(_mem_logit[76][t]->getcache()), __fixed_sigmoid(_mem_logit[77][t]->getcache()), __fixed_sigmoid(_mem_logit[78][t]->getcache()), __fixed_sigmoid(_mem_logit[79][t]->getcache()), __fixed_sigmoid(_mem_logit[80][t]->getcache()), __fixed_sigmoid(_mem_logit[81][t]->getcache())}))/__fixed_region_pop[r],0.10000000),Gaussian14775056.gen());
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
  printf("\nsample time: %fs (#iter = %d)\n",__elapsed_seconds.count(),500000000);
  swift::_print_answer();
  swift::_garbage_collection();
}
