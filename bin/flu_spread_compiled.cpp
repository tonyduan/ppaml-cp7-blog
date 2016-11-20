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
const vector<string> __vecstr_instance_TemporalTriple = {"temporal_triples[0]", "temporal_triples[1]", "temporal_triples[2]", "temporal_triples[3]", "temporal_triples[4]", "temporal_triples[5]", "temporal_triples[6]", "temporal_triples[7]", "temporal_triples[8]", "temporal_triples[9]", "temporal_triples[10]", "temporal_triples[11]", "temporal_triples[12]", "temporal_triples[13]", "temporal_triples[14]", "temporal_triples[15]", "temporal_triples[16]", "temporal_triples[17]", "temporal_triples[18]", "temporal_triples[19]", "temporal_triples[20]", "temporal_triples[21]", "temporal_triples[22]", "temporal_triples[23]", "temporal_triples[24]", "temporal_triples[25]", "temporal_triples[26]", "temporal_triples[27]", "temporal_triples[28]", "temporal_triples[29]", "temporal_triples[30]", "temporal_triples[31]", "temporal_triples[32]", "temporal_triples[33]", "temporal_triples[34]", "temporal_triples[35]", "temporal_triples[36]", "temporal_triples[37]", "temporal_triples[38]", "temporal_triples[39]", "temporal_triples[40]", "temporal_triples[41]", "temporal_triples[42]", "temporal_triples[43]", "temporal_triples[44]", "temporal_triples[45]", "temporal_triples[46]", "temporal_triples[47]", "temporal_triples[48]", "temporal_triples[49]", "temporal_triples[50]", "temporal_triples[51]", "temporal_triples[52]", "temporal_triples[53]", "temporal_triples[54]", "temporal_triples[55]", "temporal_triples[56]", "temporal_triples[57]", "temporal_triples[58]", "temporal_triples[59]", "temporal_triples[60]", "temporal_triples[61]", "temporal_triples[62]", "temporal_triples[63]", "temporal_triples[64]", "temporal_triples[65]", "temporal_triples[66]", "temporal_triples[67]", "temporal_triples[68]", "temporal_triples[69]", "temporal_triples[70]", "temporal_triples[71]", "temporal_triples[72]", "temporal_triples[73]", "temporal_triples[74]", "temporal_triples[75]", "temporal_triples[76]", "temporal_triples[77]", "temporal_triples[78]", "temporal_triples[79]", "temporal_triples[80]", "temporal_triples[81]", "temporal_triples[82]", "temporal_triples[83]", "temporal_triples[84]", "temporal_triples[85]", "temporal_triples[86]", "temporal_triples[87]", "temporal_triples[88]", "temporal_triples[89]", "temporal_triples[90]", "temporal_triples[91]", "temporal_triples[92]", "temporal_triples[93]", "temporal_triples[94]", "temporal_triples[95]", "temporal_triples[96]", "temporal_triples[97]", "temporal_triples[98]", "temporal_triples[99]", "temporal_triples[100]", "temporal_triples[101]"};
const vector<string> __vecstr_instance_SpatialTriple = {"spatial_triples[0]", "spatial_triples[1]", "spatial_triples[2]", "spatial_triples[3]", "spatial_triples[4]", "spatial_triples[5]", "spatial_triples[6]", "spatial_triples[7]", "spatial_triples[8]", "spatial_triples[9]", "spatial_triples[10]", "spatial_triples[11]", "spatial_triples[12]", "spatial_triples[13]", "spatial_triples[14]", "spatial_triples[15]", "spatial_triples[16]", "spatial_triples[17]", "spatial_triples[18]", "spatial_triples[19]", "spatial_triples[20]", "spatial_triples[21]", "spatial_triples[22]", "spatial_triples[23]", "spatial_triples[24]", "spatial_triples[25]", "spatial_triples[26]", "spatial_triples[27]", "spatial_triples[28]", "spatial_triples[29]", "spatial_triples[30]", "spatial_triples[31]", "spatial_triples[32]", "spatial_triples[33]", "spatial_triples[34]", "spatial_triples[35]", "spatial_triples[36]", "spatial_triples[37]", "spatial_triples[38]", "spatial_triples[39]", "spatial_triples[40]", "spatial_triples[41]", "spatial_triples[42]", "spatial_triples[43]", "spatial_triples[44]", "spatial_triples[45]", "spatial_triples[46]", "spatial_triples[47]", "spatial_triples[48]", "spatial_triples[49]", "spatial_triples[50]", "spatial_triples[51]", "spatial_triples[52]", "spatial_triples[53]", "spatial_triples[54]", "spatial_triples[55]", "spatial_triples[56]", "spatial_triples[57]", "spatial_triples[58]", "spatial_triples[59]", "spatial_triples[60]", "spatial_triples[61]", "spatial_triples[62]", "spatial_triples[63]", "spatial_triples[64]", "spatial_triples[65]", "spatial_triples[66]", "spatial_triples[67]", "spatial_triples[68]", "spatial_triples[69]", "spatial_triples[70]", "spatial_triples[71]", "spatial_triples[72]", "spatial_triples[73]", "spatial_triples[74]", "spatial_triples[75]", "spatial_triples[76]", "spatial_triples[77]", "spatial_triples[78]", "spatial_triples[79]", "spatial_triples[80]", "spatial_triples[81]", "spatial_triples[82]", "spatial_triples[83]", "spatial_triples[84]", "spatial_triples[85]", "spatial_triples[86]", "spatial_triples[87]", "spatial_triples[88]", "spatial_triples[89]", "spatial_triples[90]", "spatial_triples[91]", "spatial_triples[92]", "spatial_triples[93]", "spatial_triples[94]", "spatial_triples[95]", "spatial_triples[96]", "spatial_triples[97]", "spatial_triples[98]", "spatial_triples[99]", "spatial_triples[100]", "spatial_triples[101]", "spatial_triples[102]", "spatial_triples[103]", "spatial_triples[104]", "spatial_triples[105]", "spatial_triples[106]", "spatial_triples[107]", "spatial_triples[108]", "spatial_triples[109]", "spatial_triples[110]", "spatial_triples[111]", "spatial_triples[112]", "spatial_triples[113]", "spatial_triples[114]", "spatial_triples[115]", "spatial_triples[116]", "spatial_triples[117]", "spatial_triples[118]", "spatial_triples[119]", "spatial_triples[120]", "spatial_triples[121]", "spatial_triples[122]", "spatial_triples[123]", "spatial_triples[124]", "spatial_triples[125]", "spatial_triples[126]", "spatial_triples[127]", "spatial_triples[128]", "spatial_triples[129]", "spatial_triples[130]", "spatial_triples[131]", "spatial_triples[132]", "spatial_triples[133]", "spatial_triples[134]", "spatial_triples[135]", "spatial_triples[136]", "spatial_triples[137]", "spatial_triples[138]", "spatial_triples[139]", "spatial_triples[140]", "spatial_triples[141]", "spatial_triples[142]", "spatial_triples[143]", "spatial_triples[144]", "spatial_triples[145]", "spatial_triples[146]", "spatial_triples[147]", "spatial_triples[148]", "spatial_triples[149]", "spatial_triples[150]", "spatial_triples[151]", "spatial_triples[152]", "spatial_triples[153]", "spatial_triples[154]", "spatial_triples[155]", "spatial_triples[156]", "spatial_triples[157]", "spatial_triples[158]", "spatial_triples[159]", "spatial_triples[160]", "spatial_triples[161]", "spatial_triples[162]", "spatial_triples[163]", "spatial_triples[164]", "spatial_triples[165]", "spatial_triples[166]", "spatial_triples[167]", "spatial_triples[168]", "spatial_triples[169]", "spatial_triples[170]", "spatial_triples[171]", "spatial_triples[172]", "spatial_triples[173]", "spatial_triples[174]", "spatial_triples[175]", "spatial_triples[176]", "spatial_triples[177]", "spatial_triples[178]", "spatial_triples[179]", "spatial_triples[180]", "spatial_triples[181]", "spatial_triples[182]", "spatial_triples[183]", "spatial_triples[184]", "spatial_triples[185]", "spatial_triples[186]", "spatial_triples[187]", "spatial_triples[188]", "spatial_triples[189]", "spatial_triples[190]", "spatial_triples[191]", "spatial_triples[192]", "spatial_triples[193]", "spatial_triples[194]", "spatial_triples[195]", "spatial_triples[196]", "spatial_triples[197]", "spatial_triples[198]", "spatial_triples[199]", "spatial_triples[200]", "spatial_triples[201]", "spatial_triples[202]", "spatial_triples[203]", "spatial_triples[204]", "spatial_triples[205]", "spatial_triples[206]", "spatial_triples[207]", "spatial_triples[208]", "spatial_triples[209]", "spatial_triples[210]", "spatial_triples[211]", "spatial_triples[212]", "spatial_triples[213]", "spatial_triples[214]", "spatial_triples[215]", "spatial_triples[216]", "spatial_triples[217]", "spatial_triples[218]", "spatial_triples[219]", "spatial_triples[220]", "spatial_triples[221]", "spatial_triples[222]", "spatial_triples[223]", "spatial_triples[224]", "spatial_triples[225]", "spatial_triples[226]", "spatial_triples[227]", "spatial_triples[228]", "spatial_triples[229]", "spatial_triples[230]", "spatial_triples[231]", "spatial_triples[232]", "spatial_triples[233]", "spatial_triples[234]", "spatial_triples[235]", "spatial_triples[236]", "spatial_triples[237]", "spatial_triples[238]", "spatial_triples[239]", "spatial_triples[240]", "spatial_triples[241]", "spatial_triples[242]", "spatial_triples[243]", "spatial_triples[244]", "spatial_triples[245]", "spatial_triples[246]", "spatial_triples[247]", "spatial_triples[248]", "spatial_triples[249]", "spatial_triples[250]", "spatial_triples[251]", "spatial_triples[252]", "spatial_triples[253]", "spatial_triples[254]", "spatial_triples[255]", "spatial_triples[256]", "spatial_triples[257]", "spatial_triples[258]", "spatial_triples[259]", "spatial_triples[260]", "spatial_triples[261]", "spatial_triples[262]", "spatial_triples[263]", "spatial_triples[264]", "spatial_triples[265]", "spatial_triples[266]", "spatial_triples[267]", "spatial_triples[268]", "spatial_triples[269]", "spatial_triples[270]", "spatial_triples[271]", "spatial_triples[272]", "spatial_triples[273]", "spatial_triples[274]", "spatial_triples[275]", "spatial_triples[276]", "spatial_triples[277]", "spatial_triples[278]", "spatial_triples[279]", "spatial_triples[280]", "spatial_triples[281]", "spatial_triples[282]", "spatial_triples[283]", "spatial_triples[284]", "spatial_triples[285]", "spatial_triples[286]", "spatial_triples[287]", "spatial_triples[288]", "spatial_triples[289]", "spatial_triples[290]", "spatial_triples[291]", "spatial_triples[292]", "spatial_triples[293]", "spatial_triples[294]", "spatial_triples[295]", "spatial_triples[296]", "spatial_triples[297]", "spatial_triples[298]", "spatial_triples[299]", "spatial_triples[300]", "spatial_triples[301]", "spatial_triples[302]", "spatial_triples[303]", "spatial_triples[304]", "spatial_triples[305]", "spatial_triples[306]", "spatial_triples[307]", "spatial_triples[308]", "spatial_triples[309]", "spatial_triples[310]", "spatial_triples[311]", "spatial_triples[312]", "spatial_triples[313]", "spatial_triples[314]", "spatial_triples[315]", "spatial_triples[316]", "spatial_triples[317]", "spatial_triples[318]", "spatial_triples[319]", "spatial_triples[320]", "spatial_triples[321]", "spatial_triples[322]", "spatial_triples[323]", "spatial_triples[324]", "spatial_triples[325]", "spatial_triples[326]", "spatial_triples[327]", "spatial_triples[328]", "spatial_triples[329]", "spatial_triples[330]", "spatial_triples[331]", "spatial_triples[332]", "spatial_triples[333]", "spatial_triples[334]", "spatial_triples[335]", "spatial_triples[336]", "spatial_triples[337]", "spatial_triples[338]", "spatial_triples[339]", "spatial_triples[340]", "spatial_triples[341]", "spatial_triples[342]", "spatial_triples[343]", "spatial_triples[344]", "spatial_triples[345]", "spatial_triples[346]", "spatial_triples[347]", "spatial_triples[348]", "spatial_triples[349]", "spatial_triples[350]", "spatial_triples[351]", "spatial_triples[352]", "spatial_triples[353]", "spatial_triples[354]", "spatial_triples[355]", "spatial_triples[356]", "spatial_triples[357]", "spatial_triples[358]", "spatial_triples[359]", "spatial_triples[360]", "spatial_triples[361]", "spatial_triples[362]", "spatial_triples[363]", "spatial_triples[364]", "spatial_triples[365]", "spatial_triples[366]", "spatial_triples[367]", "spatial_triples[368]", "spatial_triples[369]", "spatial_triples[370]", "spatial_triples[371]", "spatial_triples[372]", "spatial_triples[373]", "spatial_triples[374]", "spatial_triples[375]", "spatial_triples[376]", "spatial_triples[377]", "spatial_triples[378]", "spatial_triples[379]", "spatial_triples[380]", "spatial_triples[381]", "spatial_triples[382]", "spatial_triples[383]", "spatial_triples[384]", "spatial_triples[385]", "spatial_triples[386]", "spatial_triples[387]", "spatial_triples[388]", "spatial_triples[389]", "spatial_triples[390]", "spatial_triples[391]", "spatial_triples[392]", "spatial_triples[393]", "spatial_triples[394]", "spatial_triples[395]", "spatial_triples[396]", "spatial_triples[397]", "spatial_triples[398]", "spatial_triples[399]", "spatial_triples[400]", "spatial_triples[401]", "spatial_triples[402]", "spatial_triples[403]", "spatial_triples[404]", "spatial_triples[405]", "spatial_triples[406]", "spatial_triples[407]", "spatial_triples[408]", "spatial_triples[409]", "spatial_triples[410]", "spatial_triples[411]", "spatial_triples[412]", "spatial_triples[413]", "spatial_triples[414]", "spatial_triples[415]", "spatial_triples[416]", "spatial_triples[417]", "spatial_triples[418]", "spatial_triples[419]", "spatial_triples[420]", "spatial_triples[421]", "spatial_triples[422]", "spatial_triples[423]", "spatial_triples[424]", "spatial_triples[425]", "spatial_triples[426]", "spatial_triples[427]", "spatial_triples[428]", "spatial_triples[429]", "spatial_triples[430]", "spatial_triples[431]", "spatial_triples[432]", "spatial_triples[433]", "spatial_triples[434]", "spatial_triples[435]", "spatial_triples[436]", "spatial_triples[437]", "spatial_triples[438]", "spatial_triples[439]", "spatial_triples[440]", "spatial_triples[441]", "spatial_triples[442]", "spatial_triples[443]", "spatial_triples[444]", "spatial_triples[445]", "spatial_triples[446]", "spatial_triples[447]", "spatial_triples[448]", "spatial_triples[449]", "spatial_triples[450]", "spatial_triples[451]", "spatial_triples[452]", "spatial_triples[453]", "spatial_triples[454]", "spatial_triples[455]", "spatial_triples[456]", "spatial_triples[457]", "spatial_triples[458]", "spatial_triples[459]", "spatial_triples[460]", "spatial_triples[461]", "spatial_triples[462]", "spatial_triples[463]", "spatial_triples[464]", "spatial_triples[465]", "spatial_triples[466]", "spatial_triples[467]", "spatial_triples[468]", "spatial_triples[469]", "spatial_triples[470]", "spatial_triples[471]", "spatial_triples[472]", "spatial_triples[473]", "spatial_triples[474]", "spatial_triples[475]", "spatial_triples[476]", "spatial_triples[477]", "spatial_triples[478]", "spatial_triples[479]", "spatial_triples[480]", "spatial_triples[481]", "spatial_triples[482]", "spatial_triples[483]", "spatial_triples[484]", "spatial_triples[485]", "spatial_triples[486]", "spatial_triples[487]", "spatial_triples[488]", "spatial_triples[489]", "spatial_triples[490]", "spatial_triples[491]", "spatial_triples[492]", "spatial_triples[493]", "spatial_triples[494]", "spatial_triples[495]", "spatial_triples[496]", "spatial_triples[497]", "spatial_triples[498]", "spatial_triples[499]", "spatial_triples[500]", "spatial_triples[501]", "spatial_triples[502]", "spatial_triples[503]", "spatial_triples[504]", "spatial_triples[505]", "spatial_triples[506]", "spatial_triples[507]", "spatial_triples[508]", "spatial_triples[509]", "spatial_triples[510]", "spatial_triples[511]", "spatial_triples[512]", "spatial_triples[513]", "spatial_triples[514]", "spatial_triples[515]", "spatial_triples[516]", "spatial_triples[517]", "spatial_triples[518]", "spatial_triples[519]", "spatial_triples[520]", "spatial_triples[521]", "spatial_triples[522]", "spatial_triples[523]", "spatial_triples[524]", "spatial_triples[525]", "spatial_triples[526]", "spatial_triples[527]", "spatial_triples[528]", "spatial_triples[529]", "spatial_triples[530]", "spatial_triples[531]", "spatial_triples[532]", "spatial_triples[533]", "spatial_triples[534]", "spatial_triples[535]", "spatial_triples[536]", "spatial_triples[537]", "spatial_triples[538]", "spatial_triples[539]", "spatial_triples[540]", "spatial_triples[541]", "spatial_triples[542]", "spatial_triples[543]", "spatial_triples[544]", "spatial_triples[545]", "spatial_triples[546]", "spatial_triples[547]", "spatial_triples[548]", "spatial_triples[549]", "spatial_triples[550]", "spatial_triples[551]", "spatial_triples[552]", "spatial_triples[553]", "spatial_triples[554]", "spatial_triples[555]", "spatial_triples[556]", "spatial_triples[557]", "spatial_triples[558]", "spatial_triples[559]", "spatial_triples[560]", "spatial_triples[561]", "spatial_triples[562]", "spatial_triples[563]", "spatial_triples[564]", "spatial_triples[565]", "spatial_triples[566]", "spatial_triples[567]", "spatial_triples[568]", "spatial_triples[569]", "spatial_triples[570]", "spatial_triples[571]", "spatial_triples[572]", "spatial_triples[573]", "spatial_triples[574]", "spatial_triples[575]", "spatial_triples[576]", "spatial_triples[577]", "spatial_triples[578]", "spatial_triples[579]", "spatial_triples[580]", "spatial_triples[581]", "spatial_triples[582]", "spatial_triples[583]", "spatial_triples[584]", "spatial_triples[585]", "spatial_triples[586]", "spatial_triples[587]", "spatial_triples[588]", "spatial_triples[589]", "spatial_triples[590]", "spatial_triples[591]", "spatial_triples[592]", "spatial_triples[593]", "spatial_triples[594]", "spatial_triples[595]", "spatial_triples[596]", "spatial_triples[597]", "spatial_triples[598]", "spatial_triples[599]", "spatial_triples[600]", "spatial_triples[601]", "spatial_triples[602]", "spatial_triples[603]", "spatial_triples[604]", "spatial_triples[605]", "spatial_triples[606]", "spatial_triples[607]", "spatial_triples[608]", "spatial_triples[609]", "spatial_triples[610]", "spatial_triples[611]", "spatial_triples[612]", "spatial_triples[613]", "spatial_triples[614]", "spatial_triples[615]", "spatial_triples[616]", "spatial_triples[617]", "spatial_triples[618]", "spatial_triples[619]", "spatial_triples[620]", "spatial_triples[621]", "spatial_triples[622]", "spatial_triples[623]", "spatial_triples[624]", "spatial_triples[625]", "spatial_triples[626]", "spatial_triples[627]", "spatial_triples[628]", "spatial_triples[629]", "spatial_triples[630]", "spatial_triples[631]", "spatial_triples[632]", "spatial_triples[633]", "spatial_triples[634]", "spatial_triples[635]", "spatial_triples[636]", "spatial_triples[637]", "spatial_triples[638]", "spatial_triples[639]", "spatial_triples[640]", "spatial_triples[641]", "spatial_triples[642]", "spatial_triples[643]", "spatial_triples[644]", "spatial_triples[645]", "spatial_triples[646]", "spatial_triples[647]", "spatial_triples[648]", "spatial_triples[649]", "spatial_triples[650]", "spatial_triples[651]", "spatial_triples[652]", "spatial_triples[653]", "spatial_triples[654]", "spatial_triples[655]", "spatial_triples[656]", "spatial_triples[657]", "spatial_triples[658]", "spatial_triples[659]", "spatial_triples[660]", "spatial_triples[661]", "spatial_triples[662]", "spatial_triples[663]", "spatial_triples[664]", "spatial_triples[665]", "spatial_triples[666]", "spatial_triples[667]", "spatial_triples[668]", "spatial_triples[669]", "spatial_triples[670]", "spatial_triples[671]", "spatial_triples[672]", "spatial_triples[673]", "spatial_triples[674]", "spatial_triples[675]", "spatial_triples[676]", "spatial_triples[677]", "spatial_triples[678]", "spatial_triples[679]", "spatial_triples[680]", "spatial_triples[681]", "spatial_triples[682]", "spatial_triples[683]", "spatial_triples[684]", "spatial_triples[685]", "spatial_triples[686]", "spatial_triples[687]", "spatial_triples[688]", "spatial_triples[689]", "spatial_triples[690]", "spatial_triples[691]", "spatial_triples[692]", "spatial_triples[693]", "spatial_triples[694]", "spatial_triples[695]", "spatial_triples[696]", "spatial_triples[697]", "spatial_triples[698]", "spatial_triples[699]", "spatial_triples[700]", "spatial_triples[701]", "spatial_triples[702]", "spatial_triples[703]", "spatial_triples[704]", "spatial_triples[705]", "spatial_triples[706]", "spatial_triples[707]", "spatial_triples[708]", "spatial_triples[709]", "spatial_triples[710]", "spatial_triples[711]", "spatial_triples[712]", "spatial_triples[713]", "spatial_triples[714]", "spatial_triples[715]", "spatial_triples[716]", "spatial_triples[717]", "spatial_triples[718]", "spatial_triples[719]", "spatial_triples[720]", "spatial_triples[721]", "spatial_triples[722]", "spatial_triples[723]", "spatial_triples[724]", "spatial_triples[725]", "spatial_triples[726]", "spatial_triples[727]", "spatial_triples[728]", "spatial_triples[729]", "spatial_triples[730]", "spatial_triples[731]", "spatial_triples[732]", "spatial_triples[733]", "spatial_triples[734]", "spatial_triples[735]", "spatial_triples[736]", "spatial_triples[737]", "spatial_triples[738]", "spatial_triples[739]", "spatial_triples[740]", "spatial_triples[741]", "spatial_triples[742]", "spatial_triples[743]", "spatial_triples[744]", "spatial_triples[745]", "spatial_triples[746]", "spatial_triples[747]", "spatial_triples[748]", "spatial_triples[749]", "spatial_triples[750]", "spatial_triples[751]", "spatial_triples[752]", "spatial_triples[753]", "spatial_triples[754]", "spatial_triples[755]", "spatial_triples[756]", "spatial_triples[757]", "spatial_triples[758]", "spatial_triples[759]", "spatial_triples[760]", "spatial_triples[761]", "spatial_triples[762]", "spatial_triples[763]", "spatial_triples[764]", "spatial_triples[765]", "spatial_triples[766]", "spatial_triples[767]", "spatial_triples[768]", "spatial_triples[769]", "spatial_triples[770]", "spatial_triples[771]", "spatial_triples[772]", "spatial_triples[773]", "spatial_triples[774]", "spatial_triples[775]", "spatial_triples[776]", "spatial_triples[777]", "spatial_triples[778]", "spatial_triples[779]", "spatial_triples[780]", "spatial_triples[781]", "spatial_triples[782]", "spatial_triples[783]", "spatial_triples[784]", "spatial_triples[785]", "spatial_triples[786]", "spatial_triples[787]", "spatial_triples[788]", "spatial_triples[789]", "spatial_triples[790]", "spatial_triples[791]", "spatial_triples[792]", "spatial_triples[793]", "spatial_triples[794]", "spatial_triples[795]", "spatial_triples[796]", "spatial_triples[797]", "spatial_triples[798]", "spatial_triples[799]", "spatial_triples[800]", "spatial_triples[801]", "spatial_triples[802]", "spatial_triples[803]", "spatial_triples[804]", "spatial_triples[805]", "spatial_triples[806]", "spatial_triples[807]", "spatial_triples[808]", "spatial_triples[809]", "spatial_triples[810]", "spatial_triples[811]", "spatial_triples[812]", "spatial_triples[813]", "spatial_triples[814]", "spatial_triples[815]", "spatial_triples[816]", "spatial_triples[817]", "spatial_triples[818]", "spatial_triples[819]", "spatial_triples[820]", "spatial_triples[821]", "spatial_triples[822]", "spatial_triples[823]", "spatial_triples[824]", "spatial_triples[825]", "spatial_triples[826]", "spatial_triples[827]", "spatial_triples[828]", "spatial_triples[829]", "spatial_triples[830]", "spatial_triples[831]", "spatial_triples[832]", "spatial_triples[833]", "spatial_triples[834]", "spatial_triples[835]", "spatial_triples[836]", "spatial_triples[837]", "spatial_triples[838]", "spatial_triples[839]", "spatial_triples[840]", "spatial_triples[841]", "spatial_triples[842]", "spatial_triples[843]", "spatial_triples[844]", "spatial_triples[845]", "spatial_triples[846]", "spatial_triples[847]", "spatial_triples[848]", "spatial_triples[849]", "spatial_triples[850]", "spatial_triples[851]", "spatial_triples[852]", "spatial_triples[853]", "spatial_triples[854]", "spatial_triples[855]", "spatial_triples[856]", "spatial_triples[857]", "spatial_triples[858]", "spatial_triples[859]", "spatial_triples[860]", "spatial_triples[861]", "spatial_triples[862]", "spatial_triples[863]", "spatial_triples[864]", "spatial_triples[865]", "spatial_triples[866]", "spatial_triples[867]", "spatial_triples[868]", "spatial_triples[869]", "spatial_triples[870]", "spatial_triples[871]", "spatial_triples[872]", "spatial_triples[873]", "spatial_triples[874]", "spatial_triples[875]", "spatial_triples[876]", "spatial_triples[877]", "spatial_triples[878]", "spatial_triples[879]", "spatial_triples[880]", "spatial_triples[881]", "spatial_triples[882]", "spatial_triples[883]", "spatial_triples[884]", "spatial_triples[885]", "spatial_triples[886]", "spatial_triples[887]", "spatial_triples[888]", "spatial_triples[889]", "spatial_triples[890]", "spatial_triples[891]", "spatial_triples[892]", "spatial_triples[893]", "spatial_triples[894]", "spatial_triples[895]", "spatial_triples[896]", "spatial_triples[897]", "spatial_triples[898]", "spatial_triples[899]", "spatial_triples[900]", "spatial_triples[901]", "spatial_triples[902]", "spatial_triples[903]", "spatial_triples[904]", "spatial_triples[905]", "spatial_triples[906]", "spatial_triples[907]", "spatial_triples[908]", "spatial_triples[909]", "spatial_triples[910]", "spatial_triples[911]", "spatial_triples[912]", "spatial_triples[913]", "spatial_triples[914]", "spatial_triples[915]", "spatial_triples[916]", "spatial_triples[917]", "spatial_triples[918]", "spatial_triples[919]", "spatial_triples[920]", "spatial_triples[921]", "spatial_triples[922]", "spatial_triples[923]", "spatial_triples[924]", "spatial_triples[925]", "spatial_triples[926]", "spatial_triples[927]", "spatial_triples[928]", "spatial_triples[929]", "spatial_triples[930]", "spatial_triples[931]", "spatial_triples[932]", "spatial_triples[933]", "spatial_triples[934]", "spatial_triples[935]", "spatial_triples[936]", "spatial_triples[937]", "spatial_triples[938]", "spatial_triples[939]", "spatial_triples[940]", "spatial_triples[941]", "spatial_triples[942]", "spatial_triples[943]", "spatial_triples[944]", "spatial_triples[945]", "spatial_triples[946]", "spatial_triples[947]", "spatial_triples[948]", "spatial_triples[949]", "spatial_triples[950]", "spatial_triples[951]", "spatial_triples[952]", "spatial_triples[953]", "spatial_triples[954]", "spatial_triples[955]", "spatial_triples[956]", "spatial_triples[957]", "spatial_triples[958]", "spatial_triples[959]", "spatial_triples[960]", "spatial_triples[961]", "spatial_triples[962]", "spatial_triples[963]", "spatial_triples[964]", "spatial_triples[965]", "spatial_triples[966]", "spatial_triples[967]", "spatial_triples[968]", "spatial_triples[969]", "spatial_triples[970]", "spatial_triples[971]", "spatial_triples[972]", "spatial_triples[973]", "spatial_triples[974]", "spatial_triples[975]", "spatial_triples[976]", "spatial_triples[977]", "spatial_triples[978]", "spatial_triples[979]", "spatial_triples[980]", "spatial_triples[981]", "spatial_triples[982]", "spatial_triples[983]", "spatial_triples[984]", "spatial_triples[985]", "spatial_triples[986]", "spatial_triples[987]", "spatial_triples[988]", "spatial_triples[989]", "spatial_triples[990]", "spatial_triples[991]", "spatial_triples[992]", "spatial_triples[993]", "spatial_triples[994]", "spatial_triples[995]", "spatial_triples[996]", "spatial_triples[997]", "spatial_triples[998]", "spatial_triples[999]", "spatial_triples[1000]", "spatial_triples[1001]", "spatial_triples[1002]", "spatial_triples[1003]", "spatial_triples[1004]", "spatial_triples[1005]", "spatial_triples[1006]", "spatial_triples[1007]", "spatial_triples[1008]", "spatial_triples[1009]", "spatial_triples[1010]", "spatial_triples[1011]", "spatial_triples[1012]", "spatial_triples[1013]", "spatial_triples[1014]", "spatial_triples[1015]", "spatial_triples[1016]", "spatial_triples[1017]", "spatial_triples[1018]", "spatial_triples[1019]", "spatial_triples[1020]", "spatial_triples[1021]", "spatial_triples[1022]", "spatial_triples[1023]", "spatial_triples[1024]", "spatial_triples[1025]", "spatial_triples[1026]", "spatial_triples[1027]", "spatial_triples[1028]", "spatial_triples[1029]", "spatial_triples[1030]", "spatial_triples[1031]", "spatial_triples[1032]", "spatial_triples[1033]", "spatial_triples[1034]", "spatial_triples[1035]", "spatial_triples[1036]", "spatial_triples[1037]", "spatial_triples[1038]", "spatial_triples[1039]", "spatial_triples[1040]", "spatial_triples[1041]", "spatial_triples[1042]", "spatial_triples[1043]", "spatial_triples[1044]", "spatial_triples[1045]", "spatial_triples[1046]", "spatial_triples[1047]", "spatial_triples[1048]", "spatial_triples[1049]", "spatial_triples[1050]", "spatial_triples[1051]", "spatial_triples[1052]", "spatial_triples[1053]", "spatial_triples[1054]", "spatial_triples[1055]", "spatial_triples[1056]", "spatial_triples[1057]", "spatial_triples[1058]", "spatial_triples[1059]", "spatial_triples[1060]", "spatial_triples[1061]", "spatial_triples[1062]", "spatial_triples[1063]", "spatial_triples[1064]", "spatial_triples[1065]", "spatial_triples[1066]", "spatial_triples[1067]", "spatial_triples[1068]", "spatial_triples[1069]", "spatial_triples[1070]", "spatial_triples[1071]", "spatial_triples[1072]", "spatial_triples[1073]", "spatial_triples[1074]", "spatial_triples[1075]", "spatial_triples[1076]", "spatial_triples[1077]", "spatial_triples[1078]", "spatial_triples[1079]", "spatial_triples[1080]", "spatial_triples[1081]", "spatial_triples[1082]", "spatial_triples[1083]", "spatial_triples[1084]", "spatial_triples[1085]", "spatial_triples[1086]", "spatial_triples[1087]", "spatial_triples[1088]", "spatial_triples[1089]", "spatial_triples[1090]", "spatial_triples[1091]", "spatial_triples[1092]", "spatial_triples[1093]", "spatial_triples[1094]", "spatial_triples[1095]", "spatial_triples[1096]", "spatial_triples[1097]", "spatial_triples[1098]", "spatial_triples[1099]", "spatial_triples[1100]", "spatial_triples[1101]", "spatial_triples[1102]", "spatial_triples[1103]", "spatial_triples[1104]", "spatial_triples[1105]", "spatial_triples[1106]", "spatial_triples[1107]", "spatial_triples[1108]", "spatial_triples[1109]", "spatial_triples[1110]", "spatial_triples[1111]", "spatial_triples[1112]", "spatial_triples[1113]", "spatial_triples[1114]", "spatial_triples[1115]", "spatial_triples[1116]", "spatial_triples[1117]", "spatial_triples[1118]", "spatial_triples[1119]", "spatial_triples[1120]", "spatial_triples[1121]", "spatial_triples[1122]", "spatial_triples[1123]", "spatial_triples[1124]", "spatial_triples[1125]", "spatial_triples[1126]", "spatial_triples[1127]", "spatial_triples[1128]", "spatial_triples[1129]", "spatial_triples[1130]", "spatial_triples[1131]", "spatial_triples[1132]", "spatial_triples[1133]", "spatial_triples[1134]", "spatial_triples[1135]", "spatial_triples[1136]", "spatial_triples[1137]", "spatial_triples[1138]", "spatial_triples[1139]", "spatial_triples[1140]", "spatial_triples[1141]", "spatial_triples[1142]", "spatial_triples[1143]", "spatial_triples[1144]", "spatial_triples[1145]", "spatial_triples[1146]", "spatial_triples[1147]", "spatial_triples[1148]", "spatial_triples[1149]", "spatial_triples[1150]", "spatial_triples[1151]", "spatial_triples[1152]", "spatial_triples[1153]", "spatial_triples[1154]", "spatial_triples[1155]", "spatial_triples[1156]", "spatial_triples[1157]", "spatial_triples[1158]", "spatial_triples[1159]", "spatial_triples[1160]", "spatial_triples[1161]", "spatial_triples[1162]", "spatial_triples[1163]", "spatial_triples[1164]", "spatial_triples[1165]", "spatial_triples[1166]", "spatial_triples[1167]", "spatial_triples[1168]", "spatial_triples[1169]", "spatial_triples[1170]", "spatial_triples[1171]", "spatial_triples[1172]", "spatial_triples[1173]", "spatial_triples[1174]", "spatial_triples[1175]", "spatial_triples[1176]", "spatial_triples[1177]", "spatial_triples[1178]", "spatial_triples[1179]", "spatial_triples[1180]", "spatial_triples[1181]", "spatial_triples[1182]", "spatial_triples[1183]", "spatial_triples[1184]", "spatial_triples[1185]", "spatial_triples[1186]", "spatial_triples[1187]", "spatial_triples[1188]", "spatial_triples[1189]", "spatial_triples[1190]", "spatial_triples[1191]", "spatial_triples[1192]", "spatial_triples[1193]", "spatial_triples[1194]", "spatial_triples[1195]", "spatial_triples[1196]", "spatial_triples[1197]", "spatial_triples[1198]", "spatial_triples[1199]", "spatial_triples[1200]", "spatial_triples[1201]", "spatial_triples[1202]", "spatial_triples[1203]", "spatial_triples[1204]", "spatial_triples[1205]", "spatial_triples[1206]", "spatial_triples[1207]", "spatial_triples[1208]", "spatial_triples[1209]", "spatial_triples[1210]", "spatial_triples[1211]", "spatial_triples[1212]", "spatial_triples[1213]", "spatial_triples[1214]", "spatial_triples[1215]", "spatial_triples[1216]", "spatial_triples[1217]", "spatial_triples[1218]", "spatial_triples[1219]", "spatial_triples[1220]", "spatial_triples[1221]", "spatial_triples[1222]", "spatial_triples[1223]", "spatial_triples[1224]", "spatial_triples[1225]", "spatial_triples[1226]", "spatial_triples[1227]", "spatial_triples[1228]", "spatial_triples[1229]", "spatial_triples[1230]", "spatial_triples[1231]", "spatial_triples[1232]", "spatial_triples[1233]", "spatial_triples[1234]", "spatial_triples[1235]", "spatial_triples[1236]", "spatial_triples[1237]", "spatial_triples[1238]", "spatial_triples[1239]", "spatial_triples[1240]", "spatial_triples[1241]", "spatial_triples[1242]", "spatial_triples[1243]", "spatial_triples[1244]", "spatial_triples[1245]", "spatial_triples[1246]", "spatial_triples[1247]", "spatial_triples[1248]", "spatial_triples[1249]", "spatial_triples[1250]", "spatial_triples[1251]", "spatial_triples[1252]", "spatial_triples[1253]", "spatial_triples[1254]", "spatial_triples[1255]", "spatial_triples[1256]", "spatial_triples[1257]", "spatial_triples[1258]", "spatial_triples[1259]", "spatial_triples[1260]", "spatial_triples[1261]", "spatial_triples[1262]", "spatial_triples[1263]", "spatial_triples[1264]", "spatial_triples[1265]", "spatial_triples[1266]", "spatial_triples[1267]", "spatial_triples[1268]", "spatial_triples[1269]", "spatial_triples[1270]", "spatial_triples[1271]", "spatial_triples[1272]", "spatial_triples[1273]", "spatial_triples[1274]", "spatial_triples[1275]", "spatial_triples[1276]", "spatial_triples[1277]", "spatial_triples[1278]", "spatial_triples[1279]", "spatial_triples[1280]", "spatial_triples[1281]", "spatial_triples[1282]", "spatial_triples[1283]", "spatial_triples[1284]", "spatial_triples[1285]", "spatial_triples[1286]", "spatial_triples[1287]", "spatial_triples[1288]", "spatial_triples[1289]", "spatial_triples[1290]", "spatial_triples[1291]", "spatial_triples[1292]", "spatial_triples[1293]", "spatial_triples[1294]", "spatial_triples[1295]", "spatial_triples[1296]", "spatial_triples[1297]", "spatial_triples[1298]", "spatial_triples[1299]", "spatial_triples[1300]", "spatial_triples[1301]", "spatial_triples[1302]", "spatial_triples[1303]", "spatial_triples[1304]", "spatial_triples[1305]", "spatial_triples[1306]", "spatial_triples[1307]", "spatial_triples[1308]", "spatial_triples[1309]", "spatial_triples[1310]", "spatial_triples[1311]", "spatial_triples[1312]", "spatial_triples[1313]", "spatial_triples[1314]", "spatial_triples[1315]", "spatial_triples[1316]", "spatial_triples[1317]", "spatial_triples[1318]", "spatial_triples[1319]", "spatial_triples[1320]", "spatial_triples[1321]", "spatial_triples[1322]", "spatial_triples[1323]", "spatial_triples[1324]", "spatial_triples[1325]", "spatial_triples[1326]", "spatial_triples[1327]", "spatial_triples[1328]", "spatial_triples[1329]", "spatial_triples[1330]", "spatial_triples[1331]", "spatial_triples[1332]", "spatial_triples[1333]", "spatial_triples[1334]", "spatial_triples[1335]", "spatial_triples[1336]", "spatial_triples[1337]", "spatial_triples[1338]", "spatial_triples[1339]", "spatial_triples[1340]", "spatial_triples[1341]", "spatial_triples[1342]", "spatial_triples[1343]", "spatial_triples[1344]", "spatial_triples[1345]", "spatial_triples[1346]", "spatial_triples[1347]", "spatial_triples[1348]", "spatial_triples[1349]", "spatial_triples[1350]", "spatial_triples[1351]", "spatial_triples[1352]", "spatial_triples[1353]", "spatial_triples[1354]", "spatial_triples[1355]", "spatial_triples[1356]", "spatial_triples[1357]", "spatial_triples[1358]", "spatial_triples[1359]", "spatial_triples[1360]", "spatial_triples[1361]", "spatial_triples[1362]", "spatial_triples[1363]", "spatial_triples[1364]", "spatial_triples[1365]", "spatial_triples[1366]", "spatial_triples[1367]", "spatial_triples[1368]", "spatial_triples[1369]", "spatial_triples[1370]", "spatial_triples[1371]", "spatial_triples[1372]", "spatial_triples[1373]", "spatial_triples[1374]", "spatial_triples[1375]", "spatial_triples[1376]", "spatial_triples[1377]", "spatial_triples[1378]", "spatial_triples[1379]", "spatial_triples[1380]", "spatial_triples[1381]", "spatial_triples[1382]", "spatial_triples[1383]", "spatial_triples[1384]", "spatial_triples[1385]", "spatial_triples[1386]", "spatial_triples[1387]", "spatial_triples[1388]", "spatial_triples[1389]", "spatial_triples[1390]", "spatial_triples[1391]", "spatial_triples[1392]", "spatial_triples[1393]", "spatial_triples[1394]", "spatial_triples[1395]", "spatial_triples[1396]", "spatial_triples[1397]", "spatial_triples[1398]", "spatial_triples[1399]", "spatial_triples[1400]", "spatial_triples[1401]", "spatial_triples[1402]", "spatial_triples[1403]", "spatial_triples[1404]", "spatial_triples[1405]", "spatial_triples[1406]", "spatial_triples[1407]", "spatial_triples[1408]", "spatial_triples[1409]", "spatial_triples[1410]", "spatial_triples[1411]", "spatial_triples[1412]", "spatial_triples[1413]", "spatial_triples[1414]", "spatial_triples[1415]", "spatial_triples[1416]", "spatial_triples[1417]", "spatial_triples[1418]", "spatial_triples[1419]", "spatial_triples[1420]", "spatial_triples[1421]", "spatial_triples[1422]", "spatial_triples[1423]", "spatial_triples[1424]", "spatial_triples[1425]", "spatial_triples[1426]", "spatial_triples[1427]", "spatial_triples[1428]", "spatial_triples[1429]", "spatial_triples[1430]", "spatial_triples[1431]", "spatial_triples[1432]", "spatial_triples[1433]", "spatial_triples[1434]", "spatial_triples[1435]", "spatial_triples[1436]", "spatial_triples[1437]", "spatial_triples[1438]", "spatial_triples[1439]", "spatial_triples[1440]", "spatial_triples[1441]", "spatial_triples[1442]", "spatial_triples[1443]", "spatial_triples[1444]", "spatial_triples[1445]", "spatial_triples[1446]", "spatial_triples[1447]", "spatial_triples[1448]", "spatial_triples[1449]", "spatial_triples[1450]", "spatial_triples[1451]", "spatial_triples[1452]", "spatial_triples[1453]", "spatial_triples[1454]", "spatial_triples[1455]", "spatial_triples[1456]", "spatial_triples[1457]", "spatial_triples[1458]", "spatial_triples[1459]", "spatial_triples[1460]", "spatial_triples[1461]", "spatial_triples[1462]", "spatial_triples[1463]", "spatial_triples[1464]", "spatial_triples[1465]", "spatial_triples[1466]", "spatial_triples[1467]", "spatial_triples[1468]", "spatial_triples[1469]", "spatial_triples[1470]", "spatial_triples[1471]", "spatial_triples[1472]", "spatial_triples[1473]", "spatial_triples[1474]", "spatial_triples[1475]", "spatial_triples[1476]", "spatial_triples[1477]", "spatial_triples[1478]", "spatial_triples[1479]"};
const vector<string> __vecstr_instance_Region = {"regions[0]", "regions[1]", "regions[2]", "regions[3]", "regions[4]", "regions[5]", "regions[6]", "regions[7]", "regions[8]", "regions[9]", "regions[10]", "regions[11]", "regions[12]", "regions[13]", "regions[14]", "regions[15]", "regions[16]", "regions[17]", "regions[18]", "regions[19]", "regions[20]", "regions[21]", "regions[22]", "regions[23]", "regions[24]"};
const vector<string> __vecstr_instance_County = {"counties[0]", "counties[1]", "counties[2]", "counties[3]", "counties[4]", "counties[5]", "counties[6]", "counties[7]", "counties[8]", "counties[9]", "counties[10]", "counties[11]", "counties[12]", "counties[13]", "counties[14]", "counties[15]", "counties[16]", "counties[17]", "counties[18]", "counties[19]", "counties[20]", "counties[21]", "counties[22]", "counties[23]", "counties[24]", "counties[25]", "counties[26]", "counties[27]", "counties[28]", "counties[29]", "counties[30]", "counties[31]", "counties[32]", "counties[33]", "counties[34]", "counties[35]", "counties[36]", "counties[37]", "counties[38]", "counties[39]", "counties[40]", "counties[41]", "counties[42]", "counties[43]", "counties[44]", "counties[45]", "counties[46]", "counties[47]", "counties[48]", "counties[49]", "counties[50]", "counties[51]", "counties[52]", "counties[53]", "counties[54]", "counties[55]", "counties[56]", "counties[57]", "counties[58]", "counties[59]", "counties[60]", "counties[61]", "counties[62]", "counties[63]", "counties[64]", "counties[65]", "counties[66]", "counties[67]", "counties[68]", "counties[69]", "counties[70]", "counties[71]", "counties[72]", "counties[73]", "counties[74]", "counties[75]", "counties[76]", "counties[77]", "counties[78]", "counties[79]", "counties[80]", "counties[81]", "counties[82]", "counties[83]", "counties[84]", "counties[85]", "counties[86]", "counties[87]", "counties[88]", "counties[89]", "counties[90]", "counties[91]", "counties[92]", "counties[93]", "counties[94]", "counties[95]", "counties[96]", "counties[97]", "counties[98]", "counties[99]", "counties[100]", "counties[101]", "counties[102]", "counties[103]", "counties[104]", "counties[105]", "counties[106]", "counties[107]", "counties[108]", "counties[109]", "counties[110]", "counties[111]", "counties[112]", "counties[113]", "counties[114]", "counties[115]", "counties[116]", "counties[117]", "counties[118]", "counties[119]", "counties[120]", "counties[121]", "counties[122]", "counties[123]", "counties[124]", "counties[125]", "counties[126]", "counties[127]", "counties[128]", "counties[129]", "counties[130]", "counties[131]", "counties[132]", "counties[133]", "counties[134]", "counties[135]", "counties[136]", "counties[137]", "counties[138]", "counties[139]", "counties[140]", "counties[141]", "counties[142]", "counties[143]", "counties[144]", "counties[145]", "counties[146]", "counties[147]", "counties[148]", "counties[149]", "counties[150]", "counties[151]", "counties[152]", "counties[153]", "counties[154]", "counties[155]", "counties[156]", "counties[157]", "counties[158]", "counties[159]", "counties[160]", "counties[161]", "counties[162]", "counties[163]", "counties[164]", "counties[165]", "counties[166]", "counties[167]", "counties[168]", "counties[169]", "counties[170]", "counties[171]", "counties[172]", "counties[173]", "counties[174]", "counties[175]", "counties[176]", "counties[177]", "counties[178]", "counties[179]", "counties[180]", "counties[181]", "counties[182]", "counties[183]", "counties[184]", "counties[185]", "counties[186]", "counties[187]", "counties[188]", "counties[189]", "counties[190]", "counties[191]", "counties[192]", "counties[193]", "counties[194]", "counties[195]", "counties[196]", "counties[197]", "counties[198]", "counties[199]", "counties[200]", "counties[201]", "counties[202]", "counties[203]", "counties[204]", "counties[205]", "counties[206]", "counties[207]", "counties[208]", "counties[209]", "counties[210]", "counties[211]", "counties[212]", "counties[213]", "counties[214]", "counties[215]", "counties[216]", "counties[217]", "counties[218]", "counties[219]", "counties[220]", "counties[221]", "counties[222]", "counties[223]", "counties[224]", "counties[225]", "counties[226]", "counties[227]", "counties[228]", "counties[229]", "counties[230]", "counties[231]", "counties[232]", "counties[233]", "counties[234]", "counties[235]", "counties[236]", "counties[237]", "counties[238]", "counties[239]", "counties[240]", "counties[241]", "counties[242]", "counties[243]", "counties[244]", "counties[245]", "counties[246]", "counties[247]", "counties[248]", "counties[249]", "counties[250]", "counties[251]", "counties[252]", "counties[253]", "counties[254]", "counties[255]", "counties[256]", "counties[257]", "counties[258]", "counties[259]", "counties[260]", "counties[261]", "counties[262]", "counties[263]", "counties[264]", "counties[265]", "counties[266]", "counties[267]", "counties[268]", "counties[269]", "counties[270]", "counties[271]", "counties[272]", "counties[273]", "counties[274]", "counties[275]", "counties[276]"};
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
TruncatedGauss TruncatedGauss28865968;
TruncatedGauss TruncatedGauss28866480;
Gaussian Gaussian28866896;
BooleanDistrib BooleanDistrib28878224;
BooleanDistrib BooleanDistrib28882224;
Gaussian Gaussian28886496;
Gaussian Gaussian28973584;
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
  for (int c = 0;c<277;c++)
  for (int t = 0;t<103;t++)
  _answer_4[c][t]->add(_mem_logit[c][t]->getval(),1);


}
void _init_storage()
{
  _mem_beta1=new _Var_beta1();
  _mem_beta2=new _Var_beta2();
  _mem_bias=new _Var_bias();
  _mem_temporal_edge.resize(0,277);
  _mem_temporal_edge.resize(1,102);
  for (int c = 0;c<277;c++)
  {
    for (int t = 0;t<102;t++)
    {
      _mem_temporal_edge[c][t]=new _Var_temporal_edge(c, t);
    }

  }

  _mem_spatial_edge.resize(0,103);
  _mem_spatial_edge.resize(1,1480);
  for (int t = 0;t<103;t++)
  {
    for (int s = 0;s<1480;s++)
    {
      _mem_spatial_edge[t][s]=new _Var_spatial_edge(t, s);
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

  _mem_region_rate.resize(0,25);
  _mem_region_rate.resize(1,103);
  for (int r = 0;r<25;r++)
  {
    for (int t = 0;t<103;t++)
    {
      _mem_region_rate[r][t]=new _Var_region_rate(r, t);
    }

  }

  TruncatedGauss28865968.init(0.10000000,0.10000000,0.000000,0.25000000);
  TruncatedGauss28866480.init(0.10000000,0.10000000,0.000000,0.25000000);
  Gaussian28866896.init(-4.00000000,0.01000000);
  _answer_4.resize(0,277);
  _answer_4.resize(1,103);
  for (int c = 0;c<277;c++)
  {
    for (int t = 0;t<103;t++)
    {
      _answer_4[c][t]=new Hist<double>(false, 20);
    }

  }

}
void _init_world()
{
  for (int c = 0;c<277;c++)
  for (int t = 0;t<102;t++)
  _util_set_evidence<char>(_mem_temporal_edge[c][t],1);


  for (int t = 0;t<103;t++)
  for (int s = 0;s<1480;s++)
  _util_set_evidence<char>(_mem_spatial_edge[t][s],1);


  for (int r = 0;r<25;r++)
  for (int t = 0;t<103;t++)
  if (__fixed_observations(t,r)>0.000000)
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
  for (int c = 0;c<277;c++)
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
  return TruncatedGauss28865968.loglikeli(val);
}
double _Var_beta1::getcachelikeli()
{
  auto _t_val = getcache();
  return TruncatedGauss28865968.loglikeli(_t_val);
}
void _Var_beta1::sample()
{
  val=TruncatedGauss28865968.gen();
}
void _Var_beta1::sample_cache()
{
  cache_val=TruncatedGauss28865968.gen();
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
  return TruncatedGauss28866480.loglikeli(val);
}
double _Var_beta2::getcachelikeli()
{
  auto _t_val = getcache();
  return TruncatedGauss28866480.loglikeli(_t_val);
}
void _Var_beta2::sample()
{
  val=TruncatedGauss28866480.gen();
}
void _Var_beta2::sample_cache()
{
  cache_val=TruncatedGauss28866480.gen();
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
  return Gaussian28866896.loglikeli(val);
}
double _Var_bias::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian28866896.loglikeli(_t_val);
}
void _Var_bias::sample()
{
  val=Gaussian28866896.gen();
}
void _Var_bias::sample_cache()
{
  cache_val=Gaussian28866896.gen();
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
  return BooleanDistrib28878224.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getval()-_mem_bias->getval())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getval()-_mem_bias->getval()))),BooleanDistrib28878224.loglikeli(val);
}
double _Var_temporal_edge::getcachelikeli()
{
  auto _t_val = getcache();
  return BooleanDistrib28878224.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getcache()-_mem_bias->getcache())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getcache()-_mem_bias->getcache()))),BooleanDistrib28878224.loglikeli(_t_val);
}
void _Var_temporal_edge::sample()
{
  val=(BooleanDistrib28878224.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getval()-_mem_bias->getval())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getval()-_mem_bias->getval()))),BooleanDistrib28878224.gen());
}
void _Var_temporal_edge::sample_cache()
{
  cache_val=(BooleanDistrib28878224.init(exp(1.00000000*__fixed_tau1*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,0)))]->getcache()-_mem_bias->getcache())*(_mem_logit[c][__fixed_toWeek(toInt(__fixed_temporal_obs(t,1)))]->getcache()-_mem_bias->getcache()))),BooleanDistrib28878224.gen());
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
  return BooleanDistrib28882224.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getval()-_mem_bias->getval())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getval()-_mem_bias->getval()))),BooleanDistrib28882224.loglikeli(val);
}
double _Var_spatial_edge::getcachelikeli()
{
  auto _t_val = getcache();
  return BooleanDistrib28882224.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getcache()-_mem_bias->getcache())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getcache()-_mem_bias->getcache()))),BooleanDistrib28882224.loglikeli(_t_val);
}
void _Var_spatial_edge::sample()
{
  val=(BooleanDistrib28882224.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getval()-_mem_bias->getval())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getval()-_mem_bias->getval()))),BooleanDistrib28882224.gen());
}
void _Var_spatial_edge::sample_cache()
{
  cache_val=(BooleanDistrib28882224.init(exp(1.00000000*__fixed_tau1*__fixed_rho*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,0)))][t]->getcache()-_mem_bias->getcache())*(_mem_logit[__fixed_toCounty(toInt(__fixed_spatial_obs(s,1)))][t]->getcache()-_mem_bias->getcache()))),BooleanDistrib28882224.gen());
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
  return Gaussian28886496.init(_mem_bias->getval()+_mem_beta1->getval()*__fixed_covariates1(c,t)+_mem_beta2->getval()*__fixed_covariates2(c,t),__fixed_D[c]*9.00000000),Gaussian28886496.loglikeli(val);
}
double _Var_logit::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian28886496.init(_mem_bias->getcache()+_mem_beta1->getcache()*__fixed_covariates1(c,t)+_mem_beta2->getcache()*__fixed_covariates2(c,t),__fixed_D[c]*9.00000000),Gaussian28886496.loglikeli(_t_val);
}
void _Var_logit::sample()
{
  val=(Gaussian28886496.init(_mem_bias->getval()+_mem_beta1->getval()*__fixed_covariates1(c,t)+_mem_beta2->getval()*__fixed_covariates2(c,t),__fixed_D[c]*9.00000000),Gaussian28886496.gen());
}
void _Var_logit::sample_cache()
{
  cache_val=(Gaussian28886496.init(_mem_bias->getcache()+_mem_beta1->getcache()*__fixed_covariates1(c,t)+_mem_beta2->getcache()*__fixed_covariates2(c,t),__fixed_D[c]*9.00000000),Gaussian28886496.gen());
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
  return Gaussian28973584.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getval()), __fixed_sigmoid(_mem_logit[1][t]->getval()), __fixed_sigmoid(_mem_logit[2][t]->getval()), __fixed_sigmoid(_mem_logit[3][t]->getval()), __fixed_sigmoid(_mem_logit[4][t]->getval()), __fixed_sigmoid(_mem_logit[5][t]->getval()), __fixed_sigmoid(_mem_logit[6][t]->getval()), __fixed_sigmoid(_mem_logit[7][t]->getval()), __fixed_sigmoid(_mem_logit[8][t]->getval()), __fixed_sigmoid(_mem_logit[9][t]->getval()), __fixed_sigmoid(_mem_logit[10][t]->getval()), __fixed_sigmoid(_mem_logit[11][t]->getval()), __fixed_sigmoid(_mem_logit[12][t]->getval()), __fixed_sigmoid(_mem_logit[13][t]->getval()), __fixed_sigmoid(_mem_logit[14][t]->getval()), __fixed_sigmoid(_mem_logit[15][t]->getval()), __fixed_sigmoid(_mem_logit[16][t]->getval()), __fixed_sigmoid(_mem_logit[17][t]->getval()), __fixed_sigmoid(_mem_logit[18][t]->getval()), __fixed_sigmoid(_mem_logit[19][t]->getval()), __fixed_sigmoid(_mem_logit[20][t]->getval()), __fixed_sigmoid(_mem_logit[21][t]->getval()), __fixed_sigmoid(_mem_logit[22][t]->getval()), __fixed_sigmoid(_mem_logit[23][t]->getval()), __fixed_sigmoid(_mem_logit[24][t]->getval()), __fixed_sigmoid(_mem_logit[25][t]->getval()), __fixed_sigmoid(_mem_logit[26][t]->getval()), __fixed_sigmoid(_mem_logit[27][t]->getval()), __fixed_sigmoid(_mem_logit[28][t]->getval()), __fixed_sigmoid(_mem_logit[29][t]->getval()), __fixed_sigmoid(_mem_logit[30][t]->getval()), __fixed_sigmoid(_mem_logit[31][t]->getval()), __fixed_sigmoid(_mem_logit[32][t]->getval()), __fixed_sigmoid(_mem_logit[33][t]->getval()), __fixed_sigmoid(_mem_logit[34][t]->getval()), __fixed_sigmoid(_mem_logit[35][t]->getval()), __fixed_sigmoid(_mem_logit[36][t]->getval()), __fixed_sigmoid(_mem_logit[37][t]->getval()), __fixed_sigmoid(_mem_logit[38][t]->getval()), __fixed_sigmoid(_mem_logit[39][t]->getval()), __fixed_sigmoid(_mem_logit[40][t]->getval()), __fixed_sigmoid(_mem_logit[41][t]->getval()), __fixed_sigmoid(_mem_logit[42][t]->getval()), __fixed_sigmoid(_mem_logit[43][t]->getval()), __fixed_sigmoid(_mem_logit[44][t]->getval()), __fixed_sigmoid(_mem_logit[45][t]->getval()), __fixed_sigmoid(_mem_logit[46][t]->getval()), __fixed_sigmoid(_mem_logit[47][t]->getval()), __fixed_sigmoid(_mem_logit[48][t]->getval()), __fixed_sigmoid(_mem_logit[49][t]->getval()), __fixed_sigmoid(_mem_logit[50][t]->getval()), __fixed_sigmoid(_mem_logit[51][t]->getval()), __fixed_sigmoid(_mem_logit[52][t]->getval()), __fixed_sigmoid(_mem_logit[53][t]->getval()), __fixed_sigmoid(_mem_logit[54][t]->getval()), __fixed_sigmoid(_mem_logit[55][t]->getval()), __fixed_sigmoid(_mem_logit[56][t]->getval()), __fixed_sigmoid(_mem_logit[57][t]->getval()), __fixed_sigmoid(_mem_logit[58][t]->getval()), __fixed_sigmoid(_mem_logit[59][t]->getval()), __fixed_sigmoid(_mem_logit[60][t]->getval()), __fixed_sigmoid(_mem_logit[61][t]->getval()), __fixed_sigmoid(_mem_logit[62][t]->getval()), __fixed_sigmoid(_mem_logit[63][t]->getval()), __fixed_sigmoid(_mem_logit[64][t]->getval()), __fixed_sigmoid(_mem_logit[65][t]->getval()), __fixed_sigmoid(_mem_logit[66][t]->getval()), __fixed_sigmoid(_mem_logit[67][t]->getval()), __fixed_sigmoid(_mem_logit[68][t]->getval()), __fixed_sigmoid(_mem_logit[69][t]->getval()), __fixed_sigmoid(_mem_logit[70][t]->getval()), __fixed_sigmoid(_mem_logit[71][t]->getval()), __fixed_sigmoid(_mem_logit[72][t]->getval()), __fixed_sigmoid(_mem_logit[73][t]->getval()), __fixed_sigmoid(_mem_logit[74][t]->getval()), __fixed_sigmoid(_mem_logit[75][t]->getval()), __fixed_sigmoid(_mem_logit[76][t]->getval()), __fixed_sigmoid(_mem_logit[77][t]->getval()), __fixed_sigmoid(_mem_logit[78][t]->getval()), __fixed_sigmoid(_mem_logit[79][t]->getval()), __fixed_sigmoid(_mem_logit[80][t]->getval()), __fixed_sigmoid(_mem_logit[81][t]->getval()), __fixed_sigmoid(_mem_logit[82][t]->getval()), __fixed_sigmoid(_mem_logit[83][t]->getval()), __fixed_sigmoid(_mem_logit[84][t]->getval()), __fixed_sigmoid(_mem_logit[85][t]->getval()), __fixed_sigmoid(_mem_logit[86][t]->getval()), __fixed_sigmoid(_mem_logit[87][t]->getval()), __fixed_sigmoid(_mem_logit[88][t]->getval()), __fixed_sigmoid(_mem_logit[89][t]->getval()), __fixed_sigmoid(_mem_logit[90][t]->getval()), __fixed_sigmoid(_mem_logit[91][t]->getval()), __fixed_sigmoid(_mem_logit[92][t]->getval()), __fixed_sigmoid(_mem_logit[93][t]->getval()), __fixed_sigmoid(_mem_logit[94][t]->getval()), __fixed_sigmoid(_mem_logit[95][t]->getval()), __fixed_sigmoid(_mem_logit[96][t]->getval()), __fixed_sigmoid(_mem_logit[97][t]->getval()), __fixed_sigmoid(_mem_logit[98][t]->getval()), __fixed_sigmoid(_mem_logit[99][t]->getval()), __fixed_sigmoid(_mem_logit[100][t]->getval()), __fixed_sigmoid(_mem_logit[101][t]->getval()), __fixed_sigmoid(_mem_logit[102][t]->getval()), __fixed_sigmoid(_mem_logit[103][t]->getval()), __fixed_sigmoid(_mem_logit[104][t]->getval()), __fixed_sigmoid(_mem_logit[105][t]->getval()), __fixed_sigmoid(_mem_logit[106][t]->getval()), __fixed_sigmoid(_mem_logit[107][t]->getval()), __fixed_sigmoid(_mem_logit[108][t]->getval()), __fixed_sigmoid(_mem_logit[109][t]->getval()), __fixed_sigmoid(_mem_logit[110][t]->getval()), __fixed_sigmoid(_mem_logit[111][t]->getval()), __fixed_sigmoid(_mem_logit[112][t]->getval()), __fixed_sigmoid(_mem_logit[113][t]->getval()), __fixed_sigmoid(_mem_logit[114][t]->getval()), __fixed_sigmoid(_mem_logit[115][t]->getval()), __fixed_sigmoid(_mem_logit[116][t]->getval()), __fixed_sigmoid(_mem_logit[117][t]->getval()), __fixed_sigmoid(_mem_logit[118][t]->getval()), __fixed_sigmoid(_mem_logit[119][t]->getval()), __fixed_sigmoid(_mem_logit[120][t]->getval()), __fixed_sigmoid(_mem_logit[121][t]->getval()), __fixed_sigmoid(_mem_logit[122][t]->getval()), __fixed_sigmoid(_mem_logit[123][t]->getval()), __fixed_sigmoid(_mem_logit[124][t]->getval()), __fixed_sigmoid(_mem_logit[125][t]->getval()), __fixed_sigmoid(_mem_logit[126][t]->getval()), __fixed_sigmoid(_mem_logit[127][t]->getval()), __fixed_sigmoid(_mem_logit[128][t]->getval()), __fixed_sigmoid(_mem_logit[129][t]->getval()), __fixed_sigmoid(_mem_logit[130][t]->getval()), __fixed_sigmoid(_mem_logit[131][t]->getval()), __fixed_sigmoid(_mem_logit[132][t]->getval()), __fixed_sigmoid(_mem_logit[133][t]->getval()), __fixed_sigmoid(_mem_logit[134][t]->getval()), __fixed_sigmoid(_mem_logit[135][t]->getval()), __fixed_sigmoid(_mem_logit[136][t]->getval()), __fixed_sigmoid(_mem_logit[137][t]->getval()), __fixed_sigmoid(_mem_logit[138][t]->getval()), __fixed_sigmoid(_mem_logit[139][t]->getval()), __fixed_sigmoid(_mem_logit[140][t]->getval()), __fixed_sigmoid(_mem_logit[141][t]->getval()), __fixed_sigmoid(_mem_logit[142][t]->getval()), __fixed_sigmoid(_mem_logit[143][t]->getval()), __fixed_sigmoid(_mem_logit[144][t]->getval()), __fixed_sigmoid(_mem_logit[145][t]->getval()), __fixed_sigmoid(_mem_logit[146][t]->getval()), __fixed_sigmoid(_mem_logit[147][t]->getval()), __fixed_sigmoid(_mem_logit[148][t]->getval()), __fixed_sigmoid(_mem_logit[149][t]->getval()), __fixed_sigmoid(_mem_logit[150][t]->getval()), __fixed_sigmoid(_mem_logit[151][t]->getval()), __fixed_sigmoid(_mem_logit[152][t]->getval()), __fixed_sigmoid(_mem_logit[153][t]->getval()), __fixed_sigmoid(_mem_logit[154][t]->getval()), __fixed_sigmoid(_mem_logit[155][t]->getval()), __fixed_sigmoid(_mem_logit[156][t]->getval()), __fixed_sigmoid(_mem_logit[157][t]->getval()), __fixed_sigmoid(_mem_logit[158][t]->getval()), __fixed_sigmoid(_mem_logit[159][t]->getval()), __fixed_sigmoid(_mem_logit[160][t]->getval()), __fixed_sigmoid(_mem_logit[161][t]->getval()), __fixed_sigmoid(_mem_logit[162][t]->getval()), __fixed_sigmoid(_mem_logit[163][t]->getval()), __fixed_sigmoid(_mem_logit[164][t]->getval()), __fixed_sigmoid(_mem_logit[165][t]->getval()), __fixed_sigmoid(_mem_logit[166][t]->getval()), __fixed_sigmoid(_mem_logit[167][t]->getval()), __fixed_sigmoid(_mem_logit[168][t]->getval()), __fixed_sigmoid(_mem_logit[169][t]->getval()), __fixed_sigmoid(_mem_logit[170][t]->getval()), __fixed_sigmoid(_mem_logit[171][t]->getval()), __fixed_sigmoid(_mem_logit[172][t]->getval()), __fixed_sigmoid(_mem_logit[173][t]->getval()), __fixed_sigmoid(_mem_logit[174][t]->getval()), __fixed_sigmoid(_mem_logit[175][t]->getval()), __fixed_sigmoid(_mem_logit[176][t]->getval()), __fixed_sigmoid(_mem_logit[177][t]->getval()), __fixed_sigmoid(_mem_logit[178][t]->getval()), __fixed_sigmoid(_mem_logit[179][t]->getval()), __fixed_sigmoid(_mem_logit[180][t]->getval()), __fixed_sigmoid(_mem_logit[181][t]->getval()), __fixed_sigmoid(_mem_logit[182][t]->getval()), __fixed_sigmoid(_mem_logit[183][t]->getval()), __fixed_sigmoid(_mem_logit[184][t]->getval()), __fixed_sigmoid(_mem_logit[185][t]->getval()), __fixed_sigmoid(_mem_logit[186][t]->getval()), __fixed_sigmoid(_mem_logit[187][t]->getval()), __fixed_sigmoid(_mem_logit[188][t]->getval()), __fixed_sigmoid(_mem_logit[189][t]->getval()), __fixed_sigmoid(_mem_logit[190][t]->getval()), __fixed_sigmoid(_mem_logit[191][t]->getval()), __fixed_sigmoid(_mem_logit[192][t]->getval()), __fixed_sigmoid(_mem_logit[193][t]->getval()), __fixed_sigmoid(_mem_logit[194][t]->getval()), __fixed_sigmoid(_mem_logit[195][t]->getval()), __fixed_sigmoid(_mem_logit[196][t]->getval()), __fixed_sigmoid(_mem_logit[197][t]->getval()), __fixed_sigmoid(_mem_logit[198][t]->getval()), __fixed_sigmoid(_mem_logit[199][t]->getval()), __fixed_sigmoid(_mem_logit[200][t]->getval()), __fixed_sigmoid(_mem_logit[201][t]->getval()), __fixed_sigmoid(_mem_logit[202][t]->getval()), __fixed_sigmoid(_mem_logit[203][t]->getval()), __fixed_sigmoid(_mem_logit[204][t]->getval()), __fixed_sigmoid(_mem_logit[205][t]->getval()), __fixed_sigmoid(_mem_logit[206][t]->getval()), __fixed_sigmoid(_mem_logit[207][t]->getval()), __fixed_sigmoid(_mem_logit[208][t]->getval()), __fixed_sigmoid(_mem_logit[209][t]->getval()), __fixed_sigmoid(_mem_logit[210][t]->getval()), __fixed_sigmoid(_mem_logit[211][t]->getval()), __fixed_sigmoid(_mem_logit[212][t]->getval()), __fixed_sigmoid(_mem_logit[213][t]->getval()), __fixed_sigmoid(_mem_logit[214][t]->getval()), __fixed_sigmoid(_mem_logit[215][t]->getval()), __fixed_sigmoid(_mem_logit[216][t]->getval()), __fixed_sigmoid(_mem_logit[217][t]->getval()), __fixed_sigmoid(_mem_logit[218][t]->getval()), __fixed_sigmoid(_mem_logit[219][t]->getval()), __fixed_sigmoid(_mem_logit[220][t]->getval()), __fixed_sigmoid(_mem_logit[221][t]->getval()), __fixed_sigmoid(_mem_logit[222][t]->getval()), __fixed_sigmoid(_mem_logit[223][t]->getval()), __fixed_sigmoid(_mem_logit[224][t]->getval()), __fixed_sigmoid(_mem_logit[225][t]->getval()), __fixed_sigmoid(_mem_logit[226][t]->getval()), __fixed_sigmoid(_mem_logit[227][t]->getval()), __fixed_sigmoid(_mem_logit[228][t]->getval()), __fixed_sigmoid(_mem_logit[229][t]->getval()), __fixed_sigmoid(_mem_logit[230][t]->getval()), __fixed_sigmoid(_mem_logit[231][t]->getval()), __fixed_sigmoid(_mem_logit[232][t]->getval()), __fixed_sigmoid(_mem_logit[233][t]->getval()), __fixed_sigmoid(_mem_logit[234][t]->getval()), __fixed_sigmoid(_mem_logit[235][t]->getval()), __fixed_sigmoid(_mem_logit[236][t]->getval()), __fixed_sigmoid(_mem_logit[237][t]->getval()), __fixed_sigmoid(_mem_logit[238][t]->getval()), __fixed_sigmoid(_mem_logit[239][t]->getval()), __fixed_sigmoid(_mem_logit[240][t]->getval()), __fixed_sigmoid(_mem_logit[241][t]->getval()), __fixed_sigmoid(_mem_logit[242][t]->getval()), __fixed_sigmoid(_mem_logit[243][t]->getval()), __fixed_sigmoid(_mem_logit[244][t]->getval()), __fixed_sigmoid(_mem_logit[245][t]->getval()), __fixed_sigmoid(_mem_logit[246][t]->getval()), __fixed_sigmoid(_mem_logit[247][t]->getval()), __fixed_sigmoid(_mem_logit[248][t]->getval()), __fixed_sigmoid(_mem_logit[249][t]->getval()), __fixed_sigmoid(_mem_logit[250][t]->getval()), __fixed_sigmoid(_mem_logit[251][t]->getval()), __fixed_sigmoid(_mem_logit[252][t]->getval()), __fixed_sigmoid(_mem_logit[253][t]->getval()), __fixed_sigmoid(_mem_logit[254][t]->getval()), __fixed_sigmoid(_mem_logit[255][t]->getval()), __fixed_sigmoid(_mem_logit[256][t]->getval()), __fixed_sigmoid(_mem_logit[257][t]->getval()), __fixed_sigmoid(_mem_logit[258][t]->getval()), __fixed_sigmoid(_mem_logit[259][t]->getval()), __fixed_sigmoid(_mem_logit[260][t]->getval()), __fixed_sigmoid(_mem_logit[261][t]->getval()), __fixed_sigmoid(_mem_logit[262][t]->getval()), __fixed_sigmoid(_mem_logit[263][t]->getval()), __fixed_sigmoid(_mem_logit[264][t]->getval()), __fixed_sigmoid(_mem_logit[265][t]->getval()), __fixed_sigmoid(_mem_logit[266][t]->getval()), __fixed_sigmoid(_mem_logit[267][t]->getval()), __fixed_sigmoid(_mem_logit[268][t]->getval()), __fixed_sigmoid(_mem_logit[269][t]->getval()), __fixed_sigmoid(_mem_logit[270][t]->getval()), __fixed_sigmoid(_mem_logit[271][t]->getval()), __fixed_sigmoid(_mem_logit[272][t]->getval()), __fixed_sigmoid(_mem_logit[273][t]->getval()), __fixed_sigmoid(_mem_logit[274][t]->getval()), __fixed_sigmoid(_mem_logit[275][t]->getval()), __fixed_sigmoid(_mem_logit[276][t]->getval())}))/__fixed_region_pop[r],0.10000000),Gaussian28973584.loglikeli(val);
}
double _Var_region_rate::getcachelikeli()
{
  auto _t_val = getcache();
  return Gaussian28973584.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getcache()), __fixed_sigmoid(_mem_logit[1][t]->getcache()), __fixed_sigmoid(_mem_logit[2][t]->getcache()), __fixed_sigmoid(_mem_logit[3][t]->getcache()), __fixed_sigmoid(_mem_logit[4][t]->getcache()), __fixed_sigmoid(_mem_logit[5][t]->getcache()), __fixed_sigmoid(_mem_logit[6][t]->getcache()), __fixed_sigmoid(_mem_logit[7][t]->getcache()), __fixed_sigmoid(_mem_logit[8][t]->getcache()), __fixed_sigmoid(_mem_logit[9][t]->getcache()), __fixed_sigmoid(_mem_logit[10][t]->getcache()), __fixed_sigmoid(_mem_logit[11][t]->getcache()), __fixed_sigmoid(_mem_logit[12][t]->getcache()), __fixed_sigmoid(_mem_logit[13][t]->getcache()), __fixed_sigmoid(_mem_logit[14][t]->getcache()), __fixed_sigmoid(_mem_logit[15][t]->getcache()), __fixed_sigmoid(_mem_logit[16][t]->getcache()), __fixed_sigmoid(_mem_logit[17][t]->getcache()), __fixed_sigmoid(_mem_logit[18][t]->getcache()), __fixed_sigmoid(_mem_logit[19][t]->getcache()), __fixed_sigmoid(_mem_logit[20][t]->getcache()), __fixed_sigmoid(_mem_logit[21][t]->getcache()), __fixed_sigmoid(_mem_logit[22][t]->getcache()), __fixed_sigmoid(_mem_logit[23][t]->getcache()), __fixed_sigmoid(_mem_logit[24][t]->getcache()), __fixed_sigmoid(_mem_logit[25][t]->getcache()), __fixed_sigmoid(_mem_logit[26][t]->getcache()), __fixed_sigmoid(_mem_logit[27][t]->getcache()), __fixed_sigmoid(_mem_logit[28][t]->getcache()), __fixed_sigmoid(_mem_logit[29][t]->getcache()), __fixed_sigmoid(_mem_logit[30][t]->getcache()), __fixed_sigmoid(_mem_logit[31][t]->getcache()), __fixed_sigmoid(_mem_logit[32][t]->getcache()), __fixed_sigmoid(_mem_logit[33][t]->getcache()), __fixed_sigmoid(_mem_logit[34][t]->getcache()), __fixed_sigmoid(_mem_logit[35][t]->getcache()), __fixed_sigmoid(_mem_logit[36][t]->getcache()), __fixed_sigmoid(_mem_logit[37][t]->getcache()), __fixed_sigmoid(_mem_logit[38][t]->getcache()), __fixed_sigmoid(_mem_logit[39][t]->getcache()), __fixed_sigmoid(_mem_logit[40][t]->getcache()), __fixed_sigmoid(_mem_logit[41][t]->getcache()), __fixed_sigmoid(_mem_logit[42][t]->getcache()), __fixed_sigmoid(_mem_logit[43][t]->getcache()), __fixed_sigmoid(_mem_logit[44][t]->getcache()), __fixed_sigmoid(_mem_logit[45][t]->getcache()), __fixed_sigmoid(_mem_logit[46][t]->getcache()), __fixed_sigmoid(_mem_logit[47][t]->getcache()), __fixed_sigmoid(_mem_logit[48][t]->getcache()), __fixed_sigmoid(_mem_logit[49][t]->getcache()), __fixed_sigmoid(_mem_logit[50][t]->getcache()), __fixed_sigmoid(_mem_logit[51][t]->getcache()), __fixed_sigmoid(_mem_logit[52][t]->getcache()), __fixed_sigmoid(_mem_logit[53][t]->getcache()), __fixed_sigmoid(_mem_logit[54][t]->getcache()), __fixed_sigmoid(_mem_logit[55][t]->getcache()), __fixed_sigmoid(_mem_logit[56][t]->getcache()), __fixed_sigmoid(_mem_logit[57][t]->getcache()), __fixed_sigmoid(_mem_logit[58][t]->getcache()), __fixed_sigmoid(_mem_logit[59][t]->getcache()), __fixed_sigmoid(_mem_logit[60][t]->getcache()), __fixed_sigmoid(_mem_logit[61][t]->getcache()), __fixed_sigmoid(_mem_logit[62][t]->getcache()), __fixed_sigmoid(_mem_logit[63][t]->getcache()), __fixed_sigmoid(_mem_logit[64][t]->getcache()), __fixed_sigmoid(_mem_logit[65][t]->getcache()), __fixed_sigmoid(_mem_logit[66][t]->getcache()), __fixed_sigmoid(_mem_logit[67][t]->getcache()), __fixed_sigmoid(_mem_logit[68][t]->getcache()), __fixed_sigmoid(_mem_logit[69][t]->getcache()), __fixed_sigmoid(_mem_logit[70][t]->getcache()), __fixed_sigmoid(_mem_logit[71][t]->getcache()), __fixed_sigmoid(_mem_logit[72][t]->getcache()), __fixed_sigmoid(_mem_logit[73][t]->getcache()), __fixed_sigmoid(_mem_logit[74][t]->getcache()), __fixed_sigmoid(_mem_logit[75][t]->getcache()), __fixed_sigmoid(_mem_logit[76][t]->getcache()), __fixed_sigmoid(_mem_logit[77][t]->getcache()), __fixed_sigmoid(_mem_logit[78][t]->getcache()), __fixed_sigmoid(_mem_logit[79][t]->getcache()), __fixed_sigmoid(_mem_logit[80][t]->getcache()), __fixed_sigmoid(_mem_logit[81][t]->getcache()), __fixed_sigmoid(_mem_logit[82][t]->getcache()), __fixed_sigmoid(_mem_logit[83][t]->getcache()), __fixed_sigmoid(_mem_logit[84][t]->getcache()), __fixed_sigmoid(_mem_logit[85][t]->getcache()), __fixed_sigmoid(_mem_logit[86][t]->getcache()), __fixed_sigmoid(_mem_logit[87][t]->getcache()), __fixed_sigmoid(_mem_logit[88][t]->getcache()), __fixed_sigmoid(_mem_logit[89][t]->getcache()), __fixed_sigmoid(_mem_logit[90][t]->getcache()), __fixed_sigmoid(_mem_logit[91][t]->getcache()), __fixed_sigmoid(_mem_logit[92][t]->getcache()), __fixed_sigmoid(_mem_logit[93][t]->getcache()), __fixed_sigmoid(_mem_logit[94][t]->getcache()), __fixed_sigmoid(_mem_logit[95][t]->getcache()), __fixed_sigmoid(_mem_logit[96][t]->getcache()), __fixed_sigmoid(_mem_logit[97][t]->getcache()), __fixed_sigmoid(_mem_logit[98][t]->getcache()), __fixed_sigmoid(_mem_logit[99][t]->getcache()), __fixed_sigmoid(_mem_logit[100][t]->getcache()), __fixed_sigmoid(_mem_logit[101][t]->getcache()), __fixed_sigmoid(_mem_logit[102][t]->getcache()), __fixed_sigmoid(_mem_logit[103][t]->getcache()), __fixed_sigmoid(_mem_logit[104][t]->getcache()), __fixed_sigmoid(_mem_logit[105][t]->getcache()), __fixed_sigmoid(_mem_logit[106][t]->getcache()), __fixed_sigmoid(_mem_logit[107][t]->getcache()), __fixed_sigmoid(_mem_logit[108][t]->getcache()), __fixed_sigmoid(_mem_logit[109][t]->getcache()), __fixed_sigmoid(_mem_logit[110][t]->getcache()), __fixed_sigmoid(_mem_logit[111][t]->getcache()), __fixed_sigmoid(_mem_logit[112][t]->getcache()), __fixed_sigmoid(_mem_logit[113][t]->getcache()), __fixed_sigmoid(_mem_logit[114][t]->getcache()), __fixed_sigmoid(_mem_logit[115][t]->getcache()), __fixed_sigmoid(_mem_logit[116][t]->getcache()), __fixed_sigmoid(_mem_logit[117][t]->getcache()), __fixed_sigmoid(_mem_logit[118][t]->getcache()), __fixed_sigmoid(_mem_logit[119][t]->getcache()), __fixed_sigmoid(_mem_logit[120][t]->getcache()), __fixed_sigmoid(_mem_logit[121][t]->getcache()), __fixed_sigmoid(_mem_logit[122][t]->getcache()), __fixed_sigmoid(_mem_logit[123][t]->getcache()), __fixed_sigmoid(_mem_logit[124][t]->getcache()), __fixed_sigmoid(_mem_logit[125][t]->getcache()), __fixed_sigmoid(_mem_logit[126][t]->getcache()), __fixed_sigmoid(_mem_logit[127][t]->getcache()), __fixed_sigmoid(_mem_logit[128][t]->getcache()), __fixed_sigmoid(_mem_logit[129][t]->getcache()), __fixed_sigmoid(_mem_logit[130][t]->getcache()), __fixed_sigmoid(_mem_logit[131][t]->getcache()), __fixed_sigmoid(_mem_logit[132][t]->getcache()), __fixed_sigmoid(_mem_logit[133][t]->getcache()), __fixed_sigmoid(_mem_logit[134][t]->getcache()), __fixed_sigmoid(_mem_logit[135][t]->getcache()), __fixed_sigmoid(_mem_logit[136][t]->getcache()), __fixed_sigmoid(_mem_logit[137][t]->getcache()), __fixed_sigmoid(_mem_logit[138][t]->getcache()), __fixed_sigmoid(_mem_logit[139][t]->getcache()), __fixed_sigmoid(_mem_logit[140][t]->getcache()), __fixed_sigmoid(_mem_logit[141][t]->getcache()), __fixed_sigmoid(_mem_logit[142][t]->getcache()), __fixed_sigmoid(_mem_logit[143][t]->getcache()), __fixed_sigmoid(_mem_logit[144][t]->getcache()), __fixed_sigmoid(_mem_logit[145][t]->getcache()), __fixed_sigmoid(_mem_logit[146][t]->getcache()), __fixed_sigmoid(_mem_logit[147][t]->getcache()), __fixed_sigmoid(_mem_logit[148][t]->getcache()), __fixed_sigmoid(_mem_logit[149][t]->getcache()), __fixed_sigmoid(_mem_logit[150][t]->getcache()), __fixed_sigmoid(_mem_logit[151][t]->getcache()), __fixed_sigmoid(_mem_logit[152][t]->getcache()), __fixed_sigmoid(_mem_logit[153][t]->getcache()), __fixed_sigmoid(_mem_logit[154][t]->getcache()), __fixed_sigmoid(_mem_logit[155][t]->getcache()), __fixed_sigmoid(_mem_logit[156][t]->getcache()), __fixed_sigmoid(_mem_logit[157][t]->getcache()), __fixed_sigmoid(_mem_logit[158][t]->getcache()), __fixed_sigmoid(_mem_logit[159][t]->getcache()), __fixed_sigmoid(_mem_logit[160][t]->getcache()), __fixed_sigmoid(_mem_logit[161][t]->getcache()), __fixed_sigmoid(_mem_logit[162][t]->getcache()), __fixed_sigmoid(_mem_logit[163][t]->getcache()), __fixed_sigmoid(_mem_logit[164][t]->getcache()), __fixed_sigmoid(_mem_logit[165][t]->getcache()), __fixed_sigmoid(_mem_logit[166][t]->getcache()), __fixed_sigmoid(_mem_logit[167][t]->getcache()), __fixed_sigmoid(_mem_logit[168][t]->getcache()), __fixed_sigmoid(_mem_logit[169][t]->getcache()), __fixed_sigmoid(_mem_logit[170][t]->getcache()), __fixed_sigmoid(_mem_logit[171][t]->getcache()), __fixed_sigmoid(_mem_logit[172][t]->getcache()), __fixed_sigmoid(_mem_logit[173][t]->getcache()), __fixed_sigmoid(_mem_logit[174][t]->getcache()), __fixed_sigmoid(_mem_logit[175][t]->getcache()), __fixed_sigmoid(_mem_logit[176][t]->getcache()), __fixed_sigmoid(_mem_logit[177][t]->getcache()), __fixed_sigmoid(_mem_logit[178][t]->getcache()), __fixed_sigmoid(_mem_logit[179][t]->getcache()), __fixed_sigmoid(_mem_logit[180][t]->getcache()), __fixed_sigmoid(_mem_logit[181][t]->getcache()), __fixed_sigmoid(_mem_logit[182][t]->getcache()), __fixed_sigmoid(_mem_logit[183][t]->getcache()), __fixed_sigmoid(_mem_logit[184][t]->getcache()), __fixed_sigmoid(_mem_logit[185][t]->getcache()), __fixed_sigmoid(_mem_logit[186][t]->getcache()), __fixed_sigmoid(_mem_logit[187][t]->getcache()), __fixed_sigmoid(_mem_logit[188][t]->getcache()), __fixed_sigmoid(_mem_logit[189][t]->getcache()), __fixed_sigmoid(_mem_logit[190][t]->getcache()), __fixed_sigmoid(_mem_logit[191][t]->getcache()), __fixed_sigmoid(_mem_logit[192][t]->getcache()), __fixed_sigmoid(_mem_logit[193][t]->getcache()), __fixed_sigmoid(_mem_logit[194][t]->getcache()), __fixed_sigmoid(_mem_logit[195][t]->getcache()), __fixed_sigmoid(_mem_logit[196][t]->getcache()), __fixed_sigmoid(_mem_logit[197][t]->getcache()), __fixed_sigmoid(_mem_logit[198][t]->getcache()), __fixed_sigmoid(_mem_logit[199][t]->getcache()), __fixed_sigmoid(_mem_logit[200][t]->getcache()), __fixed_sigmoid(_mem_logit[201][t]->getcache()), __fixed_sigmoid(_mem_logit[202][t]->getcache()), __fixed_sigmoid(_mem_logit[203][t]->getcache()), __fixed_sigmoid(_mem_logit[204][t]->getcache()), __fixed_sigmoid(_mem_logit[205][t]->getcache()), __fixed_sigmoid(_mem_logit[206][t]->getcache()), __fixed_sigmoid(_mem_logit[207][t]->getcache()), __fixed_sigmoid(_mem_logit[208][t]->getcache()), __fixed_sigmoid(_mem_logit[209][t]->getcache()), __fixed_sigmoid(_mem_logit[210][t]->getcache()), __fixed_sigmoid(_mem_logit[211][t]->getcache()), __fixed_sigmoid(_mem_logit[212][t]->getcache()), __fixed_sigmoid(_mem_logit[213][t]->getcache()), __fixed_sigmoid(_mem_logit[214][t]->getcache()), __fixed_sigmoid(_mem_logit[215][t]->getcache()), __fixed_sigmoid(_mem_logit[216][t]->getcache()), __fixed_sigmoid(_mem_logit[217][t]->getcache()), __fixed_sigmoid(_mem_logit[218][t]->getcache()), __fixed_sigmoid(_mem_logit[219][t]->getcache()), __fixed_sigmoid(_mem_logit[220][t]->getcache()), __fixed_sigmoid(_mem_logit[221][t]->getcache()), __fixed_sigmoid(_mem_logit[222][t]->getcache()), __fixed_sigmoid(_mem_logit[223][t]->getcache()), __fixed_sigmoid(_mem_logit[224][t]->getcache()), __fixed_sigmoid(_mem_logit[225][t]->getcache()), __fixed_sigmoid(_mem_logit[226][t]->getcache()), __fixed_sigmoid(_mem_logit[227][t]->getcache()), __fixed_sigmoid(_mem_logit[228][t]->getcache()), __fixed_sigmoid(_mem_logit[229][t]->getcache()), __fixed_sigmoid(_mem_logit[230][t]->getcache()), __fixed_sigmoid(_mem_logit[231][t]->getcache()), __fixed_sigmoid(_mem_logit[232][t]->getcache()), __fixed_sigmoid(_mem_logit[233][t]->getcache()), __fixed_sigmoid(_mem_logit[234][t]->getcache()), __fixed_sigmoid(_mem_logit[235][t]->getcache()), __fixed_sigmoid(_mem_logit[236][t]->getcache()), __fixed_sigmoid(_mem_logit[237][t]->getcache()), __fixed_sigmoid(_mem_logit[238][t]->getcache()), __fixed_sigmoid(_mem_logit[239][t]->getcache()), __fixed_sigmoid(_mem_logit[240][t]->getcache()), __fixed_sigmoid(_mem_logit[241][t]->getcache()), __fixed_sigmoid(_mem_logit[242][t]->getcache()), __fixed_sigmoid(_mem_logit[243][t]->getcache()), __fixed_sigmoid(_mem_logit[244][t]->getcache()), __fixed_sigmoid(_mem_logit[245][t]->getcache()), __fixed_sigmoid(_mem_logit[246][t]->getcache()), __fixed_sigmoid(_mem_logit[247][t]->getcache()), __fixed_sigmoid(_mem_logit[248][t]->getcache()), __fixed_sigmoid(_mem_logit[249][t]->getcache()), __fixed_sigmoid(_mem_logit[250][t]->getcache()), __fixed_sigmoid(_mem_logit[251][t]->getcache()), __fixed_sigmoid(_mem_logit[252][t]->getcache()), __fixed_sigmoid(_mem_logit[253][t]->getcache()), __fixed_sigmoid(_mem_logit[254][t]->getcache()), __fixed_sigmoid(_mem_logit[255][t]->getcache()), __fixed_sigmoid(_mem_logit[256][t]->getcache()), __fixed_sigmoid(_mem_logit[257][t]->getcache()), __fixed_sigmoid(_mem_logit[258][t]->getcache()), __fixed_sigmoid(_mem_logit[259][t]->getcache()), __fixed_sigmoid(_mem_logit[260][t]->getcache()), __fixed_sigmoid(_mem_logit[261][t]->getcache()), __fixed_sigmoid(_mem_logit[262][t]->getcache()), __fixed_sigmoid(_mem_logit[263][t]->getcache()), __fixed_sigmoid(_mem_logit[264][t]->getcache()), __fixed_sigmoid(_mem_logit[265][t]->getcache()), __fixed_sigmoid(_mem_logit[266][t]->getcache()), __fixed_sigmoid(_mem_logit[267][t]->getcache()), __fixed_sigmoid(_mem_logit[268][t]->getcache()), __fixed_sigmoid(_mem_logit[269][t]->getcache()), __fixed_sigmoid(_mem_logit[270][t]->getcache()), __fixed_sigmoid(_mem_logit[271][t]->getcache()), __fixed_sigmoid(_mem_logit[272][t]->getcache()), __fixed_sigmoid(_mem_logit[273][t]->getcache()), __fixed_sigmoid(_mem_logit[274][t]->getcache()), __fixed_sigmoid(_mem_logit[275][t]->getcache()), __fixed_sigmoid(_mem_logit[276][t]->getcache())}))/__fixed_region_pop[r],0.10000000),Gaussian28973584.loglikeli(_t_val);
}
void _Var_region_rate::sample()
{
  val=(Gaussian28973584.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getval()), __fixed_sigmoid(_mem_logit[1][t]->getval()), __fixed_sigmoid(_mem_logit[2][t]->getval()), __fixed_sigmoid(_mem_logit[3][t]->getval()), __fixed_sigmoid(_mem_logit[4][t]->getval()), __fixed_sigmoid(_mem_logit[5][t]->getval()), __fixed_sigmoid(_mem_logit[6][t]->getval()), __fixed_sigmoid(_mem_logit[7][t]->getval()), __fixed_sigmoid(_mem_logit[8][t]->getval()), __fixed_sigmoid(_mem_logit[9][t]->getval()), __fixed_sigmoid(_mem_logit[10][t]->getval()), __fixed_sigmoid(_mem_logit[11][t]->getval()), __fixed_sigmoid(_mem_logit[12][t]->getval()), __fixed_sigmoid(_mem_logit[13][t]->getval()), __fixed_sigmoid(_mem_logit[14][t]->getval()), __fixed_sigmoid(_mem_logit[15][t]->getval()), __fixed_sigmoid(_mem_logit[16][t]->getval()), __fixed_sigmoid(_mem_logit[17][t]->getval()), __fixed_sigmoid(_mem_logit[18][t]->getval()), __fixed_sigmoid(_mem_logit[19][t]->getval()), __fixed_sigmoid(_mem_logit[20][t]->getval()), __fixed_sigmoid(_mem_logit[21][t]->getval()), __fixed_sigmoid(_mem_logit[22][t]->getval()), __fixed_sigmoid(_mem_logit[23][t]->getval()), __fixed_sigmoid(_mem_logit[24][t]->getval()), __fixed_sigmoid(_mem_logit[25][t]->getval()), __fixed_sigmoid(_mem_logit[26][t]->getval()), __fixed_sigmoid(_mem_logit[27][t]->getval()), __fixed_sigmoid(_mem_logit[28][t]->getval()), __fixed_sigmoid(_mem_logit[29][t]->getval()), __fixed_sigmoid(_mem_logit[30][t]->getval()), __fixed_sigmoid(_mem_logit[31][t]->getval()), __fixed_sigmoid(_mem_logit[32][t]->getval()), __fixed_sigmoid(_mem_logit[33][t]->getval()), __fixed_sigmoid(_mem_logit[34][t]->getval()), __fixed_sigmoid(_mem_logit[35][t]->getval()), __fixed_sigmoid(_mem_logit[36][t]->getval()), __fixed_sigmoid(_mem_logit[37][t]->getval()), __fixed_sigmoid(_mem_logit[38][t]->getval()), __fixed_sigmoid(_mem_logit[39][t]->getval()), __fixed_sigmoid(_mem_logit[40][t]->getval()), __fixed_sigmoid(_mem_logit[41][t]->getval()), __fixed_sigmoid(_mem_logit[42][t]->getval()), __fixed_sigmoid(_mem_logit[43][t]->getval()), __fixed_sigmoid(_mem_logit[44][t]->getval()), __fixed_sigmoid(_mem_logit[45][t]->getval()), __fixed_sigmoid(_mem_logit[46][t]->getval()), __fixed_sigmoid(_mem_logit[47][t]->getval()), __fixed_sigmoid(_mem_logit[48][t]->getval()), __fixed_sigmoid(_mem_logit[49][t]->getval()), __fixed_sigmoid(_mem_logit[50][t]->getval()), __fixed_sigmoid(_mem_logit[51][t]->getval()), __fixed_sigmoid(_mem_logit[52][t]->getval()), __fixed_sigmoid(_mem_logit[53][t]->getval()), __fixed_sigmoid(_mem_logit[54][t]->getval()), __fixed_sigmoid(_mem_logit[55][t]->getval()), __fixed_sigmoid(_mem_logit[56][t]->getval()), __fixed_sigmoid(_mem_logit[57][t]->getval()), __fixed_sigmoid(_mem_logit[58][t]->getval()), __fixed_sigmoid(_mem_logit[59][t]->getval()), __fixed_sigmoid(_mem_logit[60][t]->getval()), __fixed_sigmoid(_mem_logit[61][t]->getval()), __fixed_sigmoid(_mem_logit[62][t]->getval()), __fixed_sigmoid(_mem_logit[63][t]->getval()), __fixed_sigmoid(_mem_logit[64][t]->getval()), __fixed_sigmoid(_mem_logit[65][t]->getval()), __fixed_sigmoid(_mem_logit[66][t]->getval()), __fixed_sigmoid(_mem_logit[67][t]->getval()), __fixed_sigmoid(_mem_logit[68][t]->getval()), __fixed_sigmoid(_mem_logit[69][t]->getval()), __fixed_sigmoid(_mem_logit[70][t]->getval()), __fixed_sigmoid(_mem_logit[71][t]->getval()), __fixed_sigmoid(_mem_logit[72][t]->getval()), __fixed_sigmoid(_mem_logit[73][t]->getval()), __fixed_sigmoid(_mem_logit[74][t]->getval()), __fixed_sigmoid(_mem_logit[75][t]->getval()), __fixed_sigmoid(_mem_logit[76][t]->getval()), __fixed_sigmoid(_mem_logit[77][t]->getval()), __fixed_sigmoid(_mem_logit[78][t]->getval()), __fixed_sigmoid(_mem_logit[79][t]->getval()), __fixed_sigmoid(_mem_logit[80][t]->getval()), __fixed_sigmoid(_mem_logit[81][t]->getval()), __fixed_sigmoid(_mem_logit[82][t]->getval()), __fixed_sigmoid(_mem_logit[83][t]->getval()), __fixed_sigmoid(_mem_logit[84][t]->getval()), __fixed_sigmoid(_mem_logit[85][t]->getval()), __fixed_sigmoid(_mem_logit[86][t]->getval()), __fixed_sigmoid(_mem_logit[87][t]->getval()), __fixed_sigmoid(_mem_logit[88][t]->getval()), __fixed_sigmoid(_mem_logit[89][t]->getval()), __fixed_sigmoid(_mem_logit[90][t]->getval()), __fixed_sigmoid(_mem_logit[91][t]->getval()), __fixed_sigmoid(_mem_logit[92][t]->getval()), __fixed_sigmoid(_mem_logit[93][t]->getval()), __fixed_sigmoid(_mem_logit[94][t]->getval()), __fixed_sigmoid(_mem_logit[95][t]->getval()), __fixed_sigmoid(_mem_logit[96][t]->getval()), __fixed_sigmoid(_mem_logit[97][t]->getval()), __fixed_sigmoid(_mem_logit[98][t]->getval()), __fixed_sigmoid(_mem_logit[99][t]->getval()), __fixed_sigmoid(_mem_logit[100][t]->getval()), __fixed_sigmoid(_mem_logit[101][t]->getval()), __fixed_sigmoid(_mem_logit[102][t]->getval()), __fixed_sigmoid(_mem_logit[103][t]->getval()), __fixed_sigmoid(_mem_logit[104][t]->getval()), __fixed_sigmoid(_mem_logit[105][t]->getval()), __fixed_sigmoid(_mem_logit[106][t]->getval()), __fixed_sigmoid(_mem_logit[107][t]->getval()), __fixed_sigmoid(_mem_logit[108][t]->getval()), __fixed_sigmoid(_mem_logit[109][t]->getval()), __fixed_sigmoid(_mem_logit[110][t]->getval()), __fixed_sigmoid(_mem_logit[111][t]->getval()), __fixed_sigmoid(_mem_logit[112][t]->getval()), __fixed_sigmoid(_mem_logit[113][t]->getval()), __fixed_sigmoid(_mem_logit[114][t]->getval()), __fixed_sigmoid(_mem_logit[115][t]->getval()), __fixed_sigmoid(_mem_logit[116][t]->getval()), __fixed_sigmoid(_mem_logit[117][t]->getval()), __fixed_sigmoid(_mem_logit[118][t]->getval()), __fixed_sigmoid(_mem_logit[119][t]->getval()), __fixed_sigmoid(_mem_logit[120][t]->getval()), __fixed_sigmoid(_mem_logit[121][t]->getval()), __fixed_sigmoid(_mem_logit[122][t]->getval()), __fixed_sigmoid(_mem_logit[123][t]->getval()), __fixed_sigmoid(_mem_logit[124][t]->getval()), __fixed_sigmoid(_mem_logit[125][t]->getval()), __fixed_sigmoid(_mem_logit[126][t]->getval()), __fixed_sigmoid(_mem_logit[127][t]->getval()), __fixed_sigmoid(_mem_logit[128][t]->getval()), __fixed_sigmoid(_mem_logit[129][t]->getval()), __fixed_sigmoid(_mem_logit[130][t]->getval()), __fixed_sigmoid(_mem_logit[131][t]->getval()), __fixed_sigmoid(_mem_logit[132][t]->getval()), __fixed_sigmoid(_mem_logit[133][t]->getval()), __fixed_sigmoid(_mem_logit[134][t]->getval()), __fixed_sigmoid(_mem_logit[135][t]->getval()), __fixed_sigmoid(_mem_logit[136][t]->getval()), __fixed_sigmoid(_mem_logit[137][t]->getval()), __fixed_sigmoid(_mem_logit[138][t]->getval()), __fixed_sigmoid(_mem_logit[139][t]->getval()), __fixed_sigmoid(_mem_logit[140][t]->getval()), __fixed_sigmoid(_mem_logit[141][t]->getval()), __fixed_sigmoid(_mem_logit[142][t]->getval()), __fixed_sigmoid(_mem_logit[143][t]->getval()), __fixed_sigmoid(_mem_logit[144][t]->getval()), __fixed_sigmoid(_mem_logit[145][t]->getval()), __fixed_sigmoid(_mem_logit[146][t]->getval()), __fixed_sigmoid(_mem_logit[147][t]->getval()), __fixed_sigmoid(_mem_logit[148][t]->getval()), __fixed_sigmoid(_mem_logit[149][t]->getval()), __fixed_sigmoid(_mem_logit[150][t]->getval()), __fixed_sigmoid(_mem_logit[151][t]->getval()), __fixed_sigmoid(_mem_logit[152][t]->getval()), __fixed_sigmoid(_mem_logit[153][t]->getval()), __fixed_sigmoid(_mem_logit[154][t]->getval()), __fixed_sigmoid(_mem_logit[155][t]->getval()), __fixed_sigmoid(_mem_logit[156][t]->getval()), __fixed_sigmoid(_mem_logit[157][t]->getval()), __fixed_sigmoid(_mem_logit[158][t]->getval()), __fixed_sigmoid(_mem_logit[159][t]->getval()), __fixed_sigmoid(_mem_logit[160][t]->getval()), __fixed_sigmoid(_mem_logit[161][t]->getval()), __fixed_sigmoid(_mem_logit[162][t]->getval()), __fixed_sigmoid(_mem_logit[163][t]->getval()), __fixed_sigmoid(_mem_logit[164][t]->getval()), __fixed_sigmoid(_mem_logit[165][t]->getval()), __fixed_sigmoid(_mem_logit[166][t]->getval()), __fixed_sigmoid(_mem_logit[167][t]->getval()), __fixed_sigmoid(_mem_logit[168][t]->getval()), __fixed_sigmoid(_mem_logit[169][t]->getval()), __fixed_sigmoid(_mem_logit[170][t]->getval()), __fixed_sigmoid(_mem_logit[171][t]->getval()), __fixed_sigmoid(_mem_logit[172][t]->getval()), __fixed_sigmoid(_mem_logit[173][t]->getval()), __fixed_sigmoid(_mem_logit[174][t]->getval()), __fixed_sigmoid(_mem_logit[175][t]->getval()), __fixed_sigmoid(_mem_logit[176][t]->getval()), __fixed_sigmoid(_mem_logit[177][t]->getval()), __fixed_sigmoid(_mem_logit[178][t]->getval()), __fixed_sigmoid(_mem_logit[179][t]->getval()), __fixed_sigmoid(_mem_logit[180][t]->getval()), __fixed_sigmoid(_mem_logit[181][t]->getval()), __fixed_sigmoid(_mem_logit[182][t]->getval()), __fixed_sigmoid(_mem_logit[183][t]->getval()), __fixed_sigmoid(_mem_logit[184][t]->getval()), __fixed_sigmoid(_mem_logit[185][t]->getval()), __fixed_sigmoid(_mem_logit[186][t]->getval()), __fixed_sigmoid(_mem_logit[187][t]->getval()), __fixed_sigmoid(_mem_logit[188][t]->getval()), __fixed_sigmoid(_mem_logit[189][t]->getval()), __fixed_sigmoid(_mem_logit[190][t]->getval()), __fixed_sigmoid(_mem_logit[191][t]->getval()), __fixed_sigmoid(_mem_logit[192][t]->getval()), __fixed_sigmoid(_mem_logit[193][t]->getval()), __fixed_sigmoid(_mem_logit[194][t]->getval()), __fixed_sigmoid(_mem_logit[195][t]->getval()), __fixed_sigmoid(_mem_logit[196][t]->getval()), __fixed_sigmoid(_mem_logit[197][t]->getval()), __fixed_sigmoid(_mem_logit[198][t]->getval()), __fixed_sigmoid(_mem_logit[199][t]->getval()), __fixed_sigmoid(_mem_logit[200][t]->getval()), __fixed_sigmoid(_mem_logit[201][t]->getval()), __fixed_sigmoid(_mem_logit[202][t]->getval()), __fixed_sigmoid(_mem_logit[203][t]->getval()), __fixed_sigmoid(_mem_logit[204][t]->getval()), __fixed_sigmoid(_mem_logit[205][t]->getval()), __fixed_sigmoid(_mem_logit[206][t]->getval()), __fixed_sigmoid(_mem_logit[207][t]->getval()), __fixed_sigmoid(_mem_logit[208][t]->getval()), __fixed_sigmoid(_mem_logit[209][t]->getval()), __fixed_sigmoid(_mem_logit[210][t]->getval()), __fixed_sigmoid(_mem_logit[211][t]->getval()), __fixed_sigmoid(_mem_logit[212][t]->getval()), __fixed_sigmoid(_mem_logit[213][t]->getval()), __fixed_sigmoid(_mem_logit[214][t]->getval()), __fixed_sigmoid(_mem_logit[215][t]->getval()), __fixed_sigmoid(_mem_logit[216][t]->getval()), __fixed_sigmoid(_mem_logit[217][t]->getval()), __fixed_sigmoid(_mem_logit[218][t]->getval()), __fixed_sigmoid(_mem_logit[219][t]->getval()), __fixed_sigmoid(_mem_logit[220][t]->getval()), __fixed_sigmoid(_mem_logit[221][t]->getval()), __fixed_sigmoid(_mem_logit[222][t]->getval()), __fixed_sigmoid(_mem_logit[223][t]->getval()), __fixed_sigmoid(_mem_logit[224][t]->getval()), __fixed_sigmoid(_mem_logit[225][t]->getval()), __fixed_sigmoid(_mem_logit[226][t]->getval()), __fixed_sigmoid(_mem_logit[227][t]->getval()), __fixed_sigmoid(_mem_logit[228][t]->getval()), __fixed_sigmoid(_mem_logit[229][t]->getval()), __fixed_sigmoid(_mem_logit[230][t]->getval()), __fixed_sigmoid(_mem_logit[231][t]->getval()), __fixed_sigmoid(_mem_logit[232][t]->getval()), __fixed_sigmoid(_mem_logit[233][t]->getval()), __fixed_sigmoid(_mem_logit[234][t]->getval()), __fixed_sigmoid(_mem_logit[235][t]->getval()), __fixed_sigmoid(_mem_logit[236][t]->getval()), __fixed_sigmoid(_mem_logit[237][t]->getval()), __fixed_sigmoid(_mem_logit[238][t]->getval()), __fixed_sigmoid(_mem_logit[239][t]->getval()), __fixed_sigmoid(_mem_logit[240][t]->getval()), __fixed_sigmoid(_mem_logit[241][t]->getval()), __fixed_sigmoid(_mem_logit[242][t]->getval()), __fixed_sigmoid(_mem_logit[243][t]->getval()), __fixed_sigmoid(_mem_logit[244][t]->getval()), __fixed_sigmoid(_mem_logit[245][t]->getval()), __fixed_sigmoid(_mem_logit[246][t]->getval()), __fixed_sigmoid(_mem_logit[247][t]->getval()), __fixed_sigmoid(_mem_logit[248][t]->getval()), __fixed_sigmoid(_mem_logit[249][t]->getval()), __fixed_sigmoid(_mem_logit[250][t]->getval()), __fixed_sigmoid(_mem_logit[251][t]->getval()), __fixed_sigmoid(_mem_logit[252][t]->getval()), __fixed_sigmoid(_mem_logit[253][t]->getval()), __fixed_sigmoid(_mem_logit[254][t]->getval()), __fixed_sigmoid(_mem_logit[255][t]->getval()), __fixed_sigmoid(_mem_logit[256][t]->getval()), __fixed_sigmoid(_mem_logit[257][t]->getval()), __fixed_sigmoid(_mem_logit[258][t]->getval()), __fixed_sigmoid(_mem_logit[259][t]->getval()), __fixed_sigmoid(_mem_logit[260][t]->getval()), __fixed_sigmoid(_mem_logit[261][t]->getval()), __fixed_sigmoid(_mem_logit[262][t]->getval()), __fixed_sigmoid(_mem_logit[263][t]->getval()), __fixed_sigmoid(_mem_logit[264][t]->getval()), __fixed_sigmoid(_mem_logit[265][t]->getval()), __fixed_sigmoid(_mem_logit[266][t]->getval()), __fixed_sigmoid(_mem_logit[267][t]->getval()), __fixed_sigmoid(_mem_logit[268][t]->getval()), __fixed_sigmoid(_mem_logit[269][t]->getval()), __fixed_sigmoid(_mem_logit[270][t]->getval()), __fixed_sigmoid(_mem_logit[271][t]->getval()), __fixed_sigmoid(_mem_logit[272][t]->getval()), __fixed_sigmoid(_mem_logit[273][t]->getval()), __fixed_sigmoid(_mem_logit[274][t]->getval()), __fixed_sigmoid(_mem_logit[275][t]->getval()), __fixed_sigmoid(_mem_logit[276][t]->getval())}))/__fixed_region_pop[r],0.10000000),Gaussian28973584.gen());
}
void _Var_region_rate::sample_cache()
{
  cache_val=(Gaussian28973584.init(dot(__fixed_county_map.row(r),vstack({__fixed_sigmoid(_mem_logit[0][t]->getcache()), __fixed_sigmoid(_mem_logit[1][t]->getcache()), __fixed_sigmoid(_mem_logit[2][t]->getcache()), __fixed_sigmoid(_mem_logit[3][t]->getcache()), __fixed_sigmoid(_mem_logit[4][t]->getcache()), __fixed_sigmoid(_mem_logit[5][t]->getcache()), __fixed_sigmoid(_mem_logit[6][t]->getcache()), __fixed_sigmoid(_mem_logit[7][t]->getcache()), __fixed_sigmoid(_mem_logit[8][t]->getcache()), __fixed_sigmoid(_mem_logit[9][t]->getcache()), __fixed_sigmoid(_mem_logit[10][t]->getcache()), __fixed_sigmoid(_mem_logit[11][t]->getcache()), __fixed_sigmoid(_mem_logit[12][t]->getcache()), __fixed_sigmoid(_mem_logit[13][t]->getcache()), __fixed_sigmoid(_mem_logit[14][t]->getcache()), __fixed_sigmoid(_mem_logit[15][t]->getcache()), __fixed_sigmoid(_mem_logit[16][t]->getcache()), __fixed_sigmoid(_mem_logit[17][t]->getcache()), __fixed_sigmoid(_mem_logit[18][t]->getcache()), __fixed_sigmoid(_mem_logit[19][t]->getcache()), __fixed_sigmoid(_mem_logit[20][t]->getcache()), __fixed_sigmoid(_mem_logit[21][t]->getcache()), __fixed_sigmoid(_mem_logit[22][t]->getcache()), __fixed_sigmoid(_mem_logit[23][t]->getcache()), __fixed_sigmoid(_mem_logit[24][t]->getcache()), __fixed_sigmoid(_mem_logit[25][t]->getcache()), __fixed_sigmoid(_mem_logit[26][t]->getcache()), __fixed_sigmoid(_mem_logit[27][t]->getcache()), __fixed_sigmoid(_mem_logit[28][t]->getcache()), __fixed_sigmoid(_mem_logit[29][t]->getcache()), __fixed_sigmoid(_mem_logit[30][t]->getcache()), __fixed_sigmoid(_mem_logit[31][t]->getcache()), __fixed_sigmoid(_mem_logit[32][t]->getcache()), __fixed_sigmoid(_mem_logit[33][t]->getcache()), __fixed_sigmoid(_mem_logit[34][t]->getcache()), __fixed_sigmoid(_mem_logit[35][t]->getcache()), __fixed_sigmoid(_mem_logit[36][t]->getcache()), __fixed_sigmoid(_mem_logit[37][t]->getcache()), __fixed_sigmoid(_mem_logit[38][t]->getcache()), __fixed_sigmoid(_mem_logit[39][t]->getcache()), __fixed_sigmoid(_mem_logit[40][t]->getcache()), __fixed_sigmoid(_mem_logit[41][t]->getcache()), __fixed_sigmoid(_mem_logit[42][t]->getcache()), __fixed_sigmoid(_mem_logit[43][t]->getcache()), __fixed_sigmoid(_mem_logit[44][t]->getcache()), __fixed_sigmoid(_mem_logit[45][t]->getcache()), __fixed_sigmoid(_mem_logit[46][t]->getcache()), __fixed_sigmoid(_mem_logit[47][t]->getcache()), __fixed_sigmoid(_mem_logit[48][t]->getcache()), __fixed_sigmoid(_mem_logit[49][t]->getcache()), __fixed_sigmoid(_mem_logit[50][t]->getcache()), __fixed_sigmoid(_mem_logit[51][t]->getcache()), __fixed_sigmoid(_mem_logit[52][t]->getcache()), __fixed_sigmoid(_mem_logit[53][t]->getcache()), __fixed_sigmoid(_mem_logit[54][t]->getcache()), __fixed_sigmoid(_mem_logit[55][t]->getcache()), __fixed_sigmoid(_mem_logit[56][t]->getcache()), __fixed_sigmoid(_mem_logit[57][t]->getcache()), __fixed_sigmoid(_mem_logit[58][t]->getcache()), __fixed_sigmoid(_mem_logit[59][t]->getcache()), __fixed_sigmoid(_mem_logit[60][t]->getcache()), __fixed_sigmoid(_mem_logit[61][t]->getcache()), __fixed_sigmoid(_mem_logit[62][t]->getcache()), __fixed_sigmoid(_mem_logit[63][t]->getcache()), __fixed_sigmoid(_mem_logit[64][t]->getcache()), __fixed_sigmoid(_mem_logit[65][t]->getcache()), __fixed_sigmoid(_mem_logit[66][t]->getcache()), __fixed_sigmoid(_mem_logit[67][t]->getcache()), __fixed_sigmoid(_mem_logit[68][t]->getcache()), __fixed_sigmoid(_mem_logit[69][t]->getcache()), __fixed_sigmoid(_mem_logit[70][t]->getcache()), __fixed_sigmoid(_mem_logit[71][t]->getcache()), __fixed_sigmoid(_mem_logit[72][t]->getcache()), __fixed_sigmoid(_mem_logit[73][t]->getcache()), __fixed_sigmoid(_mem_logit[74][t]->getcache()), __fixed_sigmoid(_mem_logit[75][t]->getcache()), __fixed_sigmoid(_mem_logit[76][t]->getcache()), __fixed_sigmoid(_mem_logit[77][t]->getcache()), __fixed_sigmoid(_mem_logit[78][t]->getcache()), __fixed_sigmoid(_mem_logit[79][t]->getcache()), __fixed_sigmoid(_mem_logit[80][t]->getcache()), __fixed_sigmoid(_mem_logit[81][t]->getcache()), __fixed_sigmoid(_mem_logit[82][t]->getcache()), __fixed_sigmoid(_mem_logit[83][t]->getcache()), __fixed_sigmoid(_mem_logit[84][t]->getcache()), __fixed_sigmoid(_mem_logit[85][t]->getcache()), __fixed_sigmoid(_mem_logit[86][t]->getcache()), __fixed_sigmoid(_mem_logit[87][t]->getcache()), __fixed_sigmoid(_mem_logit[88][t]->getcache()), __fixed_sigmoid(_mem_logit[89][t]->getcache()), __fixed_sigmoid(_mem_logit[90][t]->getcache()), __fixed_sigmoid(_mem_logit[91][t]->getcache()), __fixed_sigmoid(_mem_logit[92][t]->getcache()), __fixed_sigmoid(_mem_logit[93][t]->getcache()), __fixed_sigmoid(_mem_logit[94][t]->getcache()), __fixed_sigmoid(_mem_logit[95][t]->getcache()), __fixed_sigmoid(_mem_logit[96][t]->getcache()), __fixed_sigmoid(_mem_logit[97][t]->getcache()), __fixed_sigmoid(_mem_logit[98][t]->getcache()), __fixed_sigmoid(_mem_logit[99][t]->getcache()), __fixed_sigmoid(_mem_logit[100][t]->getcache()), __fixed_sigmoid(_mem_logit[101][t]->getcache()), __fixed_sigmoid(_mem_logit[102][t]->getcache()), __fixed_sigmoid(_mem_logit[103][t]->getcache()), __fixed_sigmoid(_mem_logit[104][t]->getcache()), __fixed_sigmoid(_mem_logit[105][t]->getcache()), __fixed_sigmoid(_mem_logit[106][t]->getcache()), __fixed_sigmoid(_mem_logit[107][t]->getcache()), __fixed_sigmoid(_mem_logit[108][t]->getcache()), __fixed_sigmoid(_mem_logit[109][t]->getcache()), __fixed_sigmoid(_mem_logit[110][t]->getcache()), __fixed_sigmoid(_mem_logit[111][t]->getcache()), __fixed_sigmoid(_mem_logit[112][t]->getcache()), __fixed_sigmoid(_mem_logit[113][t]->getcache()), __fixed_sigmoid(_mem_logit[114][t]->getcache()), __fixed_sigmoid(_mem_logit[115][t]->getcache()), __fixed_sigmoid(_mem_logit[116][t]->getcache()), __fixed_sigmoid(_mem_logit[117][t]->getcache()), __fixed_sigmoid(_mem_logit[118][t]->getcache()), __fixed_sigmoid(_mem_logit[119][t]->getcache()), __fixed_sigmoid(_mem_logit[120][t]->getcache()), __fixed_sigmoid(_mem_logit[121][t]->getcache()), __fixed_sigmoid(_mem_logit[122][t]->getcache()), __fixed_sigmoid(_mem_logit[123][t]->getcache()), __fixed_sigmoid(_mem_logit[124][t]->getcache()), __fixed_sigmoid(_mem_logit[125][t]->getcache()), __fixed_sigmoid(_mem_logit[126][t]->getcache()), __fixed_sigmoid(_mem_logit[127][t]->getcache()), __fixed_sigmoid(_mem_logit[128][t]->getcache()), __fixed_sigmoid(_mem_logit[129][t]->getcache()), __fixed_sigmoid(_mem_logit[130][t]->getcache()), __fixed_sigmoid(_mem_logit[131][t]->getcache()), __fixed_sigmoid(_mem_logit[132][t]->getcache()), __fixed_sigmoid(_mem_logit[133][t]->getcache()), __fixed_sigmoid(_mem_logit[134][t]->getcache()), __fixed_sigmoid(_mem_logit[135][t]->getcache()), __fixed_sigmoid(_mem_logit[136][t]->getcache()), __fixed_sigmoid(_mem_logit[137][t]->getcache()), __fixed_sigmoid(_mem_logit[138][t]->getcache()), __fixed_sigmoid(_mem_logit[139][t]->getcache()), __fixed_sigmoid(_mem_logit[140][t]->getcache()), __fixed_sigmoid(_mem_logit[141][t]->getcache()), __fixed_sigmoid(_mem_logit[142][t]->getcache()), __fixed_sigmoid(_mem_logit[143][t]->getcache()), __fixed_sigmoid(_mem_logit[144][t]->getcache()), __fixed_sigmoid(_mem_logit[145][t]->getcache()), __fixed_sigmoid(_mem_logit[146][t]->getcache()), __fixed_sigmoid(_mem_logit[147][t]->getcache()), __fixed_sigmoid(_mem_logit[148][t]->getcache()), __fixed_sigmoid(_mem_logit[149][t]->getcache()), __fixed_sigmoid(_mem_logit[150][t]->getcache()), __fixed_sigmoid(_mem_logit[151][t]->getcache()), __fixed_sigmoid(_mem_logit[152][t]->getcache()), __fixed_sigmoid(_mem_logit[153][t]->getcache()), __fixed_sigmoid(_mem_logit[154][t]->getcache()), __fixed_sigmoid(_mem_logit[155][t]->getcache()), __fixed_sigmoid(_mem_logit[156][t]->getcache()), __fixed_sigmoid(_mem_logit[157][t]->getcache()), __fixed_sigmoid(_mem_logit[158][t]->getcache()), __fixed_sigmoid(_mem_logit[159][t]->getcache()), __fixed_sigmoid(_mem_logit[160][t]->getcache()), __fixed_sigmoid(_mem_logit[161][t]->getcache()), __fixed_sigmoid(_mem_logit[162][t]->getcache()), __fixed_sigmoid(_mem_logit[163][t]->getcache()), __fixed_sigmoid(_mem_logit[164][t]->getcache()), __fixed_sigmoid(_mem_logit[165][t]->getcache()), __fixed_sigmoid(_mem_logit[166][t]->getcache()), __fixed_sigmoid(_mem_logit[167][t]->getcache()), __fixed_sigmoid(_mem_logit[168][t]->getcache()), __fixed_sigmoid(_mem_logit[169][t]->getcache()), __fixed_sigmoid(_mem_logit[170][t]->getcache()), __fixed_sigmoid(_mem_logit[171][t]->getcache()), __fixed_sigmoid(_mem_logit[172][t]->getcache()), __fixed_sigmoid(_mem_logit[173][t]->getcache()), __fixed_sigmoid(_mem_logit[174][t]->getcache()), __fixed_sigmoid(_mem_logit[175][t]->getcache()), __fixed_sigmoid(_mem_logit[176][t]->getcache()), __fixed_sigmoid(_mem_logit[177][t]->getcache()), __fixed_sigmoid(_mem_logit[178][t]->getcache()), __fixed_sigmoid(_mem_logit[179][t]->getcache()), __fixed_sigmoid(_mem_logit[180][t]->getcache()), __fixed_sigmoid(_mem_logit[181][t]->getcache()), __fixed_sigmoid(_mem_logit[182][t]->getcache()), __fixed_sigmoid(_mem_logit[183][t]->getcache()), __fixed_sigmoid(_mem_logit[184][t]->getcache()), __fixed_sigmoid(_mem_logit[185][t]->getcache()), __fixed_sigmoid(_mem_logit[186][t]->getcache()), __fixed_sigmoid(_mem_logit[187][t]->getcache()), __fixed_sigmoid(_mem_logit[188][t]->getcache()), __fixed_sigmoid(_mem_logit[189][t]->getcache()), __fixed_sigmoid(_mem_logit[190][t]->getcache()), __fixed_sigmoid(_mem_logit[191][t]->getcache()), __fixed_sigmoid(_mem_logit[192][t]->getcache()), __fixed_sigmoid(_mem_logit[193][t]->getcache()), __fixed_sigmoid(_mem_logit[194][t]->getcache()), __fixed_sigmoid(_mem_logit[195][t]->getcache()), __fixed_sigmoid(_mem_logit[196][t]->getcache()), __fixed_sigmoid(_mem_logit[197][t]->getcache()), __fixed_sigmoid(_mem_logit[198][t]->getcache()), __fixed_sigmoid(_mem_logit[199][t]->getcache()), __fixed_sigmoid(_mem_logit[200][t]->getcache()), __fixed_sigmoid(_mem_logit[201][t]->getcache()), __fixed_sigmoid(_mem_logit[202][t]->getcache()), __fixed_sigmoid(_mem_logit[203][t]->getcache()), __fixed_sigmoid(_mem_logit[204][t]->getcache()), __fixed_sigmoid(_mem_logit[205][t]->getcache()), __fixed_sigmoid(_mem_logit[206][t]->getcache()), __fixed_sigmoid(_mem_logit[207][t]->getcache()), __fixed_sigmoid(_mem_logit[208][t]->getcache()), __fixed_sigmoid(_mem_logit[209][t]->getcache()), __fixed_sigmoid(_mem_logit[210][t]->getcache()), __fixed_sigmoid(_mem_logit[211][t]->getcache()), __fixed_sigmoid(_mem_logit[212][t]->getcache()), __fixed_sigmoid(_mem_logit[213][t]->getcache()), __fixed_sigmoid(_mem_logit[214][t]->getcache()), __fixed_sigmoid(_mem_logit[215][t]->getcache()), __fixed_sigmoid(_mem_logit[216][t]->getcache()), __fixed_sigmoid(_mem_logit[217][t]->getcache()), __fixed_sigmoid(_mem_logit[218][t]->getcache()), __fixed_sigmoid(_mem_logit[219][t]->getcache()), __fixed_sigmoid(_mem_logit[220][t]->getcache()), __fixed_sigmoid(_mem_logit[221][t]->getcache()), __fixed_sigmoid(_mem_logit[222][t]->getcache()), __fixed_sigmoid(_mem_logit[223][t]->getcache()), __fixed_sigmoid(_mem_logit[224][t]->getcache()), __fixed_sigmoid(_mem_logit[225][t]->getcache()), __fixed_sigmoid(_mem_logit[226][t]->getcache()), __fixed_sigmoid(_mem_logit[227][t]->getcache()), __fixed_sigmoid(_mem_logit[228][t]->getcache()), __fixed_sigmoid(_mem_logit[229][t]->getcache()), __fixed_sigmoid(_mem_logit[230][t]->getcache()), __fixed_sigmoid(_mem_logit[231][t]->getcache()), __fixed_sigmoid(_mem_logit[232][t]->getcache()), __fixed_sigmoid(_mem_logit[233][t]->getcache()), __fixed_sigmoid(_mem_logit[234][t]->getcache()), __fixed_sigmoid(_mem_logit[235][t]->getcache()), __fixed_sigmoid(_mem_logit[236][t]->getcache()), __fixed_sigmoid(_mem_logit[237][t]->getcache()), __fixed_sigmoid(_mem_logit[238][t]->getcache()), __fixed_sigmoid(_mem_logit[239][t]->getcache()), __fixed_sigmoid(_mem_logit[240][t]->getcache()), __fixed_sigmoid(_mem_logit[241][t]->getcache()), __fixed_sigmoid(_mem_logit[242][t]->getcache()), __fixed_sigmoid(_mem_logit[243][t]->getcache()), __fixed_sigmoid(_mem_logit[244][t]->getcache()), __fixed_sigmoid(_mem_logit[245][t]->getcache()), __fixed_sigmoid(_mem_logit[246][t]->getcache()), __fixed_sigmoid(_mem_logit[247][t]->getcache()), __fixed_sigmoid(_mem_logit[248][t]->getcache()), __fixed_sigmoid(_mem_logit[249][t]->getcache()), __fixed_sigmoid(_mem_logit[250][t]->getcache()), __fixed_sigmoid(_mem_logit[251][t]->getcache()), __fixed_sigmoid(_mem_logit[252][t]->getcache()), __fixed_sigmoid(_mem_logit[253][t]->getcache()), __fixed_sigmoid(_mem_logit[254][t]->getcache()), __fixed_sigmoid(_mem_logit[255][t]->getcache()), __fixed_sigmoid(_mem_logit[256][t]->getcache()), __fixed_sigmoid(_mem_logit[257][t]->getcache()), __fixed_sigmoid(_mem_logit[258][t]->getcache()), __fixed_sigmoid(_mem_logit[259][t]->getcache()), __fixed_sigmoid(_mem_logit[260][t]->getcache()), __fixed_sigmoid(_mem_logit[261][t]->getcache()), __fixed_sigmoid(_mem_logit[262][t]->getcache()), __fixed_sigmoid(_mem_logit[263][t]->getcache()), __fixed_sigmoid(_mem_logit[264][t]->getcache()), __fixed_sigmoid(_mem_logit[265][t]->getcache()), __fixed_sigmoid(_mem_logit[266][t]->getcache()), __fixed_sigmoid(_mem_logit[267][t]->getcache()), __fixed_sigmoid(_mem_logit[268][t]->getcache()), __fixed_sigmoid(_mem_logit[269][t]->getcache()), __fixed_sigmoid(_mem_logit[270][t]->getcache()), __fixed_sigmoid(_mem_logit[271][t]->getcache()), __fixed_sigmoid(_mem_logit[272][t]->getcache()), __fixed_sigmoid(_mem_logit[273][t]->getcache()), __fixed_sigmoid(_mem_logit[274][t]->getcache()), __fixed_sigmoid(_mem_logit[275][t]->getcache()), __fixed_sigmoid(_mem_logit[276][t]->getcache())}))/__fixed_region_pop[r],0.10000000),Gaussian28973584.gen());
}
void _Var_region_rate::active_edge()
{
  _mem_logit[0][t]->add_contig(this);
  _mem_logit[100][t]->add_contig(this);
  _mem_logit[101][t]->add_contig(this);
  _mem_logit[102][t]->add_contig(this);
  _mem_logit[103][t]->add_contig(this);
  _mem_logit[104][t]->add_contig(this);
  _mem_logit[105][t]->add_contig(this);
  _mem_logit[106][t]->add_contig(this);
  _mem_logit[107][t]->add_contig(this);
  _mem_logit[108][t]->add_contig(this);
  _mem_logit[109][t]->add_contig(this);
  _mem_logit[10][t]->add_contig(this);
  _mem_logit[110][t]->add_contig(this);
  _mem_logit[111][t]->add_contig(this);
  _mem_logit[112][t]->add_contig(this);
  _mem_logit[113][t]->add_contig(this);
  _mem_logit[114][t]->add_contig(this);
  _mem_logit[115][t]->add_contig(this);
  _mem_logit[116][t]->add_contig(this);
  _mem_logit[117][t]->add_contig(this);
  _mem_logit[118][t]->add_contig(this);
  _mem_logit[119][t]->add_contig(this);
  _mem_logit[11][t]->add_contig(this);
  _mem_logit[120][t]->add_contig(this);
  _mem_logit[121][t]->add_contig(this);
  _mem_logit[122][t]->add_contig(this);
  _mem_logit[123][t]->add_contig(this);
  _mem_logit[124][t]->add_contig(this);
  _mem_logit[125][t]->add_contig(this);
  _mem_logit[126][t]->add_contig(this);
  _mem_logit[127][t]->add_contig(this);
  _mem_logit[128][t]->add_contig(this);
  _mem_logit[129][t]->add_contig(this);
  _mem_logit[12][t]->add_contig(this);
  _mem_logit[130][t]->add_contig(this);
  _mem_logit[131][t]->add_contig(this);
  _mem_logit[132][t]->add_contig(this);
  _mem_logit[133][t]->add_contig(this);
  _mem_logit[134][t]->add_contig(this);
  _mem_logit[135][t]->add_contig(this);
  _mem_logit[136][t]->add_contig(this);
  _mem_logit[137][t]->add_contig(this);
  _mem_logit[138][t]->add_contig(this);
  _mem_logit[139][t]->add_contig(this);
  _mem_logit[13][t]->add_contig(this);
  _mem_logit[140][t]->add_contig(this);
  _mem_logit[141][t]->add_contig(this);
  _mem_logit[142][t]->add_contig(this);
  _mem_logit[143][t]->add_contig(this);
  _mem_logit[144][t]->add_contig(this);
  _mem_logit[145][t]->add_contig(this);
  _mem_logit[146][t]->add_contig(this);
  _mem_logit[147][t]->add_contig(this);
  _mem_logit[148][t]->add_contig(this);
  _mem_logit[149][t]->add_contig(this);
  _mem_logit[14][t]->add_contig(this);
  _mem_logit[150][t]->add_contig(this);
  _mem_logit[151][t]->add_contig(this);
  _mem_logit[152][t]->add_contig(this);
  _mem_logit[153][t]->add_contig(this);
  _mem_logit[154][t]->add_contig(this);
  _mem_logit[155][t]->add_contig(this);
  _mem_logit[156][t]->add_contig(this);
  _mem_logit[157][t]->add_contig(this);
  _mem_logit[158][t]->add_contig(this);
  _mem_logit[159][t]->add_contig(this);
  _mem_logit[15][t]->add_contig(this);
  _mem_logit[160][t]->add_contig(this);
  _mem_logit[161][t]->add_contig(this);
  _mem_logit[162][t]->add_contig(this);
  _mem_logit[163][t]->add_contig(this);
  _mem_logit[164][t]->add_contig(this);
  _mem_logit[165][t]->add_contig(this);
  _mem_logit[166][t]->add_contig(this);
  _mem_logit[167][t]->add_contig(this);
  _mem_logit[168][t]->add_contig(this);
  _mem_logit[169][t]->add_contig(this);
  _mem_logit[16][t]->add_contig(this);
  _mem_logit[170][t]->add_contig(this);
  _mem_logit[171][t]->add_contig(this);
  _mem_logit[172][t]->add_contig(this);
  _mem_logit[173][t]->add_contig(this);
  _mem_logit[174][t]->add_contig(this);
  _mem_logit[175][t]->add_contig(this);
  _mem_logit[176][t]->add_contig(this);
  _mem_logit[177][t]->add_contig(this);
  _mem_logit[178][t]->add_contig(this);
  _mem_logit[179][t]->add_contig(this);
  _mem_logit[17][t]->add_contig(this);
  _mem_logit[180][t]->add_contig(this);
  _mem_logit[181][t]->add_contig(this);
  _mem_logit[182][t]->add_contig(this);
  _mem_logit[183][t]->add_contig(this);
  _mem_logit[184][t]->add_contig(this);
  _mem_logit[185][t]->add_contig(this);
  _mem_logit[186][t]->add_contig(this);
  _mem_logit[187][t]->add_contig(this);
  _mem_logit[188][t]->add_contig(this);
  _mem_logit[189][t]->add_contig(this);
  _mem_logit[18][t]->add_contig(this);
  _mem_logit[190][t]->add_contig(this);
  _mem_logit[191][t]->add_contig(this);
  _mem_logit[192][t]->add_contig(this);
  _mem_logit[193][t]->add_contig(this);
  _mem_logit[194][t]->add_contig(this);
  _mem_logit[195][t]->add_contig(this);
  _mem_logit[196][t]->add_contig(this);
  _mem_logit[197][t]->add_contig(this);
  _mem_logit[198][t]->add_contig(this);
  _mem_logit[199][t]->add_contig(this);
  _mem_logit[19][t]->add_contig(this);
  _mem_logit[1][t]->add_contig(this);
  _mem_logit[200][t]->add_contig(this);
  _mem_logit[201][t]->add_contig(this);
  _mem_logit[202][t]->add_contig(this);
  _mem_logit[203][t]->add_contig(this);
  _mem_logit[204][t]->add_contig(this);
  _mem_logit[205][t]->add_contig(this);
  _mem_logit[206][t]->add_contig(this);
  _mem_logit[207][t]->add_contig(this);
  _mem_logit[208][t]->add_contig(this);
  _mem_logit[209][t]->add_contig(this);
  _mem_logit[20][t]->add_contig(this);
  _mem_logit[210][t]->add_contig(this);
  _mem_logit[211][t]->add_contig(this);
  _mem_logit[212][t]->add_contig(this);
  _mem_logit[213][t]->add_contig(this);
  _mem_logit[214][t]->add_contig(this);
  _mem_logit[215][t]->add_contig(this);
  _mem_logit[216][t]->add_contig(this);
  _mem_logit[217][t]->add_contig(this);
  _mem_logit[218][t]->add_contig(this);
  _mem_logit[219][t]->add_contig(this);
  _mem_logit[21][t]->add_contig(this);
  _mem_logit[220][t]->add_contig(this);
  _mem_logit[221][t]->add_contig(this);
  _mem_logit[222][t]->add_contig(this);
  _mem_logit[223][t]->add_contig(this);
  _mem_logit[224][t]->add_contig(this);
  _mem_logit[225][t]->add_contig(this);
  _mem_logit[226][t]->add_contig(this);
  _mem_logit[227][t]->add_contig(this);
  _mem_logit[228][t]->add_contig(this);
  _mem_logit[229][t]->add_contig(this);
  _mem_logit[22][t]->add_contig(this);
  _mem_logit[230][t]->add_contig(this);
  _mem_logit[231][t]->add_contig(this);
  _mem_logit[232][t]->add_contig(this);
  _mem_logit[233][t]->add_contig(this);
  _mem_logit[234][t]->add_contig(this);
  _mem_logit[235][t]->add_contig(this);
  _mem_logit[236][t]->add_contig(this);
  _mem_logit[237][t]->add_contig(this);
  _mem_logit[238][t]->add_contig(this);
  _mem_logit[239][t]->add_contig(this);
  _mem_logit[23][t]->add_contig(this);
  _mem_logit[240][t]->add_contig(this);
  _mem_logit[241][t]->add_contig(this);
  _mem_logit[242][t]->add_contig(this);
  _mem_logit[243][t]->add_contig(this);
  _mem_logit[244][t]->add_contig(this);
  _mem_logit[245][t]->add_contig(this);
  _mem_logit[246][t]->add_contig(this);
  _mem_logit[247][t]->add_contig(this);
  _mem_logit[248][t]->add_contig(this);
  _mem_logit[249][t]->add_contig(this);
  _mem_logit[24][t]->add_contig(this);
  _mem_logit[250][t]->add_contig(this);
  _mem_logit[251][t]->add_contig(this);
  _mem_logit[252][t]->add_contig(this);
  _mem_logit[253][t]->add_contig(this);
  _mem_logit[254][t]->add_contig(this);
  _mem_logit[255][t]->add_contig(this);
  _mem_logit[256][t]->add_contig(this);
  _mem_logit[257][t]->add_contig(this);
  _mem_logit[258][t]->add_contig(this);
  _mem_logit[259][t]->add_contig(this);
  _mem_logit[25][t]->add_contig(this);
  _mem_logit[260][t]->add_contig(this);
  _mem_logit[261][t]->add_contig(this);
  _mem_logit[262][t]->add_contig(this);
  _mem_logit[263][t]->add_contig(this);
  _mem_logit[264][t]->add_contig(this);
  _mem_logit[265][t]->add_contig(this);
  _mem_logit[266][t]->add_contig(this);
  _mem_logit[267][t]->add_contig(this);
  _mem_logit[268][t]->add_contig(this);
  _mem_logit[269][t]->add_contig(this);
  _mem_logit[26][t]->add_contig(this);
  _mem_logit[270][t]->add_contig(this);
  _mem_logit[271][t]->add_contig(this);
  _mem_logit[272][t]->add_contig(this);
  _mem_logit[273][t]->add_contig(this);
  _mem_logit[274][t]->add_contig(this);
  _mem_logit[275][t]->add_contig(this);
  _mem_logit[276][t]->add_contig(this);
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
  _mem_logit[82][t]->add_contig(this);
  _mem_logit[83][t]->add_contig(this);
  _mem_logit[84][t]->add_contig(this);
  _mem_logit[85][t]->add_contig(this);
  _mem_logit[86][t]->add_contig(this);
  _mem_logit[87][t]->add_contig(this);
  _mem_logit[88][t]->add_contig(this);
  _mem_logit[89][t]->add_contig(this);
  _mem_logit[8][t]->add_contig(this);
  _mem_logit[90][t]->add_contig(this);
  _mem_logit[91][t]->add_contig(this);
  _mem_logit[92][t]->add_contig(this);
  _mem_logit[93][t]->add_contig(this);
  _mem_logit[94][t]->add_contig(this);
  _mem_logit[95][t]->add_contig(this);
  _mem_logit[96][t]->add_contig(this);
  _mem_logit[97][t]->add_contig(this);
  _mem_logit[98][t]->add_contig(this);
  _mem_logit[99][t]->add_contig(this);
  _mem_logit[9][t]->add_contig(this);
  _mem_logit[0][t]->add_child(this);
  _mem_logit[100][t]->add_child(this);
  _mem_logit[101][t]->add_child(this);
  _mem_logit[102][t]->add_child(this);
  _mem_logit[103][t]->add_child(this);
  _mem_logit[104][t]->add_child(this);
  _mem_logit[105][t]->add_child(this);
  _mem_logit[106][t]->add_child(this);
  _mem_logit[107][t]->add_child(this);
  _mem_logit[108][t]->add_child(this);
  _mem_logit[109][t]->add_child(this);
  _mem_logit[10][t]->add_child(this);
  _mem_logit[110][t]->add_child(this);
  _mem_logit[111][t]->add_child(this);
  _mem_logit[112][t]->add_child(this);
  _mem_logit[113][t]->add_child(this);
  _mem_logit[114][t]->add_child(this);
  _mem_logit[115][t]->add_child(this);
  _mem_logit[116][t]->add_child(this);
  _mem_logit[117][t]->add_child(this);
  _mem_logit[118][t]->add_child(this);
  _mem_logit[119][t]->add_child(this);
  _mem_logit[11][t]->add_child(this);
  _mem_logit[120][t]->add_child(this);
  _mem_logit[121][t]->add_child(this);
  _mem_logit[122][t]->add_child(this);
  _mem_logit[123][t]->add_child(this);
  _mem_logit[124][t]->add_child(this);
  _mem_logit[125][t]->add_child(this);
  _mem_logit[126][t]->add_child(this);
  _mem_logit[127][t]->add_child(this);
  _mem_logit[128][t]->add_child(this);
  _mem_logit[129][t]->add_child(this);
  _mem_logit[12][t]->add_child(this);
  _mem_logit[130][t]->add_child(this);
  _mem_logit[131][t]->add_child(this);
  _mem_logit[132][t]->add_child(this);
  _mem_logit[133][t]->add_child(this);
  _mem_logit[134][t]->add_child(this);
  _mem_logit[135][t]->add_child(this);
  _mem_logit[136][t]->add_child(this);
  _mem_logit[137][t]->add_child(this);
  _mem_logit[138][t]->add_child(this);
  _mem_logit[139][t]->add_child(this);
  _mem_logit[13][t]->add_child(this);
  _mem_logit[140][t]->add_child(this);
  _mem_logit[141][t]->add_child(this);
  _mem_logit[142][t]->add_child(this);
  _mem_logit[143][t]->add_child(this);
  _mem_logit[144][t]->add_child(this);
  _mem_logit[145][t]->add_child(this);
  _mem_logit[146][t]->add_child(this);
  _mem_logit[147][t]->add_child(this);
  _mem_logit[148][t]->add_child(this);
  _mem_logit[149][t]->add_child(this);
  _mem_logit[14][t]->add_child(this);
  _mem_logit[150][t]->add_child(this);
  _mem_logit[151][t]->add_child(this);
  _mem_logit[152][t]->add_child(this);
  _mem_logit[153][t]->add_child(this);
  _mem_logit[154][t]->add_child(this);
  _mem_logit[155][t]->add_child(this);
  _mem_logit[156][t]->add_child(this);
  _mem_logit[157][t]->add_child(this);
  _mem_logit[158][t]->add_child(this);
  _mem_logit[159][t]->add_child(this);
  _mem_logit[15][t]->add_child(this);
  _mem_logit[160][t]->add_child(this);
  _mem_logit[161][t]->add_child(this);
  _mem_logit[162][t]->add_child(this);
  _mem_logit[163][t]->add_child(this);
  _mem_logit[164][t]->add_child(this);
  _mem_logit[165][t]->add_child(this);
  _mem_logit[166][t]->add_child(this);
  _mem_logit[167][t]->add_child(this);
  _mem_logit[168][t]->add_child(this);
  _mem_logit[169][t]->add_child(this);
  _mem_logit[16][t]->add_child(this);
  _mem_logit[170][t]->add_child(this);
  _mem_logit[171][t]->add_child(this);
  _mem_logit[172][t]->add_child(this);
  _mem_logit[173][t]->add_child(this);
  _mem_logit[174][t]->add_child(this);
  _mem_logit[175][t]->add_child(this);
  _mem_logit[176][t]->add_child(this);
  _mem_logit[177][t]->add_child(this);
  _mem_logit[178][t]->add_child(this);
  _mem_logit[179][t]->add_child(this);
  _mem_logit[17][t]->add_child(this);
  _mem_logit[180][t]->add_child(this);
  _mem_logit[181][t]->add_child(this);
  _mem_logit[182][t]->add_child(this);
  _mem_logit[183][t]->add_child(this);
  _mem_logit[184][t]->add_child(this);
  _mem_logit[185][t]->add_child(this);
  _mem_logit[186][t]->add_child(this);
  _mem_logit[187][t]->add_child(this);
  _mem_logit[188][t]->add_child(this);
  _mem_logit[189][t]->add_child(this);
  _mem_logit[18][t]->add_child(this);
  _mem_logit[190][t]->add_child(this);
  _mem_logit[191][t]->add_child(this);
  _mem_logit[192][t]->add_child(this);
  _mem_logit[193][t]->add_child(this);
  _mem_logit[194][t]->add_child(this);
  _mem_logit[195][t]->add_child(this);
  _mem_logit[196][t]->add_child(this);
  _mem_logit[197][t]->add_child(this);
  _mem_logit[198][t]->add_child(this);
  _mem_logit[199][t]->add_child(this);
  _mem_logit[19][t]->add_child(this);
  _mem_logit[1][t]->add_child(this);
  _mem_logit[200][t]->add_child(this);
  _mem_logit[201][t]->add_child(this);
  _mem_logit[202][t]->add_child(this);
  _mem_logit[203][t]->add_child(this);
  _mem_logit[204][t]->add_child(this);
  _mem_logit[205][t]->add_child(this);
  _mem_logit[206][t]->add_child(this);
  _mem_logit[207][t]->add_child(this);
  _mem_logit[208][t]->add_child(this);
  _mem_logit[209][t]->add_child(this);
  _mem_logit[20][t]->add_child(this);
  _mem_logit[210][t]->add_child(this);
  _mem_logit[211][t]->add_child(this);
  _mem_logit[212][t]->add_child(this);
  _mem_logit[213][t]->add_child(this);
  _mem_logit[214][t]->add_child(this);
  _mem_logit[215][t]->add_child(this);
  _mem_logit[216][t]->add_child(this);
  _mem_logit[217][t]->add_child(this);
  _mem_logit[218][t]->add_child(this);
  _mem_logit[219][t]->add_child(this);
  _mem_logit[21][t]->add_child(this);
  _mem_logit[220][t]->add_child(this);
  _mem_logit[221][t]->add_child(this);
  _mem_logit[222][t]->add_child(this);
  _mem_logit[223][t]->add_child(this);
  _mem_logit[224][t]->add_child(this);
  _mem_logit[225][t]->add_child(this);
  _mem_logit[226][t]->add_child(this);
  _mem_logit[227][t]->add_child(this);
  _mem_logit[228][t]->add_child(this);
  _mem_logit[229][t]->add_child(this);
  _mem_logit[22][t]->add_child(this);
  _mem_logit[230][t]->add_child(this);
  _mem_logit[231][t]->add_child(this);
  _mem_logit[232][t]->add_child(this);
  _mem_logit[233][t]->add_child(this);
  _mem_logit[234][t]->add_child(this);
  _mem_logit[235][t]->add_child(this);
  _mem_logit[236][t]->add_child(this);
  _mem_logit[237][t]->add_child(this);
  _mem_logit[238][t]->add_child(this);
  _mem_logit[239][t]->add_child(this);
  _mem_logit[23][t]->add_child(this);
  _mem_logit[240][t]->add_child(this);
  _mem_logit[241][t]->add_child(this);
  _mem_logit[242][t]->add_child(this);
  _mem_logit[243][t]->add_child(this);
  _mem_logit[244][t]->add_child(this);
  _mem_logit[245][t]->add_child(this);
  _mem_logit[246][t]->add_child(this);
  _mem_logit[247][t]->add_child(this);
  _mem_logit[248][t]->add_child(this);
  _mem_logit[249][t]->add_child(this);
  _mem_logit[24][t]->add_child(this);
  _mem_logit[250][t]->add_child(this);
  _mem_logit[251][t]->add_child(this);
  _mem_logit[252][t]->add_child(this);
  _mem_logit[253][t]->add_child(this);
  _mem_logit[254][t]->add_child(this);
  _mem_logit[255][t]->add_child(this);
  _mem_logit[256][t]->add_child(this);
  _mem_logit[257][t]->add_child(this);
  _mem_logit[258][t]->add_child(this);
  _mem_logit[259][t]->add_child(this);
  _mem_logit[25][t]->add_child(this);
  _mem_logit[260][t]->add_child(this);
  _mem_logit[261][t]->add_child(this);
  _mem_logit[262][t]->add_child(this);
  _mem_logit[263][t]->add_child(this);
  _mem_logit[264][t]->add_child(this);
  _mem_logit[265][t]->add_child(this);
  _mem_logit[266][t]->add_child(this);
  _mem_logit[267][t]->add_child(this);
  _mem_logit[268][t]->add_child(this);
  _mem_logit[269][t]->add_child(this);
  _mem_logit[26][t]->add_child(this);
  _mem_logit[270][t]->add_child(this);
  _mem_logit[271][t]->add_child(this);
  _mem_logit[272][t]->add_child(this);
  _mem_logit[273][t]->add_child(this);
  _mem_logit[274][t]->add_child(this);
  _mem_logit[275][t]->add_child(this);
  _mem_logit[276][t]->add_child(this);
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
  _mem_logit[82][t]->add_child(this);
  _mem_logit[83][t]->add_child(this);
  _mem_logit[84][t]->add_child(this);
  _mem_logit[85][t]->add_child(this);
  _mem_logit[86][t]->add_child(this);
  _mem_logit[87][t]->add_child(this);
  _mem_logit[88][t]->add_child(this);
  _mem_logit[89][t]->add_child(this);
  _mem_logit[8][t]->add_child(this);
  _mem_logit[90][t]->add_child(this);
  _mem_logit[91][t]->add_child(this);
  _mem_logit[92][t]->add_child(this);
  _mem_logit[93][t]->add_child(this);
  _mem_logit[94][t]->add_child(this);
  _mem_logit[95][t]->add_child(this);
  _mem_logit[96][t]->add_child(this);
  _mem_logit[97][t]->add_child(this);
  _mem_logit[98][t]->add_child(this);
  _mem_logit[99][t]->add_child(this);
  _mem_logit[9][t]->add_child(this);
}
void _Var_region_rate::remove_edge()
{
  _mem_logit[0][t]->erase_contig(this);
  _mem_logit[100][t]->erase_contig(this);
  _mem_logit[101][t]->erase_contig(this);
  _mem_logit[102][t]->erase_contig(this);
  _mem_logit[103][t]->erase_contig(this);
  _mem_logit[104][t]->erase_contig(this);
  _mem_logit[105][t]->erase_contig(this);
  _mem_logit[106][t]->erase_contig(this);
  _mem_logit[107][t]->erase_contig(this);
  _mem_logit[108][t]->erase_contig(this);
  _mem_logit[109][t]->erase_contig(this);
  _mem_logit[10][t]->erase_contig(this);
  _mem_logit[110][t]->erase_contig(this);
  _mem_logit[111][t]->erase_contig(this);
  _mem_logit[112][t]->erase_contig(this);
  _mem_logit[113][t]->erase_contig(this);
  _mem_logit[114][t]->erase_contig(this);
  _mem_logit[115][t]->erase_contig(this);
  _mem_logit[116][t]->erase_contig(this);
  _mem_logit[117][t]->erase_contig(this);
  _mem_logit[118][t]->erase_contig(this);
  _mem_logit[119][t]->erase_contig(this);
  _mem_logit[11][t]->erase_contig(this);
  _mem_logit[120][t]->erase_contig(this);
  _mem_logit[121][t]->erase_contig(this);
  _mem_logit[122][t]->erase_contig(this);
  _mem_logit[123][t]->erase_contig(this);
  _mem_logit[124][t]->erase_contig(this);
  _mem_logit[125][t]->erase_contig(this);
  _mem_logit[126][t]->erase_contig(this);
  _mem_logit[127][t]->erase_contig(this);
  _mem_logit[128][t]->erase_contig(this);
  _mem_logit[129][t]->erase_contig(this);
  _mem_logit[12][t]->erase_contig(this);
  _mem_logit[130][t]->erase_contig(this);
  _mem_logit[131][t]->erase_contig(this);
  _mem_logit[132][t]->erase_contig(this);
  _mem_logit[133][t]->erase_contig(this);
  _mem_logit[134][t]->erase_contig(this);
  _mem_logit[135][t]->erase_contig(this);
  _mem_logit[136][t]->erase_contig(this);
  _mem_logit[137][t]->erase_contig(this);
  _mem_logit[138][t]->erase_contig(this);
  _mem_logit[139][t]->erase_contig(this);
  _mem_logit[13][t]->erase_contig(this);
  _mem_logit[140][t]->erase_contig(this);
  _mem_logit[141][t]->erase_contig(this);
  _mem_logit[142][t]->erase_contig(this);
  _mem_logit[143][t]->erase_contig(this);
  _mem_logit[144][t]->erase_contig(this);
  _mem_logit[145][t]->erase_contig(this);
  _mem_logit[146][t]->erase_contig(this);
  _mem_logit[147][t]->erase_contig(this);
  _mem_logit[148][t]->erase_contig(this);
  _mem_logit[149][t]->erase_contig(this);
  _mem_logit[14][t]->erase_contig(this);
  _mem_logit[150][t]->erase_contig(this);
  _mem_logit[151][t]->erase_contig(this);
  _mem_logit[152][t]->erase_contig(this);
  _mem_logit[153][t]->erase_contig(this);
  _mem_logit[154][t]->erase_contig(this);
  _mem_logit[155][t]->erase_contig(this);
  _mem_logit[156][t]->erase_contig(this);
  _mem_logit[157][t]->erase_contig(this);
  _mem_logit[158][t]->erase_contig(this);
  _mem_logit[159][t]->erase_contig(this);
  _mem_logit[15][t]->erase_contig(this);
  _mem_logit[160][t]->erase_contig(this);
  _mem_logit[161][t]->erase_contig(this);
  _mem_logit[162][t]->erase_contig(this);
  _mem_logit[163][t]->erase_contig(this);
  _mem_logit[164][t]->erase_contig(this);
  _mem_logit[165][t]->erase_contig(this);
  _mem_logit[166][t]->erase_contig(this);
  _mem_logit[167][t]->erase_contig(this);
  _mem_logit[168][t]->erase_contig(this);
  _mem_logit[169][t]->erase_contig(this);
  _mem_logit[16][t]->erase_contig(this);
  _mem_logit[170][t]->erase_contig(this);
  _mem_logit[171][t]->erase_contig(this);
  _mem_logit[172][t]->erase_contig(this);
  _mem_logit[173][t]->erase_contig(this);
  _mem_logit[174][t]->erase_contig(this);
  _mem_logit[175][t]->erase_contig(this);
  _mem_logit[176][t]->erase_contig(this);
  _mem_logit[177][t]->erase_contig(this);
  _mem_logit[178][t]->erase_contig(this);
  _mem_logit[179][t]->erase_contig(this);
  _mem_logit[17][t]->erase_contig(this);
  _mem_logit[180][t]->erase_contig(this);
  _mem_logit[181][t]->erase_contig(this);
  _mem_logit[182][t]->erase_contig(this);
  _mem_logit[183][t]->erase_contig(this);
  _mem_logit[184][t]->erase_contig(this);
  _mem_logit[185][t]->erase_contig(this);
  _mem_logit[186][t]->erase_contig(this);
  _mem_logit[187][t]->erase_contig(this);
  _mem_logit[188][t]->erase_contig(this);
  _mem_logit[189][t]->erase_contig(this);
  _mem_logit[18][t]->erase_contig(this);
  _mem_logit[190][t]->erase_contig(this);
  _mem_logit[191][t]->erase_contig(this);
  _mem_logit[192][t]->erase_contig(this);
  _mem_logit[193][t]->erase_contig(this);
  _mem_logit[194][t]->erase_contig(this);
  _mem_logit[195][t]->erase_contig(this);
  _mem_logit[196][t]->erase_contig(this);
  _mem_logit[197][t]->erase_contig(this);
  _mem_logit[198][t]->erase_contig(this);
  _mem_logit[199][t]->erase_contig(this);
  _mem_logit[19][t]->erase_contig(this);
  _mem_logit[1][t]->erase_contig(this);
  _mem_logit[200][t]->erase_contig(this);
  _mem_logit[201][t]->erase_contig(this);
  _mem_logit[202][t]->erase_contig(this);
  _mem_logit[203][t]->erase_contig(this);
  _mem_logit[204][t]->erase_contig(this);
  _mem_logit[205][t]->erase_contig(this);
  _mem_logit[206][t]->erase_contig(this);
  _mem_logit[207][t]->erase_contig(this);
  _mem_logit[208][t]->erase_contig(this);
  _mem_logit[209][t]->erase_contig(this);
  _mem_logit[20][t]->erase_contig(this);
  _mem_logit[210][t]->erase_contig(this);
  _mem_logit[211][t]->erase_contig(this);
  _mem_logit[212][t]->erase_contig(this);
  _mem_logit[213][t]->erase_contig(this);
  _mem_logit[214][t]->erase_contig(this);
  _mem_logit[215][t]->erase_contig(this);
  _mem_logit[216][t]->erase_contig(this);
  _mem_logit[217][t]->erase_contig(this);
  _mem_logit[218][t]->erase_contig(this);
  _mem_logit[219][t]->erase_contig(this);
  _mem_logit[21][t]->erase_contig(this);
  _mem_logit[220][t]->erase_contig(this);
  _mem_logit[221][t]->erase_contig(this);
  _mem_logit[222][t]->erase_contig(this);
  _mem_logit[223][t]->erase_contig(this);
  _mem_logit[224][t]->erase_contig(this);
  _mem_logit[225][t]->erase_contig(this);
  _mem_logit[226][t]->erase_contig(this);
  _mem_logit[227][t]->erase_contig(this);
  _mem_logit[228][t]->erase_contig(this);
  _mem_logit[229][t]->erase_contig(this);
  _mem_logit[22][t]->erase_contig(this);
  _mem_logit[230][t]->erase_contig(this);
  _mem_logit[231][t]->erase_contig(this);
  _mem_logit[232][t]->erase_contig(this);
  _mem_logit[233][t]->erase_contig(this);
  _mem_logit[234][t]->erase_contig(this);
  _mem_logit[235][t]->erase_contig(this);
  _mem_logit[236][t]->erase_contig(this);
  _mem_logit[237][t]->erase_contig(this);
  _mem_logit[238][t]->erase_contig(this);
  _mem_logit[239][t]->erase_contig(this);
  _mem_logit[23][t]->erase_contig(this);
  _mem_logit[240][t]->erase_contig(this);
  _mem_logit[241][t]->erase_contig(this);
  _mem_logit[242][t]->erase_contig(this);
  _mem_logit[243][t]->erase_contig(this);
  _mem_logit[244][t]->erase_contig(this);
  _mem_logit[245][t]->erase_contig(this);
  _mem_logit[246][t]->erase_contig(this);
  _mem_logit[247][t]->erase_contig(this);
  _mem_logit[248][t]->erase_contig(this);
  _mem_logit[249][t]->erase_contig(this);
  _mem_logit[24][t]->erase_contig(this);
  _mem_logit[250][t]->erase_contig(this);
  _mem_logit[251][t]->erase_contig(this);
  _mem_logit[252][t]->erase_contig(this);
  _mem_logit[253][t]->erase_contig(this);
  _mem_logit[254][t]->erase_contig(this);
  _mem_logit[255][t]->erase_contig(this);
  _mem_logit[256][t]->erase_contig(this);
  _mem_logit[257][t]->erase_contig(this);
  _mem_logit[258][t]->erase_contig(this);
  _mem_logit[259][t]->erase_contig(this);
  _mem_logit[25][t]->erase_contig(this);
  _mem_logit[260][t]->erase_contig(this);
  _mem_logit[261][t]->erase_contig(this);
  _mem_logit[262][t]->erase_contig(this);
  _mem_logit[263][t]->erase_contig(this);
  _mem_logit[264][t]->erase_contig(this);
  _mem_logit[265][t]->erase_contig(this);
  _mem_logit[266][t]->erase_contig(this);
  _mem_logit[267][t]->erase_contig(this);
  _mem_logit[268][t]->erase_contig(this);
  _mem_logit[269][t]->erase_contig(this);
  _mem_logit[26][t]->erase_contig(this);
  _mem_logit[270][t]->erase_contig(this);
  _mem_logit[271][t]->erase_contig(this);
  _mem_logit[272][t]->erase_contig(this);
  _mem_logit[273][t]->erase_contig(this);
  _mem_logit[274][t]->erase_contig(this);
  _mem_logit[275][t]->erase_contig(this);
  _mem_logit[276][t]->erase_contig(this);
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
  _mem_logit[82][t]->erase_contig(this);
  _mem_logit[83][t]->erase_contig(this);
  _mem_logit[84][t]->erase_contig(this);
  _mem_logit[85][t]->erase_contig(this);
  _mem_logit[86][t]->erase_contig(this);
  _mem_logit[87][t]->erase_contig(this);
  _mem_logit[88][t]->erase_contig(this);
  _mem_logit[89][t]->erase_contig(this);
  _mem_logit[8][t]->erase_contig(this);
  _mem_logit[90][t]->erase_contig(this);
  _mem_logit[91][t]->erase_contig(this);
  _mem_logit[92][t]->erase_contig(this);
  _mem_logit[93][t]->erase_contig(this);
  _mem_logit[94][t]->erase_contig(this);
  _mem_logit[95][t]->erase_contig(this);
  _mem_logit[96][t]->erase_contig(this);
  _mem_logit[97][t]->erase_contig(this);
  _mem_logit[98][t]->erase_contig(this);
  _mem_logit[99][t]->erase_contig(this);
  _mem_logit[9][t]->erase_contig(this);
  _mem_logit[0][t]->erase_child(this);
  _mem_logit[100][t]->erase_child(this);
  _mem_logit[101][t]->erase_child(this);
  _mem_logit[102][t]->erase_child(this);
  _mem_logit[103][t]->erase_child(this);
  _mem_logit[104][t]->erase_child(this);
  _mem_logit[105][t]->erase_child(this);
  _mem_logit[106][t]->erase_child(this);
  _mem_logit[107][t]->erase_child(this);
  _mem_logit[108][t]->erase_child(this);
  _mem_logit[109][t]->erase_child(this);
  _mem_logit[10][t]->erase_child(this);
  _mem_logit[110][t]->erase_child(this);
  _mem_logit[111][t]->erase_child(this);
  _mem_logit[112][t]->erase_child(this);
  _mem_logit[113][t]->erase_child(this);
  _mem_logit[114][t]->erase_child(this);
  _mem_logit[115][t]->erase_child(this);
  _mem_logit[116][t]->erase_child(this);
  _mem_logit[117][t]->erase_child(this);
  _mem_logit[118][t]->erase_child(this);
  _mem_logit[119][t]->erase_child(this);
  _mem_logit[11][t]->erase_child(this);
  _mem_logit[120][t]->erase_child(this);
  _mem_logit[121][t]->erase_child(this);
  _mem_logit[122][t]->erase_child(this);
  _mem_logit[123][t]->erase_child(this);
  _mem_logit[124][t]->erase_child(this);
  _mem_logit[125][t]->erase_child(this);
  _mem_logit[126][t]->erase_child(this);
  _mem_logit[127][t]->erase_child(this);
  _mem_logit[128][t]->erase_child(this);
  _mem_logit[129][t]->erase_child(this);
  _mem_logit[12][t]->erase_child(this);
  _mem_logit[130][t]->erase_child(this);
  _mem_logit[131][t]->erase_child(this);
  _mem_logit[132][t]->erase_child(this);
  _mem_logit[133][t]->erase_child(this);
  _mem_logit[134][t]->erase_child(this);
  _mem_logit[135][t]->erase_child(this);
  _mem_logit[136][t]->erase_child(this);
  _mem_logit[137][t]->erase_child(this);
  _mem_logit[138][t]->erase_child(this);
  _mem_logit[139][t]->erase_child(this);
  _mem_logit[13][t]->erase_child(this);
  _mem_logit[140][t]->erase_child(this);
  _mem_logit[141][t]->erase_child(this);
  _mem_logit[142][t]->erase_child(this);
  _mem_logit[143][t]->erase_child(this);
  _mem_logit[144][t]->erase_child(this);
  _mem_logit[145][t]->erase_child(this);
  _mem_logit[146][t]->erase_child(this);
  _mem_logit[147][t]->erase_child(this);
  _mem_logit[148][t]->erase_child(this);
  _mem_logit[149][t]->erase_child(this);
  _mem_logit[14][t]->erase_child(this);
  _mem_logit[150][t]->erase_child(this);
  _mem_logit[151][t]->erase_child(this);
  _mem_logit[152][t]->erase_child(this);
  _mem_logit[153][t]->erase_child(this);
  _mem_logit[154][t]->erase_child(this);
  _mem_logit[155][t]->erase_child(this);
  _mem_logit[156][t]->erase_child(this);
  _mem_logit[157][t]->erase_child(this);
  _mem_logit[158][t]->erase_child(this);
  _mem_logit[159][t]->erase_child(this);
  _mem_logit[15][t]->erase_child(this);
  _mem_logit[160][t]->erase_child(this);
  _mem_logit[161][t]->erase_child(this);
  _mem_logit[162][t]->erase_child(this);
  _mem_logit[163][t]->erase_child(this);
  _mem_logit[164][t]->erase_child(this);
  _mem_logit[165][t]->erase_child(this);
  _mem_logit[166][t]->erase_child(this);
  _mem_logit[167][t]->erase_child(this);
  _mem_logit[168][t]->erase_child(this);
  _mem_logit[169][t]->erase_child(this);
  _mem_logit[16][t]->erase_child(this);
  _mem_logit[170][t]->erase_child(this);
  _mem_logit[171][t]->erase_child(this);
  _mem_logit[172][t]->erase_child(this);
  _mem_logit[173][t]->erase_child(this);
  _mem_logit[174][t]->erase_child(this);
  _mem_logit[175][t]->erase_child(this);
  _mem_logit[176][t]->erase_child(this);
  _mem_logit[177][t]->erase_child(this);
  _mem_logit[178][t]->erase_child(this);
  _mem_logit[179][t]->erase_child(this);
  _mem_logit[17][t]->erase_child(this);
  _mem_logit[180][t]->erase_child(this);
  _mem_logit[181][t]->erase_child(this);
  _mem_logit[182][t]->erase_child(this);
  _mem_logit[183][t]->erase_child(this);
  _mem_logit[184][t]->erase_child(this);
  _mem_logit[185][t]->erase_child(this);
  _mem_logit[186][t]->erase_child(this);
  _mem_logit[187][t]->erase_child(this);
  _mem_logit[188][t]->erase_child(this);
  _mem_logit[189][t]->erase_child(this);
  _mem_logit[18][t]->erase_child(this);
  _mem_logit[190][t]->erase_child(this);
  _mem_logit[191][t]->erase_child(this);
  _mem_logit[192][t]->erase_child(this);
  _mem_logit[193][t]->erase_child(this);
  _mem_logit[194][t]->erase_child(this);
  _mem_logit[195][t]->erase_child(this);
  _mem_logit[196][t]->erase_child(this);
  _mem_logit[197][t]->erase_child(this);
  _mem_logit[198][t]->erase_child(this);
  _mem_logit[199][t]->erase_child(this);
  _mem_logit[19][t]->erase_child(this);
  _mem_logit[1][t]->erase_child(this);
  _mem_logit[200][t]->erase_child(this);
  _mem_logit[201][t]->erase_child(this);
  _mem_logit[202][t]->erase_child(this);
  _mem_logit[203][t]->erase_child(this);
  _mem_logit[204][t]->erase_child(this);
  _mem_logit[205][t]->erase_child(this);
  _mem_logit[206][t]->erase_child(this);
  _mem_logit[207][t]->erase_child(this);
  _mem_logit[208][t]->erase_child(this);
  _mem_logit[209][t]->erase_child(this);
  _mem_logit[20][t]->erase_child(this);
  _mem_logit[210][t]->erase_child(this);
  _mem_logit[211][t]->erase_child(this);
  _mem_logit[212][t]->erase_child(this);
  _mem_logit[213][t]->erase_child(this);
  _mem_logit[214][t]->erase_child(this);
  _mem_logit[215][t]->erase_child(this);
  _mem_logit[216][t]->erase_child(this);
  _mem_logit[217][t]->erase_child(this);
  _mem_logit[218][t]->erase_child(this);
  _mem_logit[219][t]->erase_child(this);
  _mem_logit[21][t]->erase_child(this);
  _mem_logit[220][t]->erase_child(this);
  _mem_logit[221][t]->erase_child(this);
  _mem_logit[222][t]->erase_child(this);
  _mem_logit[223][t]->erase_child(this);
  _mem_logit[224][t]->erase_child(this);
  _mem_logit[225][t]->erase_child(this);
  _mem_logit[226][t]->erase_child(this);
  _mem_logit[227][t]->erase_child(this);
  _mem_logit[228][t]->erase_child(this);
  _mem_logit[229][t]->erase_child(this);
  _mem_logit[22][t]->erase_child(this);
  _mem_logit[230][t]->erase_child(this);
  _mem_logit[231][t]->erase_child(this);
  _mem_logit[232][t]->erase_child(this);
  _mem_logit[233][t]->erase_child(this);
  _mem_logit[234][t]->erase_child(this);
  _mem_logit[235][t]->erase_child(this);
  _mem_logit[236][t]->erase_child(this);
  _mem_logit[237][t]->erase_child(this);
  _mem_logit[238][t]->erase_child(this);
  _mem_logit[239][t]->erase_child(this);
  _mem_logit[23][t]->erase_child(this);
  _mem_logit[240][t]->erase_child(this);
  _mem_logit[241][t]->erase_child(this);
  _mem_logit[242][t]->erase_child(this);
  _mem_logit[243][t]->erase_child(this);
  _mem_logit[244][t]->erase_child(this);
  _mem_logit[245][t]->erase_child(this);
  _mem_logit[246][t]->erase_child(this);
  _mem_logit[247][t]->erase_child(this);
  _mem_logit[248][t]->erase_child(this);
  _mem_logit[249][t]->erase_child(this);
  _mem_logit[24][t]->erase_child(this);
  _mem_logit[250][t]->erase_child(this);
  _mem_logit[251][t]->erase_child(this);
  _mem_logit[252][t]->erase_child(this);
  _mem_logit[253][t]->erase_child(this);
  _mem_logit[254][t]->erase_child(this);
  _mem_logit[255][t]->erase_child(this);
  _mem_logit[256][t]->erase_child(this);
  _mem_logit[257][t]->erase_child(this);
  _mem_logit[258][t]->erase_child(this);
  _mem_logit[259][t]->erase_child(this);
  _mem_logit[25][t]->erase_child(this);
  _mem_logit[260][t]->erase_child(this);
  _mem_logit[261][t]->erase_child(this);
  _mem_logit[262][t]->erase_child(this);
  _mem_logit[263][t]->erase_child(this);
  _mem_logit[264][t]->erase_child(this);
  _mem_logit[265][t]->erase_child(this);
  _mem_logit[266][t]->erase_child(this);
  _mem_logit[267][t]->erase_child(this);
  _mem_logit[268][t]->erase_child(this);
  _mem_logit[269][t]->erase_child(this);
  _mem_logit[26][t]->erase_child(this);
  _mem_logit[270][t]->erase_child(this);
  _mem_logit[271][t]->erase_child(this);
  _mem_logit[272][t]->erase_child(this);
  _mem_logit[273][t]->erase_child(this);
  _mem_logit[274][t]->erase_child(this);
  _mem_logit[275][t]->erase_child(this);
  _mem_logit[276][t]->erase_child(this);
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
  _mem_logit[82][t]->erase_child(this);
  _mem_logit[83][t]->erase_child(this);
  _mem_logit[84][t]->erase_child(this);
  _mem_logit[85][t]->erase_child(this);
  _mem_logit[86][t]->erase_child(this);
  _mem_logit[87][t]->erase_child(this);
  _mem_logit[88][t]->erase_child(this);
  _mem_logit[89][t]->erase_child(this);
  _mem_logit[8][t]->erase_child(this);
  _mem_logit[90][t]->erase_child(this);
  _mem_logit[91][t]->erase_child(this);
  _mem_logit[92][t]->erase_child(this);
  _mem_logit[93][t]->erase_child(this);
  _mem_logit[94][t]->erase_child(this);
  _mem_logit[95][t]->erase_child(this);
  _mem_logit[96][t]->erase_child(this);
  _mem_logit[97][t]->erase_child(this);
  _mem_logit[98][t]->erase_child(this);
  _mem_logit[99][t]->erase_child(this);
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
