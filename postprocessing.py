
# coding: utf-8

# In[989]:

import csv
import json
import numpy as np
import pickle
import re
import sys
import matplotlib.pyplot as plt
from collections import defaultdict


# In[990]:

def is_kernel():
    if 'IPython' not in sys.modules:
        return False
    from IPython import get_ipython
    return getattr(get_ipython(), 'kernel', None) is not None


# In[991]:

if not is_kernel():
    if len(sys.argv) <= 1:
        print("Need to specify input size.")
        sys.exit()
    INPUT_SIZE = sys.argv[1]
else:
    INPUT_SIZE = "Small"


# **Load previously saved data.**

# In[992]:

with open("log/index_to_county.pickle", "rb") as picklefile:
    index_to_county = pickle.load(picklefile)


# In[993]:

with open("log/dates.pickle", "rb") as picklefile:
    dates = pickle.load(picklefile)


# In[994]:

with open("log/index_to_region.pickle", "rb") as picklefile:
    index_to_region = pickle.load(picklefile)


# #### Parse out the county-level flu rates.

# In[995]:

OUTPUT_FILE = "out/output_%s.txt" % INPUT_SIZE.lower()


# In[996]:

pos_pattern = r"query : y\(County\[(\d+)\], Week\[(\d+)\]\)\n{2}Mean = (\d+\.\d+)\s"
neg_pattern = r"query : y\(County\[(\d+)\], Week\[(\d+)\]\)\n{2}Mean = -(\d+\.\d+)\s"

with open(OUTPUT_FILE, "r") as output_file:
    pos_searches = re.findall(pos_pattern, output_file.read())
    output_file.seek(0)
    neg_searches = re.findall(neg_pattern, output_file.read())


# In[997]:

correlations = {}
for q in pos_searches:
    correlations[int(q[0]), int(q[1])] = 1.0 * float(q[2])
for q in neg_searches:
    correlations[int(q[0]), int(q[1])] = -1.0 * float(q[2])


# In[998]:

corr_matrix = np.zeros((len(index_to_county), len(dates)))


# In[999]:

for i in range(len(dates)):
  for j in range(len(index_to_county)):
    corr_matrix[j][i] = correlations[(j,i)]


# In[1000]:

corr_matrix.shape


# In[1001]:

pos_pattern = r"query : logit\(County\[(\d+)\], Week\[(\d+)\]\)\n{2}Mean = (\d+\.\d+)\s"
neg_pattern = r"query : logit\(County\[(\d+)\], Week\[(\d+)\]\)\n{2}Mean = -(\d+\.\d+)\s"

with open(OUTPUT_FILE, "r") as output_file:
    pos_searches = re.findall(pos_pattern, output_file.read())
    output_file.seek(0)
    neg_searches = re.findall(neg_pattern, output_file.read())


# In[1002]:

predictions = {}
for q in pos_searches:
    predictions[int(q[0]), int(q[1])] = 1.0 * float(q[2])
for q in neg_searches:
    predictions[int(q[0]), int(q[1])] = -1.0 * float(q[2])


# In[1003]:

for k, v in predictions.items():
  v = 1.0 / (1.0 + np.exp(-1.0 * v))
  predictions[k] = v


# #### Write output JSON.

# In[1004]:

output_dict = {}


# In[1005]:

for (i, fips) in index_to_county.items():
    county_dict = {
        "ILI percentage %": {}
    }
    for j, t in enumerate(dates):
        county_dict["ILI percentage %"][t] = predictions[(i,j)] * 100.0
    output_dict[fips] = county_dict


# In[1006]:

with open("out/%s/CountyWeeklyILI.json" % INPUT_SIZE, "w") as jsonfile:
    jsonfile.write(json.dumps(output_dict))


# #### Evaluate

# In[1007]:

eval_data = []


# In[1008]:

with open("data/%s/eval/Flu_ILI_TRUTH.csv" % INPUT_SIZE, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        eval_data.append(row)


# In[1009]:

county_map_matrix = np.loadtxt("./data_processed/county_map.txt")
region_pop_matrix = np.loadtxt("./data_processed/region_pops.txt")


# In[1010]:

print(county_map_matrix.shape)
print(region_pop_matrix.shape)


# **Graphs**

# In[1011]:

history = []
county_level_history = []


# In[1012]:

loss = 0.0


# In[1013]:

for t, date in enumerate(dates):
    county_vector = np.array([predictions[(i, t)] for i in range(len(index_to_county))])
    region_rates = np.dot(county_map_matrix, county_vector)
    region_rates = region_rates / region_pop_matrix
    history.append(region_rates)
    county_level_history.append(county_vector)
    for i, predicted_rate in enumerate(region_rates):
        if eval_data[t][index_to_region[i]] != 'NaN':
            loss += region_pop_matrix[i] * (predicted_rate * 100 - float(eval_data[t][index_to_region[i]][0:-1]))**2


# In[1014]:

print("Total loss:", loss)


# In[1015]:

print("MSE:", loss / np.sum(region_pop_matrix) / np.sum(len(dates)))


# In[1016]:

print("RMSE:", (loss / np.sum(region_pop_matrix) / np.sum(len(dates)))**0.5)


# In[1060]:

sigmoid = lambda x: 1 / (1 + np.exp(-x))


# In[1084]:

logit = lambda y: -1 * np.log((1 - y) / y)


# In[1051]:

obs = np.loadtxt("data_processed/obs.txt")


# In[1052]:

plt.figure(figsize=(18, 6))
plt.subplot(1,2,1)
plt.plot(np.array(history)[:50,:])
plt.ylim([0, np.max(history)])
plt.title("Inferred region rates.")
plt.subplot(1,2,2)
plt.plot(obs[:50,:])
plt.ylim([0, np.max(history)])
plt.title("Observed region rates.")
plt.savefig("out/%s/region_rates.png" % INPUT_SIZE)


# In[1085]:

ind = np.random.randint(0, len(index_to_county))


# In[1086]:

priors = np.loadtxt("data_processed/priors.txt")


# In[1094]:

plt.figure(figsize=(12,12))
plt.subplot(3,2,1)
plt.plot(corr_matrix[ind,:50], '.g')
plt.plot(logit(np.array(county_level_history)[ind,:50]), '.b')
plt.title("Inferred county logits.")
plt.subplot(3,2,2)
plt.plot(priors[ind,:50], '.r')
plt.title("Prior county logits.")
plt.subplot(3,2,3)
plt.plot(np.array(county_level_history)[ind,:50], '.b')
plt.title("Inferred county rates.")
plt.subplot(3,2,4)
plt.plot(sigmoid(priors[ind,:50]), '.r')
plt.title("Observed county rates.")
plt.savefig("out/%s/county_rates.png" % INPUT_SIZE)


# In[ ]:



