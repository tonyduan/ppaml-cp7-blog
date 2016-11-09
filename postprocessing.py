
# coding: utf-8

# In[1]:

import csv
import json
import numpy as np
import pickle
import re
import sys
import matplotlib.pyplot as plt
from collections import defaultdict


# In[2]:

def is_kernel():
    if 'IPython' not in sys.modules:
        return False
    from IPython import get_ipython
    return getattr(get_ipython(), 'kernel', None) is not None


# In[3]:

if not is_kernel():
    if len(sys.argv) <= 1:
        print("Need to specify input size.")
        sys.exit()
    INPUT_SIZE = sys.argv[1]
else:
    INPUT_SIZE = "Small"


# #### Parse out the county-level flu rates.

# In[54]:

OUTPUT_FILE = "out/output_%s.txt" % INPUT_SIZE.lower()


# In[55]:

pos_pattern = r"query : logit\(County\[(\d+)\], Week\[(\d+)\]\)\n{2}Mean = (\d+\.\d+)\s"
neg_pattern = r"query : logit\(County\[(\d+)\], Week\[(\d+)\]\)\n{2}Mean = -(\d+\.\d+)\s"

with open(OUTPUT_FILE, "r") as output_file:
    pos_searches = re.findall(pos_pattern, output_file.read())
    output_file.seek(0)
    neg_searches = re.findall(neg_pattern, output_file.read())


# In[56]:

predictions = {}
for q in pos_searches:
    predictions[int(q[0]), int(q[1])] = 1.0 * float(q[2])
for q in neg_searches:
    predictions[int(q[0]), int(q[1])] = -1.0 * float(q[2])


# In[57]:

for k, v in predictions.items():
  v = 1.0 / (1.0 + np.exp(-1.0 * v))
  if v < 0:
    predictions[k] = 0
  if v > 0.5:
    predictions[k] = 0
  else:
    predictions[k] = v


# In[58]:

with open("log/index_to_county.pickle", "rb") as picklefile:
    index_to_county = pickle.load(picklefile)


# In[59]:

with open("log/dates.pickle", "rb") as picklefile:
    dates = pickle.load(picklefile)


# In[60]:

with open("log/index_to_region.pickle", "rb") as picklefile:
    index_to_region = pickle.load(picklefile)


# #### Write output JSON.

# In[61]:

output_dict = {}


# In[62]:

for (i, fips) in index_to_county.items():
    county_dict = {
        "ILI percentage %": {}
    }
    for j, t in enumerate(dates):
        county_dict["ILI percentage %"][t] = predictions[(i,j)] * 100.0
    output_dict[fips] = county_dict


# In[63]:

with open("out/%s/CountyWeeklyILI.json" % INPUT_SIZE, "w") as jsonfile:
    jsonfile.write(json.dumps(output_dict))


# #### Evaluate

# In[64]:

eval_data = []


# In[65]:

with open("data/%s/eval/Flu_ILI_TRUTH.csv" % INPUT_SIZE, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        eval_data.append(row)


# In[66]:

county_map_matrix = np.loadtxt("./data_processed/county_map.txt")
region_pop_matrix = np.loadtxt("./data_processed/region_pops.txt")


# In[67]:

print(county_map_matrix.shape)
print(region_pop_matrix.shape)


# **Graphs**

# In[68]:

history = []
county_level_history = []


# In[69]:

loss = 0.0


# In[70]:

for t, date in enumerate(dates):
    county_vector = [predictions[(i, t)] for i in index_to_county.keys()]
    region_rates = np.dot(county_map_matrix, county_vector) / np.sum(county_map_matrix, axis = 1)
    history.append(region_rates)
    county_level_history.append(county_vector)
    for i, predicted_rate in enumerate(region_rates):
        if eval_data[t][index_to_region[i]] != 'NaN':
            loss += region_pop_matrix[i] * (predicted_rate * 100 - float(eval_data[t][index_to_region[i]][0:-1]))**2


# In[71]:

print("Total loss:", loss)


# In[72]:

print("MSE:", loss / np.sum(region_pop_matrix) / np.sum(len(dates)))


# In[73]:

print("RMSE:", (loss / np.sum(region_pop_matrix) / np.sum(len(dates)))**0.5)


# In[74]:

plt.figure()
plt.plot(np.array(history)[:50,:])
plt.savefig("out/%s/output.png" % INPUT_SIZE)


# In[75]:

obs = np.loadtxt('../cp7/data_processed/obs.txt')


# In[76]:

plt.figure()
plt.plot(obs[:50,:])
plt.savefig("out/%s/ref.png" % INPUT_SIZE)


# In[ ]:



