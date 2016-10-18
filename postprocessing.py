
# coding: utf-8

# In[1]:

import json
import pickle
import re
import sys
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
        print("Need to specify output file.")
        sys.exit()
    OUTPUT_FILE = sys.argv[1]
else:
    OUTPUT_FILE = 'out/output_small.txt'


# #### Parse out the county-level flu rates.

# In[13]:

pattern = r"query : county_rate\(County\[(\d+)\], Week\[(\d+)\]\)\n{2}Mean = (\d+\.\d+)\s"

with open(OUTPUT_FILE, "r") as output_file:
    searches = re.findall(pattern, output_file.read())


# In[18]:

pattern = r"query : county_rate\(County\[(\d+)\], Week\[(\d+)\]\)\n{2}Mean = -(\d+\.\d+)\s"

with open(OUTPUT_FILE, "r") as output_file:
    neg_searches = re.findall(pattern, output_file.read())


# In[19]:

searches = searches + neg_searches


# In[5]:

predictions = {}
for q in searches:
    predictions[int(q[0]), int(q[1])] = float(q[2])


# In[6]:

with open("log/index_to_county.pickle", "rb") as picklefile:
    index_to_county = pickle.load(picklefile)


# In[7]:

with open("log/dates.pickle", "rb") as picklefile:
    dates = pickle.load(picklefile)


# #### Write output JSON.

# In[70]:

output_dict = {}


# In[71]:

for (i, fips) in index_to_county.items():
    county_dict = {
        "ILI percentage %": {}
    }
    for j, t in enumerate(dates):
        county_dict["ILI percentage %"][t] = predictions[(i,j)] * 100.0
    output_dict[fips] = county_dict


# In[77]:

with open("out/CountyWeeklyILI.json", "w") as jsonfile:
    jsonfile.write(json.dumps(output_dict))


# #### Evaluate

# In[ ]:



