#load required packages
import getpass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import pylab as plt
import functions2025 as fun

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import pylab as plt
import sklearn

from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Year and database details:
year = int(input('Enter year:'))
user = input('Enter username:')
pw = getpass.getpass("Enter password: ")
cs = 'database-details' # to connect the correct database. Not available in public.
    
# load additional files (not in database):

# observations, select year
obs_orig = pd.DataFrame(pd.read_excel(r"file.xlsx")) # Observations include the in situ data in the form of year, parcel number and the observed and augmented winter class
obs_orig=obs_orig.loc[obs_orig['vuosi']==year]

# classification of iacs codes. Rename columns and set correct datatype:
classes = pd.DataFrame(pd.read_table(r'file.tsv', sep='\t', header=None))
classes.columns = ['iacs', 'name', 'class iacs']
classes = classes.astype({'iacs': 'int64'})
print(obs_orig)
print(classes)


# get the data from database separately for previous and following year:
data1 = fun.year0(user, pw, year, cs)
data2 = fun.year1(user, pw, year, cs)


# set column names
df1 = pd.DataFrame(data1, columns=(['vuosi0tunnus', 'kokonaisala', 'year0', 'farm_id', 'field_id', 'parcel_id', 'parcel_name', 'area', 'grass_area1', 'veg_area', 'tillage_area', 'iacs', 'date']))
df2 = pd.DataFrame(data2, columns=(['vuosi1tunnus','year1', 'farm_id', 'field_id', 'parcel_id', 'parcel_name', 'area', 'grass_area2', 'iacs']))

# collect only parcels that are present each year:
counted = pd.DataFrame(df1['field_id'].value_counts())    

all2 = df2[df2['field_id'].isin(list(counted.index.values) )]

all1 = pd.DataFrame(df1)
#, columns=['year1', 'MAATILA_TUNNUS','PLVUOSI_PERUSLOHKOTUNNUS', 'KASVULOHKOTUNNUS', 'PINTAALA_x', 'permanent grass PINTAALA_x', 'PINTAALA.1', 'KEVYTMUOKATTUPINTAALA', 'iacs'])
all2 = pd.DataFrame(all2)

all1['iacs'] = all1['iacs'].fillna(0)
all2['iacs'] = all2['iacs'].fillna(0)

all1 = all1[all1['iacs'] != 0]
all2 = all2[all2['iacs'] != 0]

all1 = all1.astype({'iacs': 'int64', 'field_id': 'int64', 'farm_id': 'int64'})
all2 = all2.astype({'iacs': 'int64', 'field_id': 'int64', 'farm_id': 'int64'})
classes = classes.astype({'class iacs': 'int64'})

all1 = pd.merge(all1, classes, on=['iacs'], how = 'left')
all2 = pd.merge(all2, classes, on=['iacs'], how = 'left')

all1 = all1.rename(columns={'iacs':'iacs1', 'class iacs': 'class iacs1'})
all2 = all2.rename(columns={'iacs':'iacs2', 'class iacs': 'class iacs2'})

# merge two datasets
all = pd.merge(all1, all2, on=['farm_id', 'field_id', 'parcel_name'], how='inner')

# set correct formats to variables
all['veg_area']  = pd.to_numeric(all['veg_area'] , errors='ignore').astype('float')
all['tillage_area']  = pd.to_numeric(all['tillage_area'] , errors='ignore').astype('float')
all['grass_area1']  = pd.to_numeric(all['grass_area1'] , errors='ignore').astype('float')
all['grass_area2']  = pd.to_numeric(all['grass_area2'] , errors='ignore').astype('float')

print(all.dtypes)


report = fun.report(user, pw, year, cs)
all = pd.merge(all, report, on = ['year0', 'farm_id'], how='left')

dup = all.loc[all.duplicated(subset=['field_id', 'parcel_name'], keep=False)]
dupfields = dup["field_id"].unique()
print(dup)

all = all.apply(lambda row: row[~all['field_id'].isin(dupfields)])
all = all.reset_index()
dup = all.loc[all.duplicated(subset=['field_id', 'parcel_name'], keep=False)]

print(dup)

# set autumn classes

all = fun.autumn(all)

all.to_excel(r'file.xlsx')


print(all.loc[all['autumn']!=0])

