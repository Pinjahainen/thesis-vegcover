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

year = int(input('Enter year:'))
all = pd.DataFrame(pd.read_excel(r"file.xlsx"))
obs_orig = pd.DataFrame(pd.read_excel(r"file.xlsx"))
obs_orig=obs_orig.loc[obs_orig['vuosi']==year]
print(obs_orig)

all = fun.defwinter(all)
#get train and test samples from observations
all_orig, all_train, all_test = fun.traintest(all, obs_orig)

unknown = all_orig.loc[all_orig['winter']==10]
known = all_orig.loc[all_orig['winter']!=10]

true_predict = all_orig.loc[(all_orig['truth']>-1) & (all_orig['winter']!=10) & (all_orig['winter']<6)]

actual = true_predict['truth']
predicted = true_predict['winter']

cm_pers = metrics.confusion_matrix(actual, predicted, normalize='true')*100
cm_values = metrics.confusion_matrix(actual, predicted)
labels = ['Cultivated', 'Winter crop', 'Grass', 'Stubble from grain', 'Stubble from other crop', 'Stubble and companion plant']

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_pers, display_labels = ['Muokattu', 'Syysvilja', 'Nurmi', 'Viljasänki', 'Muu sänki', 'Sänki ja aluskasvi'])

cm_display.plot(xticks_rotation=45, cmap='Greens')
plt.show(block=False)

fun.cm_analysis(actual, predicted, r'file.jpg', [0,1,2,3,4,5], ymap=None, figsize=(22,20))

df = all_orig
counts = np.zeros((8,10))


for i in range(8):
    for j in range(10):
        newdata = df.loc[df['S0']==i]
        newdata = newdata.loc[newdata['S1']==j]
        n = len(newdata['field_id'])
        counts[i][j] = n


countsdf = pd.DataFrame(counts, columns=(['S1=0', 'S1=1', 'S1=2', 'S1=3', 'S1=4', 'S1=5', 'S1=6', 'S1=7', 'S1=8', 'S1=9']), index=(['S0=0', 'S0=1', 'S0=2', 'S0=3', 'S0=4', 'S0=5', 'S0=6', 'S0=7']))

print(countsdf)


winter_final = pd.DataFrame(pd.read_excel(r'file.xlsx'))    
w0 = winter_final.loc[0, :].values.flatten().tolist()
w1 = winter_final.loc[1, :].values.flatten().tolist()
w2 = winter_final.loc[2, :].values.flatten().tolist()
w3 = winter_final.loc[3, :].values.flatten().tolist()
w4 = winter_final.loc[4, :].values.flatten().tolist()
w5 = winter_final.loc[5, :].values.flatten().tolist()
w6 = winter_final.loc[6, :].values.flatten().tolist()


autumn_final = pd.DataFrame(pd.read_excel(r'file.xlsx'))

A0 = autumn_final.loc[0, :].values.flatten().tolist()
A1 = autumn_final.loc[1, :].values.flatten().tolist()
A2 = autumn_final.loc[2, :].values.flatten().tolist()

frq_S0, frq_S1 = fun.freq(all_orig)

# Defining Bayesian Structure
model = BayesianNetwork([('S1', 'S1_real'), ('WinterKill', 'S1_real'), ('S0', 'Winter'), ('S1_real', 'Winter'), ('Winter', 'Autumn'), ('Report', 'Autumn')])
sr = 0.979 # success rate of 
# Defining the CPDs:
cpd_WK = TabularCPD('WinterKill', 2, [[1-sr], [sr]])
cpd_S0 = TabularCPD('S0', len(frq_S0), frq_S0) 
cpd_S1 = TabularCPD('S1', len(frq_S1), frq_S1)
cpd_report = TabularCPD('Report', 2, [[0.5],[0.5]])
cpd_S1R = TabularCPD('S1_real', 10, [
                        ####
                        ####
                            [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # syysvilja
                            [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # kevätkylvöinen yksivuotinen
                            [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # aluskasvi
                            [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0], # muu kasvipeite
                            [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], # nurmi
                            [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0], # pysyvä kasvipeite
                            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0], # kumina
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0], # sänkikesanto
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0], # kevätkylvöinen ei suorakylvettävä
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]], # syysvilja tai kevätkylvöinen
                  evidence=['S1','WinterKill'], evidence_card=[len(frq_S1),2])
cpd_winter = TabularCPD('Winter', 7,
                            [w0, # Winter = 0 "Ploughed or Tillaged"
                            w1, # 1 "Winter crop"
                            w2,  # 2 "Grass"
                            w3, # 3 "Stubble from cereal"
                            w4, # 4 "Stubble from other crop"
                            w5, # 5 "Stubble and undersown plant"
                            w6], # 6 other
                            evidence=['S0', 'S1_real'], evidence_card=[len(frq_S0), 10])
cpd_autumn = TabularCPD('Autumn', 3,
                        [A0, A1, A2],
                        evidence = ['Winter', 'Report'], evidence_card=[7,2])


 
# Associating the CPDs with the network structure.
model.add_cpds(cpd_S0, cpd_S1, cpd_WK, cpd_S1R, cpd_winter, cpd_autumn, cpd_report)

model.check_model()

from pgmpy.inference import VariableElimination
 
infer = VariableElimination(model)
posterior_p = infer.query(variables=['Winter'], evidence={'S0': 0, 'S1': 4, 'Autumn':0, 'Report':1})
print(posterior_p)



df = unknown.reset_index()

df['onraportti'] = df['onraportti'].fillna(0)

df = df.astype({'onraportti': 'int64'})



n = len(df['S0'])-1
print(n)

nS0 = 9
nS1 = 11
nA = 3
nR = 2
s0 = df['S0']
s1 = df['S1']
autumn = df['autumn']
report = df['onraportti']


choises = list(range(nS0*nS1*nA*nR))

conditions = [(s0 == a) & (s1 == b) & (autumn == c) & (report == d) for a in range(nS0) for b in range(nS1) for c in range(nA) for d in range(nR)]

df['group'] = np.select(conditions, choises, 100000)


#n=7
n = len(choises)
print(n)
new_df = pd. DataFrame(columns=df.columns)
for i in range(n):
    #if i == 1165:
    #    i=1166
    group = df.loc[df['group']==i]
    if len(group['S0'])==0:
        pass
    else:
        a = group['S0'].iloc[-1]
        b = group['S1'].iloc[-1]
        c = group['autumn'].iloc[-1]
        d = group['onraportti'].iloc[-1]

        infer = VariableElimination(model)
        posterior = infer.query(variables=['Winter'], evidence={'S0': a, 'S1': b, 'Autumn': c, 'Report': d})
        elements = range(7)
        # S1 = 7,8 causes NaN, transform them to uniform distribution
        probabilities = pd.DataFrame(posterior.values)
        probabilities = probabilities.fillna(1/7)
    
        group['winter'] = np.random.choice(elements, len(group['winter']), p=list(probabilities[0]))
        new_df = pd.concat([new_df, group])



print(new_df)


concat_df = pd.concat([new_df,known])

concat_df = concat_df.drop(['level_0', 'index'], axis=1)
concat_df['truth'] = concat_df['truth'].fillna(10)

concat_df = concat_df.astype({'truth': 'int64'})
concat_df.reset_index()


true_predict11 = concat_df.loc[concat_df['truth']<10]

actual = true_predict11['truth']
predicted = list(true_predict11['winter'])


fun.cm_analysis(actual, predicted, r'file.jpg', [0,1,2,3,4,5], ymap=None, figsize=(22,20))

true_predict22 = new_df.loc[new_df['truth']<10]

actual22 = true_predict22['truth']
predicted22 = list(true_predict22['winter'])

fun.cm_analysis(actual22, predicted22, r'file.jpg', [0,1,2,3,4,5], ymap=None, figsize=(22,20))


traindf = all_train.loc[all_train['truth'].isin([0,1,2,3,4,5,6])]
traindf = traindf[['S0', 'S1', 'autumn', 'onraportti', 'truth']]
traindf = traindf.fillna(0)
traindf['WinterKill'] = 0
traindf['S1_real'] = traindf['S1']

traindf.rename(columns = {'onraportti':'Report', 'autumn':'Autumn', 'truth': 'Winter'}, inplace = True)
traindf

# train the model with simulated data based on in situ train set
model.fit_update(traindf)


df = unknown.reset_index()
df['onraportti'] = df['onraportti'].fillna(0)
df = df.astype({'onraportti': 'int64'})

n = len(df['S0'])-1
print(n)

nA = 3
nR = 2
s0 = df['S0']
s1 = df['S1']
autumn = df['autumn']
report = df['onraportti']

choises = list(range(nS0*nS1*nA*nR))

conditions = [(s0 == a) & (s1 == b) & (autumn == c) & (report == d) for a in range(nS0) for b in range(nS1) for c in range(nA) for d in range(nR)]

df['group'] = np.select(conditions, choises, 100000)




n = len(choises)
print(n)
new_df = pd. DataFrame(columns=df.columns)
for i in range(n):
    #if i == 1165:
    #    i=1166
    group = df.loc[df['group']==i]
    if len(group['S0'])==0:
        pass
    else:
        a = group['S0'].iloc[-1]
        b = group['S1'].iloc[-1]
        c = group['autumn'].iloc[-1]
        d = group['onraportti'].iloc[-1]

        infer = VariableElimination(model)
        posterior = infer.query(variables=['Winter'], evidence={'S0': a, 'S1': b, 'Autumn': c, 'Report': d})
        elements = range(7)
        # S1 = 7,8 causes NaN, transform them to uniform distribution
        probabilities = pd.DataFrame(posterior.values)
        probabilities = probabilities.fillna(1/7)
    
        group['winter'] = np.random.choice(elements, len(group['winter']), p=list(probabilities[0]))
        new_df = pd.concat([new_df, group])



concat_df = pd.concat([new_df,known])

concat_df = concat_df.drop(['level_0', 'index'], axis=1)
concat_df['truth'] = concat_df['truth'].fillna(10)

concat_df = concat_df.astype({'truth': 'int64'})
concat_df.reset_index()

true_predict11 = concat_df.loc[concat_df['truth']<10]

actual1 = true_predict11['truth']
predicted1 = list(true_predict11['winter'])

fun.cm_analysis(actual1, predicted1, r'file.jpg', [0,1,2,3,4,5], ymap=None, figsize=(22,20))

true_predict22 = new_df.loc[new_df['truth']<10]

actual22 = true_predict22['truth']
predicted22 = list(true_predict22['winter'])

fun.cm_analysis(actual22, predicted22, r'file.jpg', [0,1,2,3,4,5], ymap=None, figsize=(22,20))


print(sklearn.metrics.balanced_accuracy_score(actual,predicted))


print("Kokonaisala (a):")
print(concat_df['kokonaisala'].sum())

print("Muokatut:")
print(concat_df.loc[concat_df['winter']==0]["kokonaisala"].sum())
print(concat_df.loc[concat_df['winter']==0]["kokonaisala"].sum()/concat_df['kokonaisala'].sum())

print("Syysvilja:")
print(concat_df.loc[concat_df['winter']==1]["kokonaisala"].sum())
print(concat_df.loc[concat_df['winter']==1]["kokonaisala"].sum()/concat_df['kokonaisala'].sum())

print("Nurmi:")
print(concat_df.loc[concat_df['winter']==2]["kokonaisala"].sum())
print(concat_df.loc[concat_df['winter']==2]["kokonaisala"].sum()/concat_df['kokonaisala'].sum())

print("Viljasänki:")
print(concat_df.loc[concat_df['winter']==3]["kokonaisala"].sum())
print(concat_df.loc[concat_df['winter']==3]["kokonaisala"].sum()/concat_df['kokonaisala'].sum())

print("Muu sänki:")
print(concat_df.loc[concat_df['winter']==4]["kokonaisala"].sum())
print(concat_df.loc[concat_df['winter']==4]["kokonaisala"].sum()/concat_df['kokonaisala'].sum())

print("Sänki ja aluskasvi:")
print(concat_df.loc[concat_df['winter']==5]["kokonaisala"].sum())
print(concat_df.loc[concat_df['winter']==5]["kokonaisala"].sum()/concat_df['kokonaisala'].sum())

print("Muu:")
print(concat_df.loc[concat_df['winter']>5]["kokonaisala"].sum())
print(concat_df.loc[concat_df['winter']>5]["kokonaisala"].sum()/concat_df['kokonaisala'].sum())
