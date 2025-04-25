import oracledb
import pandas as pd
import numpy as np
import collections

def year0(user, pw, year, cs):
    with oracledb.connect(user=user, password=pw, dsn=cs) as conn:
        with conn.cursor() as c:
            c.execute("""SELECT DISTINCT K.PLVUOSIILM_TUNNUS, P.PINTAALA, K.VUOSI, K.MAATILA_TUNNUS, P.PLVUOSI_PERUSLOHKOTUNNUS, K.TUNNUS, K.KASVULOHKOTUNNUS, K.PINTAALA, K.PYSYVANNURMENPINTAALA, T.PINTAALA, T.KEVYTMUOKATTUPINTAALA, KVI_KASVIKOODI, T.PAIPVM
            FROM MAATILAMGR.MLA_KASVULOHKOILMOITUS K 
            left join MAATILAMGR.MLA_PERUSLOHKOVUOSIILMOITUS P 
            ON K.PLVUOSIILM_TUNNUS = P.TUNNUS AND K.VUOSI = P.PLVUOSI_VUOSI
            LEFT JOIN MAATILAMGR.MLA_PERUSLOHKOTOIMENPIDEILM T
            ON T.PL_PERUSLOHKOTUNNUS = P.PLVUOSI_PERUSLOHKOTUNNUS and T.VUOSI = P. PLVUOSI_VUOSI
            WHERE K.VUOSI = :yr AND (T.SITOIM_KOODI = 48 OR T.SITOIM_KOODI IS NULL) AND K.ASIANTILA > 0""", [year-1]) 
            data1 = c.fetchall()
    return data1
    

def year1(user, pw, year, cs):
    with oracledb.connect(user=user, password=pw, dsn=cs) as conn:
        with conn.cursor() as c:
            c.execute("""SELECT DISTINCT K.PLVUOSIILM_TUNNUS, K.VUOSI, K.MAATILA_TUNNUS, P.PLVUOSI_PERUSLOHKOTUNNUS, K.TUNNUS, K.KASVULOHKOTUNNUS, K.PINTAALA, K.PYSYVANNURMENPINTAALA, KVI_KASVIKOODI
            FROM MAATILAMGR.MLA_KASVULOHKOILMOITUS K 
            left join MAATILAMGR.MLA_PERUSLOHKOVUOSIILMOITUS P 
            ON K.PLVUOSIILM_TUNNUS = P.TUNNUS AND K.VUOSI = P.PLVUOSI_VUOSI
            WHERE K.VUOSI = :yr AND K.ASIANTILA > 0""", [year]) 
            data2 = c.fetchall()
    return data2
    

def report(user, pw, year, cs):
    # mark the farms which has done the autumn report:
    with oracledb.connect(user=user, password=pw, dsn=cs) as conn:
        with conn.cursor() as c:
            c.execute("""SELECT DISTINCT M.VUOSI, M.MAATILA_TUNNUS 
            FROM MAATILAMGR.MLA_PERUSLOHKOTOIMENPIDEILM M 
            WHERE M.VUOSI=:yr""", [year-1]) 
            report = pd.DataFrame(c.fetchall())
    report.columns = ['year0', 'farm_id']
    report = report.astype({'farm_id': 'int64'})
    report['onraportti'] = 1
    return report



def autumn(all):
    # input: all S0 and S1 data merged
    # output: all S0 and S1 data merged with autumn classification
    pysyva1 = all['grass_area1']
    pysyva2 = all['grass_area2']
    kasvipeite = all['veg_area']
    kevytmuokattu = all['tillage_area']
    kokonaisala = all['kokonaisala']
    ala = all['area_x']

    conditions = [(kasvipeite > kokonaisala*0.98)
            , (kevytmuokattu > kokonaisala*0.98)
            , (pysyva1 > ala*0.98) & (pysyva2 > ala*0.98)]
    choises = [2,1,2]

    all['autumn'] = np.select(conditions, choises, 0)


    d_s0 = {0:0,1:1,2:0,21:0,3:1,4:4,5:2,6:2,7:5,9:3,10:2,11:3,13:100,15:0,16:6}
    d_s1 = {0:0,1:0,2:1,21:8,3:1,4:7,5:2,6:4,7:3,9:5,10:3,11:5,13:100,15:6,16:9}

    all['S0'] = all['class iacs1'].map(d_s0)
    all['S1'] = all['class iacs2'].map(d_s1)

    return all



def defwinter(all):
    # input: all S0 and S1 data merged
    # output: all S0 and S1 data merged with definite winter classes
    autumn = all['autumn']
    s0 = all['S0']
    s1 = all['S1']
    pysyva1 = all['grass_area1']
    pysyva2 = all['grass_area2']
    kasvipeite = all['veg_area']
    kevytmuokattu = all['tillage_area']
    iacsx = all['class iacs1']
    iacsy = all['class iacs2']

    conditions = [(s1 == 100) | (s0 == 100) # kasvihuone, merkataan talvipeitteiden ulkopuolelle = 100
              , (s0 == 5) & (s1.isin([3,5])) & (autumn == 2) # s0 6 muu kasvipeite ja s1 pysyvä tai muu kasvipeite = talvi 6 muu kasvipeite
              , (s0 == 3) & (s1.isin([1,3,5,6,7,8])) & (autumn == 2) #s0 4 pysyvä ja s1 ei syysvilja tai nurmi: talvi 6 muu kasvipeite
              , (s0 == 3) & (s1 == 5) # s0 pysyvä ja s1 pysyvä = talvi 6 muu kasvip
              , (s0 == 2) & (s1.isin([1,2,4,7])) & (autumn == 2) # s0 2 nurmi ja s1 ei syysvilja, muu kasvipeite, sänki tjsp = talvi 2 nurmi
              , (s0 == 1) & (s1.isin([1,6,7])) & (autumn == 2) # 
              , (s0 == 1) & (s1 == 6) # s0 1 muu viljelysk ja s1 7 sänki = talvi 4 muu sänki
              , (s0 == 2) & (s1 == 6) # s0 nurmi ja s1 sänki = talvi 2 nurmi
              , (s0 == 0) & (s1 == 6) # s0 vilja ja s1 sänki = talvi 3 viljasänki
              , (s0 == 0) & (s1.isin([1,6,7])) & (autumn == 2)
              , (pysyva1 > 0) & (pysyva2 > 0) # ilmoitettu pysyvä nurmipeite = talviluokka 3 nurmi
              , (s1 == 0) # kevät 1 syysvilja = talviluokka 2 syysvilja
              , (s1 == 6) & (s0.isin([1,2,3,4,5,6])) & (autumn == 2) # kevät 1 sänki, kevät 0 muu kuin vilja = talviluokka 5 muu sänki
              , (autumn == 1) # ilmoitettu kevytmuokatuksi = talviluokka 0 muokattu
              , (s0 == 4) & (s1 == 8) & (autumn == 2) # sadonkorjuussa paljas maa, kevät 1 kevät- tai syysvilja ja syysilmoitus 2 kasvipeite = talvi 1 syysvilja
              ]
    choises = [100,6,6,6,2,4,4,2,3,3,2,1,4,0,1]

    all['winter'] = np.select(conditions, choises, 10)
    return all

def traintest(all, obs_orig):
    # split the observations to train and test sets
    obs = obs_orig.sample(frac=0.76, replace=True, random_state=1) #training data
    ids_in_trainset = obs.parcel_id.unique()
    obs_test = obs_orig[~obs_orig['parcel_id'].isin(ids_in_trainset)] #test data

    obs.columns = ['year1', 'farm_id', 'field_id', 'parcel_id_y', 'truth']
    obs_orig.columns = ['year1', 'farm_id', 'field_id', 'parcel_id_y', 'truth']
    obs_test.columns = ['year1', 'farm_id', 'field_id', 'parcel_id_y', 'truth']
    all_train = pd.merge(all, obs, on = ['year1', 'farm_id', 'field_id', 'parcel_id_y'], how='left') # only train set observations
    all_orig = pd.merge(all, obs_orig, on = ['year1', 'farm_id', 'field_id', 'parcel_id_y'], how='left') # all the observations
    all_test = pd.merge(all, obs_test, on = ['year1', 'farm_id', 'field_id', 'parcel_id_y'], how='left') # only test set observations
    return all_orig, all_train, all_test

import sys

def extractDigits(lst):
    return [[el] for el in lst]
                 

def freq(all):
    s = all.S0
    counter = collections.Counter(s)
    frq_s0 = list([counter[x]/len(s) for x in sorted(counter.keys())])

    s = all.S1
    counter = collections.Counter(s)
    frq_s1 = list([counter[x]/len(s) for x in sorted(counter.keys())])

    return extractDigits(frq_s0), extractDigits(frq_s1)



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix



def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10), plt_title=None):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
      ### my addition:
      plt_title: Plot title, can be: None, dict or string. If None, accuracy_score is given as title.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    
    ### added plot title:
    if plt_title is None: ### Automatic title is accuracy_score
        title_str = "Acc={:.2f}%".format(accuracy_score(y_true, y_pred)*100)
    elif type(plt_title) is dict: ### but you can make your own dictionary of metrics names and values, e.g. {'acc': 0.8, 'f1':0.75, etc.}
        title_str =  ""
        for kk in plt_title.keys():
            title_str = title_str + kk + "={:.2f}%, ".format(plt_title[kk]*100)
        title_str = title_str[:-2]
    elif type(plt_title) is str: ### or you can make your own string
        title_str = plt_title
    else: ### Error otherwise
        print("Title format suported are string and dict!")
        assert False
    plt.title(title_str, fontsize=40)
    
    sns.set(font_scale=2.5) ### adaptive fontsize
    sns.heatmap(cm_perc, annot=annot, fmt='', ax=ax, cmap="Greens") ### options: YlGnBu, jet, summer
    plt.savefig(filename)
    plt.show(block=False)
