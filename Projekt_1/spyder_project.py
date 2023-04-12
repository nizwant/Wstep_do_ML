#%% Zaimportowanie pakietów
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import date

#%% Wczytanie danych
df = pd.read_csv('data/Android_Malware.csv')
df.head()

#%% Dzielenie na train/test/ walidacje
X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(
    df[df.columns.values[:-1]],
    df[df.columns.values[-1]],
    test_size=0.4, random_state=213)

X_test, X_validation, y_test, y_validation = train_test_split(
    X_test_tmp,
    y_test_tmp,
    test_size=0.5, random_state=7)

#%% Do zamiany wszytskich df na X_train
df = X_train 
df["Label"] = y_train

df.info()

#%% jak wygladaja najpopularniejsze wartosci w kazdej z kolumn
for col in df.columns:
    print(f"Top 10 most common values in {col}:")
    print(df[col].value_counts().nlargest(10))
    print()
    
#%% ilosc unikalnych wartosci w kazdej z kolumn    
with pd.option_context('display.max_rows', None):
    print(df.nunique(dropna=False))

#%% rozklad przewidywanej cechy
df.iloc[:, -1].hist()

"""
"Adware - niechciane oprogramowanie służące do wyświetlania reklam na Twoim ekranie".
Adware generuje przychody dla swoich twórców dzięki automatycznemu wyświetlaniu
internetowych reklam w interfejsie użytkownika danej aplikacji lub w formie okienek 
wyskakujących podczas procesu instalacji.

Scareware - type of malware attack that claims to have detected a virus or 
other issue on a device and directs the user to download or buy malicious software 
to resolve the problem. Generally speaking, scareware is the gateway to a more 
intricate cyberattack and not an attack in and of itself.

Benign - having no harmful influence or effect, safe

SMS malware - any malicious software delivered to victims by text messaging. 
While malware may be delivered to mobile devices via email and many other means, 
SMS malware is promoted by a text.
"""

#%% Usunięcie kolumn ID, i dwóch kolumn w których są same zera ale zapisane na różne sposoby (int, str, float)  
df = df.drop(columns = ["Unnamed: 0", "Flow ID", " CWE Flag Count", "Fwd Avg Bytes/Bulk"])

# Ta kolumna ma wartości które są int i float, zrzutujemy je na int  
df[" Down/Up Ratio"] = df[" Down/Up Ratio"].astype('int')
    
#%% Kodowanie IP
    
le = LabelEncoder()
df[" Source IP"] = le.fit_transform(df[" Source IP"])
df[" Destination IP"] = le.fit_transform(df[" Destination IP"])

# plan na binarne kodowanie + zmienic dodac adresy sieci

#%% Kodowanie protokołu, niestety dane nie są nigdzie wyjaśnione więc nie wiemy która 
#liczba co znaczy, można jednak użyć OneHotEncoder, trochę to powoduje że nie ma 
#znaczenie ze nie wiemy co to znaczy

one_hot = pd.get_dummies(df[' Protocol'])
df = df.drop(' Protocol',axis = 1)
df = df.join(one_hot)

#%% parametryzacja dat

df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], format="%d/%m/%Y %H:%M:%S")
df['month'] = df[' Timestamp'].dt.month
df['day_of_week'] = df[' Timestamp'].dt.dayofweek
df['day'] = df[' Timestamp'].dt.day
df['hour'] = df[' Timestamp'].dt.hour
df['minutes'] = df[' Timestamp'].dt.minute
df = df.drop(columns = [' Timestamp'])

#%% kodowanie Y
mapping_dict = {"Benign": 1,
                "Android_Scareware":2,
                "Android_Adware": 3,
                "Android_SMS_Malware":4}

df["Label"] = pd.Series(df["Label"]).map(mapping_dict)

#%% Korelacja kolumn

dataplot=sb.heatmap(df.corr())
plt.figure(figsize=(24,19))
plt.show()

#%% Dodanie kolumn dla każdaj zmiennej true == moda false w przeciwnym przypadku. 
#Mamy kolumny gdzie 80/90% rekordów to jedna wartość a inne występują po 3/4 razy. 
#Zadnych wartościowych informacji z tego nie ma ale z takich boolowskich kolumn 
#może da się wyciągnąć więcej

cols_to_loop_over = df.columns
for col in cols_to_loop_over:
    mode = df[col].mode()[0]
    df[str(col)+"_is_mode"] = np.where(df[col] == mode, 1, 0)

# Jeszcze raz korelacja 

dataplot=sb.heatmap(df.corr())
plt.figure(figsize=(24,19))
plt.show()

#%% Usuwanie kolumn skorelowanych, możliwe że pominiemy ten krok, gdyż planujemy korzystać z lasów losowych 

def select_correlated_columns_to_remove(df, corr_treshold):
    cor_lis = []
    cor = df.corr()
    for i, _ in enumerate(cor.columns):
        for j, col_name_to_drop in enumerate(cor.columns):
            if i < j and (abs(cor.iloc[i,j]) > corr_treshold or math.isnan(cor.iloc[i,j])):
                cor_lis.append(col_name_to_drop)
    return cor_lis

cor_list = select_correlated_columns_to_remove(df, 0.9)
df = df.drop(cor_list, axis = 1)
df

#%%Liczenie gini dla pojedynczych kolumn i usuwanie tych o niskiej predykcyjności

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

df_sample = df.sample(10000) #próbka oszacować gini. Dla 100 000 za długo się liczy
lower_gini_list = []
cor2 = df.corr()
for i in range(len(cor2.columns)):
    for j in range(len(cor2.columns)):
        if (i < j and abs(cor2.iloc[i,j]) > 0.7):
            gini_i = gini(df_sample.iloc[:,i])
            gini_j = gini(df_sample.iloc[:,j])
            if (gini_i > gini_j): 
                lower_gini_list.append(df.columns[j])
            else: 
                lower_gini_list.append(df.columns[i])
                
df = df.drop(lower_gini_list, axis = 1)
df

#%% Usuwanie kolumn o małej wariancji 

df.columns = [str(i) for i in df.columns]
from sklearn.feature_selection import VarianceThreshold
threshold = 0.2
sel = VarianceThreshold(threshold=threshold)
sel.fit_transform(df)


#%%

from xgboost import XGBClassifier # Inna paczka niż sklearn!





model=XGBClassifier(random_state=1,
                    learning_rate=0.01, # Szybkość "uczenia" się
                    booster='gbtree', # Jaki model wykorzystujemy (drzewo - gbtree, liniowe - gblinear)
                    max_depth=4 # Maksymalna głębokość drzewa 
                    )
model.fit(X_train, y_train)
model.score(X_test,y_test)