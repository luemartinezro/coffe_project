# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
from unidecode import unidecode
import re
# carga de datos

"""
Id: ide catador
    codigo de muestra
    
"""

# Tiene la identificacion del cafe hecha por los catadores. Asignación de categorias a cada cafe, le asigna un cualidad
descrip = pd.read_csv("descriptores.csv")
descrip.head()

# muestras de cafe distintas. Sere de puntajes va 6-10 cada columna es una variable
puntajes = pd.read_csv("puntajes.csv")
puntajes.head()


#
descrip.shape, puntajes.shape

# General review
descrip.info()
puntajes.info()

# resumenes

puntajes.iloc[:, 3:14].describe()

# samples
descrip.sample(10)

puntajes.sample(10)
#===================================================================================
#===================================================================================
# created a functions

# statistics sumary
def summary(df:pd.DataFrame, var1:str):
    mean = df[var1].mean()
    median = df[var1].median()
    std = df[var1].std()
    var = df[var1].var()
    min = df[var1].min()
    max = df[var1].max()
    q25 = df[var1].quantile(0.25)
    q75= df[var1].quantile(0.75)    
    
    return mean, median, std, var, min,max,q25,q75



# graph hist

def hist_df(df:pd.DataFrame):
    fig, axes = plt.subplots(nrows=len(df.columns), figsize=(8, 6 * len(df.columns)))

    for i, col in enumerate(df.columns):
        sns.histplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f'histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        
    plt.tight_layout
    plt.show()

def hist(df:pd.DataFrame, var1:float):
    graph = sns.histplot(data=df, x=var1, bins=15, kde=True)
    return graph

# boxplot
def boxplot(df:pd.DataFrame):
    df.iloc[:, 3:14].plot(
        kind='box',
        subplots=True,
        sharey=False,
        figsize=(15,6) 
    )
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    
# Graph pairplot
def pairplot(df:pd.DataFrame, hue:str):
    return sns.pairplot(df, hue=hue) 

# graph scatter

def scatter(df:pd.DataFrame):
    pd.plotting.scatter_matrix(df.iloc[:, 3:14], figsize=(15,15))
    plt.show()
   


# Define a function to remove accents
def remove_accents(s):
    return unidecode(s)


#===========================================================================================================================================

# sumary statistics
summary(puntajes, 'Fragrance_aroma')
for col in puntajes.iloc[:, 3:14]:
     summary(puntajes, col)

# graps hist
hist(puntajes, 'TOTAL_SCORE')
# multiple hist
hist_df(puntajes)
# boxplot     
boxplot(puntajes)
#scatter
scatter(puntajes)   

#==============================================================================================================================================
#=========================== transformation data
# NLP Tranformation
descrip['Global atributte'].unique() # 7 resultados
descrip['Descriptor'].unique()  # organizar casos como: te, té, Té verde, Te verde, herbaL, 'Panela', ' Panela', Cítrico, Cítrica, Rancio, Rancia, cremoo, Corto, corto,
                                # rugoso, Rugoso, espacios en blanco adelante
                                
# transform data--normalizations
descrip['Descriptor'] = descrip['Descriptor'].str.lower()
descrip['Descriptor'] = descrip['Descriptor'].str.lstrip()
descrip['Descriptor'] = descrip['Descriptor'].str.rstrip()     
# Apply the remove_accents function to the 'Descriptor' column
descrip['Descriptor'] = descrip['Descriptor'].apply(remove_accents)
# replace values
descrip['Descriptor'] = descrip['Descriptor'].replace("limoncillo", "limon")
descrip['Descriptor'] = descrip['Descriptor'].replace("limonaria", "limon")
descrip['Descriptor'] = descrip['Descriptor'].replace("citrica", "citrico")
descrip['Descriptor'] = descrip['Descriptor'].replace("naanja", "naranja")
descrip['Descriptor'] = descrip['Descriptor'].replace("pina", "pino")
descrip['Descriptor'] = descrip['Descriptor'].replace("cremoo", "crema")
descrip['Descriptor'] = descrip['Descriptor'].replace("cremoso", "crema")
descrip['Descriptor'] = descrip['Descriptor'].replace("agria", "agrio")
descrip['Descriptor'] = descrip['Descriptor'].replace("Madera", "maderoso")
descrip['Descriptor'] = descrip['Descriptor'].replace("madera", "maderoso")
descrip['Descriptor'] = descrip['Descriptor'].replace("Cana", "cana")
descrip['Descriptor'] = descrip['Descriptor'].replace("rancia", "rancio")
descrip['Descriptor'] = descrip['Descriptor'].replace("meloso", "melaza")
descrip['Descriptor'] = descrip['Descriptor'].replace("afruta", "fruta")
descrip['Descriptor'] = descrip['Descriptor'].replace("afrutado", "fruta")
descrip['Descriptor'] = descrip['Descriptor'].replace("floran", "floral")
descrip['Descriptor'] = descrip['Descriptor'].replace("aspera", "aspero")
descrip['Descriptor'] = descrip['Descriptor'].replace("amarga", "amargo")
descrip['Descriptor'] = descrip['Descriptor'].replace("medio", "media")
descrip['Descriptor'] = descrip['Descriptor'].replace("jugosa", "jugoso")


### global atributte
descrip['Global atributte'] = descrip['Global atributte'].str.lower()
descrip['Global atributte'] = descrip['Global atributte'].str.lstrip()
descrip['Global atributte'] = descrip['Global atributte'].str.rstrip()     
# Apply the remove_accents function to the 'Descriptor' column
descrip['Global atributte'] = descrip['Global atributte'].apply(remove_accents)

                            
descrip['Descriptor'].unique()    
    """
    correcciones pendientes:
    cítrico, cítrica, naanja, cremoo, 'caña', agria, pina
    

    """
    
descrip['Global atributte'].unique()
                             
                             
                             
#======================================================================================================                             
# correlation
puntajes.iloc[:, 3:13].corr()

# filter

dfsuperior = puntajes[puntajes["TOTAL_SCORE"] > 83]
dfsuperior


Corr50 = puntajes.iloc[:, 3:13].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(Corr50, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Top 50 Companies')
plt.show()

# group by

dgpuntajes = puntajes.groupby(by=['CODIGO_MUESTRA']).mean()
dgpuntajes.head()
dgpuntajes.shape
dgpuntajes.reset_index()

# drop duplicates

#descrip = descrip.drop_duplicates(subset=["CODIGO_MUESTRA"])
#descrip.head()
#descrip.shape


# join

dcompleto = puntajes.merge(descrip, how="left", on=["CODIGO_MUESTRA", "ID_Catador"])
dcompleto.head()
dcompleto.shape


# join with group by

#dgrouped = dgpuntajes.merge(descrip, how="left", on=["CODIGO_MUESTRA"])
#dgrouped.head()
#dgrouped.shape



pairplot(dcompleto, hue="Descriptor")    


# Gap statistic for k means

X =  dcompleto.iloc[:, 3:14]


# Elbow method for k means

def optimalk_elbow(data, n_min, n_max):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(n_min, n_max), timings=True)
    visualizer.fit(data)
    visualizer.show()
    k = visualizer.elbow_value_
    return k
    
k_opt = optimalk_elbow(X, 2, 10)



def optimalk(data, n_min, n_max, metric):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(n_min, n_max), metric=metric ,timings=True)
    visualizer.fit(data)
    visualizer.show()
    k = visualizer.elbow_value_
    return k
    
k_opt1 = optimalk(X, 2, 10, 'silhouette') # 'silhouette', 'calinski_harabasz'


# kmeans model
def kmeansModel(data, nclusters, random_state):
    kmodel = KMeans(n_clusters= nclusters, random_state=random_state)
    y_kmeans = kmodel.fit(data)
    labels = kmodel.labels_
    return y_kmeans, labels
    

clusterN = kmeansModel(X, k_opt, 0)      
clusterN

# Assuming clusterN is your tuple
clusterN_array = clusterN[1]  # Accessing the array from the tuple

# Creating a DataFrame with the array as a column
df = pd.DataFrame({'cluster_label': clusterN_array})
len(df)


# Add cluste to dataset 
dcompleto['clustern'] = df
dcompleto.head()


# filter per cluster
dcompleto['CODIGO_MUESTRA'].unique()


dcompleto[dcompleto['CODIGO_MUESTRA'] == 823190]



# 402076:OK, 658049:OK, 653024:OK, 833548:ok, 148225:ok, 718140:ok, 
# 658145:OK, 518207:ok, 777098:ok, 398017:ok, 394317:ok, 402076:ok, 
# 258242:ok, 148007:ok, 871037:ok, 862129:ok,  513700:ok, 862047:ok

# 402052: 1-3 Not ok, 518308: 1-2 not ok, 513056:1-3 no ok,
# 826611: 2-3 not ok, 394477: 1-3 not ok, 518524:0-3 no ok,
# 718047: 0-3 not ok, 394100: 2-3 not ok, 777172:2-3 no ok, 
# 394148: 1-3 not ok, 823190: 1-3 not ok

# clusters
# cluster 1
c1= dcompleto[dcompleto['clustern'] == 0]
c1
c1.CODIGO_MUESTRA.unique()

# cluster 2
c2= dcompleto[dcompleto['clustern'] == 1]
c2
c2.CODIGO_MUESTRA.unique()

# cluster 3
c3= dcompleto[dcompleto['clustern'] == 2]
c3
c3.CODIGO_MUESTRA.unique()
# cluster 4
c4= dcompleto[dcompleto['clustern'] == 3]
c4
c4.CODIGO_MUESTRA.unique()

### conteos

freq_descriptor = dcompleto['Descriptor'].value_counts()
freq_descriptor.head(50)

# 
freq_global_attr = dcompleto['Global atributte'].value_counts()
freq_global_attr.head(10)

# frecuencia combinada --- las preguntas pueden estar orientadas hacia la combinación de estas variables. Capaz que las mas frecuentes puedan
# brindar ideas de las primeras preguntas (1. atributo, 2. Descriptor)
freq_combination = dcompleto.groupby(['Global atributte', 'Descriptor']).size().reset_index(name='Count').sort_values(by="Descriptor", ascending=False)

sorted_combination_counts = freq_combination.sort_values(by='Count', ascending=False)
sorted_combination_counts.head(50)
