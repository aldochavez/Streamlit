import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title("Streamlit Ejemplo")

st.write("""
# Explorando Diferentes Clasificadores
Quien es el mejor?
""")

nombreDataset = st.sidebar.selectbox("Seleccionar Dataset", ("Iris", "Breast Cancer", "Wine dataset"))
nombreClasificador = st.sidebar.selectbox("Seleccionar Clasificador", ("KNN", "SVC", "Random Forest"))

st.write(nombreDataset)
st.write(nombreClasificador)

data = None
if nombreDataset == 'Iris':
    data = datasets.load_iris()
elif nombreDataset == 'Wine':
    data = datasets.load_wine()
else:
    data = datasets.load_breast_cancer()
X = data.data
y = data.target

st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

params = dict()
if nombreClasificador == 'SVC':
    C = st.sidebar.slider('C', 0.01, 10.0)
    params['C'] = C
elif nombreClasificador == 'KNN':
    K = st.sidebar.slider('K', 1, 15)
    params['n_neighbors'] = K
else:
    max_depth = st.sidebar.slider('max_depth', 2, 15)
    params['max_depth'] = max_depth
    n_estimators = st.sidebar.slider('n_estimators', 1, 100)
    params['n_estimators'] = n_estimators

clf = None
if nombreClasificador == 'SVC':
    clf = SVC(**params)
elif nombreClasificador == 'KNN':
    clf = KNeighborsClassifier(**params)
else:
    clf = RandomForestClassifier(**params, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {nombreClasificador}')
st.write(f'Accuracy =', acc)

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)