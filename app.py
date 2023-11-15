import warnings
import streamlit as st
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

file = './dataSets/housing.csv'
columns_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(file, names=columns_names, delim_whitespace=True)

def visualize_data():
    st.markdown('<p style="font-size:26px;color:purple;font-weight:bold">Visualisation des données de Diabète :</p>', unsafe_allow_html=True)
    st.header('Dimension : ')
    st.write(data.shape)
    st.header('Informations generales : ')
    st.write(data.describe())
    st.header('Histogramme : ')
    plt.figure(figsize=(8, 8))
    data.hist()
    st.pyplot(plt)

def display_prediction_and_metrics(model, df, selectedClassMetrics, X, Y):
    prediction = model.predict(df)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Prediction :')
        st.write(f'Le prix de la maison avec ces caractéristiques est estimé à {(prediction[0]*10000).round(2)} $')
    with col2:
        metricReg(model, selectedClassMetrics, X, Y)

def metricReg(model, selectedRegMetrics, X, Y):
    st.subheader('Les métriques :')
    if selectedRegMetrics:
        if "EQM" in selectedRegMetrics:
            st.markdown('<p style="font-size:20px;color:#dabfc5;font-weight:bold">Erreur Quadratique Moyenne(EQM)</p>', unsafe_allow_html=True)
            eqm = cross_val_score(model, X, Y, scoring = "neg_mean_squared_error")
            st.write(eqm.mean().round(3))
        if "EAM" in selectedRegMetrics:
            st.markdown('<p style="font-size:20px;color:#dabfc5;font-weight:bold">Erreur Absolue Moyenne(EAM)</p>', unsafe_allow_html=True)
            eam = cross_val_score(model, X, Y, scoring = "neg_mean_absolute_error")
            st.write(eam.mean().round(3))
        if "R2" in selectedRegMetrics:
            st.markdown('<p style="font-size:20px;color:#dabfc5;font-weight:bold">Le coefficient de Détermination(R²)</p>', unsafe_allow_html=True)
            r2 = cross_val_score(model, X, Y, scoring = "r2")
            st.write(r2.mean().round(3))
    else:
        st.write('Veuillez choisir une métrique pour mesurer la qualité des prédictions et évaluer la performance du modèle.')

def user_input_features():
    st.sidebar.markdown('<p style="font-size:16px;color:purple;font-weight:bold">Veuillez entrer vos valeurs :</p>', unsafe_allow_html=True)
    crim = st.sidebar.slider('CRIM', 0.0, 88.9, 3.6)
    zn = st.sidebar.slider('ZN', 0.0, 100.0, 11.3)
    indus = st.sidebar.slider('INDUS', 0.4, 27.7, 11.1)
    chas = st.sidebar.slider('CHAS', 0.0, 1.0, 0.0)
    nox = st.sidebar.slider('NOX', 0.3, 0.8, 0.5)
    rm = st.sidebar.slider('RM', 3.5, 8.7, 6.2)
    age = st.sidebar.slider('AGE', 2.9, 100.0, 68.5)
    dis = st.sidebar.slider('DIS', 1.1, 12.1, 1.1)
    rad = st.sidebar.slider('RAD', 1.0, 24.0, 9.5)
    tax = st.sidebar.slider('TAX', 187.0, 711.0, 408.2)
    ptrato = st.sidebar.slider('PTRATIO', 12.6, 22.0, 18.4)
    b = st.sidebar.slider('B', 0.3, 396.9, 356.6)
    lstat = st.sidebar.slider('LSTAT', 0.3, 37.9, 12.6)
    data = {
        'CRIM' : crim,
        'ZN' : zn,
        'INDUS' : indus,
        'CHAS' : chas,
        'NOX' : nox,
        'RM' : rm,
        'AGE' : age,
        'DIS' : dis,
        'RAD' : rad,
        'TAX' : tax,
        'PTRATIO' : ptrato,
        'B' : b,
        'LSTAT' : lstat,
    }
    #TRANSFERER LES DATAS AU DATAFRAME
    features = pd.DataFrame(data, index=[0])
    return features

def main():
    st.write(
        """
        # Application de prédiction et de visualisation de prix de maisons
        (Regression)
        """
    )
    options = ["Les options", "Visualisation des données", "Prediction de prix des maisons"]

    st.markdown('<p style="font-size:16px;color:purple;font-weight:bold">Choisissez une option :</p>', unsafe_allow_html=True)
    selected_option = st.selectbox("", options)

    if selected_option != "Les options":
        if selected_option == "Visualisation des données":
            visualize_data()
        else:
            st.sidebar.markdown('<p style="font-size:16px;color:purple;font-weight:bold">Choisissez un algorithme :</p>', unsafe_allow_html=True)
            RegAlgo = ["Les algorithms", "La régression linéaire", "Le Lasso", "Les arbres de décision"]
            algo = st.sidebar.selectbox("", RegAlgo)
            RegMetric = ["EQM", "EAM", "R2"]
            st.sidebar.markdown('<p style="font-size:16px;color:purple;font-weight:bold">Sélectionnez une ou plusieurs metrics :</p>', unsafe_allow_html=True)
            selectedRegMetrics = st.sidebar.multiselect("", RegMetric)
            df = user_input_features()
            st.subheader("User Input Parameters")
            st.write(df)

            array = data.values
            X = array[:, 0:-1]
            Y = array[:, -1]

            if algo != "Les algorithms":
                if algo == "La régression linéaire":
                    model = LinearRegression()
                elif algo == "Le Lasso":
                    model = Lasso()
                else:
                    model = DecisionTreeRegressor()
                model.fit(X, Y)
            display_prediction_and_metrics(model, df, selectedRegMetrics, X, Y)

    else:
        st.subheader("")

if __name__ == '__main__':
    main()



