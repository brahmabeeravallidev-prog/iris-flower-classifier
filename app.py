import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

st.title("Iris Flower Classifier")
st.info("Enter flower measurements and click predict")
# User inputs
sl = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)

sw = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)

pl = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)

pw = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

# Prediction
if st.button("Predict"):

    sample = np.array([[sl, sw, pl, pw]])

    prediction = model.predict(sample)

    flower = iris.target_names[prediction][0]

    st.success(f"Predicted Flower: {flower}")
