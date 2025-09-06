import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

def getDataset(dataset):
    if dataset == "Binary":
        X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=6)
        return X,y
    elif dataset == "Multiclass":
        X,y = make_blobs(n_samples=200, n_features=2, centers=3, random_state=2)
        return X,y

def initial_graph(X, y, ax):
    ax.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

def getRegressor(X_train, y_train, lr_params):
    log_reg = LogisticRegression(**lr_params)
    log_reg.fit(X_train, y_train)
    return log_reg

def plotDecisionRegions(X, y, clf, ax):
    # a = np.arange(X[:,0].min()-1, X[:,0].max()+1, step=0.01)
    # b = np.arange(X[:,1].min()-1, X[:,1].max()+1, step=0.01)

    # XX, YY = np.meshgrid(a,b)
    # input_array = np.array([XX.ravel(), YY.ravel()]).T

    # labels = log_reg.predict(input_array)

    # ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    # ax.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
    # graph=st.pyplot(fig)

    plot_decision_regions(X, y, clf=log_reg, legend=2, ax=ax)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

# st.title('Visualize your Model')
st.markdown("<h1 style='text-align: center;'>Visualize Logistic Regression</h1>", unsafe_allow_html=True)

# Horizontal line below the title
st.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.header('User Input Panel')

data = st.sidebar.selectbox(
    "Select Dataset",
    ["Binary", "Multiclass"],
    index=0
)

regularization = st.sidebar.selectbox(
    "Regularization",
    ['none', 'l1', 'l2', 'elasticnet'],
    index=2
)

c = st.sidebar.number_input(
    "c",
    value=1.0
)

solver = st.sidebar.selectbox(
    "Solver",
    ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    index=0
)

max_iter = int(st.sidebar.number_input(
    'Max Interations',
    value=500
))

l1_ratio = st.sidebar.slider(
    'l1_ratio',
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01
)

lr_params = {
    'penalty': regularization,
    'C': c,
    'solver': solver,
    'max_iter': max_iter,
    'l1_ratio': l1_ratio,
    'multi_class': 'multinomial'
}

if regularization != 'elasticnet':
    lr_params.pop('l1_ratio')

fig, ax = plt.subplots(figsize=(6, 4))
X, y = getDataset(data)
initial_graph(X, y, ax)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
graph = st.pyplot(fig)

if st.sidebar.button('Let\'s Run'):
    graph.empty()
    
    log_reg = getRegressor(X_train, y_train, lr_params)
    y_pred = log_reg.predict(X_test)

    fig, ax = plt.subplots(figsize=(6,4))
    plotDecisionRegions(X, y, log_reg, ax)
    graph.pyplot(fig)

    st.write('Accuracy of model: ', accuracy_score(y_test, y_pred))
