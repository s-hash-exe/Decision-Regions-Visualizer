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

def validate_params(solver_rules, solver, penalty, multi_class):
    rules = solver_rules.get(solver)
    if rules is None:
        return False, f"Unknown solver: {solver}"

    if penalty not in rules["penalty"]:
        return False, f"Solver '{solver}' does not support penalty '{penalty}'."

    if multi_class == "multinomial" and not rules["multinomial"]:
        return False, f"Solver '{solver}' does not support multinomial classification."

    return True, "Valid combination"

def plotDecisionRegions(X, y, log_reg, ax):
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

def run():
    page_keys = [
        'data', 'penalty', 'c', 'solver', 'max_iter', 
        'multi_class', 'l1_ratio'
    ]
    if st.sidebar.button("🔄 Refresh", key='refresh_log_reg'):
        for key in page_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # st.title('Visualize your Model')
    st.markdown("<h1 style='text-align: center;'>Visualize Logistic Regression</h1>", unsafe_allow_html=True)

    # Horizontal line below the title
    st.markdown("<hr>", unsafe_allow_html=True)

    st.sidebar.header('User Input Panel')

    data = st.sidebar.selectbox(
        "Select Dataset",
        ["Binary", "Multiclass"],
        index=0,
        key='data'
    )

    penalty = st.sidebar.selectbox(
        "Penalty",
        ['none', 'l1', 'l2', 'elasticnet'],
        index=2,
        key='penalty'
    )

    c = st.sidebar.number_input(
        "c",
        value=1.0
    )

    solver = st.sidebar.selectbox(
        "Solver",
        ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        index=0,
        key='solver'
    )

    max_iter = int(st.sidebar.number_input(
        'Max Interations',
        value=500,
        key='max_iter'
    ))

    multi_class = st.sidebar.selectbox(
        'Multi-Class',
        ['auto', 'ovr', 'multinomial'],
        index=0,
        key='multi_class'
    )

    l1_ratio = st.sidebar.slider(
        'l1_ratio',
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        key='l1_ratio'
    )

    solver_rules = {
        "lbfgs": {"penalty": ["l2", "none"], "multinomial": True},
        "liblinear": {"penalty": ["l1", "l2"], "multinomial": False},
        "newton-cg": {"penalty": ["l2", "none"], "multinomial": True},
        "newton-cholesky": {"penalty": ["l2", "none"], "multinomial": True},
        "sag": {"penalty": ["l2", "none"], "multinomial": True},
        "saga": {"penalty": ["elasticnet", "l1", "l2", "none"], "multinomial": True},
    }

    lr_params = {
        'penalty': penalty,
        'C': c,
        'solver': solver,
        'max_iter': max_iter,
        'l1_ratio': l1_ratio,
        'multi_class': multi_class
    }

    if penalty != 'elasticnet':
        lr_params.pop('l1_ratio')


    fig, ax = plt.subplots(figsize=(6, 4))
    X, y = getDataset(data)
    initial_graph(X, y, ax)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    graph = st.pyplot(fig)

    if st.sidebar.button('Let\'s Run', key='run_log_reg'):
        graph.empty()
        valid, msg = validate_params(solver_rules, solver, penalty, multi_class)
        if not valid:
            st.error(msg)
        else:
            log_reg = getRegressor(X_train, y_train, lr_params)
            y_pred = log_reg.predict(X_test)

            fig, ax = plt.subplots(figsize=(6,4))
            plotDecisionRegions(X, y, log_reg, ax)
            graph.pyplot(fig)

            st.write('Accuracy of model: ', accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    run()