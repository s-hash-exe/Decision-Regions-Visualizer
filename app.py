import streamlit as st

def show_home():
    st.title("Machine Learning Visualizer")
    st.write("Select an algorithm to explore:")

    cols = st.columns(2) 
    algorithms = ["Logistic Regression"]

    for i, algo in enumerate(algorithms):
        with cols[i % 2]:
            if st.button(algo, key=algo):
                st.session_state['page'] = algo

def show_logistic_reg():
    from models import logistic_reg
    logistic_reg.run()

    if st.button("← Back to Home"):
        st.session_state['page'] = "home"

if 'page' not in st.session_state:
    st.session_state['page'] = "home"

if st.session_state['page'] == "home":
    show_home()
elif st.session_state['page'] == "Logistic Regression":
    show_logistic_reg()
elif st.session_state['page'] == "SVM":
    pass
elif st.session_state['page'] == "Decision Tree":
    pass
elif st.session_state['page'] == "KNN":
    pass
