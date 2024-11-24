import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("CSV Data Visualization")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Select chart type
    chart_type = st.selectbox("Select a chart type", ["Line Chart", "Bar Chart", "Histogram"])

    # Select column(s) to visualize
    numeric_columns = df.select_dtypes(include=["float", "int"]).columns
    selected_column = st.selectbox("Select a column", numeric_columns)

    if chart_type == "Line Chart":
        st.line_chart(df[selected_column])
    elif chart_type == "Bar Chart":
        st.bar_chart(df[selected_column])
    elif chart_type == "Histogram":
        # Create a histogram
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], ax=ax, kde=True)
        st.pyplot(fig)
