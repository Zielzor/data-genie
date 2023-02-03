import streamlit as st
import pandas as pd
import tempfile

def filter_data():
    st.set_page_config(page_title="Filter Data", page_icon=":mag_right:", layout="wide")
    st.title("Filter Data")
    uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv" if uploaded_file.name.endswith(".csv") else ".xlsx") as temp:
            temp.write(uploaded_file.read())
            temp.seek(0)
            data = pd.read_csv(temp.name) if temp.name.endswith(".csv") else pd.read_excel(temp.name)
            column_names = data.columns.tolist()
            selected_columns = st.multiselect("Select columns to filter by", column_names)
            show_all = st.checkbox("Show all values of selected columns")
            if show_all:
                st.write(data[selected_columns])
            else:
                selected_values = []
                for column in selected_columns:
                    selected_values.append((column, st.text_input(f"Enter a value to filter by in {column}")))
                if st.button("Filter"):
                    filtered_data = data.copy()
                    for column, value in selected_values:
                        filtered_data = filtered_data[filtered_data[column] == value]
                    st.dataframe(filtered_data)

if __name__ == '__main__':
    filter_data()
