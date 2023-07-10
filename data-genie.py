import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("Data Science Web App")
    st.write("Upload a CSV/XLSX/TXT file")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]

        # Read file
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_extension == "txt":
            df = pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("Invalid file format. Please upload a CSV, XLSX, or TXT file.")
            return

        # Display the data
        st.subheader("Data Preview")
        st.write(df.head())

        # Filter the dataset
        st.subheader("Filter Dataset")
        filter_col = st.selectbox("Select column to filter", df.columns)
        filter_value = st.text_input("Enter filter value")

        filtered_df = df[df[filter_col] == filter_value]
        st.write(filtered_df)

        # Display unique values for selected columns
        st.subheader("Unique Values")
        unique_cols = st.multiselect("Select columns for unique values", df.columns)
        for col in unique_cols:
            st.subheader(f"Unique values for column: {col}")
            st.write(df[col].unique())

        # Display values of selected columns as histograms
        st.subheader("Histogram")
        hist_cols = st.multiselect("Select columns for histogram", df.columns)
        for col in hist_cols:
            st.subheader(f"Histogram for column: {col}")
            plt.hist(df[col])
            st.pyplot()

        # Pivot table
        st.subheader("Pivot Table")
        pivot_cols = st.multiselect("Select columns for pivot table", df.columns)
        pivot_table = df.pivot_table(index=pivot_cols)
        st.write(pivot_table)

        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        # Export filtered/transformed dataset
        st.subheader("Export Filtered/Transformed Dataset")
        export_format = st.selectbox("Select export format", ["csv", "xlsx", "txt"])
        export_filename = st.text_input("Enter export file name")

        if st.button("Export"):
            if export_format == "csv":
                filtered_df.to_csv(export_filename + ".csv", index=False)
            elif export_format == "xlsx":
                filtered_df.to_excel(export_filename + ".xlsx", index=False)
            elif export_format == "txt":
                filtered_df.to_csv(export_filename + ".txt", sep="\t", index=False)
            st.success("Export successful!")

if __name__ == "__main__":
    main()
