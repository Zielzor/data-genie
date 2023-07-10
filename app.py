import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Data Web App")
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

        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        # Data info
        st.subheader("Dataset Info")
        st.write(df.dtypes)

        #Show random sample
        st.subheader("Random sample")
        sample_df = df.sample(50)
        st.write(sample_df)

         #Export random sample
        st.subheader("Export Random Sample Dataset")
        export_format = st.selectbox("Select export format", ["csv", "xlsx", "txt"], key="random-export")
        export_filename = st.text_input("Enter export file name")

        if st.button("Export Sample"):
            if export_format == "csv":
                filtered_df.to_csv(export_filename + ".csv", index=False)
            elif export_format == "xlsx":
                filtered_df.to_excel(export_filename + ".xlsx", index=False)
            elif export_format == "txt":
                filtered_df.to_csv(export_filename + ".txt", sep="\t", index=False)
            st.success("Export successful!")

        # Filter the dataset
        st.subheader("Filter Dataset")
        columns = st.multiselect("Columns: ", df.columns)
        filter = st.radio("Chose by:", ("include","exclude"))

        if filter == "exclude":
            columns = [col for col in df.columns if col not in columns]
        
        filtered_df = df[columns]
        filtered_df


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
        def show_pivot_table(df):
            st.subheader("Create Pivot Table")
            available_columns = df.columns.tolist()

            # Select columns for index, columns, and values
            index_col = st.selectbox("Select index column", available_columns)
            columns_col = st.selectbox("Select columns column", available_columns)
            values_col = st.selectbox("Select values column", available_columns)

            # Generate the pivot table
            pivot_table = pd.pivot_table(df, index=index_col, columns=columns_col, values=values_col)

            # Display the pivot table
            st.dataframe(pivot_table)

        show_pivot_table(df)

        # Export filtered/transformed dataset
        st.subheader("Export Filtered/Transformed Dataset")
        export_format = st.selectbox("Select export format", ["csv", "xlsx", "txt"],key="export-format")
        export_filename = st.text_input("Enter export file name", key="export-name")

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