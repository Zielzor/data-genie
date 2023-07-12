import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

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

        #displaying Desc and Info side by side
        col1, col2 = st.columns(2)

        with col1:
            # Descriptive statistics
            st.subheader("Descriptive Statistics")
            st.write(df.describe())
        
        with col2:
            # Data info
            st.subheader("Dataset Info")
            st.write(df.dtypes)

        #Correlactions heatmap
        st.subheader("Detailed Correlations")
        correlation_view = st.radio("Correlation View",("Table", "Heatmap"))

        if correlation_view == "Table":
            st.write(df.corr())
        elif correlation_view == "Heatmap":
            fig,ax = plt.subplots(figsize=(10,8)) #adjust the size
            cmap = sns.diverging_palette(220,20, as_cmap=True) #Color pallete
            sns.heatmap(df.corr(), annot=False,cmap=cmap, linewidths=0.5, annot_kws={"fontsize":10},ax=ax)
            st.pyplot(fig)
         
        
        #Show random sample
        st.subheader("Random sample")
        sample_df = df.sample(50)
        st.write(sample_df)

         #Export random sample
        st.sidebar.subheader("Export Random Sample Dataset")
        export_format = st.sidebar.selectbox("Select export format", ["csv", "xlsx", "txt"], key="random-export")
        export_filename = st.sidebar.text_input("Enter export file name")

        if st.sidebar.button("Export Sample"):
            if export_format == "csv":
                filtered_df.to_csv(export_filename + ".csv", index=False)
            elif export_format == "xlsx":
                filtered_df.to_excel(export_filename + ".xlsx", index=False)
            elif export_format == "txt":
                filtered_df.to_csv(export_filename + ".txt", sep="\t", index=False)
            st.sidebar.success("Export successful!")

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
        st.subheader("Pivot Table")
        pivot_cols = st.multiselect("Select columns for pivot table", df.columns)
        pivot_values = st.selectbox("Select values for pivot table",df.columns)
        pivot_agg = st.selectbox("Select aggregation function", ["sum","count"])
        
        if pivot_cols and pivot_values and pivot_agg:
            pivot_table = df.pivot_table(index=pivot_cols, values=pivot_values, aggfunc=pivot_agg)
            st.write(pivot_table)
            

       
        # Export filtered/transformed dataset
        st.sidebar.subheader("Export Filtered/Transformed Dataset")
        export_format = st.sidebar.selectbox("Select export format", ["csv", "xlsx", "txt"],key="export-format")
        export_filename = st.sidebar.text_input("Enter export file name", key="export-name")

        if st.sidebar.button("Export"):
            if export_format == "csv":
                filtered_df.to_csv(export_filename + ".csv", index=False)
            elif export_format == "xlsx":
                filtered_df.to_excel(export_filename + ".xlsx", index=False)
            elif export_format == "txt":
                filtered_df.to_csv(export_filename + ".txt", sep="\t", index=False)
            st.sidebar.success("Export successful!")

if __name__ == "__main__":
    main()