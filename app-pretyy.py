import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

# Function to get or create session state
def get_session_state():
    session_state = st.session_state
    if "kmeans_page" not in session_state:
        session_state.kmeans_page = False
    return session_state

def main():
    st.title("Data Science Web App")

    # Check if the "Go to K-Means" button has been clicked
    session_state = get_session_state()
    # Check if the "Go to K-Means" button has been clicked
    if "kmeans_page" not in session_state:
        session_state.kmeans_page = False

    # Check if the "Go to Main Page" button has been clicked
    if "main_page" not in session_state:
        session_state.main_page = False

    
    # Create the "Go to Main Page" button
    if st.button("Statistics"):
        session_state.main_page = True
        session_state.kmeans_page = False
    # Create the "Go to K-Means" button
    if st.button("K-Means"):
        session_state.kmeans_page = True
        session_state.main_page = False


    # Render the appropriate page based on the button click
    if session_state.kmeans_page:
        render_kmeans_page()
    elif session_state.main_page:
        render_main_page()


def render_main_page():
    st.subheader("Statistacal functions")
    st.write("This section allows for statistical insights/data manipulation")

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



def render_kmeans_page():
    st.subheader("K-Means Page")
    st.write("This is the K-Means page")

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

        #Display corr
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

        # Feature selection
        all_features = df.columns.tolist()
        include_features = st.multiselect("Include features", all_features)
        exclude_features = st.multiselect("Exclude features", all_features, default=[])

        if not include_features:
            st.warning("Please select at least one feature to include.")
            return
        
        selected_features = list(set(include_features) - set(exclude_features))
        if len(selected_features) == 0:
            st.warning("All selected features are excluded. Please modify your selections.")
            return

        selected_df = df[selected_features]

        # K-Means clustering
        num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(selected_df)
        cluster_labels = kmeans.labels_
        df["Cluster"] = cluster_labels

        # Display the clustered data
        st.subheader("Clustered Data")
        st.write(df)

        # Export the dataset
        export_format = st.sidebar.selectbox("Select export format", ["csv", "xlsx", "txt"])
        export_filename = st.sidebar.text_input("Enter export file name")

        if st.sidebar.button("Export"):
            if export_filename:
                if export_format == "csv":
                    df.to_csv(export_filename + ".csv", index=False)
                elif export_format == "xlsx":
                    df.to_excel(export_filename + ".xlsx", index=False)
                elif export_format == "txt":
                    df.to_csv(export_filename + ".txt", sep="\t", index=False)
                st.sidebar.success("Export successful!")
            else:
                st.sidebar.error("Please enter a valid export file name.")

        # Perform PCA for visualization
        if len(selected_features) >= 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(selected_df)
            pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
            pca_df["Cluster"] = cluster_labels

            # Display PCA scatter plot
            st.subheader("PCA of Clustering")
            plt.figure(figsize=(8, 6))
            for cluster in range(num_clusters):
                cluster_data = pca_df[pca_df["Cluster"] == cluster]
                plt.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"Cluster {cluster}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA of Clustering")
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning("PCA requires at least two selected features for visualization.")

# Run the app
if __name__ == "__main__":
    main()
