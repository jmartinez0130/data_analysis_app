import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from scipy import stats
from io import BytesIO
import base64
from datetime import datetime, timedelta

# Function to create dummy air pollution data
def create_dummy_air_pollution_data():
    np.random.seed(42)
    dates = [datetime.today() - timedelta(days=i) for i in range(100)]
    data = {
        'Date': dates,
        'PM2.5': np.random.normal(50, 15, 100),
        'PM10': np.random.normal(100, 25, 100),
        'NO2': np.random.normal(40, 10, 100),
        'CO': np.random.normal(1.0, 0.2, 100),
        'Temperature': np.random.normal(20, 5, 100),
        'Humidity': np.random.normal(50, 10, 100)
    }
    df = pd.DataFrame(data)
    return df

# Function to convert DataFrame to CSV and download
def download_csv(df, filename="file.csv"):
    csv_data = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv_data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

import plotly.io as pio

# Function to download figures as PNG using Plotly and kaleido
def download_figure(fig, filename="figure.png"):
    buffer = BytesIO()
    fig.write_image(buffer, format="png")  # Uses kaleido to write image
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Figure</a>'
    return href


# Title and Introduction
st.title('Interactive EDA for Air Pollution Data 🌍💨')
st.write("""
Welcome to the **Interactive EDA Application** for Air Pollution Data. This tool helps you:
- Upload your air pollution dataset (CSV)
- Explore, clean, and preprocess your data
- Visualize relationships, detect outliers, and generate correlation plots
- Train basic machine learning models on air pollution data.
Use the sidebar to navigate through different features.
""")

# Sidebar - Action Selection
st.sidebar.title('📋 Actions')

# Sidebar - Step 1: Upload Data or Use Sample Dataset
st.sidebar.header("📂 Step 1: Upload Data")
use_sample = st.sidebar.checkbox("Use Sample Air Pollution Data")
data = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if use_sample:
    df = create_dummy_air_pollution_data()
    st.success("Sample air pollution dataset loaded.")
else:
    if data is not None:
        df = pd.read_csv(data)
        st.success("File uploaded successfully.")
    else:
        st.warning("Please upload a CSV file or use the sample data to proceed.")
        df = None

# Ensure Data is Available for Analysis
if df is not None:
    st.write("### Preview of the Dataset")
    st.write(df.head())

    # Use Tabs for Easy Navigation between Data Cleaning, Visualization, etc.
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Cleaning", "Visualizations", "Correlation Plot", "Outlier Detection", "Machine Learning"])

    # Tab 1: Data Cleaning
    with tab1:
        st.header("🧹 Data Cleaning")

        # Drop Columns
        st.subheader("Drop Columns")
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        if columns_to_drop:
            df.drop(columns=columns_to_drop, axis=1, inplace=True)
            st.write("### Data after Dropping Columns")
            st.write(df.head())

        # Rename Columns
        st.subheader("Rename Columns")
        columns_to_rename = st.multiselect("Select columns to rename", df.columns)
        new_names = st.text_input("Enter new names (comma-separated) for the selected columns")
        if columns_to_rename and new_names:
            new_names_list = new_names.split(',')
            df.rename(columns=dict(zip(columns_to_rename, new_names_list)), inplace=True)
            st.write("### Data after Renaming Columns")
            st.write(df.head())

        # Change Data Types
        st.subheader("Change Data Types")
        columns_to_change_type = st.multiselect("Select columns to change data type", df.columns)
        new_type = st.selectbox("Select the new data type", ('int64', 'float64', 'object'))
        if columns_to_change_type:
            df[columns_to_change_type] = df[columns_to_change_type].astype(new_type)
            st.write("### Data after Changing Data Type")
            st.write(df.head())

        # Handle Missing Values
        st.subheader("Handle Missing Values")
        columns_to_impute = st.multiselect("Select columns to impute missing values", df.columns)
        impute_method = st.selectbox('Select the imputation method', ['Linear Interpolation', 'Mean Imputation', 'Median Imputation'])
        if columns_to_impute:
            if impute_method == 'Linear Interpolation':
                df[columns_to_impute] = df[columns_to_impute].interpolate(method='linear')
            elif impute_method == 'Mean Imputation':
                imputer = SimpleImputer(strategy='mean')
                df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
            elif impute_method == 'Median Imputation':
                imputer = SimpleImputer(strategy='median')
                df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
            st.write("### Data after Imputation")
            st.write(df.head())

        st.markdown(download_csv(df, "cleaned_data.csv"), unsafe_allow_html=True)

    # Tab 2: Data Visualizations
    with tab2:
        st.header("📊 Data Visualizations")

        # Time-Series Plots for Air Pollution Data
        st.subheader("Time-Series Visualization")
        time_series_var = st.selectbox("Select variable for time-series plot", ['PM2.5', 'PM10', 'NO2', 'CO', 'Temperature', 'Humidity'])
        time_chart = px.line(df, x='Date', y=time_series_var, title=f"Time-Series of {time_series_var}")
        st.plotly_chart(time_chart)

        # Download Plot
        st.markdown(download_figure(time_chart, f"time_series_{time_series_var}.png"), unsafe_allow_html=True)

        # Other Visualization Options
        st.subheader("Generate Other Interactive Plots")
        x_axis = st.selectbox("Select X axis", df.columns)
        y_axis = st.selectbox("Select Y axis", df.columns)
        plot_type = st.selectbox("Select Plot Type", ['Scatter Plot', 'Line Chart', 'Bar Chart', 'Histogram'])
        
        if st.button("Generate Plot"):
            st.write(f"### {plot_type} for {x_axis} vs {y_axis}")
            if plot_type == 'Scatter Plot':
                fig = px.scatter(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)
            elif plot_type == 'Line Chart':
                fig = px.line(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)
            elif plot_type == 'Bar Chart':
                fig = px.bar(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)
            elif plot_type == 'Histogram':
                fig = px.histogram(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)

            # Download Plot
            st.markdown(download_figure(fig, f"{plot_type.lower()}_{x_axis}_vs_{y_axis}.png"), unsafe_allow_html=True)

    # Tab 3: Correlation Plot
    with tab3:
        st.header("📈 Correlation Plot")

        st.subheader("Generate Correlation Heatmap")
        corr = df.corr()
        plt.figure(figsize=(10, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot()

        # Download correlation matrix
        st.markdown(download_csv(corr, "correlation_matrix.csv"), unsafe_allow_html=True)

    # Tab 4: Outlier Detection
    with tab4:
        st.header("🚨 Outlier Detection")

        # Info about Outlier Detection Methods
        st.info("""
        **Outlier Detection Methods**:
        - **Isolation Forest**: Detects outliers by isolating anomalies through random partitioning of data points.
        - **Z-score**: Measures how many standard deviations a data point is from the mean; values beyond a threshold (e.g., 3) are considered outliers.
        """)

        # Detect Outliers in Selected Columns
        columns_to_find_outliers = st.multiselect("Select columns for outlier detection", df.columns)
        outlier_method = st.selectbox("Select Outlier Detection Method", ['Isolation Forest', 'Z-score'])
        
        if st.button("Find Outliers"):
            if outlier_method == 'Isolation Forest' and columns_to_find_outliers:
                iso_forest = IsolationForest(contamination='auto')
                outliers = iso_forest.fit_predict(df[columns_to_find_outliers])
                df['Outlier'] = outliers
                st.write("### Detected Outliers")
                st.write(df[df['Outlier'] == -1])  # Display outliers

                # Visualize Outliers
                fig = px.scatter(df, x=columns_to_find_outliers[0], y=columns_to_find_outliers[1], color=df['Outlier'].astype(str))
                st.plotly_chart(fig)

                st.markdown(download_csv(df, "outliers_data.csv"), unsafe_allow_html=True)

            elif outlier_method == 'Z-score' and columns_to_find_outliers:
                z_scores = np.abs(stats.zscore(df[columns_to_find_outliers]))
                outliers = (z_scores > 3).any(axis=1)
                df['Outlier'] = outliers
                st.write("### Detected Outliers")
                st.write(df[outliers])

                # Visualize Outliers
                fig = px.scatter(df, x=columns_to_find_outliers[0], y=columns_to_find_outliers[1], color=df['Outlier'].astype(str))
                st.plotly_chart(fig)

                st.markdown(download_csv(df, "outliers_data.csv"), unsafe_allow_html=True)

    # Tab 5: Machine Learning
    with tab5:
        st.header("📈 Machine Learning")

        st.info("""
        **Machine Learning Models**:
        - **Linear Regression**: A linear approach to modeling the relationship between a dependent variable and one or more independent variables.
        - **Ridge Regression**: A variation of linear regression that introduces a regularization term to prevent overfitting by penalizing large coefficients.
        - **Lasso Regression**: Another regularization method that can shrink coefficients to zero, effectively performing feature selection.
        """)

        # Select Features and Target for Model Training
        features = st.multiselect("Select Features for Model", df.columns)
        target = st.selectbox("Select Target Variable", df.columns)
        model_type = st.selectbox("Select Model Type", ['Linear Regression', 'Ridge Regression', 'Lasso Regression'])
        
        if st.button("Train Model") and features and target:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            if model_type == 'Linear Regression':
                model = LinearRegression()
            elif model_type == 'Ridge Regression':
                model = Ridge()
            elif model_type == 'Lasso Regression':
                model = Lasso()

            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            st.write(f"### Model Trained: R² Score = {score:.2f}")

    # Save Cleaned Data
    st.sidebar.header("💾 Save Data")
    if st.sidebar.button("Save Data as CSV"):
        df.to_csv('cleaned_air_pollution_data.csv', index=False)
        st.success("Data saved as 'cleaned_air_pollution_data.csv'")

else:
    st.info("Please upload a dataset or select the sample dataset to begin analysis.")
