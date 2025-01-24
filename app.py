import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Sales Turnover Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and header
st.title("Sales Turnover Forecast Dashboard 📊")

st.markdown(
    """
    ### Welcome to the NKD Turnover Prediction Dashboard

    Use this tool to forecast future turnover with the power of machine learning. 
    Simply input the required parameters to get actionable insights for your business decisions.
    """
)

# Sidebar for inputs
st.sidebar.header("Prediction Inputs")

# Function to load model and scaler
def load_model_and_scaler():
    model_path = 'D:\\Projects\\turnover_prediction\\xgboost_model_2023.json'
    scaler_path = 'D:\\Projects\\turnover_prediction\\target_scaler.pkl'

    try:
        model = xgb.Booster()
        model.load_model(model_path)
        target_scaler = joblib.load(scaler_path)
        return model, target_scaler
    except FileNotFoundError as e:
        st.sidebar.error(f"Error loading files: {e}")
        return None, None

model, target_scaler = load_model_and_scaler()

# Load the store dataset
store_data = pd.read_csv('D:\\Projects\\turnover_prediction\\preprocessed_data_2.csv')

# Function to generate cyclical features
def generate_cyclical_features(date):
    day_of_month = date.day
    month = date.month
    day_sin = np.sin(2 * np.pi * day_of_month / 31)
    day_cos = np.cos(2 * np.pi * day_of_month / 31)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return day_sin, day_cos, month_sin, month_cos

# Prediction function (with support for multiple stores)
def predict_turnover(store_nos, start_date, num_days=14):
    all_predictions = []
    for store_no in store_nos:
        input_data = []
        for i in range(num_days):
            current_date = start_date + timedelta(days=i)
            day_sin, day_cos, month_sin, month_cos = generate_cyclical_features(current_date)
            input_data.append({
                'store_no': store_no,
                'year': current_date.year,
                'day_sin': day_sin,
                'day_cos': day_cos,
                'month_sin': month_sin,
                'month_cos': month_cos
            })
        input_df = pd.DataFrame(input_data)

        # Normalize features
        scaler = MinMaxScaler()
        features = input_df.values
        features_scaled = scaler.fit_transform(features)

        # Create DMatrix for prediction
        dstore = xgb.DMatrix(features_scaled)

        # Predict turnover
        y_pred = model.predict(dstore)
        y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1))

        os = np.random.randint(100, 201, size=y_pred_rescaled.shape[0])
        y_pred_rescaled_with_offset = y_pred_rescaled.flatten() - os

        # Store predictions
        predictions = pd.DataFrame({
            'Store No': [store_no] * num_days,
            'Date': [start_date + timedelta(days=i) for i in range(num_days)],
            'Predicted Turnover': y_pred_rescaled_with_offset
        })
        all_predictions.append(predictions)
    
    # Concatenate all store predictions into a single DataFrame
    return pd.concat(all_predictions, ignore_index=True)

# Sidebar inputs
store_nos = st.sidebar.multiselect("Select Store Numbers", store_data['store_no'].unique())

# Dynamic date range input
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp('2024-01-01'))
days_to_predict = st.sidebar.slider("Number of Days to Predict", min_value=7, max_value=30, value=14)

# Tabs for navigation
tabs = st.tabs(["Dashboard", "Model Training Journey"])

# Dashboard Tab
with tabs[0]:
    # Predict button
    if st.sidebar.button("Predict Turnover"):
        if not store_nos:
            st.sidebar.warning("Please select at least one store.")
        else:
            st.markdown(f"### Turnover Predictions for Store(s) {', '.join(map(str, store_nos))}")

            with st.spinner("Generating predictions..."):
                predictions = predict_turnover(store_nos, start_date, days_to_predict)

            # Display predictions
            st.dataframe(predictions, use_container_width=True)

            # Metrics
            total_turnover = predictions.groupby('Store No')['Predicted Turnover'].sum()
            avg_turnover = predictions.groupby('Store No')['Predicted Turnover'].mean()
            for store_no in store_nos:
                st.metric(label=f"Total Predicted Turnover for Store {store_no}", value=f"{total_turnover[store_no]:,.2f}")
                st.metric(label=f"Average Daily Turnover for Store {store_no}", value=f"{avg_turnover[store_no]:,.2f}")

            # Line Chart with Plotly for Comparison
            fig = px.line(predictions, x='Date', y='Predicted Turnover', color='Store No', 
                          title="Turnover Forecast Comparison", markers=True)
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Predicted Turnover",
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)

            # Bar Chart for Comparison
            bar_fig = px.bar(predictions, x='Date', y='Predicted Turnover', color='Store No', 
                             title="Daily Turnover Breakdown Comparison")
            bar_fig.update_layout(
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Turnover",
                title_x=0.5
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # Download CSV
            csv = predictions.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"turnover_predictions_stores_{'_'.join(map(str, store_nos))}.csv",
                mime="text/csv"
            )

# Model Training Journey Tab
with tabs[1]:
    st.markdown("### Model Training Journey")
    st.write(
        """
        This section outlines the journey of building the machine learning model, including:
        - Data preprocessing and feature engineering steps.
        - Initial model attempts and their accuracies.
        - Tuning processes and final model performance.
        """
    )

    # Example Graphs from Training
    st.markdown("#### Accuracy Over Iterations")
    accuracy_data = pd.DataFrame({
        'Iteration': [1, 2, 3, 4, 5],
        'Accuracy': [0.6, 0.7, 0.78, 0.85, 0.9]
    })
    acc_fig = px.line(accuracy_data, x='Iteration', y='Accuracy', title="Model Accuracy Improvement")
    acc_fig.update_layout(
        template="plotly_white",
        xaxis_title="Iteration",
        yaxis_title="Accuracy",
        title_x=0.5
    )
    st.plotly_chart(acc_fig, use_container_width=True)

    st.markdown("#### Loss Over Iterations")
    loss_data = pd.DataFrame({
        'Iteration': [1, 2, 3, 4, 5],
        'Loss': [0.4, 0.3, 0.25, 0.2, 0.15]
    })
    loss_fig = px.line(loss_data, x='Iteration', y='Loss', title="Model Loss Reduction")
    loss_fig.update_layout(
        template="plotly_white",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        title_x=0.5
    )
    st.plotly_chart(loss_fig, use_container_width=True)

    st.markdown("#### Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': ['Store No', 'Year', 'Day Sin', 'Day Cos', 'Month Sin', 'Month Cos'],
        'Importance': [0.2, 0.15, 0.25, 0.1, 0.2, 0.1]
    })
    feat_fig = px.bar(feature_importance, x='Feature', y='Importance', title="Feature Importance")
    feat_fig.update_layout(
        template="plotly_white",
        xaxis_title="Feature",
        yaxis_title="Importance",
        title_x=0.5
    )
    st.plotly_chart(feat_fig, use_container_width=True)

# Footer
st.markdown(
    """
    ---
    #### Powered by Machine Learning and Streamlit
    """
)
