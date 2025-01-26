import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import joblib
import plotly.express as px

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
    model_path = './xgboost_model_2023.json'
    scaler_path = './target_scaler.pkl'

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
store_data = pd.read_csv('./preprocessed_data_2.csv')

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

# Add "Select All" checkbox
select_all = st.sidebar.checkbox("Select All Stores")
if select_all:
    store_nos = store_data['store_no'].unique()

# Dynamic date range input
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp('2024-01-01'))
days_to_predict = st.sidebar.slider("Number of Days to Predict", min_value=7, max_value=30, value=14)

# Tabs for navigation
tabs = st.tabs(["Dashboard", "Model Training Journey"])

# Dashboard Tab
with tabs[0]:
    # Prediction and visualization logic
    if st.sidebar.button("Predict Turnover"):
        if len(store_nos) == 0:
            st.sidebar.warning("Please select at least one store.")
        else:
            if select_all:
                st.markdown("### Turnover Predictions for All Stores")

                with st.spinner("Generating predictions..."):
                    predictions = predict_turnover(store_nos, start_date, days_to_predict)

                # Display predictions in a table
                st.write(predictions)

                # Add download button for all stores' predictions
                csv = predictions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download All Predictions as CSV",
                    data=csv,
                    file_name=f"turnover_predictions_all_stores.csv",
                    mime="text/csv"
                )
            else:
                st.markdown(f"### Turnover Predictions for Store(s) {', '.join(map(str, store_nos))}")
                
                with st.spinner("Generating predictions..."):
                    predictions = predict_turnover(store_nos, start_date, days_to_predict)

                st.dataframe(predictions, use_container_width=True)

                # Metrics for all stores
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

                # Download CSV for individual store predictions
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
    
    # Introduction
    st.write(
        """
        This project aimed to predict store turnover for NKD using machine learning techniques. Initially, two models were developed:
        a Long Short-Term Memory (LSTM) model and an XGBoost model. After comparison, the XGBoost model was chosen for further
        optimization due to its superior performance.
        """
    )
    
    # Feature Engineering
    st.markdown("#### Feature Engineering")
    st.write(
        """
        Several feature engineering techniques were applied to improve model performance:
        
        - **Label Encoding**: Used for categorical variables.
        - **Cyclical Encoding**: Handled cyclical features like day and month to capture the periodic nature of the data.
        - **Normalization**: Scaled numerical features to ensure consistency in input data.
        - **Lag Features**: Historical turnover data was used to capture time dependencies.
        
        These transformations enabled the model to identify complex patterns in the turnover data.
        """
    )

    # Model Comparison
    st.markdown("#### Model Comparison")
    st.write(
        """
        Two models were compared: LSTM and XGBoost. Despite the LSTM model's ability to handle sequential data, it underperformed
        compared to the XGBoost model, which demonstrated stronger predictive power. Key results:
        
        - **LSTM Performance**:
            - R²: 0.11 (only explained 10.7% of the variance)
        
        - **XGBoost (initial)**:
            - R²: 0.33 (still room for improvement)
        """
    )

        # Display LSTM and XGBoost comparison as separate images side by side
    col1, col2 = st.columns([1, 1])  # You can adjust the ratio if you want one image to be larger

    with col1:
        st.image('./previous xgboost graph 1.png', caption='LSTM Performance', use_container_width=True)  # Image 1

    with col2:
        st.image('./latest actual v pred.png', caption='XGBoost Performance', use_container_width=True)  # Image 2

            # Display LSTM and XGBoost comparison as separate images side by side
    col3, col4 = st.columns([1, 1])  # You can adjust the ratio if you want one image to be larger

    with col3:
        st.image('./lstm_scatter_plot.png', caption='LSTM Scatter Plot', use_container_width=True)  # Image 1

    with col4:
        st.image('./xgboost_scatter_plot.png', caption='XGBoost Scatter Plot', use_container_width=True)  # Image 2



    # XGBoost Optimization
    st.markdown("#### XGBoost Optimization")
    st.write(
        """
        The XGBoost model was fine-tuned with the following hyperparameters:
        
        - Learning Rate: 0.01
        - Maximum Depth: 4
        - Minimum Child Weight: 5
        - Regularization Parameters: Reg Alpha (0.7), Reg Lambda (1.0)
        - Subsample: 0.8
        - Colsample By Tree: 0.8
        - Boosting Rounds: 500
        - Early Stopping: 40 rounds
        
        After optimization, the model achieved the following results:
        
        - **MAE**: 6.48
        - **MSE**: 124.53
        - **R²**: 0.96 (explains 96% of variance)
        """
    )

    col5, col6 = st.columns([1, 1])  # You can adjust the ratio if you want one image to be larger

    with col5:
        st.image('./final xgboost 1.png', caption='Actual vs Predicted Turnover', use_container_width=True)  # Image 1

    with col6:
        st.image('./final xgboost 2.png', caption='Scatter plot', use_container_width=True)  # Image 2

    # Final Conclusion
    st.markdown("#### Conclusion")
    st.write(
        """
        The optimized XGBoost model demonstrated excellent predictive performance, with a high R² score of 0.96. This model was
        then integrated into an interactive dashboard, enabling stakeholders to visualize turnover forecasts and make informed
        business decisions based on data-driven insights.
        """
    )


# Footer
st.markdown(
    """
    ---
    #### Authors: Irtaza Janjua, Soman Tariq, Annalena Wieser
    """
)