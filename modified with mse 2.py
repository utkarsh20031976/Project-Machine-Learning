# AI-Based Electricity Demand Forecasting System
# Complete implementation with data generation, model training, and Streamlit deployment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. DATA GENERATION
# ===============================

def generate_electricity_data(n_samples=8737):
    """
    Generate synthetic electricity demand data similar to Delhi power system
    """
    np.random.seed(42)
    
    # Generate date range
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    data = []
    
    for i, date in enumerate(dates):
        # Base load varies by hour of day (MW)
        hour = date.hour
        base_load = 800 + 400 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 6 PM
        
        # Seasonal variation
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Weekend effect
        is_weekend = date.weekday() >= 5
        weekend_factor = 0.85 if is_weekend else 1.0
        
        # Temperature effect (synthetic)
        temp_base = 25 + 10 * np.sin(2 * np.pi * day_of_year / 365)
        temperature = temp_base + np.random.normal(0, 5)
        temp_factor = 1 + 0.02 * abs(temperature - 22)  # Cooling/heating effect
        
        # Calculate final load
        load = base_load * seasonal_factor * weekend_factor * temp_factor
        load += np.random.normal(0, 30)  # Add noise
        
        # Generate other features
        humidity = max(30, min(90, 60 + np.random.normal(0, 15)))
        wind_speed = max(0, np.random.exponential(8))
        rainfall = max(0, np.random.exponential(2) if np.random.random() < 0.3 else 0)
        
        # Holiday/Festival flags
        is_holiday = np.random.random() < 0.05  # 5% chance
        is_festival = np.random.random() < 0.02  # 2% chance
        
        # Real estate development
        dev_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
        
        data.append({
            'DateTime': date,
            'Load_MW': max(200, load),  # Minimum load
            'Temperature_C': temperature,
            'Humidity_Percent': humidity,
            'WindSpeed_kmh': wind_speed,
            'Rainfall_mm': rainfall,
            'Hour': hour,
            'DayOfWeek': date.weekday(),
            'Month': date.month,
            'IsWeekend': int(is_weekend),
            'IsHoliday': int(is_holiday),
            'IsFestival': int(is_festival),
            'RealEstateDev': dev_level
        })
    
    return pd.DataFrame(data)

# ===============================
# 2. FEATURE ENGINEERING
# ===============================

def engineer_features(df):
    """
    Perform feature engineering on the dataset
    """
    df = df.copy()
    
    # Label encode categorical variables
    le = LabelEncoder()
    df['RealEstateDev_Encoded'] = le.fit_transform(df['RealEstateDev'])
    
    # Create additional time-based features
    df['HourSin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['HourCos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Temperature interaction features
    df['TempHumidity'] = df['Temperature_C'] * df['Humidity_Percent'] / 100
    df['TempSquared'] = df['Temperature_C'] ** 2
    
    return df

# ===============================
# 3. MODEL TRAINING
# ===============================

class ElectricityForecastingSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.results = {}
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Feature columns (excluding target and non-numeric)
        self.feature_columns = [
            'Temperature_C', 'Humidity_Percent', 'WindSpeed_kmh', 'Rainfall_mm',
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsHoliday', 'IsFestival',
            'RealEstateDev_Encoded', 'HourSin', 'HourCos', 'MonthSin', 'MonthCos',
            'TempHumidity', 'TempSquared'
        ]
        
        X = df[self.feature_columns]
        y = df['Load_MW']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all three models"""
        
        # 1. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Calculate all metrics including MSE
        rf_mse = mean_squared_error(y_test, rf_pred)
        
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {
            'MAE': mean_absolute_error(y_test, rf_pred),
            'MSE': rf_mse,
            'RMSE': np.sqrt(rf_mse),
            'predictions': rf_pred
        }
        
        # 2. XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        # Calculate all metrics including MSE
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = {
            'MAE': mean_absolute_error(y_test, xgb_pred),
            'MSE': xgb_mse,
            'RMSE': np.sqrt(xgb_mse),
            'predictions': xgb_pred
        }
        
        # 3. Support Vector Regression
        print("Training SVR...")
        svr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=100, gamma='scale'))
        ])
        svr_pipeline.fit(X_train, y_train)
        svr_pred = svr_pipeline.predict(X_test)
        
        # Calculate all metrics including MSE
        svr_mse = mean_squared_error(y_test, svr_pred)
        
        self.models['SVR'] = svr_pipeline
        self.results['SVR'] = {
            'MAE': mean_absolute_error(y_test, svr_pred),
            'MSE': svr_mse,
            'RMSE': np.sqrt(svr_mse),
            'predictions': svr_pred
        }
        
        # Store test data for visualization
        self.y_test = y_test
        self.X_test = X_test
        
        return self.results
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        if model_name in ['Random Forest', 'XGBoost']:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                return importance_df
        return None
    
    def predict(self, model_name, features):
        """Make prediction with specified model"""
        if model_name in self.models:
            # Ensure features are in the right order
            feature_array = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
            prediction = self.models[model_name].predict(feature_array)[0]
            return max(0, prediction)  # Ensure non-negative
        return None

# ===============================
# 4. STREAMLIT APPLICATION
# ===============================

def main():
    st.set_page_config(
        page_title="AI-Based Electricity Demand Forecasting",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ AI-Based Electricity Demand Forecasting System by Utkarsh Singh")
    st.markdown("**Delhi Power System - Real-time Load Prediction**")
    
    # Initialize session state
    if 'system' not in st.session_state:
        with st.spinner("Loading forecasting system..."):
            # Generate data
            df = generate_electricity_data()
            df_engineered = engineer_features(df)
            
            # Train models
            system = ElectricityForecastingSystem()
            X_train, X_test, y_train, y_test = system.prepare_data(df_engineered)
            results = system.train_models(X_train, X_test, y_train, y_test)
            
            st.session_state.system = system
            st.session_state.df = df_engineered
            st.session_state.results = results
    
    system = st.session_state.system
    df = st.session_state.df
    results = st.session_state.results
    
    # Sidebar for model selection and inputs
    st.sidebar.header("🔧 Model Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Forecasting Model:",
        ["XGBoost", "Random Forest", "SVR"]
    )
    
    st.sidebar.header("📊 Input Parameters")
    
    # Input parameters
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (°C)", -5, 50, 25)
        humidity = st.slider("Humidity (%)", 20, 100, 60)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
        rainfall = st.slider("Rainfall (mm)", 0, 50, 0)
    
    with col2:
        hour = st.selectbox("Hour of Day", range(24), 12)
        day_of_week = st.selectbox("Day of Week", 
                                 ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], 0)
        month = st.selectbox("Month", range(1, 13), 6)
        real_estate = st.selectbox("Real Estate Development", ["Low", "Medium", "High"], 1)
    
    is_weekend = st.sidebar.checkbox("Weekend", value=day_of_week in ["Sat", "Sun"])
    is_holiday = st.sidebar.checkbox("Public Holiday")
    is_festival = st.sidebar.checkbox("Festival Day")
    
    # Prepare input features
    le_mapping = {"Low": 0, "Medium": 1, "High": 2}
    dow_mapping = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    
    input_features = {
        'Temperature_C': temperature,
        'Humidity_Percent': humidity,
        'WindSpeed_kmh': wind_speed,
        'Rainfall_mm': rainfall,
        'Hour': hour,
        'DayOfWeek': dow_mapping[day_of_week],
        'Month': month,
        'IsWeekend': int(is_weekend),
        'IsHoliday': int(is_holiday),
        'IsFestival': int(is_festival),
        'RealEstateDev_Encoded': le_mapping[real_estate],
        'HourSin': np.sin(2 * np.pi * hour / 24),
        'HourCos': np.cos(2 * np.pi * hour / 24),
        'MonthSin': np.sin(2 * np.pi * month / 12),
        'MonthCos': np.cos(2 * np.pi * month / 12),
        'TempHumidity': temperature * humidity / 100,
        'TempSquared': temperature ** 2
    }
    
    # Make prediction
    prediction = system.predict(model_choice, input_features)
    
    # Main content area
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.subheader(f"🎯 {model_choice} Prediction")
        st.metric(
            label="Predicted Electricity Load",
            value=f"{prediction:.2f} MW",
            delta=f"Model: {model_choice}"
        )
        
        # Show model performance with MSE
        st.subheader("📈 Model Performance")
        perf_data = {
            'Metric': ['MAE (MW)', 'MSE (MW²)', 'RMSE (MW)'],
            'Value': [f"{results[model_choice]['MAE']:.2f}", 
                     f"{results[model_choice]['MSE']:.2f}",
                     f"{results[model_choice]['RMSE']:.2f}"]
        }
        st.table(pd.DataFrame(perf_data))
    
    with col2:
        st.subheader("🔍 Input Summary")
        st.write(f"**Time:** {hour}:00")
        st.write(f"**Day:** {day_of_week}")
        st.write(f"**Month:** {month}")
        st.write(f"**Temp:** {temperature}°C")
        st.write(f"**Humidity:** {humidity}%")
        st.write(f"**Development:** {real_estate}")
        
        if is_weekend:
            st.write("🏠 **Weekend**")
        if is_holiday:
            st.write("🎉 **Holiday**")
        if is_festival:
            st.write("🎊 **Festival**")
    
    with col3:
        st.subheader("📊 Model Comparison")
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'MAE': result['MAE'],
                'MSE': result['MSE'],
                'RMSE': result['RMSE']
            })
        
        comp_df = pd.DataFrame(comparison_data)
        fig = px.bar(comp_df, x='Model', y=['MAE', 'MSE', 'RMSE'], 
                    title="Model Performance Comparison",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional MSE Analysis Section
    st.header("📊 Detailed Performance Analysis")
    
    # Create metrics comparison table
    st.subheader("🎯 Complete Model Performance Metrics")
    
    performance_df = pd.DataFrame(comparison_data)
    performance_df = performance_df.round(2)
    
    # Add ranking for each metric
    performance_df['MAE_Rank'] = performance_df['MAE'].rank()
    performance_df['MSE_Rank'] = performance_df['MSE'].rank()
    performance_df['RMSE_Rank'] = performance_df['RMSE'].rank()
    
    st.dataframe(performance_df, use_container_width=True)
    
    # Best model identification
    best_mae_model = performance_df.loc[performance_df['MAE'].idxmin(), 'Model']
    best_mse_model = performance_df.loc[performance_df['MSE'].idxmin(), 'Model']
    best_rmse_model = performance_df.loc[performance_df['RMSE'].idxmin(), 'Model']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Best MAE", best_mae_model, f"{performance_df.loc[performance_df['Model'] == best_mae_model, 'MAE'].iloc[0]:.2f} MW")
    with col2:
        st.metric("🏆 Best MSE", best_mse_model, f"{performance_df.loc[performance_df['Model'] == best_mse_model, 'MSE'].iloc[0]:.2f} MW²")
    with col3:
        st.metric("🏆 Best RMSE", best_rmse_model, f"{performance_df.loc[performance_df['Model'] == best_rmse_model, 'RMSE'].iloc[0]:.2f} MW")
    
    # Detailed Analysis Section
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction Visualization", "Feature Importance", "Historical Data", "Error Analysis"])
    
    with tab1:
        # Show recent predictions vs actual
        n_points = 50
        actual = system.y_test.iloc[:n_points].values
        predicted = results[model_choice]['predictions'][:n_points]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(n_points)),
            y=actual,
            mode='lines+markers',
            name='Actual Load',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(n_points)),
            y=predicted,
            mode='lines+markers',
            name=f'{model_choice} Prediction',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f"{model_choice} - Predicted vs Actual Load (Last 50 Test Points)",
            xaxis_title="Time Points",
            yaxis_title="Load (MW)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if model_choice in ['Random Forest', 'XGBoost']:
            importance_df = system.get_feature_importance(model_choice)
            if importance_df is not None:
                fig = px.bar(
                    importance_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f"{model_choice} - Top 10 Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is not available for SVR model.")
    
    with tab3:
        st.subheader("📈 Historical Load Patterns")
        
        # Show load by hour of day
        hourly_avg = df.groupby('Hour')['Load_MW'].mean().reset_index()
        fig = px.line(hourly_avg, x='Hour', y='Load_MW', 
                     title="Average Load by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show load by day of week
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_avg = df.groupby('DayOfWeek')['Load_MW'].mean().reset_index()
        daily_avg['DayName'] = daily_avg['DayOfWeek'].map(lambda x: dow_names[x])
        
        fig = px.bar(daily_avg, x='DayName', y='Load_MW',
                    title="Average Load by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔍 Error Analysis")
        
        # Calculate residuals for current model
        actual = system.y_test.values
        predicted = results[model_choice]['predictions']
        residuals = actual - predicted
        
        # Residual plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predicted,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', opacity=0.6)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title=f"{model_choice} - Residual Plot",
            xaxis_title="Predicted Load (MW)",
            yaxis_title="Residuals (MW)",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(residuals, nbins=30, title="Distribution of Residuals")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error statistics
            error_stats = {
                'Statistic': ['Mean Error', 'Std Error', 'Min Error', 'Max Error', 'Mean Absolute Error'],
                'Value': [
                    f"{np.mean(residuals):.2f} MW",
                    f"{np.std(residuals):.2f} MW",
                    f"{np.min(residuals):.2f} MW",
                    f"{np.max(residuals):.2f} MW",
                    f"{np.mean(np.abs(residuals)):.2f} MW"
                ]
            }
            st.subheader("Error Statistics")
            st.table(pd.DataFrame(error_stats))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **AI-Based Electricity Demand Forecasting System**  
    Built with Machine Learning for Delhi Power System  
    Models: Random Forest, XGBoost, Support Vector Regression  
    Metrics: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error)
    """)

if __name__ == "__main__":
    main()
