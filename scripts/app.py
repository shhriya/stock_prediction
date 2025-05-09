import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
from datetime import date, timedelta
import pandas_market_calendars as mcal
from main import (
    CONFIG, get_date_range, download_data, is_stationary,
    decompose_series, fit_sarima_model, forecast, prepare_forecast_dataframe
)
#hi
def main():
    st.title('Stock Market Forecasting App')
    st.subheader('This app forecasts the stock market price of selected companies.')
    
    st.sidebar.header('Select the parameters from below')
    
    # Get date range from BigQuery
    try:
        min_date, max_date = get_date_range(CONFIG)
    except Exception as e:
        st.error(f"Error fetching date range: {str(e)}")
        min_date = date(2020, 1, 1)
        max_date = date.today()
    
    start_date = st.sidebar.date_input('Start date', min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input('End date', max_date, min_value=min_date, max_value=max_date)
    
    # Select ticker
    ticker = st.sidebar.selectbox('Select the ticker symbol', ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "NFLX", "META", "JPM"])
    
    # Download and show data
    data = download_data(CONFIG, ticker, start_date, end_date)
    st.subheader(f"Data for {ticker}")
    st.write(data)
    
    # Plot the data
    st.header('Data Visualization')
    fig = px.line(data, x='date', y=data.columns, title='Stock Price Overview', width=1000, height=600)
    st.plotly_chart(fig)
    
    # Subset column
    column = st.selectbox("Select column for forecasting", data.columns[1:])
    data = data[['date', column]]
    
    st.write("Selected Data")
    st.write(data)
    
    # Stationarity test
    st.header('Is the data stationary?')
    stationary = is_stationary(data[column])
    st.write("Stationary:" if stationary else "Not stationary")
    
    # Decomposition
    # st.header("Decomposition of the data")
    decomposition = decompose_series(data[column])
    # st.write(decomposition.plot())
    
    st.write("## Decomposed Components")
    st.plotly_chart(px.line(x=data["date"], y=decomposition.trend, title='Trend'))
    st.plotly_chart(px.line(x=data["date"], y=decomposition.seasonal, title='Seasonality'))
    st.plotly_chart(px.line(x=data["date"], y=decomposition.resid, title='Residuals'))
    
    # Model training with default parameters (p=1, d=1, q=2, seasonal_order=7)
    st.header("Model Training")
    
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    st.write(f"Training on {train_size} data points, holding out {len(data) - train_size} for testing")
    
    # Train the model
    model = fit_sarima_model(data, column)
    
    # Show model summary
    # st.header("Model Summary")
    # st.write(model.summary())
    
    # Calculate and show metrics on test data
    test_data = data[column][train_size:]
    if len(test_data) > 0:
        st.subheader("Model Performance on Test Data")
    
        # Get predictions for test period with exogenous variables
        test_exog = model._exog[train_size:]
        test_predictions = model.get_forecast(steps=len(test_data), exog=test_exog)
        test_mean = (test_predictions.predicted_mean * model._train_iqr) + model._train_median
    
        # Calculate error metrics
        mse = np.mean((test_data - test_mean) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_data - test_mean))
        mape = np.mean(np.abs((test_data - test_mean) / test_data)) * 100
    
        # Calculate percentage of predictions within different error ranges
        errors = np.abs((test_data - test_mean) / test_data) * 100
        within_5_percent = np.mean(errors <= 5) * 100
        within_10_percent = np.mean(errors <= 10) * 100
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.write("Absolute Error Metrics:")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    
        with col2:
            st.write("Relative Error Metrics:")
            st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            st.write(f"Predictions within 5% of actual: {within_5_percent:.1f}%")
            st.write(f"Predictions within 10% of actual: {within_10_percent:.1f}%")
    
        # Plot actual vs predicted values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['date'][train_size:], y=test_data, name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data['date'][train_size:], y=test_mean, name='Predicted', line=dict(color='red')))
        fig.update_layout(title='Model Performance on Test Data', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
    
    
    st.subheader("Model Diagnostics")
    st.write(f"**AIC:** {model.aic}")
    st.write(f"**BIC:** {model.bic}")
    
    # # Model Diagnostics for Training Period
    # st.subheader("Residual Analysis (Training Period)")
    
    # Get residuals for training period
    train_residuals = model.resid
    train_dates = data['date'][:train_size]
    
    # # Plot 1: Histogram of residuals
    # fig_hist = ff.create_distplot(
    #     [train_residuals.dropna()],
    #     group_labels=['Training Residuals'],
    #     bin_size=0.5
    # )
    # fig_hist.update_layout(title='Distribution of Training Residuals')
    # st.plotly_chart(fig_hist, use_container_width=True)
    
    # # Plot 2: Residuals over time
    # residuals_fig = px.line(
    #     x=train_dates,
    #     y=train_residuals,
    #     title='Residuals Over Time (Training Period)'
    # )
    # residuals_fig.add_hline(y=0, line_dash="dash", line_color="red")
    # st.plotly_chart(residuals_fig)
    
    # # Plot 3: Autocorrelation plot
    # fig_acf, ax = plt.subplots(figsize=(10, 4))
    # plot_acf(train_residuals.dropna(), ax=ax, lags=40)
    # plt.title('Autocorrelation of Training Residuals')
    # buf = io.BytesIO()
    # fig_acf.savefig(buf, format="png")
    # st.image(buf)
    
    
    # Forecasting
    st.markdown("<p style='color:green; font-size: 30px; font-weight: bold;'>Forecasting</p>", unsafe_allow_html=True)
    st.write("Note: Forecasts will be shown only for business days (Monday-Friday)")
    trading_days = st.number_input('Trading days to forecast', 1, 365, 10)
    
    try:
        if len(data) > 0:
            # Start forecasting from the last actual data point
            start_forecast_date = data['date'].iloc[-1]
            if start_forecast_date.weekday() >= 5:  # If last date is weekend
                start_forecast_date += pd.Timedelta(days=(7 - start_forecast_date.weekday()))  # Move to Monday
        
            # Generate forecasts with confidence intervals
            preds, lower, upper = forecast(model, data, column, 0, trading_days)
        
            if preds is not None:
                # Create forecast dataframe with confidence intervals
                pred_df = pd.DataFrame({
                    'date': pd.date_range(start=start_forecast_date, periods=len(preds), freq='B'),
                    'predicted_mean': preds,
                    'lower_ci': lower,
                    'upper_ci': upper
                })
            
                if len(pred_df) > trading_days:
                    pred_df = pred_df.head(trading_days)
                
                # Show forecast statistics
                # st.subheader("Forecast Statistics")
                avg_range = (pred_df['upper_ci'] - pred_df['lower_ci']).mean()
                max_range = (pred_df['upper_ci'] - pred_df['lower_ci']).max()
            
                col1, col2 = st.columns(2)
            else:
                st.error("Failed to generate predictions")
                pred_df = pd.DataFrame(columns=['date', 'predicted_mean', 'lower_ci', 'upper_ci'])
        else:
            st.error("Not enough data points for forecasting")
            pred_df = pd.DataFrame(columns=['date', 'predicted_mean', 'lower_ci', 'upper_ci'])
    
    except Exception as e:
        st.error(f"Error during forecasting: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        pred_df = pd.DataFrame(columns=['date', 'predicted_mean', 'lower_ci', 'upper_ci'])
    
    st.write("Predictions")
    st.write(pred_df)
    
    # Plot predictions with confidence intervals
    fig = go.Figure()
    
    # Plot actual data
    fig.add_trace(go.Scatter(
        x=data["date"],
        y=data[column],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Plot predicted mean
    fig.add_trace(go.Scatter(
        x=pred_df["date"],
        y=pred_df["predicted_mean"],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence interval as a shaded area
    fig.add_trace(go.Scatter(
        x=pred_df["date"].tolist() + pred_df["date"].tolist()[::-1],
        y=pred_df["upper_ci"].tolist() + pred_df["lower_ci"].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,0,0,0)'),
        name='95% Confidence Interval'
    ))
    
    # Update layout
    fig.update_layout(
        title='Stock Price Forecast with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Price',
        width=1200,
        height=500,
        showlegend=True,
        hovermode='x'
    )
    
    st.plotly_chart(fig)



if __name__ == "__main__":
    main()
