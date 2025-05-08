import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import pandas as pd
import io
from datetime import date
from train_model import (
    CONFIG, get_date_range, download_data, is_stationary,
    decompose_series, fit_sarima_model, forecast, prepare_forecast_dataframe
)
 
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
st.header("Decomposition of the data")
decomposition = decompose_series(data[column])
st.write(decomposition.plot())
 
st.write("## Decomposed Components")
st.plotly_chart(px.line(x=data["date"], y=decomposition.trend, title='Trend'))
st.plotly_chart(px.line(x=data["date"], y=decomposition.seasonal, title='Seasonality'))
st.plotly_chart(px.line(x=data["date"], y=decomposition.resid, title='Residuals'))
 
# Model training with default parameters (p=1, d=1, q=2, seasonal_order=7)
model = fit_sarima_model(data[column])
st.header("Model Summary")
st.write(model.summary())
 
 
st.subheader("Model Diagnostics")
st.write(f"**AIC:** {model.aic}")
st.write(f"**BIC:** {model.bic}")
 
# Residuals
residuals = model.resid
 
st.subheader("Residual Analysis")
 
# Plot 1: Histogram of residuals
fig_hist = ff.create_distplot([residuals.dropna()], group_labels=['Residuals'], bin_size=0.5)
st.plotly_chart(fig_hist, use_container_width=True)
 
# Plot 2: Residuals over time
st.plotly_chart(px.line(x=data['date'], y=residuals, title='Residuals Over Time'))
 
# Plot 3: Autocorrelation plot
fig_acf, ax = plt.subplots()
plot_acf(residuals.dropna(), ax=ax, lags=40)
buf = io.BytesIO()
fig_acf.savefig(buf, format="png")
st.image(buf)
 
 
# Forecasting
st.markdown("<p style='color:green; font-size: 30px; font-weight: bold;'>Forecasting</p>", unsafe_allow_html=True)
forecast_period = st.number_input('Days to forecast', 1, 365, 10)
 
# Start forecasting from the next day after the last date
start_forecast_date = end_date + pd.Timedelta(days=1)
preds = forecast(model, len(data), len(data) + forecast_period - 1)  # Subtract 1 to not include the last date
pred_df = prepare_forecast_dataframe(preds, start_date=start_forecast_date)
 
st.write("Predictions")
st.write(pred_df)
 
# Plot predictions
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["predicted_mean"], mode='lines', name='Forecasted', line=dict(color='red')))
fig.update_layout(title='Actual vs Forecast', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
st.plotly_chart(fig)
 