import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stock-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_model(self, model_path='stock_dl_model.h5'):
        try:
            self.model = load_model(model_path)
            return True
        except:
            return False
    
    def download_stock_data(self, stock, start_date, end_date):
        """Download stock data from Yahoo Finance"""
        try:
            df = yf.download(stock, start=start_date, end=end_date)
            # Ensure we have a clean DataFrame with single column names
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return None
    
    def prepare_data(self, data, lookback=100):
        """Prepare data for model prediction"""
        # Ensure data is 1D array
        if isinstance(data, pd.Series):
            data_values = data.values
        else:
            data_values = data
            
        data_array = self.scaler.fit_transform(data_values.reshape(-1, 1))
        
        x_test = []
        for i in range(lookback, len(data_array)):
            x_test.append(data_array[i-lookback:i, 0])
        
        return np.array(x_test)
    
    def predict(self, x_data):
        """Make predictions using the loaded model"""
        if self.model is None:
            return None
        
        predictions = self.model.predict(x_data)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions.flatten()

def safe_float_format(value, default=0.0):
    """Safely format float values handling various data types"""
    try:
        if hasattr(value, 'iloc'):
            value = value.iloc[0] if hasattr(value, 'iloc') else value
        if hasattr(value, 'values'):
            value = value.values[0] if len(value.values) > 0 else default
        return float(value)
    except (ValueError, TypeError, IndexError):
        return default

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = StockPredictor()
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Stock selection
    stock_symbol = st.sidebar.text_input("Stock Symbol", value="GOOG").upper()
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=dt.date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=dt.date(2024, 12, 31))
    
    # Model loading
    st.sidebar.subheader("Model Settings")
    model_loaded = predictor.load_model()
    
    if model_loaded:
        st.sidebar.success("✅ Model loaded successfully!")
    else:
        st.sidebar.warning("⚠️ Model file not found. Predictions will not be available.")
    
    # Main content
    if st.sidebar.button("Analyze Stock") or 'data_loaded' in st.session_state:
        
        with st.spinner("Downloading stock data..."):
            df = predictor.download_stock_data(stock_symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            st.session_state.data_loaded = True
            
            # Ensure we have proper numeric values
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Display basic info - FIXED VERSION
            st.markdown(f'<div class="stock-card">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Safe extraction of current price
                try:
                    current_price = float(df['Close'].iloc[-1])
                    st.metric("Current Price", f"${current_price:.2f}")
                except (ValueError, TypeError, IndexError) as e:
                    st.metric("Current Price", "N/A")
                    st.error(f"Error reading current price: {e}")
            
            with col2:
                # Safe calculation of daily change
                try:
                    if len(df) > 1:
                        current_price = float(df['Close'].iloc[-1])
                        prev_price = float(df['Close'].iloc[-2])
                        price_change = current_price - prev_price
                        change_percent = (price_change / prev_price) * 100
                        st.metric("Daily Change", f"${price_change:.2f}", f"{change_percent:.2f}%")
                    else:
                        st.metric("Daily Change", "N/A")
                except (ValueError, TypeError, IndexError):
                    st.metric("Daily Change", "N/A")
            
            with col3:
                # Safe volume display
                try:
                    volume = int(df['Volume'].iloc[-1])
                    st.metric("Volume", f"{volume:,}")
                except (ValueError, TypeError, IndexError):
                    st.metric("Volume", "N/A")
            
            with col4:
                st.metric("Data Points", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Price Chart", 
                "🕯️ Candlestick", 
                "📈 Technical Indicators", 
                "🤖 Model Prediction", 
                "📋 Raw Data"
            ])
            
            with tab1:
                st.subheader(f"{stock_symbol} Price History")
                
                # Price chart with moving averages
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Ensure we have numeric data for plotting
                close_prices = pd.to_numeric(df['Close'], errors='coerce')
                ax.plot(df.index, close_prices, label='Close Price', linewidth=2)
                
                # Calculate moving averages
                ma100 = close_prices.rolling(100, min_periods=1).mean()
                ma200 = close_prices.rolling(200, min_periods=1).mean()
                
                ax.plot(df.index, ma100, label='MA 100', linewidth=1, alpha=0.7)
                ax.plot(df.index, ma200, label='MA 200', linewidth=1, alpha=0.7)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (USD)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{stock_symbol} Stock Price with Moving Averages')
                
                st.pyplot(fig)
                
                # Volume chart
                st.subheader("Trading Volume")
                fig_vol, ax_vol = plt.subplots(figsize=(12, 4))
                
                volume_data = pd.to_numeric(df['Volume'], errors='coerce')
                ax_vol.fill_between(df.index, volume_data, alpha=0.3, color='orange')
                ax_vol.plot(df.index, volume_data, color='orange', alpha=0.8)
                ax_vol.set_xlabel('Date')
                ax_vol.set_ylabel('Volume')
                ax_vol.set_title(f'{stock_symbol} Trading Volume')
                ax_vol.grid(True, alpha=0.3)
                st.pyplot(fig_vol)
            
            with tab2:
                st.subheader("Candlestick Chart")
                
                # Reset index for candlestick
                df_reset = df.reset_index()
                
                # Ensure numeric data for candlestick
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=df_reset['Date'] if 'Date' in df_reset.columns else df_reset.index,
                    open=pd.to_numeric(df_reset['Open'], errors='coerce'),
                    high=pd.to_numeric(df_reset['High'], errors='coerce'),
                    low=pd.to_numeric(df_reset['Low'], errors='coerce'),
                    close=pd.to_numeric(df_reset['Close'], errors='coerce'),
                    name='Price'
                )])
                
                fig_candle.update_layout(
                    title=f"{stock_symbol} Candlestick Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    xaxis_rangeslider_visible=False,
                    height=600
                )
                
                st.plotly_chart(fig_candle, use_container_width=True)
            
            with tab3:
                st.subheader("Technical Indicators")
                
                # Calculate indicators
                df_indicators = df.copy()
                close_prices = pd.to_numeric(df_indicators['Close'], errors='coerce')
                
                df_indicators['MA_50'] = close_prices.rolling(50, min_periods=1).mean()
                df_indicators['MA_100'] = close_prices.rolling(100, min_periods=1).mean()
                df_indicators['EMA_50'] = close_prices.ewm(span=50, adjust=False).mean()
                df_indicators['EMA_100'] = close_prices.ewm(span=100, adjust=False).mean()
                
                fig_tech = make_subplots(
                    rows=2, cols=1, 
                    subplot_titles=('Moving Averages', 'Exponential Moving Averages'),
                    vertical_spacing=0.1
                )
                
                # Moving averages
                fig_tech.add_trace(
                    go.Scatter(x=df_indicators.index, y=close_prices, 
                              name='Close Price', line=dict(color='black')), 
                    row=1, col=1
                )
                fig_tech.add_trace(
                    go.Scatter(x=df_indicators.index, y=df_indicators['MA_50'], 
                              name='MA 50', line=dict(color='blue')), 
                    row=1, col=1
                )
                fig_tech.add_trace(
                    go.Scatter(x=df_indicators.index, y=df_indicators['MA_100'], 
                              name='MA 100', line=dict(color='red')), 
                    row=1, col=1
                )
                
                # Exponential moving averages
                fig_tech.add_trace(
                    go.Scatter(x=df_indicators.index, y=close_prices, 
                              name='Close Price', line=dict(color='black'), 
                              showlegend=False), 
                    row=2, col=1
                )
                fig_tech.add_trace(
                    go.Scatter(x=df_indicators.index, y=df_indicators['EMA_50'], 
                              name='EMA 50', line=dict(color='green')), 
                    row=2, col=1
                )
                fig_tech.add_trace(
                    go.Scatter(x=df_indicators.index, y=df_indicators['EMA_100'], 
                              name='EMA 100', line=dict(color='orange')), 
                    row=2, col=1
                )
                
                fig_tech.update_layout(height=800, showlegend=True, 
                                     title_text=f"{stock_symbol} Technical Indicators")
                st.plotly_chart(fig_tech, use_container_width=True)
            
            with tab4:
                st.subheader("LSTM Model Predictions")
                
                if model_loaded:
                    # Prepare data for prediction
                    lookback = 100
                    if len(df) > lookback:
                        # Use last portion of data for testing
                        test_data = df['Close'].iloc[-500:]  # Last 500 points
                        
                        with st.spinner("Making predictions..."):
                            x_test = predictor.prepare_data(test_data, lookback)
                            
                            if len(x_test) > 0:
                                x_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
                                predictions = predictor.predict(x_test_reshaped)
                                
                                # Create comparison plot
                                actual_prices = test_data.iloc[lookback:].values
                                
                                # Ensure we have matching lengths
                                min_length = min(len(actual_prices), len(predictions))
                                actual_prices = actual_prices[:min_length]
                                predictions = predictions[:min_length]
                                
                                fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
                                ax_pred.plot(actual_prices, label='Actual Price', linewidth=2)
                                ax_pred.plot(predictions, label='Predicted Price', linewidth=2, alpha=0.8)
                                ax_pred.set_xlabel('Time Steps')
                                ax_pred.set_ylabel('Price (USD)')
                                ax_pred.legend()
                                ax_pred.grid(True, alpha=0.3)
                                ax_pred.set_title('Actual vs Predicted Stock Prices')
                                
                                st.pyplot(fig_pred)
                                
                                # Prediction metrics
                                mse = np.mean((actual_prices - predictions) ** 2)
                                st.metric("Mean Squared Error", f"{mse:.2f}")
                                
                            else:
                                st.warning("Not enough data for predictions after preprocessing.")
                    else:
                        st.warning(f"Not enough historical data for predictions. Need more than {lookback} data points, but only have {len(df)}.")
                else:
                    st.error("Model not loaded. Please ensure the model file 'stock_dl_model.h5' is available in the directory.")
            
            with tab5:
                st.subheader("Raw Data")
                
                # Display data with proper formatting
                display_df = df.copy()
                # Convert numeric columns to proper format
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_cols:
                    if col in display_df.columns:
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                
                st.dataframe(display_df.tail(100))  # Show last 100 rows
                
                # Data statistics
                st.subheader("Data Statistics")
                st.dataframe(display_df.describe())
                
                # Download data
                csv = display_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{stock_symbol}_stock_data.csv",
                    mime="text/csv"
                )
        
        else:
            st.error("Failed to download stock data. Please check the stock symbol and date range.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Stock Price Prediction App using LSTM Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()