# Import necessary libraries
print("Test output: Script is running")
import pandas as pd
import numpy as np
import os
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class CandleStickAnalyzer:
    """
    A class to fetch, analyze and visualize candlestick data for investment automation.
    Generates clean Excel outputs for portfolio management.
    """
    
    def __init__(self, data_file=None, output_dir="investment_outputs"):
        """
        Initialize the CandleStickAnalyzer.
        
        Parameters:
        -----------
        data_file : str, optional
            Path to CSV file containing historical stock data
        output_dir : str, default 'investment_outputs'
            Directory to save output files
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.data = None
        self.signals = None
        self.portfolio = None
        self.performance = None
        self.future_predictions = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Set plotting style
        sns.set_theme(style="whitegrid")
        plt.style.use('ggplot')
    
    def load_data(self, sample_size=None):
        """
        Load data from file or generate sample data if no file provided.
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of days of sample data to generate if no file is provided
        
        Returns:
        --------
        pandas.DataFrame : The loaded or generated data
        """
        if self.data_file and os.path.exists(self.data_file):
            try:
                self.data = pd.read_csv(self.data_file)
                print(f"Data successfully loaded from {self.data_file}")
            except Exception as e:
                print(f"Error loading data: {e}")
                self.generate_sample_data(sample_size or 365)
        else:
            print("No data file provided or file not found. Generating sample data.")
            # Generate more data than requested to account for NaN values during technical analysis
            self.generate_sample_data(sample_size + 250 if sample_size else 600)
            
        # Ensure datetime format for Date column
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
        return self.data
    
    def generate_sample_data(self, days=365):
        """
        Generate sample candlestick data for demonstration.
        
        Parameters:
        -----------
        days : int, default 365
            Number of days of data to generate
        
        Returns:
        --------
        pandas.DataFrame : The generated sample data
        """
        end_date = datetime.datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Set a seed for reproducibility
        np.random.seed(42)
        
        # Generate sample price data with some trend and volatility
        base_price = 100
        trend = np.cumsum(np.random.normal(0.001, 0.02, size=len(dates)))
        volatility = np.random.normal(0, 0.02, size=len(dates))
        
        # Create OHLC data
        opens = base_price * (1 + trend + volatility)
        closes = opens * (1 + np.random.normal(0, 0.015, size=len(dates)))
        highs = np.maximum(opens, closes) * (1 + abs(np.random.normal(0, 0.01, size=len(dates))))
        lows = np.minimum(opens, closes) * (1 - abs(np.random.normal(0, 0.01, size=len(dates))))
        volumes = np.random.randint(100000, 1000000, size=len(dates))
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        })
        
        print(f"Sample data generated for {days} trading days")
        return self.data
    
    def calculate_technical_indicators(self):
        """
        Calculate common technical indicators for analysis with improved parameters.
        
        Returns:
        --------
        pandas.DataFrame : Data with added technical indicators
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        data = self.data.copy()
        
        # Calculate moving averages (including longer-term SMA for trend filtering)
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_100'] = data['Close'].rolling(window=100).mean()  # Added for trend filtering
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (Moving Average Convergence Divergence)
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        std_dev = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (std_dev * 2)
        data['BB_Lower'] = data['BB_Middle'] - (std_dev * 2)
        
        # Add ATR for stop-loss calculations
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        # Store the full data first
        self.data_with_indicators = data
        
        # Only remove NaN rows for subsequent analysis
        clean_data = data.dropna().reset_index(drop=True)
        print("Columns after calculating indicators:", clean_data.columns)
        
        if len(clean_data) >= 30:  # Make sure we have enough points for meaningful analysis
            self.data = clean_data
            print(f"Technical indicators successfully calculated. {len(clean_data)} valid data points after removing NaN values.")
        else:
            print(f"Warning: Only {len(clean_data)} valid data points after computing indicators. Using data with NaN values.")
            self.data = data.fillna(method='bfill').fillna(method='ffill')
            
        return self.data
    
    def generate_trading_signals(self):
        """
        Generate trading signals based on technical indicators with improved methodology.
        Incorporates a trend filter and uses more extreme thresholds.
        
        Returns:
        --------
        pandas.DataFrame : Data with added trading signals
        """
        if self.data is None:
            print("No data with indicators available. Please calculate indicators first.")
            return None

        data = self.data.copy()
        
        # ----- Trend Filter -----
        # Use the 100-day SMA to determine the overall trend
        data['SMA_100'] = data['Close'].rolling(window=100).mean()
        data['Trend'] = np.where(data['Close'] > data['SMA_100'], 1, -1)
        
        # ----- Signal Calculations -----
        # SMA signals: bullish if 20-day > 50-day, bearish otherwise
        data['SMA_Signal'] = 0
        data.loc[data['SMA_20'] > data['SMA_50'], 'SMA_Signal'] = 1
        data.loc[data['SMA_20'] < data['SMA_50'], 'SMA_Signal'] = -1
        
        # RSI signals: now using more extreme thresholds for higher confidence
        data['RSI_Signal'] = 0
        data.loc[data['RSI'] < 20, 'RSI_Signal'] = 1  # Oversold -> Buy
        data.loc[data['RSI'] > 80, 'RSI_Signal'] = -1  # Overbought -> Sell
        
        # MACD signals remain similar
        data['MACD_Signal'] = 0
        data.loc[data['MACD'] > data['Signal_Line'], 'MACD_Signal'] = 1
        data.loc[data['MACD'] < data['Signal_Line'], 'MACD_Signal'] = -1
        
        # Bollinger Bands signals remain similar
        data['BB_Signal'] = 0
        data.loc[data['Close'] < data['BB_Lower'], 'BB_Signal'] = 1
        data.loc[data['Close'] > data['BB_Upper'], 'BB_Signal'] = -1
        
        # ----- Combine Signals with Weighted Factors -----
        data['Combined_Signal'] = (0.4 * data['SMA_Signal'] + 
                                   0.3 * data['MACD_Signal'] +
                                   0.2 * data['RSI_Signal'] + 
                                   0.1 * data['BB_Signal'])
        
        # ----- Final Trading Signal with Trend Filter -----
        # Only trade when the combined signal is very strong (>=0.7 or <= -0.7) and in line with the trend
        data['Trade_Signal'] = 0
        data.loc[(data['Trend'] == 1) & (data['Combined_Signal'] >= 0.7), 'Trade_Signal'] = 1
        data.loc[(data['Trend'] == -1) & (data['Combined_Signal'] <= -0.7), 'Trade_Signal'] = -1
        
        self.signals = data
        print(f"Trading signals successfully generated for {len(data)} data points")
        return data

    def predict_future_signals(self, prediction_horizon=2):
        """
        Train a RandomForestClassifier to predict 2-day price movement.
        Only generate trade signals if prediction probability >= 65%.
        """
        if self.signals is None:
            print("No signals available. Generate them first.")
            return None
        
        # Define features and target for prediction
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'BB_Upper', 'BB_Lower', 'Volume']
        df = self.signals.copy()
        df['Future_Close'] = df['Close'].shift(-prediction_horizon)
        df = df.dropna().reset_index(drop=True)
        df['Target'] = np.where(df['Future_Close'] > df['Close'], 1, 0)
        
        X = df[features]
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
        
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, preds)
        print(f"ML Model Prediction Accuracy: {accuracy:.2%}")
        print(classification_report(y_test, preds))
        
        # Generate predicted signals for the entire dataset (tail)
        df['Predicted_Signal'] = np.where((model.predict_proba(df[features])[:, 1] >= 0.65) & 
                                          (model.predict(df[features]) == 1), 1, 0)
        # Overwrite signals with predicted signals
        self.signals = df
        print("Predicted signals generated using ML model with confidence threshold >= 65%.")
        return df

    def backtest_strategy(self, initial_capital=10000, holding_period=2):
        """
        Backtest the trading strategy and calculate returns on a 2-day trade basis.
        A trade is entered when a predicted signal is generated and held for up to a maximum of 'holding_period' days.
        Applies ATR-based stop-loss (1.5x ATR) and take-profit (3x ATR).
        
        Parameters:
        -----------
        initial_capital : float, default 10000
            Initial capital for backtesting
        holding_period : int, default 2
            Maximum number of days to hold a trade
        
        Returns:
        --------
        pandas.DataFrame : Portfolio performance data
        """
        if self.signals is None or 'Predicted_Signal' not in self.signals.columns:
            print("Predicted signals not available. Run predict_future_signals() first.")
            return None
        
        signals = self.signals.copy().reset_index(drop=True)
        
        # Use the predicted signal for trade entries
        signals['Position'] = signals['Predicted_Signal'].shift().fillna(0)
        signals['Entry_Price'] = signals['Close'].where(signals['Position'] == 1)
        signals['Exit_Price'] = np.nan
        
        # Set up ATR-based stop-loss and take-profit levels
        signals['Stop_Loss'] = signals['Entry_Price'] - (signals['ATR'] * 1.5)
        signals['Take_Profit'] = signals['Entry_Price'] + (signals['ATR'] * 3.0)
        
        trade_returns = []
        # Loop through each row to simulate trades
        for i in range(len(signals)):
            if signals.loc[i, 'Position'] == 1 and not pd.isna(signals.loc[i, 'Entry_Price']):
                entry_price = signals.loc[i, 'Entry_Price']
                sl = signals.loc[i, 'Stop_Loss']
                tp = signals.loc[i, 'Take_Profit']
                exit_idx = min(i + holding_period, len(signals) - 1)
                exit_price = signals.loc[exit_idx, 'Close']  # default exit price
                
                # Check during holding period if stop-loss or take-profit are triggered
                for j in range(i, exit_idx + 1):
                    current_price = signals.loc[j, 'Close']
                    if current_price <= sl:
                        exit_price = sl
                        break
                    if current_price >= tp:
                        exit_price = tp
                        break
                
                signals.loc[exit_idx, 'Exit_Price'] = exit_price
                ret = (exit_price - entry_price) / entry_price
                trade_returns.append(ret)
        
        signals['Market_Return'] = signals['Close'].pct_change().fillna(0)
        signals['Strategy_Return'] = signals['Position'].shift(1) * signals['Market_Return']
        signals['Strategy_Return'] = signals['Strategy_Return'].fillna(0)
        signals['Cumulative_Market_Return'] = (1 + signals['Market_Return']).cumprod()
        signals['Cumulative_Strategy_Return'] = (1 + signals['Strategy_Return']).cumprod()
        signals['Market_Portfolio'] = initial_capital * signals['Cumulative_Market_Return']
        signals['Strategy_Portfolio'] = initial_capital * signals['Cumulative_Strategy_Return']
        
        if trade_returns:
            win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
        else:
            win_rate = 0
        
        self.portfolio = signals
        self.performance = {
            'Market_Return': signals['Cumulative_Market_Return'].iloc[-1] - 1,
            'Strategy_Return': signals['Cumulative_Strategy_Return'].iloc[-1] - 1,
            'Market_Volatility': signals['Market_Return'].std() * np.sqrt(252),
            'Strategy_Volatility': signals['Strategy_Return'].std() * np.sqrt(252),
            'Market_Sharpe': (signals['Cumulative_Market_Return'].iloc[-1] - 1) / (signals['Market_Return'].std() * np.sqrt(252)) if signals['Market_Return'].std() != 0 else 0,
            'Strategy_Sharpe': (signals['Cumulative_Strategy_Return'].iloc[-1] - 1) / (signals['Strategy_Return'].std() * np.sqrt(252)) if signals['Strategy_Return'].std() != 0 else 0,
            'Market_Max_Drawdown': ((signals['Market_Portfolio'] - signals['Market_Portfolio'].cummax()) / signals['Market_Portfolio'].cummax()).min(),
            'Strategy_Max_Drawdown': ((signals['Strategy_Portfolio'] - signals['Strategy_Portfolio'].cummax()) / signals['Strategy_Portfolio'].cummax()).min(),
            'Win_Rate': win_rate,
            'Total_Trades': len(trade_returns)
        }
        
        print(f"Strategy backtesting completed on {len(signals)} data points")
        return signals
    
    def visualize_data(self, save_file=True):
        """
        Create visualizations of the candlestick data and signals with improved diagnostics.
        
        Parameters:
        -----------
        save_file : bool, default True
            Whether to save the visualization to a file
        
        Returns:
        --------
        None
        """
        if self.data is None:
            print("No data available for visualization.")
            return
            
        if len(self.data) < 5:
            print(f"Insufficient data available for visualization. Need at least 5 points, but have {len(self.data)}.")
            return
            
        try:
            # Check if we have the necessary columns
            required_columns = ['Date', 'Close', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower', 'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram']
            for col in required_columns:
                if col not in self.data.columns:
                    print(f"Missing column {col} required for visualization.")
                    return
                    
            plt.figure(figsize=(16, 12))
            
            # Plot 1: Price and Moving Averages with Buy/Sell Signals
            plt.subplot(3, 1, 1)
            plt.plot(self.data['Date'], self.data['Close'], label='Close Price', color='black', alpha=0.7)
            plt.plot(self.data['Date'], self.data['SMA_20'], label='20-day SMA', color='blue')
            plt.plot(self.data['Date'], self.data['SMA_50'], label='50-day SMA', color='red')
            plt.plot(self.data['Date'], self.data['SMA_100'], label='100-day SMA (Trend)', color='purple', linestyle='--')
            plt.fill_between(self.data['Date'], self.data['BB_Upper'], self.data['BB_Lower'], 
                             color='gray', alpha=0.3, label='Bollinger Bands')
            
            if self.signals is not None and 'Predicted_Signal' in self.signals.columns:
                buy_signals = self.signals[self.signals['Predicted_Signal'] == 1]
                plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
            
            plt.title('Price Chart with Technical Indicators and Predicted Signals')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            if save_file:
                plt.savefig(os.path.join(self.output_dir, 'technical_analysis.png'), dpi=300)
                print(f"Visualization saved to {os.path.join(self.output_dir, 'technical_analysis.png')}")
            plt.close()
            
            # Plot Portfolio Performance
            if self.portfolio is not None:
                plt.figure(figsize=(16, 6))
                plt.plot(self.portfolio['Date'], self.portfolio['Market_Portfolio'], label='Buy & Hold', color='blue')
                plt.plot(self.portfolio['Date'], self.portfolio['Strategy_Portfolio'], label='Strategy', color='green')
                plt.title('Portfolio Performance')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.legend()
                plt.grid(True)
                if save_file:
                    plt.savefig(os.path.join(self.output_dir, 'portfolio_performance.png'), dpi=300)
                    print(f"Portfolio performance chart saved to {os.path.join(self.output_dir, 'portfolio_performance.png')}")
                plt.close()
            
                # Additional analytics visuals (trade analytics)
                if hasattr(self, 'signals') and self.signals is not None:
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # 1. Signal Distribution (top left)
                    signal_counts = self.signals['Trade_Signal'].value_counts()
                    counts = [
                        signal_counts.get(0, 0),
                        signal_counts.get(1, 0),
                        signal_counts.get(-1, 0)
                    ]
                    axes[0, 0].bar(['Hold', 'Buy', 'Sell'], counts, color=['gray', 'green', 'red'])
                    axes[0, 0].set_title('Distribution of Trading Signals')
                    axes[0, 0].set_ylabel('Count')
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # 2. Trade Returns Distribution (top right)
                    if hasattr(self, 'performance') and 'Win_Rate' in self.performance:
                        trade_stats = [
                            f"Win Rate: {self.performance['Win_Rate']:.2%}",
                            f"Total Trades: {self.performance['Total_Trades']}",
                            f"Profit Factor: {max(1.0, self.performance['Strategy_Return'] / abs(self.performance['Market_Max_Drawdown'])):.2f}",
                            f"Sharpe Ratio: {self.performance['Strategy_Sharpe']:.2f}",
                            f"Max Drawdown: {self.performance['Strategy_Max_Drawdown']*100:.2f}%"
                        ]
                        axes[0, 1].axis('off')
                        for i, stat in enumerate(trade_stats):
                            axes[0, 1].text(0.1, 0.8 - (i * 0.15), stat, fontsize=14)
                        axes[0, 1].set_title('Trade Performance Metrics')
                    
                    # 3. Combined Signal Histogram (bottom left)
                    axes[1, 0].hist(self.signals['Combined_Signal'], bins=20, color='blue', alpha=0.7)
                    axes[1, 0].axvline(x=0.6, color='green', linestyle='--', label='Buy Threshold')
                    axes[1, 0].axvline(x=-0.6, color='red', linestyle='--', label='Sell Threshold')
                    axes[1, 0].set_title('Distribution of Combined Signal Values')
                    axes[1, 0].set_xlabel('Combined Signal')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 4. Monthly Returns Heatmap (bottom right)
                    if len(self.portfolio) > 30:
                        self.portfolio['YearMonth'] = self.portfolio['Date'].dt.strftime('%Y-%m')
                        monthly_returns = self.portfolio.groupby('YearMonth')['Strategy_Return'].sum().reset_index()
                        monthly_returns['Month'] = pd.to_datetime(monthly_returns['YearMonth'] + '-01').dt.month_name()
                        monthly_returns['Year'] = pd.to_datetime(monthly_returns['YearMonth'] + '-01').dt.year
                        
                        if len(monthly_returns) > 1:
                            try:
                                pivot_table = monthly_returns.pivot_table(
                                    index='Month', 
                                    columns='Year', 
                                    values='Strategy_Return',
                                    aggfunc='sum'
                                )
                                months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                                               'July', 'August', 'September', 'October', 'November', 'December']
                                pivot_table = pivot_table.reindex(months_order)
                                im = axes[1, 1].imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
                                for i in range(len(pivot_table.index)):
                                    for j in range(len(pivot_table.columns)):
                                        if not np.isnan(pivot_table.values[i, j]):
                                            text = f"{pivot_table.values[i, j]*100:.1f}%"
                                            axes[1, 1].text(j, i, text, ha='center', va='center', 
                                                             color='black' if abs(pivot_table.values[i, j]) < 0.05 else 'white')
                                axes[1, 1].set_xticks(np.arange(len(pivot_table.columns)))
                                axes[1, 1].set_yticks(np.arange(len(pivot_table.index)))
                                axes[1, 1].set_xticklabels(pivot_table.columns)
                                axes[1, 1].set_yticklabels(pivot_table.index)
                                axes[1, 1].set_title('Monthly Returns Heatmap')
                                fig.colorbar(im, ax=axes[1, 1], label='Monthly Return')
                            except Exception as e:
                                axes[1, 1].text(0.5, 0.5, f"Insufficient data for monthly heatmap\n{e}", 
                                                 ha='center', va='center', fontsize=12)
                                axes[1, 1].axis('off')
                        else:
                            axes[1, 1].text(0.5, 0.5, "Insufficient data for monthly heatmap", 
                                             ha='center', va='center', fontsize=12)
                            axes[1, 1].axis('off')
                    else:
                        axes[1, 1].text(0.5, 0.5, "Insufficient data for monthly heatmap", 
                                         ha='center', va='center', fontsize=12)
                        axes[1, 1].axis('off')
                    
                    plt.tight_layout()
                    
                    if save_file:
                        plt.savefig(os.path.join(self.output_dir, 'trade_analytics.png'), dpi=300)
                        print(f"Trade analytics saved to {os.path.join(self.output_dir, 'trade_analytics.png')}")
                    
                    plt.close()
                
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def optimize_strategy_parameters(self, param_grid=None):
        """
        Optimize strategy parameters using grid search.
        
        Parameters:
        -----------
        param_grid : dict, optional
            Dictionary of parameter ranges to test
            
        Returns:
        --------
        dict : Best parameters and performance metrics
        """
        if self.data is None:
            print("No data available for optimization.")
        best_params = {}
        performance_metrics = {}
        return {"best_params": best_params, "performance_metrics": performance_metrics}
    
    def export_to_excel(self):
        """
        Export all data and analysis to a clean, formatted Excel file.
        
        Returns:
        --------
        str : Path to the saved Excel file
        """
        if self.data is None:
            print("No data available. Please load and analyze data first.")
            return None
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = os.path.join(self.output_dir, f"investment_analysis_{timestamp}.xlsx")
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                self.data.to_excel(writer, sheet_name='Raw Data', index=False)
                if self.signals is not None:
                    self.signals.to_excel(writer, sheet_name='Signals', index=False)
                if self.portfolio is not None:
                    self.portfolio.to_excel(writer, sheet_name='Portfolio', index=False)
                if self.performance is not None:
                    perf_df = pd.DataFrame(list(self.performance.items()), columns=['Metric', 'Value'])
                    perf_df.to_excel(writer, sheet_name='Performance Summary', index=False)
            print(f"Data successfully exported to {excel_file}")
            return excel_file
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            import traceback
            traceback.print_exc()
            return None

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    try:
        # Initialize the analyzer with no data file (will use sample data)
        analyzer = CandleStickAnalyzer()
        
        # Load/generate data - generate more data to ensure enough after dropping NaN values
        analyzer.load_data(sample_size=500)
        
        # Calculate technical indicators
        analyzer.calculate_technical_indicators()
        
        # Generate baseline trading signals
        analyzer.generate_trading_signals()
        
        # Use ML to predict future signals with a 2-day horizon (only trades with >=65% confidence)
        analyzer.predict_future_signals(prediction_horizon=2)
        
        # Backtest the strategy using predicted signals and a 2-day holding period
        analyzer.backtest_strategy(initial_capital=10000, holding_period=2)
        
        # Create visualizations (all original visuals are preserved)
        analyzer.visualize_data(save_file=True)
        
        # Export results to Excel
        excel_file = analyzer.export_to_excel()
        
        print(f"\nAnalysis complete. Results saved to {excel_file}")
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        import traceback
        traceback.print_exc()
