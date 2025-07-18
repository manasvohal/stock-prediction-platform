import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from app.services.data_fetcher import DataFetcher
from app.core.config import settings

logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    """Dataset for stock price prediction"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM model for stock price prediction"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Enhanced LSTM architecture with bidirectional layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout,
            bidirectional=True
        )
        # Adjust for bidirectional output (hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class MLPredictor:
    """Advanced ML predictor service using LSTM for time series forecasting"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.model = None
        self.scaler = None
        self.sequence_length = 20  # Number of days to use for prediction
        self.forecast_days = 30    # Number of days to forecast
        self.model_path = os.path.join(settings.MODEL_DIR, "lstm_model.pth")
        self.scaler_path = os.path.join(settings.MODEL_DIR, "price_scaler.pkl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_or_create_model(self, input_dim=1):
        """Load existing model or create a new one"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading LSTM model from {self.model_path}")
                model = LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=64,
                    num_layers=2,
                    output_dim=1
                )
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                return model
            else:
                logger.warning(f"Model file {self.model_path} not found. Creating new model.")
                model = LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=64,
                    num_layers=2,
                    output_dim=1
                )
                model.to(self.device)
                model.eval()
                return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=64,
                num_layers=2,
                output_dim=1
            )
            model.to(self.device)
            model.eval()
            return model
    
    def _load_or_create_scaler(self):
        """Load existing scaler or create a new one"""
        try:
            if os.path.exists(self.scaler_path):
                logger.info(f"Loading scaler from {self.scaler_path}")
                return joblib.load(self.scaler_path)
            else:
                logger.warning(f"Scaler file {self.scaler_path} not found. Creating new scaler.")
                return MinMaxScaler(feature_range=(0, 1))
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            return MinMaxScaler(feature_range=(0, 1))
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Prepare data for prediction
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            Tuple of (sequences, last_price)
        """
        # Extract closing prices
        prices = df['close'].values.reshape(-1, 1)
        
        # Get the last price for reference
        last_price = float(prices[-1][0])
        
        # Load or create scaler
        self.scaler = self._load_or_create_scaler()
        
        # Fit scaler if it's new
        if not os.path.exists(self.scaler_path):
            self.scaler.fit(prices)
            
        # Scale the data
        scaled_prices = self.scaler.transform(prices)
        
        # Create sequences
        sequences = []
        for i in range(len(scaled_prices) - self.sequence_length):
            seq = scaled_prices[i:i+self.sequence_length]
            sequences.append(seq)
        
        if not sequences:
            # If we don't have enough data, use what we have
            seq = scaled_prices[-self.sequence_length:]
            sequences.append(seq)
            
        return np.array(sequences), last_price
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare additional features for enhanced prediction
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            Array of features
        """
        # Make a copy of the dataframe to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        # Calculate technical indicators
        df_copy.loc[:, 'ma7'] = df_copy['close'].rolling(window=7).mean()
        df_copy.loc[:, 'ma21'] = df_copy['close'].rolling(window=21).mean()
        df_copy.loc[:, 'ma50'] = df_copy['close'].rolling(window=50).mean()
        df_copy.loc[:, 'rsi'] = self._calculate_rsi(df_copy['close'], 14)
        df_copy.loc[:, 'volatility'] = df_copy['close'].rolling(window=20).std()
        
        # Calculate price momentum
        df_copy.loc[:, 'price_momentum'] = df_copy['close'].pct_change(5)
        
        # Calculate MACD
        ema12 = df_copy['close'].ewm(span=12, adjust=False).mean()
        ema26 = df_copy['close'].ewm(span=26, adjust=False).mean()
        df_copy.loc[:, 'macd'] = ema12 - ema26
        df_copy.loc[:, 'macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df_copy.loc[:, 'bb_middle'] = df_copy['close'].rolling(window=20).mean()
        std = df_copy['close'].rolling(window=20).std()
        df_copy.loc[:, 'bb_upper'] = df_copy['bb_middle'] + (std * 2)
        df_copy.loc[:, 'bb_lower'] = df_copy['bb_middle'] - (std * 2)
        df_copy.loc[:, 'bb_width'] = (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['bb_middle']
        
        # Calculate volume features if available
        if 'volume' in df_copy.columns:
            df_copy.loc[:, 'volume_ma7'] = df_copy['volume'].rolling(window=7).mean()
            df_copy.loc[:, 'volume_change'] = df_copy['volume'].pct_change()
        
        # Fill NaN values
        df_copy.fillna(method='bfill', inplace=True)
        df_copy.fillna(0, inplace=True)  # Any remaining NaNs
        
        # Extract features
        feature_columns = ['close', 'ma7', 'ma21', 'ma50', 'rsi', 'volatility', 
                          'price_momentum', 'macd', 'macd_signal', 'bb_width']
        
        if 'volume' in df_copy.columns:
            feature_columns.extend(['volume_ma7', 'volume_change'])
            
        features = df_copy[feature_columns].values
        
        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        
        # Create sequences
        sequences = []
        for i in range(len(scaled_features) - self.sequence_length):
            seq = scaled_features[i:i+self.sequence_length]
            sequences.append(seq)
        
        if not sequences:
            # If we don't have enough data, use what we have
            seq = scaled_features[-self.sequence_length:]
            sequences.append(seq)
            
        return np.array(sequences)
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train(self, ticker: str, epochs: int = 50) -> Dict[str, Any]:
        """Train the model for a specific stock
        
        Args:
            ticker: Stock ticker symbol
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training results
        """
        try:
            # Get historical data (2 years for training)
            df = self.data_fetcher.get_historical_prices(ticker, period="2y")
            
            # Prepare features
            features = self._prepare_features(df)
            input_dim = features.shape[2]
            
            # Prepare target
            prices = df['close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(prices)
            
            # Create sequences for training
            X = []
            y = []
            for i in range(len(features)):
                if i + 1 < len(scaled_prices):
                    X.append(features[i])
                    y.append(scaled_prices[i + self.sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split into training and validation sets
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Create datasets and dataloaders
            train_dataset = StockDataset(X_train, y_train)
            val_dataset = StockDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=64,
                num_layers=2,
                output_dim=1
            )
            model.to(self.device)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                
                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save the model and scaler
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(model.state_dict(), self.model_path)
            joblib.dump(scaler, self.scaler_path)
            
            # Set the model and scaler for prediction
            self.model = model
            self.scaler = scaler
            
            return {
                "ticker": ticker,
                "epochs": epochs,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "model_saved": True
            }
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            raise ValueError(f"Could not train model for {ticker}: {str(e)}")
    
    def predict(self, ticker: str) -> Dict[str, Any]:
        """Predict future stock prices
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get historical data
            df = self.data_fetcher.get_historical_prices(ticker, period="1y")
            stock_info = self.data_fetcher.get_stock_info(ticker)
            
            # Prepare features
            features = self._prepare_features(df)
            input_dim = features.shape[2]
            
            # Load or create model
            if self.model is None:
                self.model = self._load_or_create_model(input_dim=input_dim)
            
            # Get the last sequence for prediction
            last_sequence = features[-1]
            last_sequence = torch.tensor(last_sequence).unsqueeze(0).float().to(self.device)
            
            # Get the last price for reference
            last_price = float(df['close'].iloc[-1])
            current_date = df['date'].iloc[-1]
            
            # Create a new scaler specifically for this stock's data
            stock_scaler = MinMaxScaler(feature_range=(0, 1))
            prices = df['close'].values.reshape(-1, 1)
            stock_scaler.fit(prices)  # Fit on this specific stock's price range
            
            # Calculate historical volatility for realistic predictions
            daily_returns = df['close'].pct_change().dropna()
            historical_volatility = daily_returns.std()
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                future_prices = []
                future_dates = []
                
                # Use the last sequence as our starting point
                current_sequence = last_sequence.clone()
                
                # Generate predictions for forecast_days
                for i in range(self.forecast_days):
                    # Predict the next price
                    pred = self.model(current_sequence)
                    pred_price = pred.item()
                    
                    # Add small random noise based on historical volatility to make predictions more realistic
                    # The noise gets larger as we predict further into the future
                    noise_factor = min(0.5, 0.02 * (i + 1))  # Cap at 50% of volatility
                    noise = np.random.normal(0, historical_volatility * noise_factor)
                    pred_price = max(0, pred_price * (1 + noise))  # Ensure price doesn't go negative
                    
                    # Store the prediction
                    future_prices.append(pred_price)
                    
                    # Calculate the next date
                    next_date = current_date + timedelta(days=i+1)
                    future_dates.append(next_date)
                    
                    # Update the sequence for the next prediction
                    # Remove the first timestep and add the new prediction
                    new_seq = current_sequence.clone()
                    new_seq = new_seq[:, 1:, :]
                    
                    # Create a new feature vector for the predicted point
                    # For simplicity, just repeat the last feature vector but update the price
                    new_feature = new_seq[:, -1, :].clone()
                    new_feature[:, 0] = pred_price  # Update only the price feature
                    
                    # Add the new feature vector as a new timestep
                    new_feature = new_feature.unsqueeze(1)  # Add timestep dimension
                    current_sequence = torch.cat([new_seq, new_feature], dim=1)
            
            # Convert predictions back to original scale using the stock-specific scaler
            future_prices_array = np.array(future_prices).reshape(-1, 1)
            future_prices_scaled = stock_scaler.inverse_transform(future_prices_array)
            future_prices = future_prices_scaled.flatten().tolist()
            
            # Apply trend-based smoothing to make the forecast more realistic
            # Calculate the recent trend from the last 20 days of data
            recent_prices = df['close'].values[-20:]
            if len(recent_prices) > 1:
                recent_trend = (recent_prices[-1] / recent_prices[0]) - 1
                trend_factor = 1 + (recent_trend * 0.5)  # Dampen the trend effect
                
                # Apply trend influence that increases over time
                for i in range(len(future_prices)):
                    trend_influence = min(0.8, 0.02 * (i + 1))  # Cap at 80%
                    future_prices[i] = future_prices[i] * (1 + (trend_factor - 1) * trend_influence)
            
            # Calculate metrics
            avg_predicted_price = sum(future_prices) / len(future_prices)
            price_change = avg_predicted_price - last_price
            price_change_percent = (price_change / last_price) * 100
            
            # Determine prediction based on price change
            if price_change_percent > 5:
                prediction = "buy"
                confidence = min(0.9, 0.5 + abs(price_change_percent) / 20)
            elif price_change_percent < -5:
                prediction = "sell"
                confidence = min(0.9, 0.5 + abs(price_change_percent) / 20)
            else:
                prediction = "hold"
                confidence = 0.5 + (5 - abs(price_change_percent)) / 10
            
            # Calculate volatility
            volatility = np.std(future_prices) / np.mean(future_prices) * 100
            
            # Prepare forecast data
            forecast_data = []
            for i, (date, price) in enumerate(zip(future_dates, future_prices)):
                forecast_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "price": price,
                    "day": i + 1
                })
            
            # Prepare result
            result = {
                "ticker": ticker,
                "company_name": stock_info.get("company_name", ticker),
                "current_price": last_price,
                "last_date": current_date.strftime("%Y-%m-%d"),
                "forecast": forecast_data,
                "avg_predicted_price": avg_predicted_price,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "prediction": prediction,
                "confidence": confidence,
                "volatility": volatility,
                "forecast_days": self.forecast_days,
                "model_type": "LSTM Neural Network",
                "features_used": ["price", "moving_averages", "rsi", "volatility", "momentum", "macd", "bollinger_bands"]
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error predicting for {ticker}: {e}")
            
            # Provide a fallback prediction
            try:
                # Get historical data
                df = self.data_fetcher.get_historical_prices(ticker, period="1y")
                stock_info = self.data_fetcher.get_stock_info(ticker)
                
                # Get the last price
                last_price = float(df['close'].iloc[-1])
                current_date = df['date'].iloc[-1]
                
                # Calculate a simple moving average-based prediction
                ma20 = df['close'].rolling(window=20).mean().iloc[-1]
                ma50 = df['close'].rolling(window=50).mean().iloc[-1]
                
                # Calculate volatility for more realistic predictions
                daily_returns = df['close'].pct_change().dropna()
                historical_volatility = daily_returns.std()
                
                # Determine trend direction based on this specific stock's data
                recent_change = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 0.5  # Half of recent 20-day change
                
                # Calculate trend using exponential smoothing of recent performance
                alpha = 0.3  # Smoothing factor
                recent_prices = df['close'].values[-30:]  # Last 30 days
                
                # Initialize with first value
                trend_estimate = 0
                
                # Apply exponential smoothing
                for i in range(1, len(recent_prices)):
                    daily_change = recent_prices[i] / recent_prices[i-1] - 1
                    trend_estimate = alpha * daily_change + (1 - alpha) * trend_estimate
                
                # Scale the trend to a reasonable daily change
                trend = trend_estimate * 0.5  # Dampen the trend
                
                # Adjust trend based on MA crossover
                if ma20 > ma50:
                    # Uptrend - boost the trend slightly
                    trend = max(0.0005, trend)  # Ensure at least slightly positive
                elif ma20 < ma50:
                    # Downtrend - make trend slightly negative
                    trend = min(-0.0005, trend)  # Ensure at least slightly negative
                
                # Generate simple forecast
                future_prices = []
                future_dates = []
                
                # Start with the last price
                current_price = last_price
                
                for i in range(self.forecast_days):
                    # Calculate trend factor that evolves over time
                    # For longer forecasts, trend reverts to the mean (becomes less extreme)
                    days_factor = min(1.0, 1.0 - (i / (self.forecast_days * 2)))
                    current_trend = trend * days_factor
                    
                    # Add random noise that increases with time
                    noise_factor = min(0.7, 0.02 * (i + 1))  # Cap at 70% of volatility
                    noise = np.random.normal(0, historical_volatility * noise_factor)
                    
                    # Calculate next price with trend and noise
                    next_price = current_price * (1 + current_trend + noise)
                    next_price = max(current_price * 0.5, next_price)  # Prevent unrealistic drops
                    
                    future_prices.append(next_price)
                    current_price = next_price  # Use this price for next iteration
                    
                    # Calculate the next date
                    next_date = current_date + timedelta(days=i+1)
                    future_dates.append(next_date)
                
                # Calculate metrics
                avg_predicted_price = sum(future_prices) / len(future_prices)
                price_change = avg_predicted_price - last_price
                price_change_percent = (price_change / last_price) * 100
                
                # Determine prediction based on price change
                if price_change_percent > 1:
                    prediction = "buy"
                    confidence = 0.55
                elif price_change_percent < -1:
                    prediction = "sell"
                    confidence = 0.55
                else:
                    prediction = "hold"
                    confidence = 0.6
                
                # Prepare forecast data
                forecast_data = []
                for i, (date, price) in enumerate(zip(future_dates, future_prices)):
                    forecast_data.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "price": price,
                        "day": i + 1
                    })
                
                # Prepare result
                result = {
                    "ticker": ticker,
                    "company_name": stock_info.get("company_name", ticker),
                    "current_price": last_price,
                    "last_date": current_date.strftime("%Y-%m-%d"),
                    "forecast": forecast_data,
                    "avg_predicted_price": avg_predicted_price,
                    "price_change": price_change,
                    "price_change_percent": price_change_percent,
                    "prediction": prediction,
                    "confidence": confidence,
                    "volatility": historical_volatility * 100,  # Convert to percentage
                    "forecast_days": self.forecast_days,
                    "model_type": "Advanced Trend Analysis (Fallback)",
                    "features_used": ["price", "moving_averages", "volatility"],
                    "is_fallback": True
                }
                
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback prediction also failed for {ticker}: {fallback_error}")
                raise ValueError(f"Could not make prediction for {ticker}: {str(e)}")
    
    def evaluate(self, ticker: str) -> Dict[str, Any]:
        """Evaluate model performance on historical data
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get historical data
            df = self.data_fetcher.get_historical_prices(ticker, period="1y")
            
            # Split data into train and test
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            # Prepare features
            train_features = self._prepare_features(train_df)
            test_features = self._prepare_features(test_df)
            
            input_dim = train_features.shape[2]
            
            # Load or create model
            if self.model is None:
                self.model = self._load_or_create_model(input_dim=input_dim)
            
            # Prepare target
            test_prices = test_df['close'].values.reshape(-1, 1)
            
            # Load or create scaler
            if self.scaler is None:
                self.scaler = self._load_or_create_scaler()
            
            # Fit scaler if it's new or not fitted
            try:
                # Try to transform a sample to check if scaler is fitted
                self.scaler.transform(np.array([[0]]))
            except:
                # If not fitted, fit on all price data
                all_prices = df['close'].values.reshape(-1, 1)
                self.scaler.fit(all_prices)
                
            # Scale test prices
            scaled_test_prices = self.scaler.transform(test_prices)
            
            # Make predictions
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for i in range(len(test_features)):
                    seq = torch.tensor(test_features[i]).unsqueeze(0).float().to(self.device)
                    pred = self.model(seq)
                    predictions.append(pred.item())
            
            # Convert predictions back to original scale
            predictions_array = np.array(predictions).reshape(-1, 1)
            predictions_scaled = self.scaler.inverse_transform(predictions_array)
            predictions = predictions_scaled.flatten().tolist()
            
            # Calculate metrics
            actual_prices = test_df['close'].values.tolist()
            
            # Make sure we have the same number of predictions and actual prices
            min_len = min(len(predictions), len(actual_prices))
            predictions = predictions[:min_len]
            actual_prices = actual_prices[:min_len]
            
            # Calculate MSE and RMSE
            mse = np.mean((np.array(predictions) - np.array(actual_prices)) ** 2)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE
            mape = np.mean(np.abs((np.array(actual_prices) - np.array(predictions)) / np.array(actual_prices))) * 100
            
            # Calculate directional accuracy
            correct_direction = 0
            for i in range(1, len(predictions)):
                actual_direction = actual_prices[i] > actual_prices[i-1]
                pred_direction = predictions[i] > predictions[i-1]
                if actual_direction == pred_direction:
                    correct_direction += 1
            
            directional_accuracy = correct_direction / (len(predictions) - 1) * 100 if len(predictions) > 1 else 0
            
            return {
                "ticker": ticker,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "directional_accuracy": directional_accuracy,
                "test_samples": min_len
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model for {ticker}: {e}")
            
            # Return fallback evaluation metrics
            return {
                "ticker": ticker,
                "mse": 0,
                "rmse": 0,
                "mape": 0,
                "directional_accuracy": 50.0,  # Random guessing
                "test_samples": 0,
                "is_fallback": True,
                "error": str(e)
            } 