import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

def calculate_moving_averages(df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
    """Calculate moving averages for the given periods
    
    Args:
        df: DataFrame with price data
        periods: List of periods to calculate moving averages for
        
    Returns:
        Dictionary with moving averages
    """
    result = {}
    for period in periods:
        result[f"ma{period}"] = df["close"].rolling(window=period).mean()
    return result

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)
    
    Args:
        df: DataFrame with price data
        period: Period for RSI calculation
        
    Returns:
        Series with RSI values
    """
    delta = df["close"].diff()
    
    # Make two series: one for gains, one for losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # First value is sum of gains
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        df: DataFrame with price data
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        
    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
    # Calculate EMAs
    ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def identify_support_resistance(df: pd.DataFrame, window: int = 10, threshold: float = 0.02) -> Dict[str, List[float]]:
    """Identify support and resistance levels
    
    Args:
        df: DataFrame with price data
        window: Window size for peak detection
        threshold: Threshold for grouping levels (percentage of price)
        
    Returns:
        Dictionary with support and resistance levels
    """
    # Find local minima and maxima
    df["min"] = df["low"].rolling(window=window, center=True).min()
    df["max"] = df["high"].rolling(window=window, center=True).max()
    
    # Identify potential support levels (local minima)
    support_levels = []
    for i in range(window, len(df) - window):
        if df["low"].iloc[i] == df["min"].iloc[i] and df["low"].iloc[i] < df["low"].iloc[i-1] and df["low"].iloc[i] < df["low"].iloc[i+1]:
            support_levels.append(df["low"].iloc[i])
    
    # Identify potential resistance levels (local maxima)
    resistance_levels = []
    for i in range(window, len(df) - window):
        if df["high"].iloc[i] == df["max"].iloc[i] and df["high"].iloc[i] > df["high"].iloc[i-1] and df["high"].iloc[i] > df["high"].iloc[i+1]:
            resistance_levels.append(df["high"].iloc[i])
    
    # Group nearby levels
    support_levels = group_levels(support_levels, threshold)
    resistance_levels = group_levels(resistance_levels, threshold)
    
    # Sort levels
    support_levels.sort()
    resistance_levels.sort()
    
    # Get the most recent price
    current_price = df["close"].iloc[-1]
    
    # Filter levels based on current price
    support_levels = [level for level in support_levels if level < current_price]
    resistance_levels = [level for level in resistance_levels if level > current_price]
    
    # Take the 3 closest levels
    support_levels = sorted(support_levels, key=lambda x: current_price - x)[:3]
    resistance_levels = sorted(resistance_levels, key=lambda x: x - current_price)[:3]
    
    return {
        "support": support_levels,
        "resistance": resistance_levels
    }

def group_levels(levels: List[float], threshold: float) -> List[float]:
    """Group nearby price levels
    
    Args:
        levels: List of price levels
        threshold: Threshold for grouping (percentage)
        
    Returns:
        List of grouped price levels
    """
    if not levels:
        return []
    
    # Sort levels
    sorted_levels = sorted(levels)
    
    # Group levels
    grouped_levels = []
    current_group = [sorted_levels[0]]
    
    for i in range(1, len(sorted_levels)):
        # If the current level is close to the previous one, add to the current group
        if (sorted_levels[i] - sorted_levels[i-1]) / sorted_levels[i-1] <= threshold:
            current_group.append(sorted_levels[i])
        else:
            # Otherwise, calculate the average of the current group and start a new group
            grouped_levels.append(sum(current_group) / len(current_group))
            current_group = [sorted_levels[i]]
    
    # Add the last group
    if current_group:
        grouped_levels.append(sum(current_group) / len(current_group))
    
    return grouped_levels

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands
    
    Args:
        df: DataFrame with price data
        window: Window size for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        Tuple of (middle band, upper band, lower band)
    """
    middle_band = df["close"].rolling(window=window).mean()
    std_dev = df["close"].rolling(window=window).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return middle_band, upper_band, lower_band 