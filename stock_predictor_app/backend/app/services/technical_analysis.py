import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from app.services.data_fetcher import DataFetcher
from app.utils.indicators import (
    calculate_moving_averages, 
    calculate_rsi, 
    calculate_macd, 
    identify_support_resistance,
    calculate_bollinger_bands
)

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Service for analyzing technical indicators of stocks"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
    
    def analyze(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Perform technical analysis on a stock
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with technical analysis results
        """
        try:
            # Get historical price data
            df = self.data_fetcher.get_historical_prices(ticker, period)
            
            if df.empty:
                raise ValueError(f"No historical data available for {ticker}")
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Identify patterns
            patterns = self._identify_patterns(df)
            
            # Determine trend
            trend, trend_strength = self._determine_trend(df, indicators)
            
            # Generate summary
            summary = self._generate_summary(ticker, indicators, trend, trend_strength, patterns)
            
            # Prepare result
            result = {
                "ticker": ticker,
                "period": period,
                "last_price": df["close"].iloc[-1],
                "price_change": self._calculate_price_change(df),
                "price_change_percent": self._calculate_price_change_percent(df),
                "volume_avg": df["volume"].mean(),
                "indicators": indicators,
                "trend": trend,
                "trend_strength": trend_strength,
                "patterns": patterns,
                "support_resistance": indicators["support_resistance"],
                "summary": summary
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in technical analysis for {ticker}: {e}")
            raise ValueError(f"Could not perform technical analysis for {ticker}: {str(e)}")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            # Calculate moving averages
            ma_periods = [20, 50, 200]
            moving_averages = calculate_moving_averages(df, ma_periods)
            
            # Calculate RSI
            rsi = calculate_rsi(df, period=14)
            
            # Calculate MACD
            macd, signal, histogram = calculate_macd(df)
            
            # Identify support and resistance levels
            support_resistance = identify_support_resistance(df)
            
            # Calculate Bollinger Bands
            ma20, upper_band, lower_band = calculate_bollinger_bands(df)
            
            # Calculate Average True Range (ATR)
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Calculate volume indicators
            volume_sma = df["volume"].rolling(window=20).mean().iloc[-1]
            volume_change = ((df["volume"].iloc[-1] / volume_sma) - 1) * 100
            
            return {
                "moving_averages": {
                    "ma20": moving_averages["ma20"].iloc[-1] if not moving_averages["ma20"].empty else None,
                    "ma50": moving_averages["ma50"].iloc[-1] if not moving_averages["ma50"].empty else None,
                    "ma200": moving_averages["ma200"].iloc[-1] if not moving_averages["ma200"].empty else None,
                },
                "rsi": rsi.iloc[-1] if not rsi.empty else None,
                "macd": {
                    "macd": macd.iloc[-1] if not macd.empty else None,
                    "signal": signal.iloc[-1] if not signal.empty else None,
                    "histogram": histogram.iloc[-1] if not histogram.empty else None,
                },
                "bollinger_bands": {
                    "upper": upper_band.iloc[-1] if not upper_band.empty else None,
                    "lower": lower_band.iloc[-1] if not lower_band.empty else None,
                    "width": ((upper_band.iloc[-1] - lower_band.iloc[-1]) / ma20.iloc[-1]) if not upper_band.empty and not lower_band.empty and not ma20.empty and ma20.iloc[-1] != 0 else None
                },
                "atr": atr,
                "volume": {
                    "current": df["volume"].iloc[-1],
                    "average": volume_sma,
                    "change_percent": volume_change
                },
                "support_resistance": support_resistance
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {
                "moving_averages": {"ma20": None, "ma50": None, "ma200": None},
                "rsi": None,
                "macd": {"macd": None, "signal": None, "histogram": None},
                "bollinger_bands": {"upper": None, "lower": None, "width": None},
                "atr": None,
                "volume": {"current": None, "average": None, "change_percent": None},
                "support_resistance": {"support": [], "resistance": []}
            }
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify chart patterns"""
        try:
            patterns = {
                "detected": [],
                "descriptions": {}
            }
            
            # Simple pattern detection (this could be expanded with more sophisticated algorithms)
            
            # Check for double bottom
            if len(df) >= 40:
                # Look for two bottoms with similar prices
                rolling_min = df["low"].rolling(window=10).min()
                bottoms = []
                
                for i in range(10, len(df) - 10):
                    if df["low"].iloc[i] == rolling_min.iloc[i] and df["low"].iloc[i] < df["low"].iloc[i-5] and df["low"].iloc[i] < df["low"].iloc[i+5]:
                        bottoms.append((i, df["low"].iloc[i]))
                
                if len(bottoms) >= 2:
                    # Check if we have two bottoms with similar prices
                    for i in range(len(bottoms) - 1):
                        for j in range(i + 1, len(bottoms)):
                            if abs(bottoms[i][0] - bottoms[j][0]) > 15:  # Bottoms should be separated
                                price_diff_pct = abs(bottoms[i][1] - bottoms[j][1]) / bottoms[i][1]
                                if price_diff_pct < 0.03:  # 3% tolerance for price similarity
                                    patterns["detected"].append("double_bottom")
                                    patterns["descriptions"]["double_bottom"] = "Double bottom pattern detected, which is typically bullish."
                                    break
                        if "double_bottom" in patterns["detected"]:
                            break
            
            # Check for bullish engulfing
            for i in range(1, len(df)):
                if df["close"].iloc[i-1] < df["open"].iloc[i-1] and df["close"].iloc[i] > df["open"].iloc[i]:
                    if df["open"].iloc[i] <= df["close"].iloc[i-1] and df["close"].iloc[i] > df["open"].iloc[i-1]:
                        patterns["detected"].append("bullish_engulfing")
                        patterns["descriptions"]["bullish_engulfing"] = "Bullish engulfing pattern detected in recent candles, which may indicate a potential reversal to the upside."
                        break
            
            # Check for bearish engulfing
            for i in range(1, len(df)):
                if df["close"].iloc[i-1] > df["open"].iloc[i-1] and df["close"].iloc[i] < df["open"].iloc[i]:
                    if df["open"].iloc[i] >= df["close"].iloc[i-1] and df["close"].iloc[i] < df["open"].iloc[i-1]:
                        patterns["detected"].append("bearish_engulfing")
                        patterns["descriptions"]["bearish_engulfing"] = "Bearish engulfing pattern detected in recent candles, which may indicate a potential reversal to the downside."
                        break
            
            return patterns
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return {"detected": [], "descriptions": {}}
    
    def _determine_trend(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Tuple[str, float]:
        """Determine the current trend and its strength"""
        try:
            # Get moving averages
            ma20 = indicators["moving_averages"]["ma20"]
            ma50 = indicators["moving_averages"]["ma50"]
            ma200 = indicators["moving_averages"]["ma200"]
            
            current_price = df["close"].iloc[-1]
            
            # Initialize trend variables
            trend = "sideways"
            trend_strength = 0.5  # Neutral
            
            # Determine trend based on moving averages
            if ma20 and ma50 and ma200:
                # Strong uptrend: price > ma20 > ma50 > ma200
                if current_price > ma20 > ma50 > ma200:
                    trend = "strong_uptrend"
                    trend_strength = 0.9
                # Uptrend: price > ma50 > ma200
                elif current_price > ma50 > ma200:
                    trend = "uptrend"
                    trend_strength = 0.7
                # Strong downtrend: price < ma20 < ma50 < ma200
                elif current_price < ma20 < ma50 < ma200:
                    trend = "strong_downtrend"
                    trend_strength = 0.1
                # Downtrend: price < ma50 < ma200
                elif current_price < ma50 < ma200:
                    trend = "downtrend"
                    trend_strength = 0.3
                # Potential reversal up: price > ma20 but price < ma50
                elif current_price > ma20 and current_price < ma50:
                    trend = "potential_reversal_up"
                    trend_strength = 0.6
                # Potential reversal down: price < ma20 but price > ma50
                elif current_price < ma20 and current_price > ma50:
                    trend = "potential_reversal_down"
                    trend_strength = 0.4
                else:
                    trend = "sideways"
                    trend_strength = 0.5
            
            # Adjust trend strength based on RSI
            rsi = indicators["rsi"]
            if rsi is not None:
                if rsi > 70 and trend.endswith("uptrend"):
                    trend_strength -= 0.1  # Overbought condition in uptrend
                elif rsi < 30 and trend.endswith("downtrend"):
                    trend_strength += 0.1  # Oversold condition in downtrend
            
            # Adjust trend strength based on MACD
            macd_val = indicators["macd"]["macd"]
            macd_signal = indicators["macd"]["signal"]
            if macd_val is not None and macd_signal is not None:
                # MACD crossing above signal line is bullish
                if macd_val > macd_signal and macd_val > 0:
                    trend_strength += 0.1
                # MACD crossing below signal line is bearish
                elif macd_val < macd_signal and macd_val < 0:
                    trend_strength -= 0.1
            
            # Ensure trend_strength is between 0 and 1
            trend_strength = max(0, min(1, trend_strength))
            
            return trend, trend_strength
        except Exception as e:
            logger.error(f"Error determining trend: {e}")
            return "unknown", 0.5
    
    def _calculate_price_change(self, df: pd.DataFrame) -> float:
        """Calculate absolute price change"""
        if len(df) >= 2:
            return df["close"].iloc[-1] - df["close"].iloc[0]
        return 0.0
    
    def _calculate_price_change_percent(self, df: pd.DataFrame) -> float:
        """Calculate percentage price change"""
        if len(df) >= 2 and df["close"].iloc[0] != 0:
            return ((df["close"].iloc[-1] / df["close"].iloc[0]) - 1) * 100
        return 0.0
    
    def _generate_summary(
        self, 
        ticker: str,
        indicators: Dict[str, Any],
        trend: str,
        trend_strength: float,
        patterns: Dict[str, Any]
    ) -> str:
        """Generate a human-readable summary of the technical analysis"""
        # Trend description
        trend_descriptions = {
            "strong_uptrend": f"{ticker} is in a strong uptrend with all major moving averages aligned bullishly",
            "uptrend": f"{ticker} is in an uptrend, trading above its 50-day and 200-day moving averages",
            "strong_downtrend": f"{ticker} is in a strong downtrend with all major moving averages aligned bearishly",
            "downtrend": f"{ticker} is in a downtrend, trading below its 50-day and 200-day moving averages",
            "potential_reversal_up": f"{ticker} may be forming a bullish reversal, recently crossing above its 20-day moving average",
            "potential_reversal_down": f"{ticker} may be forming a bearish reversal, recently crossing below its 20-day moving average",
            "sideways": f"{ticker} is trading sideways with no clear directional trend",
            "unknown": f"Technical trend for {ticker} could not be determined"
        }
        
        summary = trend_descriptions.get(trend, trend_descriptions["unknown"])
        
        # Add RSI information
        rsi = indicators["rsi"]
        if rsi is not None:
            if rsi > 70:
                summary += f". RSI at {rsi:.1f} indicates overbought conditions"
            elif rsi < 30:
                summary += f". RSI at {rsi:.1f} indicates oversold conditions"
            else:
                summary += f". RSI at {rsi:.1f} is neutral"
        
        # Add MACD information
        macd_val = indicators["macd"]["macd"]
        macd_signal = indicators["macd"]["signal"]
        if macd_val is not None and macd_signal is not None:
            if macd_val > macd_signal:
                summary += ". MACD is above its signal line, suggesting bullish momentum"
            else:
                summary += ". MACD is below its signal line, suggesting bearish momentum"
        
        # Add pattern information
        if patterns["detected"]:
            pattern_list = ", ".join([p.replace("_", " ").title() for p in patterns["detected"]])
            summary += f". Chart patterns detected: {pattern_list}"
        
        # Add volume information
        volume_change = indicators["volume"]["change_percent"]
        if volume_change is not None:
            if volume_change > 50:
                summary += f". Trading with significantly higher than average volume ({volume_change:.1f}% above average)"
            elif volume_change < -50:
                summary += f". Trading with significantly lower than average volume ({-volume_change:.1f}% below average)"
        
        return summary 