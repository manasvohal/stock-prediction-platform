import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from app.services.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

class FundamentalAnalysis:
    """Service for analyzing fundamental data of stocks"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
    
    def analyze(self, ticker: str) -> Dict[str, Any]:
        """Perform fundamental analysis on a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with fundamental analysis results
        """
        try:
            # Get stock info and financial statements
            stock_info = self.data_fetcher.get_stock_info(ticker)
            financials = self.data_fetcher.get_financial_statements(ticker)
            
            # Extract key financial statements
            income_stmt = financials["income_statement"]
            balance_sheet = financials["balance_sheet"]
            cash_flow = financials["cash_flow"]
            
            # Calculate key metrics
            valuation_metrics = self._calculate_valuation_metrics(stock_info, income_stmt, balance_sheet)
            growth_metrics = self._calculate_growth_metrics(income_stmt, cash_flow)
            financial_health_metrics = self._calculate_financial_health(balance_sheet, income_stmt)
            
            # Generate summary
            summary = self._generate_summary(
                stock_info, 
                valuation_metrics, 
                growth_metrics, 
                financial_health_metrics
            )
            
            # Combine all metrics
            result = {
                **valuation_metrics,
                **growth_metrics,
                **financial_health_metrics,
                "summary": summary,
                "company_name": stock_info["company_name"],
                "sector": stock_info["sector"],
                "industry": stock_info["industry"],
                "market_cap": stock_info["market_cap"],
                "current_price": stock_info["current_price"],
                "currency": stock_info["currency"],
                "business_summary": stock_info["business_summary"]
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {ticker}: {e}")
            raise ValueError(f"Could not perform fundamental analysis for {ticker}: {str(e)}")
    
    def _calculate_valuation_metrics(
        self, 
        stock_info: Dict[str, Any], 
        income_stmt: pd.DataFrame, 
        balance_sheet: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate valuation metrics"""
        try:
            # Get the most recent data (first column)
            latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
            latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            
            # Extract metrics from stock info
            pe_ratio = stock_info.get("trailingPE", None)
            pb_ratio = stock_info.get("priceToBook", None)
            ps_ratio = stock_info.get("priceToSalesTrailing12Months", None)
            
            # Calculate additional metrics if not available in stock info
            if pe_ratio is None and "Net Income" in latest_income and stock_info["market_cap"] > 0:
                net_income = latest_income["Net Income"]
                if net_income > 0:
                    pe_ratio = stock_info["market_cap"] / net_income
            
            if pb_ratio is None and "Total Stockholder Equity" in latest_balance and stock_info["market_cap"] > 0:
                equity = latest_balance["Total Stockholder Equity"]
                if equity > 0:
                    pb_ratio = stock_info["market_cap"] / equity
            
            # Calculate valuation score (lower is better)
            valuation_score = 0.5  # Default neutral score
            
            # Adjust based on industry averages (simplified)
            sector = stock_info["sector"]
            if sector and pe_ratio and pb_ratio:
                # These are simplified industry benchmarks
                sector_pe_benchmarks = {
                    "Technology": 30,
                    "Healthcare": 25,
                    "Consumer Cyclical": 20,
                    "Financial Services": 15,
                    "Industrials": 18,
                    "Communication Services": 22,
                    "Consumer Defensive": 17,
                    "Energy": 12,
                    "Basic Materials": 14,
                    "Utilities": 16,
                    "Real Estate": 20
                }
                
                benchmark_pe = sector_pe_benchmarks.get(sector, 20)
                
                # Calculate score (0 to 1, lower is better value)
                if pe_ratio > 0:
                    pe_score = min(1, pe_ratio / (benchmark_pe * 2))
                    pb_score = min(1, pb_ratio / 5)  # Simplified PB benchmark
                    valuation_score = (pe_score * 0.7) + (pb_score * 0.3)
            
            return {
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "ps_ratio": ps_ratio,
                "valuation_score": valuation_score,
                "dividend_yield": stock_info.get("dividendYield", 0) * 100 if stock_info.get("dividendYield") else 0,
                "peg_ratio": stock_info.get("pegRatio", None)
            }
        except Exception as e:
            logger.error(f"Error calculating valuation metrics: {e}")
            return {
                "pe_ratio": None,
                "pb_ratio": None,
                "ps_ratio": None,
                "valuation_score": 0.5,
                "dividend_yield": 0,
                "peg_ratio": None
            }
    
    def _calculate_growth_metrics(
        self, 
        income_stmt: pd.DataFrame, 
        cash_flow: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate growth metrics"""
        try:
            # Need at least 2 years of data
            if income_stmt.shape[1] < 2 or cash_flow.shape[1] < 2:
                return {
                    "revenue_growth": None,
                    "earnings_growth": None,
                    "fcf_growth": None,
                    "growth_score": 0.5
                }
            
            # Calculate revenue growth
            if "Total Revenue" in income_stmt.index:
                latest_revenue = income_stmt.loc["Total Revenue", income_stmt.columns[0]]
                prev_revenue = income_stmt.loc["Total Revenue", income_stmt.columns[1]]
                revenue_growth = ((latest_revenue - prev_revenue) / prev_revenue) * 100 if prev_revenue > 0 else 0
            else:
                revenue_growth = None
            
            # Calculate earnings growth
            if "Net Income" in income_stmt.index:
                latest_earnings = income_stmt.loc["Net Income", income_stmt.columns[0]]
                prev_earnings = income_stmt.loc["Net Income", income_stmt.columns[1]]
                earnings_growth = ((latest_earnings - prev_earnings) / abs(prev_earnings)) * 100 if prev_earnings != 0 else 0
            else:
                earnings_growth = None
            
            # Calculate free cash flow growth
            if "Free Cash Flow" in cash_flow.index:
                latest_fcf = cash_flow.loc["Free Cash Flow", cash_flow.columns[0]]
                prev_fcf = cash_flow.loc["Free Cash Flow", cash_flow.columns[1]]
                fcf_growth = ((latest_fcf - prev_fcf) / abs(prev_fcf)) * 100 if prev_fcf != 0 else 0
            else:
                fcf_growth = None
            
            # Calculate growth score (higher is better)
            growth_score = 0.5  # Default neutral score
            
            # Adjust based on growth rates
            if revenue_growth is not None and earnings_growth is not None:
                # Weight revenue growth and earnings growth
                rev_score = min(1, max(0, (revenue_growth + 10) / 40))  # Scale -10% to 30% to 0-1
                earn_score = min(1, max(0, (earnings_growth + 20) / 60))  # Scale -20% to 40% to 0-1
                growth_score = (rev_score * 0.4) + (earn_score * 0.6)
            
            return {
                "revenue_growth": revenue_growth,
                "earnings_growth": earnings_growth,
                "fcf_growth": fcf_growth,
                "growth_score": growth_score
            }
        except Exception as e:
            logger.error(f"Error calculating growth metrics: {e}")
            return {
                "revenue_growth": None,
                "earnings_growth": None,
                "fcf_growth": None,
                "growth_score": 0.5
            }
    
    def _calculate_financial_health(
        self, 
        balance_sheet: pd.DataFrame, 
        income_stmt: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate financial health metrics"""
        try:
            # Get the most recent data
            latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            latest_income = income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
            
            # Calculate debt-to-equity ratio
            if "Total Debt" in latest_balance and "Total Stockholder Equity" in latest_balance:
                total_debt = latest_balance["Total Debt"]
                total_equity = latest_balance["Total Stockholder Equity"]
                debt_to_equity = total_debt / total_equity if total_equity > 0 else float('inf')
            else:
                debt_to_equity = None
            
            # Calculate current ratio
            if "Current Assets" in latest_balance and "Current Liabilities" in latest_balance:
                current_assets = latest_balance["Current Assets"]
                current_liabilities = latest_balance["Current Liabilities"]
                current_ratio = current_assets / current_liabilities if current_liabilities > 0 else float('inf')
            else:
                current_ratio = None
            
            # Calculate return on equity (ROE)
            if "Net Income" in latest_income and "Total Stockholder Equity" in latest_balance:
                net_income = latest_income["Net Income"]
                total_equity = latest_balance["Total Stockholder Equity"]
                roe = (net_income / total_equity) * 100 if total_equity > 0 else None
            else:
                roe = None
            
            # Calculate financial health score (higher is better)
            health_score = 0.5  # Default neutral score
            
            if debt_to_equity is not None and current_ratio is not None:
                # Lower debt-to-equity is better (0-3 scale)
                de_score = max(0, min(1, 1 - (debt_to_equity / 3)))
                
                # Higher current ratio is better (1-3 scale)
                cr_score = max(0, min(1, (current_ratio - 1) / 2))
                
                # Combine scores
                health_score = (de_score * 0.5) + (cr_score * 0.5)
            
            return {
                "debt_to_equity": debt_to_equity,
                "current_ratio": current_ratio,
                "roe": roe,
                "financial_health_score": health_score
            }
        except Exception as e:
            logger.error(f"Error calculating financial health metrics: {e}")
            return {
                "debt_to_equity": None,
                "current_ratio": None,
                "roe": None,
                "financial_health_score": 0.5
            }
    
    def _generate_summary(
        self, 
        stock_info: Dict[str, Any],
        valuation_metrics: Dict[str, Any],
        growth_metrics: Dict[str, Any],
        financial_health_metrics: Dict[str, Any]
    ) -> str:
        """Generate a human-readable summary of the fundamental analysis"""
        company_name = stock_info["company_name"]
        
        # Valuation assessment
        if valuation_metrics["valuation_score"] < 0.3:
            valuation_text = f"{company_name} appears undervalued based on traditional metrics"
        elif valuation_metrics["valuation_score"] < 0.7:
            valuation_text = f"{company_name} is fairly valued relative to the market"
        else:
            valuation_text = f"{company_name} appears overvalued based on traditional metrics"
        
        # Growth assessment
        if growth_metrics["growth_score"] > 0.7:
            growth_text = "showing strong growth in revenue and earnings"
        elif growth_metrics["growth_score"] > 0.3:
            growth_text = "showing moderate growth"
        else:
            growth_text = "showing weak or negative growth"
        
        # Financial health assessment
        if financial_health_metrics["financial_health_score"] > 0.7:
            health_text = "with a strong balance sheet and financial position"
        elif financial_health_metrics["financial_health_score"] > 0.3:
            health_text = "with an adequate financial position"
        else:
            health_text = "with potential financial health concerns"
        
        # Combine assessments
        summary = f"{valuation_text}, {growth_text}, {health_text}."
        
        # Add dividend info if applicable
        if valuation_metrics["dividend_yield"] > 0:
            summary += f" The company offers a {valuation_metrics['dividend_yield']:.2f}% dividend yield."
        
        return summary 