import axios from 'axios';

const API_URL = 'http://localhost:8001/api/v1';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface StockInfo {
  ticker: string;
  company_name: string;
  sector: string;
  industry: string;
  country: string;
  market_cap: number;
  current_price: number;
  currency: string;
  exchange: string;
  website: string;
  logo_url: string;
  business_summary: string;
}

export interface ChartData {
  ticker: string;
  period: string;
  data: {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    dividends: number;
    stock_splits: number;
  }[];
}

export interface FundamentalData {
  pe_ratio?: number;
  pb_ratio?: number;
  ps_ratio?: number;
  peg_ratio?: number;
  dividend_yield?: number;
  valuation_score?: number;
  revenue_growth?: number;
  earnings_growth?: number;
  fcf_growth?: number;
  growth_score?: number;
  debt_to_equity?: number;
  current_ratio?: number;
  roe?: number;
  financial_health_score?: number;
  summary?: string;
  company_name: string;
  sector: string;
  industry: string;
  market_cap: number;
  current_price: number;
  currency: string;
  business_summary: string;
}

export interface TechnicalData {
  ticker: string;
  period: string;
  last_price: number;
  price_change: number;
  price_change_percent: number;
  volume_avg: number;
  indicators: {
    moving_averages: {
      ma20: number;
      ma50: number;
      ma200: number;
    };
    rsi: number;
    macd: {
      macd: number;
      signal: number;
      histogram: number;
    };
    bollinger_bands: {
      upper: number;
      lower: number;
      width: number;
    };
    atr: number;
    volume: {
      current: number;
      average: number;
      change_percent: number;
    };
    support_resistance: {
      support: number[];
      resistance: number[];
    };
  };
  trend: string;
  trend_strength: number;
  patterns: {
    detected: string[];
    descriptions: Record<string, string>;
  };
  support_resistance: {
    support: number[];
    resistance: number[];
  };
  summary: string;
}

export interface PredictionData {
  ticker: string;
  prediction: 'buy' | 'hold' | 'sell';
  confidence: number;
  predicted_return: number;
  explanation: string;
  prediction_date: string;
  time_horizon: string;
  fundamental_factors: {
    pe_ratio: number;
    growth_score: number;
    valuation_score: number;
    financial_health_score: number;
  };
  technical_factors: {
    rsi: number;
    trend: string;
    trend_strength: number;
    macd: number;
    macd_signal: number;
  };
  sentiment_factors: {
    sentiment_score: number;
    sentiment_label: string;
    sentiment_magnitude: number;
  };
  feature_importance: Record<string, number>;
}

export interface SentimentData {
  ticker: string;
  sentiment_score: number;
  sentiment_label: string;
  sentiment_magnitude: number;
  article_count: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  summary: string;
  articles: {
    title: string;
    description: string;
    url: string;
    publishedAt: string;
    source: {
      name: string;
    };
    preprocessed_text: string;
    sentiment_score: number;
    sentiment_label: string;
  }[];
}

export const fetchStockInfo = async (ticker: string): Promise<StockInfo> => {
  const response = await api.get(`/stocks/${ticker}`);
  return response.data;
};

export const fetchChartData = async (ticker: string, period: string): Promise<ChartData> => {
  const response = await api.get(`/stocks/${ticker}/chart?period=${period}`);
  return response.data;
};

export const fetchFundamentals = async (ticker: string): Promise<FundamentalData> => {
  const response = await api.get(`/fundamentals/${ticker}`);
  return response.data;
};

export const fetchTechnicals = async (ticker: string, period: string): Promise<TechnicalData> => {
  const response = await api.get(`/technical/${ticker}?period=${period}`);
  return response.data;
};

export const fetchPrediction = async (ticker: string): Promise<PredictionData> => {
  const response = await api.get(`/predict/${ticker}`);
  return response.data;
};

export const fetchSentiment = async (ticker: string, days: number): Promise<SentimentData> => {
  const response = await api.get(`/sentiment/${ticker}?days=${days}`);
  return response.data;
};

export default api; 