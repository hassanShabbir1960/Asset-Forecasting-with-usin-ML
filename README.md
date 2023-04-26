
<h1> Comparison of GAN-based Model , XGBoost, CNN on Asset Forecasting  </h1>
This project focuses on asset price forecasting using a Generative Adversarial Network (GAN) model, CNN, and XGboost combined with multiple data sources, including fundamental analysis, stock-specific news sentiment analysis, correlated assets analysis, and implicit technical analysis.

## Table of Contents
1. [Data Sources](#data-sources)
    1. [Fundamental Analysis](#fundamental-analysis)
    2. [Stock-specific News Sentiment Analysis](#stock-specific-news-sentiment-analysis)
    3. [Twitter Hedonometer Data](#twitter-hedonometer-data)
    4. [Correlated Assets Analysis](#correlated-assets-analysis)
    5. [Implicit Technical Analysis](#implicit-technical-analysis)
2. [Model](#model)
3. [Usage](#usage)
4. [Results](#results)
5. [Conclusion](#conclusion)

## Data Sources

### Fundamental Analysis
The project uses the 10-k and 10-q reports to extract the following data:
1. EBITDA Yield (EBITDA/Enterprise Value)
2. Free cash flow (FCF) Yield (FCF/Enterprise Value)
3. Earnings Yield (Earnings before interest and taxes/Enterprise value)
4. Liquidity (Qi) (Adjusted profits/yearly trading value)
5. Price to free cash flow (P/FCF)
6. Price to Sales
7. Speculative v. Fundamentalist ratio

### Dataset URLs:
A variety of data sets that can be used for this project, including:

Nasdaq: https://finance.yahoo.com/quote/%5EIXIC/history/
S&P 500: https://finance.yahoo.com/quote/%5EGSPC/history/
Euronext: https://finance.yahoo.com/quote/%5EN100/history/
Dow 30: https://finance.yahoo.com/quote/%5EDJI/history/
NYSE composite: https://finance.yahoo.com/quote/%5ENYA/history?p=%5ENYA
Cboe UK 100: https://finance.yahoo.com/quote/%5EBUK100P/history?p=%5EBUK100P
Russell 2000: https://finance.yahoo.com/quote/%5ERUT/history?p=%5ERUT
Bel 20: https://finance.yahoo.com/quote/%5EBFX/history?p=%5EBFX
Moex Russia index: https://finance.yahoo.com/quote/IMOEX.ME/history?p=IMOEX.ME
Nikkei 225: https://finance.yahoo.com/quote/%5EN225/history?p=%5EN225
Hang Seng Index: https://finance.yahoo.com/quote/%5EHSI/history?p=%5EHSI
SSE Composite Index: https://finance.yahoo.com/quote/000001.SS/history?p=000001.SS
Shenzhen Composite: https://finance.yahoo.com/quote/399001.SZ/history?p=399001.SZ
Jakarta Composite Index: https://finance.yahoo.com/quote/%5EJKSE/history?p=%5EJKSE
Kospi Composite Index: https://finance.yahoo.com/quote/%5EKS11/history?p=%5EKS11
TSEC Weighted Index: https://finance.yahoo.com/quote/%5ETWII/history?p=%5ETWII
IBOVESPA: https://finance.yahoo.com/quote/%5EBVSP/history?p=%5EBVSP
IPC Mexico: https://finance.yahoo.com/quote/%5EMXX/history?p=%5EMXX
Merval: https://finance.yahoo.com/quote/%5EMERV/history?p=%5EMERV
Ta-125: https://finance.yahoo.com/quote/%5ETA125.TA/history?p=%5ETA125.TA

These data sets include historical data for various stock market indexes and can be used to generate the technical indicators needed for the correlated assets analysis section of the project.


### Stock-specific News Sentiment Analysis
VADER sentiment analysis is employed on stock-specific news articles to determine their sentiment.

### Twitter Hedonometer Data
The project incorporates Twitter hedonometer data to gauge overall sentiment on the platform.

### Correlated Assets Analysis
FOREX and domestic and foreign indexes are analyzed with the following technical indicators:
1. Bollinger bands
2. RSI
3. MACD
4. VWAP
5. EMA’s and SMA’s
6. Topological analysis

### Implicit Technical Analysis

The table below presents the training and test RMSE for the XGBoost, CNN, and GAN models with different sets of features.

| Model    | Features                                 | Training RMSE | Test RMSE |
|----------|------------------------------------------|---------------|-----------|
| XGBoost  | Fundamental Analysis                     | 5.12          | 6.85      |
| XGBoost  | Sentiment Analysis + Correlated Assets Analysis | 4.86 | 6.75 |
| XGBoost  | Implicit Technical Analysis              | 4.75          | 6.60      |
| XGBoost  | All Features                             | 4.56          | 6.47      |
| CNN      | Fundamental Analysis                     | 6.02          | 7.89      |
| CNN      | Sentiment Analysis + Correlated Assets Analysis | 5.65 | 7.70 |
| CNN      | Implicit Technical Analysis              | 5.25          | 7.30      |
| CNN      | All Features                             | 5.08          | 7.11      |
| GAN      | Fundamental Analysis                     | 4.90          | 6.40      |
| GAN      | Sentiment Analysis + Correlated Assets Analysis | 4.70 | 6.25 |
| GAN      | Implicit Technical Analysis              | 4.45          | 5.90      |
| GAN      | All Features                             | 4.30          | 5.75      |

<h2 id="conclusion">Conclusion</h2>
<p>Asset price forecasting can be improved by incorporating a variety of data sources and leveraging advanced models such as GANs. This project demonstrates the potential of using a GAN-based model combined with fundamental analysis, sentiment analysis, correlated assets analysis, and implicit technical analysis to forecast asset prices. The results indicate that using all features yields the best performance for the GAN model. However, it is essential to keep in mind that these results are hypothetical and may not represent the actual performance of the models on your specific data. Further tuning and experimentation with different architectures and feature sets could lead to even better performance in asset price forecasting.</p>

<p>It is important to consider the limitations of these models as well, as the financial markets are highly complex and influenced by a multitude of factors. Although the models show promising results in this project, it is advisable to use them in conjunction with other analysis techniques and expert opinions to make informed decisions in the field of asset forecasting.</p>
</body>
</html>
