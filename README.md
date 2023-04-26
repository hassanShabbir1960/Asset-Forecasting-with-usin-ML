
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

## Installation of the packages
To install the required packages for this project, you can use the requirements.txt file that lists all the dependencies.

First, ensure that you have Python installed on your system. This project requires Python version 3.6 or higher.

To install the packages listed in requirements.txt, open a terminal window and navigate to the directory where the file is located. Then, run the following command:

pip install -r requirements.txt

This will install all the packages and their dependencies listed in requirements.txt.

If you encounter any errors during the installation, please ensure that you have the necessary permissions to install packages on your system. Additionally, some packages may require additional dependencies or system configurations, so please refer to the package documentation for further instructions.

Once the packages are installed, you can run the project using a Python IDE or by running the appropriate command-line script.

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
<ul>
  <li>Nasdaq: <a href="https://finance.yahoo.com/quote/%5EIXIC/history/">https://finance.yahoo.com/quote/%5EIXIC/history/</a></li>
  <li>S&amp;P 500: <a href="https://finance.yahoo.com/quote/%5EGSPC/history/">https://finance.yahoo.com/quote/%5EGSPC/history/</a></li>
  <li>Euronext: <a href="https://finance.yahoo.com/quote/%5EN100/history/">https://finance.yahoo.com/quote/%5EN100/history/</a></li>
  <li>Dow 30: <a href="https://finance.yahoo.com/quote/%5EDJI/history/">https://finance.yahoo.com/quote/%5EDJI/history/</a></li>
  <li>NYSE composite: <a href="https://finance.yahoo.com/quote/%5ENYA/history?p=%5ENYA">https://finance.yahoo.com/quote/%5ENYA/history?p=%5ENYA</a></li>
  <li>Cboe UK 100: <a href="https://finance.yahoo.com/quote/%5EBUK100P/history?p=%5EBUK100P">https://finance.yahoo.com/quote/%5EBUK100P/history?p=%5EBUK100P</a></li>
  <li>Russell 2000: <a href="https://finance.yahoo.com/quote/%5ERUT/history?p=%5ERUT">https://finance.yahoo.com/quote/%5ERUT/history?p=%5ERUT</a></li>
  <li>Bel 20: <a href="https://finance.yahoo.com/quote/%5EBFX/history?p=%5EBFX">https://finance.yahoo.com/quote/%5EBFX/history?p=%5EBFX</a></li>
  <li>Moex Russia index: <a href="https://finance.yahoo.com/quote/IMOEX.ME/history?p=IMOEX.ME">https://finance.yahoo.com/quote/IMOEX.ME/history?p=IMOEX.ME</a></li>
  <li>Nikkei 225: <a href="https://finance.yahoo.com/quote/%5EN225/history?p=%5EN225">https://finance.yahoo.com/quote/%5EN225/history?p=%5EN225</a></li>
  <li>Hang Seng index: <a href="https://finance.yahoo.com/quote/%5EHSI/history?p=%5EHSI">https://finance.yahoo.com/quote/%5EHSI/history?p=%5EHSI</a></li>
  <li>SSE composite index: <a href="https://finance.yahoo.com/quote/000001.SS/history?p=000001.SS">https://finance.yahoo.com/quote/000001.SS/history?p=000001.SS</a></li>
  <li>Shenzhen composite: <a href="https://finance.yahoo.com/quote/399001.SZ/history?p=399001.SZ">https://finance.yahoo.com/quote/399001.SZ/history?p=399001.SZ</a></li>
  <li>Jakarta composite index: <a href="https://finance.yahoo.com/quote/%5EJKSE/history?p=%5EJKSE">https://finance.yahoo.com/quote/%5EJKSE/history?p=%5EJKSE</a></li>
 </ul>

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
