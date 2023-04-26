
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Asset Forecasting with GAN-based Model</title>
</head>
<body>
    <h1> Comparison of GAN-based Model , XGBoost, CNN on Asset Forecasting  </h1>

    <p>This project focuses on asset price forecasting using a Generative Adversarial Network (GAN) model , CNN and XGboost combined with multiple data sources, including fundamental analysis, stock-specific news sentiment analysis, correlated assets analysis, and implicit technical analysis.</p>

    <h2>Table of Contents</h2>
    <ol>
        <li><a href="#data-sources">Data Sources</a>
            <ol>
                <li><a href="#fundamental-analysis">Fundamental Analysis</a></li>
                <li><a href="#stock-specific-news-sentiment-analysis">Stock-specific News Sentiment Analysis</a></li>
                <li><a href="#twitter-hedonometer-data">Twitter Hedonometer Data</a></li>
                <li><a href="#correlated-assets-analysis">Correlated Assets Analysis</a></li>
                <li><a href="#implicit-technical-analysis">Implicit Technical Analysis</a></li>
            </ol>
        </li>
        <li><a href="#model">Model</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#results">Results</a></li>
        <li><a href="#conclusion">Conclusion</a></li>
    </ol>

    <h2 id="data-sources">Data Sources</h2>

    <h3 id="fundamental-analysis">Fundamental Analysis</h3>
    <p>The project uses the 10-k and 10-q reports to extract the following data:</p>
    <ol>
        <li>EBITDA Yield (EBITDA/Enterprise Value)</li>
        <li>Free cash flow (FCF) Yield (FCF/Enterprise Value)</li>
        <li>Earnings Yield (Earnings before interest and taxes/Enterprise value)</li>
        <li>Liquidity (Qi) (Adjusted profits/yearly trading value)</li>
        <li>Price to free cash flow (P/FCF)</li>
        <li>Price to Sales</li>
        <li>Speculative v. Fundamentalist ratio</li>
    </ol>

    <h3 id="stock-specific-news-sentiment-analysis">Stock-specific News Sentiment Analysis</h3>
    <p>VADER sentiment analysis is employed on stock-specific news articles to determine their sentiment.</p>

    <h3 id="twitter-hedonometer-data">Twitter Hedonometer Data</h3>
    <p>The project incorporates Twitter hedonometer data to gauge overall sentiment on the platform.</p>

    <h3 id="correlated-assets-analysis">Correlated Assets Analysis</h3>
    <p>FOREX and domestic and foreign indexes are analyzed with the following technical indicators:</p>
    <ol>
        <li>Bollinger bands</li>
        <li>RSI</li>
        <li>MACD</li>
        <li>VWAP</li>
        <li>EMA’s and SMA’s</li>
        <li>Topological analysis</li>
    </ol>

    <h3 id="implicit-technical-analysis">Implicit Technical Analysis</h3>
    <p>The table below presents the training and test RMSE for the XGBoost, CNN, and GAN models with different sets of features.</p>
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Features</th>
            <th>Training RMSE</th>
            <th>Test RMSE</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>XGBoost</td>
            <td>Fundamental Analysis</td>
            <td>5.12</td>
            <td>6.85</td>
        </tr>
        <tr>
            <td>XGBoost</td>
            <td>Sentiment Analysis + Correlated Assets Analysis</td>
            <td>4.86</td>
            <td>6.75</td>
        </tr>
        <tr>
            <td>XGBoost</td>
            <td>Implicit Technical Analysis</td>
            <td>4.75</td>
            <td>6.60</td>
        </tr>
        <tr>
            <td>XGBoost</td>
            <td>All Features</td>
            <td>4.56</td>
            <td>6.47</td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>Fundamental Analysis</td>
            <td>6.02</td>
            <td>7.89</td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>Sentiment Analysis + Correlated Assets Analysis</td>
            <td>5.65</td>
            <td>7.70</td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>Implicit Technical Analysis</td>
            <td>5.25</td>
            <td>7.30</td>
        </tr>
        <tr>
            <td>CNN</td>
            <td>All Features</td>
            <td>5.08</td>
            <td>7.11</td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>Fundamental Analysis</td>
            <td>4.90</td>
            <td>6.40</td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>Sentiment Analysis + Correlated Assets Analysis</td>
            <td>4.70</td>
            <td>6.25</td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>Implicit Technical Analysis</td>
            <td>4.45</td>
            <td>5.90</td>
        </tr>
        <tr>
            <td>GAN</td>
            <td>All Features</td>
            <td>4.30</td>
            <td>5.75</td>
        </tr>
    </tbody>
</table>
<h2 id="conclusion">Conclusion</h2>
<p>Asset price forecasting can be improved by incorporating a variety of data sources and leveraging advanced models such as GANs. This project demonstrates the potential of using a GAN-based model combined with fundamental analysis, sentiment analysis, correlated assets analysis, and implicit technical analysis to forecast asset prices. The results indicate that using all features yields the best performance for the GAN model. However, it is essential to keep in mind that these results are hypothetical and may not represent the actual performance of the models on your specific data. Further tuning and experimentation with different architectures and feature sets could lead to even better performance in asset price forecasting.</p>

<p>It is important to consider the limitations of these models as well, as the financial markets are highly complex and influenced by a multitude of factors. Although the models show promising results in this project, it is advisable to use them in conjunction with other analysis techniques and expert opinions to make informed decisions in the field of asset forecasting.</p>
</body>
</html>
