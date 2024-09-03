                                  Stock Price and News Sentiment Analysis

Project Overview

This project analyzes the relationship between stock price movements and news sentiment. The main objective is to determine if there is any correlation between the sentiment of news articles related to a specific stock and the stock's daily returns. The analysis is conducted using historical stock price data and sentiment analysis of news headlines.

Data Description
1. Stock Price Data
Source: Historical stock price data for META.
Columns:
Date: The date of the stock price record.
Open: The opening price of the stock on the given date.
High: The highest price of the stock on the given date.
Low: The lowest price of the stock on the given date.
Close: The closing price of the stock on the given date.
Volume: The number of shares traded on the given date.
2. News Data
Source: News headlines related to META.
Columns:
Date: The publication date of the news article.
Headline: The text of the news headline.
Sentiment Score: The sentiment score assigned to each headline (positive, negative, or neutral).

Analysis Workflow
Data Loading:

Load stock price data and news headlines data into pandas DataFrames.
Data Preprocessing:

Convert date columns to datetime format.
Align stock prices and news data by date.
Calculate daily stock returns.
Sentiment Analysis:

Perform sentiment analysis on news headlines using TextBlob or NLTK.
Assign a sentiment score (positive, negative, neutral) to each headline.
Technical Indicators:

Calculate technical indicators such as Simple Moving Averages (SMA) and Relative Strength Index (RSI) using TA-Lib.
Correlation Analysis:

Analyze the correlation between news sentiment scores and daily stock returns.
Generate visualizations to understand the relationship between sentiment and stock price movements.

Visualizations
Daily Stock Returns:

Visual representation of the percentage change in stock prices over time.
Correlation between News Sentiment and Stock Returns:

Scatter plot showing the relationship between sentiment scores and daily stock returns.
Conclusion
The analysis reveals that there is no strong or consistent correlation between news sentiment and daily stock returns for META. This suggests that other factors might have a more significant impact on stock price movements.