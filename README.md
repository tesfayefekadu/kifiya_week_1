# Financial-news-Stock-Analysis

## Project Overview
This project is focused on analyzing a large corpus of financial news data to discover correlations between news sentiment and stock market movements. 

### Main Objectives
1. Perform sentiment analysis on financial news headlines.
2. Establish statistical correlations between news sentiment and stock price movements.
3. Provide actionable insights and investment strategies based on your analysis.


## Folder Structure

```plaintext
├── .vscode/
│   └── settings.json  
├── app/
│   └── app.py     # streamlit app         
├── .github/
│   └── workflows/
│       └── unittests.yml      # GitHub Actions
├── .gitignore                 # directories to be ignored by git
├── requirements.txt           # contains dependencies for the project
├── README.md                  
├── src/
│   ├── __init__.py
│   
├── notebooks/
│   ├── __init__.py 
├── tests/
└── scripts/
    ├── __init__.py
    ├── eda_analysis.py # Script for financial news analysis 
    ├── financial_analyzer.py # script for the stock data analysis    
    ├── sentiment_analysis.py # script for financial news and stock price integration analysis
    └── README.md             # Documentation for the scripts directory
```
## Setup

1. Clone the repository:
   ```
   git clone 
   cd KIFIYA_WEEK_1
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Streamlit App

To run the Streamlit app locally:

1. Ensure you're in the project directory and your virtual environment is activated (if you're using one).

2. Run the following command:
   ```
   streamlit run app.py
   ```

## Data Sources

- Historical stock data: Retrieved from Yahoo Finance
- News sentiment data: Collected from financial news sources 


## Contact

For any questions or feedback, please open an issue on this GitHub repository.