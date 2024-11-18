import logging
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
import matplotlib.dates as mdates
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configure logging
logging.basicConfig(
    filename='news_analysis.log',  # Save log to a file
    filemode='a',  # Append to the file
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# load data 
def load_data(file_path):
    logging.info("Loading data from file: %s", file_path)
    df = pd.read_csv(file_path)
    logging.info("Data loaded successfully with %d rows and %d columns", df.shape[0], df.shape[1])
    return df

# headline length
def headline_length(df):
    logging.info("Analyzing headline length statistics.")
    df['headline_length'] = df['headline'].str.len()
    stats = df['headline_length'].describe()
    logging.info("Headline length analysis complete.")
    return stats

# count number of articles per publisher 
def articles_per_publisher(df):
    logging.info("Counting articles per publisher.")
    counts = df['publisher'].value_counts()
    logging.info("Article count per publisher complete.")
    return counts

# analyze publication dates 
def publication_dates(df):
    logging.info("Analyzing publication dates.")
    daily_counts = df.groupby(df['date'].dt.date).size()
    top_days = daily_counts.nlargest(5)
    weekday_counts = df['date'].dt.day_name().value_counts()
    monthly_counts = df.groupby(df['date'].dt.to_period('M').dt.to_timestamp()).size()
    
    logging.info("Publication dates analysis complete.")
    return {
        'daily_counts': daily_counts,
        'top_days': top_days,
        'weekday_counts': weekday_counts,
        'monthly_counts': monthly_counts
    }

# Plot the publication trends
def plot_publication_trends(date_analysis):

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Daily trend
    date_analysis['daily_counts'].plot(ax=axes[0, 0])
    axes[0, 0].set_title('Daily Article Count')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Articles')
    
    # Top days
    date_analysis['top_days'].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Top 5 Days with Most Articles')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Number of Articles')
    
    # Weekday distribution
    date_analysis['weekday_counts'].plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Article Distribution by Weekday')
    axes[1, 0].set_xlabel('Weekday')
    axes[1, 0].set_ylabel('Number of Articles')
    
    # Monthly trend
    monthly_counts = date_analysis['monthly_counts']
    monthly_counts.plot(ax=axes[1, 1])
    axes[1, 1].set_title('Monthly Article Count')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Articles')
    
    # Format x-axis to show months
    axes[1, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    
    plt.tight_layout()
    return fig

# sentiment analysis on the 'headline' column 
def sentiment_analysis(df, text_column='headline'):
    logging.info("Performing sentiment analysis on column: %s", text_column)
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(x))
    df['sentiment'] = df['sentiment_scores'].apply(lambda x: 'positive' if x['compound'] > 0 else ('negative' if x['compound'] < 0 else 'neutral'))
    logging.info("Sentiment analysis complete.")
    return df

# Topic Modeling on 'headline' column
def perform_topic_modeling(df, text_column='headline', num_topics=5, num_words=10):
    logging.info("Performing topic modeling on column: %s", text_column)
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df[text_column])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    logging.info("Topic modeling complete. %d topics identified.", num_topics)
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [(words[i], topic[i]) for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(top_words)
    return topics

# Time Series Analysis - Analyze the distribution of publication times throughout the day
def analyze_publication_times(df):
    logging.info("Analyzing distribution of publication times throughout the day.")
    df['hour'] = df['date'].dt.hour
    hourly_distribution = df['hour'].value_counts().sort_index()
    logging.info("Distribution analysis by hour complete.")
    
    plt.figure(figsize=(12, 6))
    hourly_distribution.plot(kind='bar')
    plt.title('Distribution of Article Publications by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    peak_hour = hourly_distribution.idxmax()
    logging.info("The peak publication hour is %d:00.", peak_hour)
    return f"The peak publication hour is {peak_hour}:00"

# Identify days with unusually high publication frequency
def identify_publication_spikes(df, threshold=2):
    logging.info("Identifying days with unusually high publication frequency.")
    daily_counts = df.groupby(df['date'].dt.date).size()
    mean_publications = daily_counts.mean()
    std_publications = daily_counts.std()
    
    spikes = daily_counts[daily_counts > mean_publications + threshold * std_publications]
    logging.info("Publication spikes identified: %d days.", len(spikes))
    return spikes

# Publisher Analysis - Analyze the contribution and type of news from different publishers
def analyze_publishers(df):
    logging.info("Analyzing contribution by different publishers.")
    publisher_counts = df['publisher'].value_counts()
    top_publishers = publisher_counts.head(10)
    logging.info("Top 10 publishers analysis complete.")
    
    plt.figure(figsize=(12, 6))
    top_publishers.plot(kind='bar')
    plt.title('Top 10 Publishers by Number of Articles')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return top_publishers

# Identify unique domains if email addresses are used as publisher names
def analyze_publisher_domains(df):
    logging.info("Analyzing unique domains in publisher names.")
    def extract_domain(email):
        try:
            return email.split('@')[1]
        except:
            return email
    
    df['domain'] = df['publisher'].apply(extract_domain)
    domain_counts = df['domain'].value_counts()
    logging.info("Publisher domain analysis complete.")
    
    plt.figure(figsize=(12, 6))
    domain_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Publisher Domains')
    plt.xlabel('Domain')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return domain_counts

# Analyze differences in the type of news reported by top publishers
def analyze_news_types_by_publisher(df, top_n=5):
    logging.info("Analyzing news types reported by top %d publishers.", top_n)
    stop_words = set(stopwords.words('english'))
    top_publishers = df['publisher'].value_counts().head(top_n).index
    
    for publisher in top_publishers:
        logging.info("Analyzing news type for publisher: %s", publisher)
        publisher_headlines = df[df['publisher'] == publisher]['headline']
        words = []
        for headline in publisher_headlines:
            tokens = word_tokenize(headline.lower())
            words.extend([word for word in tokens if word.isalnum() and word not in stop_words])
        
        word_freq = Counter(words)
        
        logging.info("Top 10 most common words for %s: %s", publisher, word_freq.most_common(10))
        
        print(f"\nTop 10 most common words for {publisher}:")
        for word, count in word_freq.most_common(10):
            print(f"{word}: {count}")
