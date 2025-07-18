import re
import string
import nltk
from typing import List, Dict, Any
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text: str) -> str:
    """Clean and normalize text
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text: str) -> str:
    """Remove stopwords from text
    
    Args:
        text: Cleaned text string
        
    Returns:
        Text with stopwords removed
    """
    if not text:
        return ""
    
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    
    # Keep words that are not stopwords
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    return ' '.join(filtered_text)

def lemmatize_text(text: str) -> str:
    """Lemmatize text
    
    Args:
        text: Text with stopwords removed
        
    Returns:
        Lemmatized text
    """
    if not text:
        return ""
    
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    
    # Lemmatize each word
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    
    return ' '.join(lemmatized_text)

def preprocess_text(text: str, remove_stops: bool = True, lemmatize: bool = True) -> str:
    """Full preprocessing pipeline
    
    Args:
        text: Raw text string
        remove_stops: Whether to remove stopwords
        lemmatize: Whether to lemmatize text
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Remove stopwords if requested
    if remove_stops:
        cleaned_text = remove_stopwords(cleaned_text)
    
    # Lemmatize if requested
    if lemmatize:
        cleaned_text = lemmatize_text(cleaned_text)
    
    return cleaned_text

def preprocess_news_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preprocess a list of news articles
    
    Args:
        articles: List of news articles
        
    Returns:
        List of articles with preprocessed text
    """
    processed_articles = []
    
    for article in articles:
        # Extract title and description
        title = article.get("title", "")
        description = article.get("description", "")
        
        # Combine title and description
        full_text = f"{title} {description}"
        
        # Preprocess text
        preprocessed_text = preprocess_text(full_text, remove_stops=False, lemmatize=False)
        
        # Create new article with preprocessed text
        processed_article = article.copy()
        processed_article["preprocessed_text"] = preprocessed_text
        
        processed_articles.append(processed_article)
    
    return processed_articles 