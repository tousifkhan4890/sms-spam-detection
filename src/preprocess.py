"""
Text preprocessing utilities for SMS Spam Detection project.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Cleans raw SMS text for model input:
    - Lowercases text
    - Removes non-alphabetic characters
    - Removes stopwords
    - Applies stemming

    Args:
        text (str): Raw message text.

    Returns:
        str: Cleaned, stemmed message text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)