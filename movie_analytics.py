#!/usr/bin/env python3
"""
Simple Movie Review Analytics
Analyzes IMDB dataset with static data evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MovieReviewAnalytics:
    def __init__(self, csv_path='IMDB-Dataset.csv'):
        """Initialize with dataset path"""
        self.csv_path = csv_path
        self.df = None
        self.positive_reviews = None
        self.negative_reviews = None
        
    def load_data(self):
        """Load and basic preprocessing of the dataset"""
        print("🔄 Loading IMDB dataset...")
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {self.df.shape}")
            print(f"📝 Columns: {list(self.df.columns)}")
            return True
        except FileNotFoundError:
            print("❌ Dataset file not found! Please ensure IMDB-Dataset.csv is in the current directory.")
            return False
    
    def basic_statistics(self):
        """Generate basic statistics about the dataset"""
        print("\n" + "="*60)
        print("📈 BASIC DATASET STATISTICS")
        print("="*60)
        
        # Dataset info
        print(f"Total reviews: {len(self.df):,}")
        print(f"Unique reviews: {self.df['review'].nunique():,}")
        print(f"Duplicate reviews: {len(self.df) - self.df['review'].nunique():,}")
        
        # Sentiment distribution
        sentiment_counts = self.df['sentiment'].value_counts()
        print(f"\n🎭 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {sentiment}: {count:,} ({percentage:.1f}%)")
        
        # Review length statistics
        self.df['review_length'] = self.df['review'].str.len()
        self.df['word_count'] = self.df['review'].str.split().str.len()
        
        print(f"\n📏 Review Length Statistics:")
        print(f"  Average characters: {self.df['review_length'].mean():.0f}")
        print(f"  Median characters: {self.df['review_length'].median():.0f}")
        print(f"  Min characters: {self.df['review_length'].min()}")
        print(f"  Max characters: {self.df['review_length'].max():,}")
        
        print(f"\n📝 Word Count Statistics:")
        print(f"  Average words: {self.df['word_count'].mean():.0f}")
        print(f"  Median words: {self.df['word_count'].median():.0f}")
        print(f"  Min words: {self.df['word_count'].min()}")
        print(f"  Max words: {self.df['word_count'].max():,}")
    
    def visualize_data(self):
        """Create visualizations of the dataset"""
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Sentiment Distribution
        sentiment_counts = self.df['sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                      autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # 2. Review Length Distribution
        axes[0, 1].hist(self.df['review_length'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Distribution of Review Lengths (Characters)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Number of Characters')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Word Count Distribution
        axes[1, 0].hist(self.df['word_count'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('Distribution of Word Counts', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Words')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Box plot of review lengths by sentiment
        self.df.boxplot(column='word_count', by='sentiment', ax=axes[1, 1])
        axes[1, 1].set_title('Word Count by Sentiment', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Sentiment')
        axes[1, 1].set_ylabel('Word Count')
        
        plt.tight_layout()
        plt.savefig('movie_review_analytics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Visualizations saved as 'movie_review_analytics.png'")
    
    def text_preprocessing(self, text):
        """Simple text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def analyze_vocabulary(self):
        """Analyze vocabulary and word frequencies"""
        print("\n" + "="*60)
        print("📚 VOCABULARY ANALYSIS")
        print("="*60)
        
        # Preprocess text
        self.df['clean_review'] = self.df['review'].apply(self.text_preprocessing)
        
        # Separate positive and negative reviews
        self.positive_reviews = self.df[self.df['sentiment'] == 'positive']['clean_review']
        self.negative_reviews = self.df[self.df['sentiment'] == 'negative']['clean_review']
        
        # Count words
        all_words = ' '.join(self.df['clean_review']).split()
        positive_words = ' '.join(self.positive_reviews).split()
        negative_words = ' '.join(self.negative_reviews).split()
        
        print(f"Total unique words: {len(set(all_words)):,}")
        print(f"Positive review words: {len(positive_words):,}")
        print(f"Negative review words: {len(negative_words):,}")
        
        # Most common words
        word_freq = Counter(all_words)
        positive_freq = Counter(positive_words)
        negative_freq = Counter(negative_words)
        
        print(f"\n🔥 Top 10 Most Common Words Overall:")
        for word, count in word_freq.most_common(10):
            print(f"  {word}: {count:,}")
        
        print(f"\n😊 Top 10 Words in Positive Reviews:")
        for word, count in positive_freq.most_common(10):
            print(f"  {word}: {count:,}")
        
        print(f"\n😞 Top 10 Words in Negative Reviews:")
        for word, count in negative_freq.most_common(10):
            print(f"  {word}: {count:,}")
    
    def baseline_models(self):
        """Train and evaluate baseline models"""
        print("\n" + "="*60)
        print("🤖 BASELINE MODEL EVALUATION")
        print("="*60)
        
        # Prepare data
        X = self.df['clean_review']
        y = self.df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train):,}")
        print(f"Test set size: {len(X_test):,}")
        
        # TF-IDF Vectorization
        print("\n🔄 Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            model.fit(X_train_tfidf, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            print(f"✅ {name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Classification report
            print(f"\n📊 {name} Classification Report:")
            print(classification_report(y_test, y_pred))
        
        # Best model
        best_model = max(results, key=results.get)
        best_accuracy = results[best_model]
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        return results
    
    def sample_predictions(self):
        """Test with sample movie reviews"""
        print("\n" + "="*60)
        print("🧪 SAMPLE PREDICTIONS")
        print("="*60)
        
        # Simple rule-based classifier for demo
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'brilliant', 'outstanding', 'superb', 'perfect', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'boring', 
                         'disappointing', 'waste', 'poor', 'hate', 'stupid', 'pathetic']
        
        sample_reviews = [
            "This movie was absolutely amazing! The acting was superb and the plot was engaging.",
            "Terrible movie. Poor acting and boring storyline. Waste of time.",
            "The movie was okay, nothing special but not bad either.",
            "One of the best films I've ever seen! Highly recommended!",
            "Completely disappointing. Expected much better from this director."
        ]
        
        print("📝 Simple Rule-Based Classification:")
        print("-" * 40)
        
        for i, review in enumerate(sample_reviews, 1):
            clean_review = self.text_preprocessing(review).split()
            
            positive_score = sum(1 for word in clean_review if word in positive_words)
            negative_score = sum(1 for word in clean_review if word in negative_words)
            
            if positive_score > negative_score:
                sentiment = "POSITIVE"
                confidence = positive_score / (positive_score + negative_score + 1)
            elif negative_score > positive_score:
                sentiment = "NEGATIVE"
                confidence = negative_score / (positive_score + negative_score + 1)
            else:
                sentiment = "NEUTRAL"
                confidence = 0.5
            
            print(f"\n{i}. Review: \"{review}\"")
            print(f"   Prediction: {sentiment} (Confidence: {confidence:.2%})")
            print(f"   Positive words: {positive_score}, Negative words: {negative_score}")
    
    def run_complete_analysis(self):
        """Run the complete analytics pipeline"""
        print("🎬" * 20)
        print("MOVIE REVIEW SENTIMENT ANALYTICS")
        print("🎬" * 20)
        
        # Load data
        if not self.load_data():
            return
        
        # Run all analyses
        self.basic_statistics()
        self.visualize_data()
        self.analyze_vocabulary()
        baseline_results = self.baseline_models()
        self.sample_predictions()
        
        # Summary
        print("\n" + "="*60)
        print("📋 ANALYSIS SUMMARY")
        print("="*60)
        print(f"✅ Dataset loaded: {len(self.df):,} reviews")
        print(f"✅ Baseline model accuracies:")
        for model, accuracy in baseline_results.items():
            print(f"   {model}: {accuracy*100:.2f}%")
        print(f"✅ Visualizations saved as 'movie_review_analytics.png'")
        print(f"✅ Analysis complete!")

def main():
    """Main function to run the analytics"""
    analyzer = MovieReviewAnalytics()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
