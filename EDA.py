import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import seaborn as sns
import empath
from collections import defaultdict
import re
df = pd.read_csv('Chatbot Support Research Study (Responses) - Form Responses 1.csv')

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', kde=True, color='skyblue', edgecolor='black')

plt.title('Distribution of Participant Ages', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
# Add mean age line
mean_age = df['Age'].mean()
plt.axvline(x=mean_age, color='red', linestyle='--', label=f'Mean Age: {mean_age:.2f}')

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Mean Age: {mean_age:.2f}")
print(f"Median Age: {df['Age'].median():.2f}")
print(f"Age Range: {df['Age'].min()} to {df['Age'].max()}")
print(f"Standard Deviation: {df['Age'].std():.2f}")

gender_counts = df['Gender'].value_counts()
labels = gender_counts.keys()
sizes = gender_counts.values

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightpink'])
plt.axis('equal')  
plt.title('Gender Distribution')
plt.show()

df['Frequency of Using Customer Services'].value_counts()
df['Have you ever used a Chatbot for customer service/support ?'].value_counts()
df['Have you ever used a Chatbot for customer service/support ?'].isna().sum()
df['Have you ever used a Chatbot for customer service/support ?'].fillna('No',inplace = True)

df['How professional did you find the human representative?'].value_counts()
df['Which support type do you believe is more efficient in resolving issues?'].value_counts()


liwc_categories = {
    'positive_emotions': ['happy', 'joy', 'excellent', 'good', 'love', 'like', 'satisfied'],
    'negative_emotions': ['sad', 'angry', 'hate', 'terrible', 'awful', 'unsatisfied', 'frustrat'],
    'cognitive_processes': ['think', 'know', 'consider', 'understand', 'solve', 'resolve'],
    'time': ['quick', 'fast', 'slow', 'time', 'wait', 'immediate', 'delay'],
    'communication': ['talk', 'chat', 'speak', 'communicate', 'explain', 'clarify'],
    'technology': ['bot', 'chatbot', 'system', 'automated', 'ai', 'tech'],
    'human': ['person', 'human', 'representative', 'agent', 'staff'],
    'improvement': ['better', 'improve', 'enhance', 'upgrade', 'fix', 'change']
}

def analyze_text(text):
    if not isinstance(text, str) or pd.isna(text):
        return None
    
    words = re.findall(r'\b\w+\b', text.lower())
    
    category_counts = defaultdict(int)
    for word in words:
        for category, category_words in liwc_categories.items():
            if any(category_word in word for category_word in category_words):
                category_counts[category] += 1
    
    total_words = len(words)
    if total_words == 0:
        return None
    
    category_percentages = {
        category: (count / total_words) * 100 
        for category, count in category_counts.items()
    }
    
    return category_percentages

# Columns to analyze
columns_to_analyze = [
    'What did you like about the chatbot support?',
    'What improvements do you suggest for chatbot support?',
    'What did you like about the human support?',
    'What improvements do you suggest for human support?'
]

results = {}
for column in columns_to_analyze:
    column_results = df[column].apply(analyze_text).dropna()
    if not column_results.empty:
        results[column] = column_results.apply(pd.Series).mean()
    else:
        print(f"No valid data for analysis in column: {column}")

for column, analysis in results.items():
    print(f"\nLIWC-recreated Analysis for: {column}")
    for category, percentage in sorted(analysis.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {percentage:.2f}%")

def calculate_sentiment(x):
    if x is None:
        return None
    return x.get('positive_emotions', 0) - x.get('negative_emotions', 0)

df['chatbot_sentiment'] = df['What did you like about the chatbot support?'].apply(analyze_text).apply(calculate_sentiment)
df['human_sentiment'] = df['What did you like about the human support?'].apply(analyze_text).apply(calculate_sentiment)

print("\nOverall Sentiment Analysis:")
print(f"Average Chatbot Sentiment: {df['chatbot_sentiment'].mean():.2f}")
print(f"Average Human Support Sentiment: {df['human_sentiment'].mean():.2f}")

lexicon = empath.Empath()

def analyze_column(column_name):
    all_text = ' '.join(df[column_name].dropna().astype(str))
    analysis = lexicon.analyze(all_text, normalize=True)
    sorted_categories = sorted(analysis.items(), key=lambda x: x[1], reverse=True)
    top_categories = dict(sorted_categories[:10])
    return top_categories

columns_to_analyze = [
    'What did you like about the chatbot support?',
    'What improvements do you suggest for chatbot support?',
    'What did you like about the human support?',
    'What improvements do you suggest for human support?'
]

results = {}
for column in columns_to_analyze:
    results[column] = analyze_column(column)

for column, analysis in results.items():
    print(f"\nEmpath Analysis for: {column}")
    for category, score in analysis.items():
        print(f"{category}: {score:.4f}")
