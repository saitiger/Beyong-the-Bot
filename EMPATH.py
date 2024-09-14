import pandas as pd
import empath
from collections import defaultdict

lexicon = empath.Empath()

# Define custom categories
custom_categories = {
    "automation_sentiment": ["bot", "chatbot", "automated", "quick", "fast", "efficient", "24/7", "instant"],
    "human_touch": ["empathy", "understanding", "personalized", "human", "representative", "patient", "listen"],
    "problem_resolution": ["solve", "resolution", "fixed", "answered", "clarified", "resolved", "solution"],
    "ease_of_use": ["easy", "simple", "straightforward", "intuitive", "user-friendly", "complicated", "difficult"],
    "wait_time_perception": ["wait", "queue", "immediate", "prompt", "delay", "quick", "slow"],
    "technical_comprehension": ["understand", "comprehend", "grasp", "confusion", "clear", "explain", "technical"],
    "flexibility_in_communication": ["adapt", "flexible", "natural", "variety", "options", "customize"],
    "support_availability": ["available", "24/7", "anytime", "accessible", "convenient"],
    "escalation_needs": ["transfer", "escalate", "supervisor", "additional help", "human support"],
    "customer_frustration": ["frustrated", "annoyed", "angry", "dissatisfied", "upset", "disappointed"],
    "support_channel_preference": ["chat", "voice", "email", "phone", "text", "in-person"],
    "issue_complexity": ["simple", "complex", "complicated", "straightforward", "advanced", "basic"]
}

# Add custom categories to Empath
for category, terms in custom_categories.items():
    lexicon.create_category(category, terms)

def analyze_column(column_name):
    all_text = ' '.join(df[column_name].dropna().astype(str))
    
    analysis = lexicon.analyze(all_text, categories=list(custom_categories.keys()) + lexicon.cats, normalize=True)
    
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

# Print results
for column, analysis in results.items():
    print(f"\nEmpath Analysis for: {column}")
    for category, score in analysis.items():
        print(f"{category}: {score:.4f}")
