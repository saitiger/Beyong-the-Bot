import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import AnovaRM

df = pd.read_csv('User Survey Google Form Data.csv')

# 1. Repeated Measures ANOVA

df_long = pd.melt(df[['Satisfaction_Chatbot', 'Satisfaction_Human']], 
                  var_name='Support_Type', value_name='Satisfaction')
df_long['Subject'] = np.repeat(range(len(df)), 2)

aov_rm = AnovaRM(df_long, 'Satisfaction', 'Subject', within=['Support_Type']).fit()
print("Repeated Measures ANOVA Results:")
print(aov_rm.summary())

# 2. Paired Sample t-test
t_stat, p_value = stats.ttest_rel(df['Satisfaction_Chatbot'], df['Satisfaction_Human'])
print("\nPaired Sample t-test Results:")
print(f"t-statistic: {t_stat}, p-value: {p_value}")

# 3. Correlation Tests
pearson_r, pearson_p = stats.pearsonr(df['Chatbot_Accuracy'], df['Satisfaction_Chatbot'])
print("\nPearson Correlation (Chatbot Accuracy vs Satisfaction):")
print(f"r: {pearson_r}, p-value: {pearson_p}")

# Spearman correlation between frequency of use and satisfaction
spearman_rho, spearman_p = stats.spearmanr(df['Frequency_of_Use'], df['Satisfaction_Overall'])
print("\nSpearman Correlation (Frequency of Use vs Overall Satisfaction):")
print(f"rho: {spearman_rho}, p-value: {spearman_p}")

# 4. Order Effect Analysis
t_stat_order, p_value_order = stats.ttest_ind(
    df[df['Order'] == 0]['Satisfaction_Overall'],
    df[df['Order'] == 1]['Satisfaction_Overall']
)
print("\nIndependent t-test for Order Effect:")
print(f"t-statistic: {t_stat_order}, p-value: {p_value_order}")

# 5. Descriptive Statistics
print("\nDescriptive Statistics:")
print(df[['Satisfaction_Chatbot', 'Satisfaction_Human']].describe())
