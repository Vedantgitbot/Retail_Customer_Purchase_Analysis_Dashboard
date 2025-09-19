import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats import weightstats as stests

df = pd.read_csv("cleaned_transactions.csv")

groups = [df.loc[df['Age Group'] == g, 'Total Amount'] for g in df['Age Group'].dropna().unique()]
anova_result = stats.f_oneway(*groups)
print("ANOVA F-statistic:", anova_result.statistic)
print("ANOVA p-value:", anova_result.pvalue)

model = ols('Q("Total Amount") ~ C(Q("Age Group"))', data=df).fit()
tukey = sm.stats.multicomp.pairwise_tukeyhsd(df['Total Amount'], df['Age Group'])
print(tukey)

t_stat, t_pval = stats.ttest_1samp(df['Total Amount'], 50)
print("One-sample T-test statistic:", t_stat)
print("One-sample T-test p-value:", t_pval)

male = df[df['Gender'] == "Male"]["Total Amount"]
female = df[df['Gender'] == "Female"]["Total Amount"]
t2_stat, t2_pval = stats.ttest_ind(male, female, equal_var=False)
print("Two-sample T-test statistic (Male vs Female):", t2_stat)
print("Two-sample T-test p-value:", t2_pval)

ztest, zpval = stests.ztest(male, female)
print("Z-test statistic (Male vs Female):", ztest)
print("Z-test p-value:", zpval)
print("✅ Statistical analysis completed.")
