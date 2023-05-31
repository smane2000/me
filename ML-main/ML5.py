#!pip install numpy pandas mlxtend 
import os
import csv
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = []
with open(r'/root/Market_Basket_Optimisation.csv') as file: 
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        data.append(row)

print(len(data))

te = TransactionEncoder()
x = te.fit_transform(data)

print(x)
print(te.columns_)

df = pd.DataFrame(x, columns=te.columns_)

print(df)

freq_itemset = apriori(df, min_support=0.01, use_colnames=True)

print(freq_itemset)

rules = association_rules(freq_itemset, metric='confidence', min_threshold=0.10)
rules = rules[['antecedents', 'consequents', 'support', 'confidence']]

print(rules)

rules[rules['antecedents'] == {'cake'}]['consequents']

print(rules[rules['antecedents'] == {'cake'}]['consequents'])