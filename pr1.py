import pandas as pd
from itertools import combinations


df = pd.read_csv("youtube_transactions.csv")
transactions = df['items'].apply(lambda x: set(x.split(';'))).tolist()
num_transactions = len(transactions)


all_items = set()
for t in transactions:
    all_items.update(t)
all_items = list(all_items)


min_support = 0.1
frequent_itemsets = {}


for item in all_items:
    count = sum(1 for t in transactions if item in t)
    support = count / num_transactions
    if support >= min_support:
        frequent_itemsets[frozenset([item])] = support


for item1, item2 in combinations(all_items, 2):
    count = sum(1 for t in transactions if item1 in t and item2 in t)
    support = count / num_transactions
    if support >= min_support:
        frequent_itemsets[frozenset([item1, item2])] = support


rules = []
for itemset in frequent_itemsets:
    if len(itemset) < 2:
        continue
    for i in range(1, len(itemset)):
        for antecedent in combinations(itemset, i):
            antecedent = frozenset(antecedent)
            consequent = itemset - antecedent
            support_itemset = frequent_itemsets[itemset]
            support_antecedent = frequent_itemsets.get(antecedent,
                                                       sum(1 for t in transactions if
                                                           antecedent.issubset(t)) / num_transactions)
            support_consequent = frequent_itemsets.get(consequent,
                                                       sum(1 for t in transactions if
                                                           consequent.issubset(t)) / num_transactions)

            confidence = support_itemset / support_antecedent if support_antecedent > 0 else 0
            conviction = (1 - support_consequent) / (1 - confidence) if confidence != 1 else float('inf')
            lift = confidence / support_consequent if support_consequent > 0 else 0
            leverage = support_itemset - support_antecedent * support_consequent

            rules.append({
                'antecedent': set(antecedent),
                'consequent': set(consequent),
                'support': support_itemset,
                'confidence': confidence,
                'conviction': conviction,
                'lift': lift,
                'leverage': leverage
            })


for r in rules:
    ant = ', '.join(r['antecedent'])
    cons = ', '.join(r['consequent'])
    print(f"Правило: {ant} -> {cons}")
    print(f"Support: {r['support']:.3f}, Confidence: {r['confidence']:.3f}, Lift: {r['lift']:.3f}, Leverage: {r['leverage']:.3f}, Conviction: {r['conviction']:.3f}")
    print("-" * 60)