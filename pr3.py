import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, hamming_loss, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv("youtube_transactions.csv")


df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df["items"] = df["items"].apply(lambda x: x.split(";"))

mlb_X = MultiLabelBinarizer()
X = mlb_X.fit_transform(df["items"])



def create_labels(items):
    labels = []

    if "Machine Learning" in items or "Deep Learning" in items or "AI" in items:
        if random.random() < 0.85:
            labels.append("AI_User")
        else:
            labels.append("Other")


    if "Gaming" in items or "CS2" in items or "Minecraft" in items:
        if random.random() < 0.8:
            labels.append("Gamer")
        else:
            labels.append("Other")


    if "Cooking" in items or "Food" in items or "Baking" in items:
        if random.random() < 0.85:
            labels.append("Food_Lover")
        else:
            labels.append("Other")


    if "Workout" in items or "Fitness" in items or "Gym" in items:
        if random.random() < 0.85:
            labels.append("Fitness_User")
        else:
            labels.append("Other")


    if len(labels) == 0:
        labels.append("Other")

    return labels

mlb_y = MultiLabelBinarizer()
y = mlb_y.fit_transform(df["items"].apply(create_labels))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model_lr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)

print("Logistic Regression")
print(classification_report(y_test, y_pred))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("F1 Score (micro):", f1_score(y_test, y_pred, average="micro"))

rf = RandomForestClassifier(n_estimators=100, random_state=42)

ensemble = OneVsRestClassifier(
    VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("rf", rf)
        ],
        voting="soft"
    )
)

ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)

print("\nEnsemble")
print(classification_report(y_test, y_pred_ens))
print("Hamming Loss:", hamming_loss(y_test, y_pred_ens))
print("F1 Score (micro):", f1_score(y_test, y_pred_ens, average="micro"))

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

df["cluster"] = clusters

print("\nCluster distribution:")
print(df["cluster"].value_counts())

sil_score = silhouette_score(X, clusters)
print("Silhouette Score:", sil_score)


print("\nComparison: clusters vs AI_User class")


ai_index = list(mlb_y.classes_).index("AI_User")
ai_labels = y[:, ai_index]

comparison = pd.crosstab(ai_labels, clusters)
print(comparison)

print("\nCluster vs other labels distribution:")
print(pd.DataFrame({
    "cluster": clusters,
    "AI_User": ai_labels
}).groupby("cluster").mean())
