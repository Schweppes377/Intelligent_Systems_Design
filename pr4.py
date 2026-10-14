import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score



df = pd.read_csv("youtube_transactions.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df["items"] = df["items"].apply(lambda x: x.split(";"))

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["items"])


plt.figure()
plt.bar(range(10), X.sum(axis=0)[:10])
plt.title("Top features frequency")
plt.show()


pca_vis = PCA(n_components=2)
X_2d = pca_vis.fit_transform(X)

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.title("PCA visualization of dataset")
plt.show()



np.random.seed(42)

y = (
    X[:, 0] * 0.4 +
    X[:, 1] * 0.3 +
    X[:, 2] * 0.2 +
    np.random.normal(0, 0.1, X.shape[0])
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def evaluate(name, model):
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    end = time.time()

    print(f"\n {name} ")
    print("MAE:", round(mean_absolute_error(y_test, pred), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 4))
    print("R2:", round(r2_score(y_test, pred), 4))
    print("Time:", round(end - start, 4), "sec")

    return pred

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Gradient Boosting": GradientBoostingRegressor()
}

results = {}

for name, model in models.items():
    results[name] = evaluate(name, model)



plt.figure()
plt.plot(y_test, label="True")
plt.plot(results["Linear Regression"], label="Linear")
plt.plot(results["Gradient Boosting"], label="Boosting")
plt.legend()
plt.title("Regression comparison")
plt.show()



y_class = (y > np.median(y)).astype(int)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_c, y_train_c)
pred_c = clf.predict(X_test_c)

print("\nBONUS: Logistic Regression Classification")
print("Accuracy:", accuracy_score(y_test_c, pred_c))

pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X)

print("\nExplained variance ratio:", pca.explained_variance_ratio_)
