import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "titanic_dataset_700_records.csv"))


# print(df.head())
# print(df.isnull().sum())


df["passenger_class"] = df["passenger_class"].fillna(df["passenger_class"].mode()[0])
df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
df["age"] = df["age"].fillna(df["age"].mean())
df["siblings_spouses"] = df["siblings_spouses"].fillna(df["siblings_spouses"].median())
df["parents_children"] = df["parents_children"].fillna(df["parents_children"].median())
df["ticket_fare"] = df["ticket_fare"].fillna(df["ticket_fare"].mean())
df["embarkation_port"] = df["embarkation_port"].fillna(df["embarkation_port"].mode()[0])


x = df.drop("survived", axis=1)
y = df["survived"]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("=" * 50)


# ---------------- Encoding ----------------
gender_encoder = LabelEncoder()
port_encoder = LabelEncoder()

x_train["gender"] = gender_encoder.fit_transform(x_train["gender"])
x_test["gender"] = gender_encoder.transform(x_test["gender"])

x_train["embarkation_port"] = port_encoder.fit_transform(x_train["embarkation_port"])
x_test["embarkation_port"] = port_encoder.transform(x_test["embarkation_port"])


# ---------------- Logistic Regression ----------------
# logistic = LogisticRegression()

# logistic.fit(x_train, y_train)
# logistic_prediction = logistic.predict(x_test)

# print("Logistic_Accuracy :", accuracy_score(y_test, logistic_prediction))

# print("=" * 50)


# # ---------------- Decision Tree ----------------
# decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)

# decision_tree.fit(x_train, y_train)
# decisiontree_prediction = decision_tree.predict(x_test)

# print("Decision Tree Accuracy :", accuracy_score(y_test, decisiontree_prediction))

# print("=" * 50)


# ---------------- Random Forest ----------------
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest.fit(x_train, y_train)
randomforest_prediction = random_forest.predict(x_test)

print("Random Forest Accuracy :", accuracy_score(y_test, randomforest_prediction))

print("=" * 50)

# ---------------- Save Model & Objects ----------------
joblib.dump(random_forest, os.path.join(MODEL_DIR, "titanic_model.pkl"))
joblib.dump(gender_encoder, os.path.join(MODEL_DIR, "gender_encoder.pkl"))
joblib.dump(port_encoder, os.path.join(MODEL_DIR, "port_encoder.pkl"))


print("Model and Encoders Saved Successfully")