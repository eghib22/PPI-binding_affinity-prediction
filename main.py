import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# -----------------------------
# Amino acid list
# -----------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# -----------------------------
# Feature extraction functions
# -----------------------------
def aa_composition(seq):
    seq = seq.upper()
    length = len(seq)
    return [seq.count(aa) / length if length > 0 else 0 for aa in AMINO_ACIDS]

def extract_features(df):
    features = []
    for _, row in df.iterrows():
        f = []
        f.append(len(row["seqA"]))
        f.append(len(row["seqB"]))
        f.extend(aa_composition(row["seqA"]))
        f.extend(aa_composition(row["seqB"]))
        features.append(f)
    return np.array(features)

# -----------------------------
# Load data
# -----------------------------

# Read the PPB-Affinity Excel file
data = pd.read_excel("data/PPB-Affinity.xlsx")

# Keep only the columns we need
data = data[["Ligand Name", "Receptor Name", "KD(M)"]]

# Rename them to match the rest of your code
data = data.rename(columns={
    "Ligand Name": "seqA",
    "Receptor Name": "seqB",
    "KD(M)": "affinity"
})

# Drop rows with missing values
data = data.dropna(subset=["seqA", "seqB", "affinity"])

# Ensure protein names are strings (some may be parsed as numbers)
data["seqA"] = data["seqA"].astype(str)
data["seqB"] = data["seqB"].astype(str)

# Convert KD to -log10 scale (pKd) for better regression
# Lower KD = higher affinity, so -log10 gives higher values for stronger binders
data["affinity"] = -np.log10(data["affinity"])

# Quick check
print("Columns:", data.columns)
print("Number of rows:", len(data))
print(data.head())


X = extract_features(data)
y = data["affinity"].values

# -----------------------------
# Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Baseline model
# -----------------------------
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5
baseline_corr, _ = pearsonr(y_test, baseline_pred)

# -----------------------------
# Linear Regression model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)
ml_pred = model.predict(X_test)

ml_rmse = mean_squared_error(y_test, ml_pred) ** 0.5
ml_corr, _ = pearsonr(y_test, ml_pred)

# -----------------------------
# Save results
# -----------------------------
results = pd.DataFrame({
    "Model": ["Baseline", "Linear Regression"],
    "RMSE": [baseline_rmse, ml_rmse],
    "Pearson Correlation": [baseline_corr, ml_corr]
})

results.to_csv("results/results.csv", index=False)

# -----------------------------
# Plot predictions
# -----------------------------
plt.figure()
plt.scatter(y_test, ml_pred)
plt.xlabel("True pKd (-log10 KD)")
plt.ylabel("Predicted pKd (-log10 KD)")
plt.title("PPB-Affinity: Linear Regression Predictions")
plt.savefig("results/prediction_plot.png")

print("=== RESULTS ===")
print(results)
print("\nSaved:")
print("- results/results.csv")
print("- results/prediction_plot.png")
