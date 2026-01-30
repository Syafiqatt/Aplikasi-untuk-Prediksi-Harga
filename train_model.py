import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("retail_store_inventory.csv")

# =========================
# RENAME COLUMNS
# =========================
df.columns = df.columns.str.lower().str.replace(" ", "_")

# =========================
# CLEAN DATA
# =========================
df = df.dropna()
df = df[df["price"] > 0]
# Tambahan: Hapus data demand yang aneh (negatif)
df = df[df["demand_forecast"] >= 0]

# =========================
# FEATURE ENGINEERING
# =========================
df["demand"] = df["demand_forecast"]
df["stock"] = df["inventory_level"]
df["competition"] = df["competitor_pricing"] # Ini adalah HARGA pesaing

# =========================
# FINAL DATAFRAME
# =========================
x = df[["demand", "stock", "competition"]]
y = df["price"]

# =========================
# TRAIN MODEL
# =========================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Menggunakan parameter yang lebih kuat
model = RandomForestRegressor(
    n_estimators=200,   # Ditambah agar lebih presisi
    max_depth=15,       
    random_state=42
)

model.fit(x_train, y_train)

# Evaluasi
predictions = model.predict(x_test)

r2 = r2_score(y_test, predictions)
print("R-squared (R²) Model:", round(r2, 4))
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae:.2f}") 
# MAE rendah (sekitar 2.4) berarti prediksi rata-rata meleset cuma $2.4

# =========================
# SAVE MODEL
# =========================
with open("pricing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as pricing_model.pkl")