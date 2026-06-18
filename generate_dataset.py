"""
AgriSense AI — Dataset Generator
Generates realistic synthetic dataset based on:
- Rajasthan crop growing conditions
- ICAR disease risk research papers
- Rajasthan Agriculture Department field reports
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 1200  # Total records

# ── Crop distribution (Rajasthan realistic proportions) ──────
crops = np.random.choice(
    ["Wheat", "Mustard", "Bajra", "Soybean", "Chickpea", "Maize"],
    size=N,
    p=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
)

# ── Season mapping ───────────────────────────────────────────
season_map = {
    "Wheat": "Rabi", "Mustard": "Rabi", "Chickpea": "Rabi",
    "Bajra": "Kharif", "Soybean": "Kharif", "Maize": "Kharif"
}
seasons = [season_map[c] for c in crops]

# ── District mapping ─────────────────────────────────────────
district_pool = {
    "Wheat":    ["Sriganganagar", "Hanumangarh", "Bikaner", "Churu"],
    "Mustard":  ["Bharatpur", "Alwar", "Dausa", "Tonk"],
    "Bajra":    ["Jodhpur", "Barmer", "Jaisalmer", "Nagaur"],
    "Soybean":  ["Kota", "Baran", "Jhalawar", "Bundi"],
    "Chickpea": ["Ajmer", "Pali", "Nagaur", "Sikar"],
    "Maize":    ["Udaipur", "Dungarpur", "Rajsamand", "Banswara"],
}
districts = [np.random.choice(district_pool[c]) for c in crops]

# ── Crop age by growth stage ─────────────────────────────────
def get_crop_age(crop):
    stages = {
        "Wheat":    [(0,20,"Seedling"),(21,50,"Vegetative"),(51,80,"Flowering"),(81,120,"Maturity")],
        "Mustard":  [(0,20,"Seedling"),(21,45,"Vegetative"),(46,75,"Flowering"),(76,110,"Maturity")],
        "Bajra":    [(0,15,"Seedling"),(16,40,"Vegetative"),(41,65,"Flowering"),(66,90,"Maturity")],
        "Soybean":  [(0,20,"Seedling"),(21,50,"Vegetative"),(51,80,"Flowering"),(81,110,"Maturity")],
        "Chickpea": [(0,20,"Seedling"),(21,50,"Vegetative"),(51,80,"Flowering"),(81,110,"Maturity")],
        "Maize":    [(0,20,"Seedling"),(21,45,"Vegetative"),(46,70,"Flowering"),(71,100,"Maturity")],
    }
    idx = np.random.choice(len(stages[crop]), p=[0.15, 0.30, 0.35, 0.20])
    stage = stages[crop][idx]
    return np.random.randint(stage[0], stage[1]+1), stage[2]

ages_stages = [get_crop_age(c) for c in crops]
crop_ages   = [x[0] for x in ages_stages]
growth_stages = [x[1] for x in ages_stages]

# ── Weather features (season-realistic) ─────────────────────
temperatures, humidities, rainfalls, wind_speeds = [], [], [], []
for i in range(N):
    s = seasons[i]
    if s == "Rabi":   # Nov–Apr: cooler, drier
        temperatures.append(round(np.random.uniform(8, 35), 1))
        humidities.append(round(np.random.uniform(30, 85), 1))
        rainfalls.append(round(np.random.uniform(0, 25), 1))
    else:             # Kharif: Jul–Oct: hot, humid, rainy
        temperatures.append(round(np.random.uniform(22, 42), 1))
        humidities.append(round(np.random.uniform(50, 98), 1))
        rainfalls.append(round(np.random.uniform(0, 50), 1))
    wind_speeds.append(round(np.random.uniform(0, 28), 1))

temperatures = np.array(temperatures)
humidities   = np.array(humidities)
rainfalls    = np.array(rainfalls)
wind_speeds  = np.array(wind_speeds)
soil_moistures = np.clip(
    rainfalls * 1.2 + humidities * 0.3 + np.random.normal(0, 5, N), 10, 95
).round(1)

# ── Previous disease incidence (field history) ───────────────
prev_disease = np.random.choice([0, 1], size=N, p=[0.65, 0.35])

# ── Risk label generation (domain-based rules) ───────────────
def compute_risk(i):
    score = 0
    h = humidities[i]; t = temperatures[i]
    r = rainfalls[i];  sm = soil_moistures[i]
    w = wind_speeds[i]; a = crop_ages[i]
    p = prev_disease[i]

    # Humidity
    if h > 85: score += 4
    elif h > 75: score += 3
    elif h > 65: score += 1

    # Temperature (disease-favourable ranges)
    if 10 <= t <= 28: score += 3
    elif 28 < t <= 35: score += 1
    elif t > 38: score -= 2

    # Rainfall
    if r > 30: score += 3
    elif r > 15: score += 2
    elif r > 5:  score += 1

    # Soil moisture
    if sm > 75: score += 2
    elif sm > 55: score += 1

    # Wind (low wind = poor air circulation = more disease)
    if w < 5: score += 1

    # Crop age (flowering stage most vulnerable)
    if 45 <= a <= 85: score += 2
    elif 20 <= a < 45: score += 1

    # Previous disease history
    if p == 1: score += 2

    # Risk thresholds
    if score >= 12: return "High"
    elif score >= 7: return "Medium"
    else: return "Low"

risk_labels = [compute_risk(i) for i in range(N)]

# ── Disease name (based on crop + risk) ──────────────────────
disease_map = {
    "Wheat":    {"High": "Yellow Rust",       "Medium": "Powdery Mildew", "Low": "None"},
    "Mustard":  {"High": "Alternaria Blight", "Medium": "White Rust",     "Low": "None"},
    "Bajra":    {"High": "Downy Mildew",      "Medium": "Ergot",          "Low": "None"},
    "Soybean":  {"High": "Rust",              "Medium": "Bacterial Pustule","Low": "None"},
    "Chickpea": {"High": "Ascochyta Blight",  "Medium": "Wilt",           "Low": "None"},
    "Maize":    {"High": "Turcicum Blight",   "Medium": "Common Rust",    "Low": "None"},
}
likely_diseases = [disease_map[crops[i]][risk_labels[i]] for i in range(N)]

# ── Assemble DataFrame ───────────────────────────────────────
df = pd.DataFrame({
    "record_id":            [f"AGR{str(i+1).zfill(4)}" for i in range(N)],
    "district":             districts,
    "crop":                 crops,
    "season":               seasons,
    "growth_stage":         growth_stages,
    "crop_age_days":        crop_ages,
    "temperature_C":        temperatures,
    "humidity_pct":         humidities,
    "rainfall_mm":          rainfalls,
    "soil_moisture_pct":    soil_moistures,
    "wind_speed_kmh":       wind_speeds,
    "prev_disease_history": prev_disease,
    "likely_disease":       likely_diseases,
    "risk_level":           risk_labels,
})

# ── Save full dataset ────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/crop_disease_dataset.csv", index=False)

# ── Train/Test split (80/20) ─────────────────────────────────
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["risk_level"])
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv",  index=False)

print(f"✅ Dataset generated: {len(df)} records")
print(f"   Train: {len(train_df)} | Test: {len(test_df)}")
print(f"\nRisk Distribution:")
print(df["risk_level"].value_counts())
print(f"\nCrop Distribution:")
print(df["crop"].value_counts())
print(f"\nSample rows:")
print(df.head(3).to_string())
