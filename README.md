# 🌾 AgriSense AI

**1M1B × IBM SkillsBuild AI for Sustainability Internship**
Built by Nishant Sharma | Arya College of Engineering, Jaipur | 2026

## Project Structure
agrisense-ai/
├── app.py                          # Streamlit web app
├── requirements.txt                # Dependencies
├── generate_dataset.py             # Dataset generation script
├── crop_disease_dataset.csv        # Full dataset (1200 records)
├── train.csv                       # Training set (960 records)
├── test.csv                        # Test set (240 records)
├── agrisense_model.pkl             # Trained Random Forest model
├── AgriSense_AI_Model_Training.ipynb  # Jupyter notebook (EDA + Training)
├── 01_dataset_overview.png         # Chart: dataset distribution
├── 02_feature_analysis.png         # Chart: correlation + boxplot
├── 03_district_risk.png            # Chart: district risk analysis
├── 04_model_evaluation.png         # Chart: confusion matrix + comparison
└── 05_feature_importance.png       # Chart: feature importance

## How to Run
pip install -r requirements.txt
streamlit run app.py

## SDG Alignment
- SDG 2: Zero Hunger
- SDG 15: Life on Land
