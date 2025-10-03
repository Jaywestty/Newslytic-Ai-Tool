#%%
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # goes up from src/
MODEL_PATH = os.path.join(BASE_DIR, "Model/classifier_model_20251003_154212.pkl")
# Load pipeline
pipeline = joblib.load(MODEL_PATH)

print("✅ Headline Classifier model loaded successfully")
# %%
sample_text = [
    "Man dies of gunshot inside Akwa Ibom church, police begin probe",
    "Another win for Afrobeats, fans celebrate Davido’s Maryland show",
    "Three Ebonyi Police Officers Face Interrogation Over Alleged Baby Sale For N25Million",
    "Six suspected cultists arrested in Anambra black spot raid"
]

# %%
# Make predictions
predictions = pipeline.predict(sample_text)
# %%
# Print results
for text, pred in zip(sample_text, predictions):
    label = "Crime" if pred == 1 else "Non-Crime"
    print(f"\"{text}\" → Prediction: {label}")

# %%
