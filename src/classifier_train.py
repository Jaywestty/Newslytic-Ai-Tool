# %%
#import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  BernoulliNB
from sklearn.pipeline import Pipeline
import os
import joblib
from datetime import datetime

# %%
#read the dataset
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # goes up from src/
DATA_PATH = os.path.join(BASE_DIR, "Datasets", "CrimeVsNoCrimeArticles.csv")

df = pd.read_csv(DATA_PATH)
# %%
#replace null values with a null string
df_cl = df.where((pd.notnull(df)), '')
df_cl.head()

# %%
#seperate into X and y which is or features and Target variables
X = df_cl['title']
y = df_cl['is_crime_report']

#split into Test and Train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# %%
#Feature extraction
vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=2,
    max_df=0.9,
    max_features=5000,
    ngram_range=(1, 2)
)

# %%
#build model
model = BernoulliNB()
# %%

#build pipeline
pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('Classifier', model)
    ])
pipeline.fit(X_train, y_train)
# %%
#Save the model
# Path to your triage folder
# Build path to triage models folder
MODEL_DIR = os.path.join(BASE_DIR, "Model")
#os.makedirs(MODEL_DIR, exist_ok=True)  # create folder if it doesn't exist
MODEL_PATH = "Model/classifier_model.pkl"

# Add timestamp to filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = os.path.join(MODEL_DIR, f"classifier_model_{timestamp}.pkl")

with open(MODEL_PATH, 'wb') as f:
    joblib.dump(pipeline, f)

print('Model saved succesfully')
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
