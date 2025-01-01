import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Add this line
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


# Preload and preprocess the dataset (as in your code)
df = pd.read_csv("hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

df1 = df.drop(columns=['flags', 'category', 'response'], inplace=False)
df1[['instruction', 'intent']] = df1[['instruction', 'intent']].apply(lambda x: x.str.lower())
df2 = df1.drop_duplicates()
df2['label'] = pd.factorize(df2['intent'])[0]
categories = df2['intent'].unique()

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df2[['instruction']], df2['label'])
df2_resampled = pd.DataFrame({'instruction': X_resampled['instruction'], 'label': y_resampled})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df2_resampled['instruction'], df2_resampled['label'], test_size=0.2, random_state=42
)

# Vectorize text
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train models
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_tfidf, y_train)

rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Streamlit app
st.title("Instruction Intent Classification")
st.write("Enter an instruction to classify its intent using different models.")

# Input box
user_input = st.text_input("Enter your instruction:")

if user_input:
    instruction_tfidf = tfidf.transform([user_input])
    
    # Naive Bayes Prediction
    nb_pred = nb_model.predict(instruction_tfidf)
    nb_result = categories[nb_pred[0]]
    st.write(f"**Naive Bayes Prediction:** {nb_result}")
    
    # SVM Prediction
    svm_pred = svm_model.predict(instruction_tfidf)
    svm_result = categories[svm_pred[0]]
    st.write(f"**Support Vector Machine Prediction:** {svm_result}")
    
    # Random Forest Prediction
    rf_pred = rf_model.predict(instruction_tfidf)
    rf_result = categories[rf_pred[0]]
    st.write(f"**Random Forest Prediction:** {rf_result}")
