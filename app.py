import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Page Title
st.title(" E-Commerce Customer Sentiment Analyzer")
st.write("Analyze customer reviews instantly using Machine Learning & NLP.")

# 2. Load Data and Train a Simple Model (MAKE SURE IT RETURNS df)
@st.cache_data
def load_model():
    df = pd.read_csv("reviews.csv")
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(df['review'], df['sentiment'])
    return model, df  # <-- Return BOTH model and df here!

# Unpack both returned values
model, df = load_model()

# 3. Show Sample Data in the App (This will now work!)
with st.expander("View Training Dataset"):
    st.dataframe(df)

# 4. User Input Section
st.subheader("Test a New Product Review")
user_input = st.text_area("Enter customer feedback or review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter review to analyze.")
    else:
        # Predict sentiment
        prediction = model.predict([user_input])[0]

        # Display Results
        if prediction == 1:
            st.success(" Positive Sentiment: The customer is satisfied!")
        else:
            st.error(" Negative Sentiment: This indicates a customer pain point or issue.")