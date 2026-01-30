import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
tensorflow==2.15
numpy
pandas
scikit-learn
streamlit


# Load files
books = pd.read_csv("books.csv")
ratings = pd.read_csv("ratings.csv")
model = load_model("ann_model.h5")

# Encode IDs (same logic as training)
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()

ratings['user_enc'] = user_encoder.fit_transform(ratings['user_id'])
ratings['book_enc'] = book_encoder.fit_transform(ratings['book_id'])

books['book_enc'] = book_encoder.transform(books['book_id'])

st.title("📚 ANN Book Recommendation Website")

# User selection
user_id = st.selectbox("Select User", ratings['user_id'].unique())

if st.button("Recommend"):
    u_enc = user_encoder.transform([user_id])[0]

    user_input = np.full(len(books), u_enc)
    book_input = books['book_enc'].values

    preds = model.predict([user_input, book_input], batch_size=1024, verbose=0)

    books['score'] = preds.flatten()
    top = books.sort_values('score', ascending=False).head(5)

    st.subheader("Recommended Books")
    for _, row in top.iterrows():
        st.write("👉", row['title'])
