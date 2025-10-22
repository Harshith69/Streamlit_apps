import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM

# Page configuration
st.set_page_config(page_title="Next Word Prediction")

# Display loading status
st.write("Loading model and tokenizer...")

try:
    # Get absolute path of the current file
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Define model and tokenizer paths
    model_path = os.path.join(base_path, 'next_word_lstm_model_with_early_stopping.h5')
    tokenizer_path = os.path.join(base_path, 'tokenizer.pickle')

    # Load model with fallback for time_major error
    try:
        model = load_model(model_path, compile=False)
    except TypeError as e:
        if "time_major" in str(e):
            # Define a compatible LSTM class that ignores 'time_major'
            class CompatibleLSTM(LSTM):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('time_major', None)
                    super().__init__(*args, **kwargs)
            model = load_model(model_path, compile=False, custom_objects={'LSTM': CompatibleLSTM})
        else:
            raise e

    st.success("✅ Model loaded successfully!")

    # Load tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    st.success("✅ Tokenizer loaded successfully!")

except Exception as e:
    st.error(f"❌ Loading failed: {str(e)}")
    st.info("""
    **Troubleshooting tips:**
    1. Ensure `.h5` and `.pickle` files are in the same folder as app.py
    2. Check if the working directory is correct when running Streamlit
    3. The model might be built with a different TensorFlow version
    4. Re-train or re-save the model with your current TensorFlow version if needed
    """)
    st.stop()

# --------------------------------------------------------
# Function to predict the next word
# --------------------------------------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len - 1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return None
    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------------------------------------
# Streamlit app UI
# --------------------------------------------------------
st.title("🔮 Next Word Prediction with LSTM")
st.write("Enter a sequence of words and predict what comes next!")

input_text = st.text_input("Enter text:", "To be or not to")

# Determine sequence length dynamically from model
max_sequence_len = model.input_shape[1] + 1

if st.button("Predict Next Word"):
    if input_text.strip():
        with st.spinner('Predicting...'):
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word and not next_word.startswith("Error"):
            st.success(f"**Next word:** {next_word}")
        else:
            st.error(f"Prediction failed: {next_word}")
    else:
        st.warning("Please enter some text first!")
