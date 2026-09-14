"""
app.py — English → Urdu Translator (Streamlit)
================================================
A simple web interface for the seq2seq translation model from Week 11 Day 5.
It loads the trained model FROM DISK (the `translator/` folder) and translates
English sentences to Urdu.

--- HOW TO RUN ---
1. First, save your trained model by running the SAVE cell (see save_model_snippet.py
   or the save cell added to the Day 5 notebook). This creates a `translator/` folder:
       translator/encoder.keras
       translator/decoder_step.keras
       translator/config.pkl
2. Put this app.py next to that `translator/` folder.
3. Install the requirements:
       pip install streamlit tensorflow
4. Run:
       streamlit run app.py
5. It opens in your browser at http://localhost:8501
"""

import os
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

MODEL_DIR = "translator"

# ----------------------------------------------------------------------
# Load the model + tokenizers once, and cache so it doesn't reload every click
# ----------------------------------------------------------------------
@st.cache_resource
def load_translator():
    encoder = load_model(os.path.join(MODEL_DIR, "encoder.keras"))
    decoder_step = load_model(os.path.join(MODEL_DIR, "decoder_step.keras"))
    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)
    return encoder, decoder_step, cfg


# ----------------------------------------------------------------------
# Translate one English sentence to Urdu (same logic as the notebook)
# ----------------------------------------------------------------------
def translate(sentence, encoder, decoder_step, cfg):
    en_tok = cfg["en_tok"]
    ur_tok = cfg["ur_tok"]
    MAX_EN = cfg["MAX_EN"]
    MAX_UR = cfg["MAX_UR"]
    idx_to_word = {i: w for w, i in ur_tok.word_index.items()}

    # 1. encode the English sentence into a context state
    seq = pad_sequences(en_tok.texts_to_sequences([sentence.lower()]),
                        maxlen=MAX_EN, padding="post")
    state = encoder.predict(seq, verbose=0)

    # 2. generate Urdu one word at a time, starting from <sos>
    word = np.array([[ur_tok.word_index["<sos>"]]])
    output = []
    for _ in range(MAX_UR):
        probs, state = decoder_step.predict([word, state], verbose=0)
        idx = int(probs[0, -1].argmax())
        w = idx_to_word.get(idx, "")
        if w == "<eos>" or w == "":
            break
        output.append(w)
        word = np.array([[idx]])       # feed the prediction back in
    return " ".join(output)


# ----------------------------------------------------------------------
# The interface
# ----------------------------------------------------------------------
st.set_page_config(page_title="English → Urdu Translator", page_icon="🌐")

st.title("🌐 English → Urdu Translator")
st.caption("A sequence-to-sequence (encoder–decoder GRU) model, trained in Week 11.")

# check the model exists before doing anything
if not os.path.isdir(MODEL_DIR):
    st.error(
        f"Could not find the `{MODEL_DIR}/` folder. "
        "Run the SAVE cell in the Day 5 notebook first, then put this app next to the folder."
    )
    st.stop()

encoder, decoder_step, cfg = load_translator()

# input
english = st.text_input("Enter an English sentence:", value="i am happy")

col1, col2 = st.columns([1, 3])
with col1:
    go = st.button("Translate", type="primary")

# a few example buttons
st.write("Or try an example:")
examples = ["how are you", "what is your name", "i am a student", "i love my country"]
ex_cols = st.columns(len(examples))
for c, ex in zip(ex_cols, examples):
    if c.button(ex):
        english = ex
        go = True

# translate
if go and english.strip():
    with st.spinner("Translating..."):
        urdu = translate(english, encoder, decoder_step, cfg)
    st.markdown("**English:**  " + english)
    # show Urdu large and right-aligned (Urdu is a right-to-left script)
    st.markdown(
        f"<div style='font-size:28px; text-align:right; direction:rtl; "
        f"margin-top:10px;'>{urdu}</div>",
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Note: this is a small classroom model trained on a limited dataset — "
    "translations are approximate, not production quality."
)
