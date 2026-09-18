import streamlit as st
import requests

# FastAPI URL
API_URL = "https://cavalry-rasping-unvalued.ngrok-free.dev/generate"

st.title("🤖 Text Generation App")

# Prompt input
prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Artificial intelligence is..."
)

# Max tokens input
max_tokens = st.number_input(
    "Maximum tokens:",
    min_value=1,
    max_value=500,
    value=50,
    step=1
)

# Generate button
if st.button("Generate Text"):

    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:

        data = {
            "prompt": prompt,
            "max_new_tokens": max_tokens
        }

        try:
            response = requests.post(
                API_URL,
                json=data,
                timeout=120
            )

            if response.status_code == 200:

                result = response.json()

                st.subheader("Generated Text")

                st.write(result["generated_text"])

            else:
                st.error(
                    f"API Error: {response.status_code}"
                )
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")