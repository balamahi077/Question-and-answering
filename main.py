import streamlit as st
import openai
import requests
from io import BytesIO
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
import re

# --- Page Config ---
st.set_page_config(page_title=" Medicine Assistant", layout="centered")
st.title(" Medicine Strip Assistant")
st.caption("Image → Table → Translation → Audio for Elderly Care")

# --- Inputs ---
openai_api = st.text_input("🔐 Enter your OpenAI API Key", type="password")
input_type = st.radio("📷 How would you like to provide the image?", ["Upload Image", "Paste Image URL"])
user_lang = st.text_input("🌐 Output Language (e.g., en, hi, kn)", value="en")

image_bytes = None

# --- Image Loader ---
def load_image():
    if input_type == "Upload Image":
        uploaded = st.file_uploader("Upload a medicine strip image", type=["jpg", "jpeg", "png"])
        return uploaded.read() if uploaded else None
    else:
        url = st.text_input("Paste image URL here:")
        if url:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    st.image(BytesIO(response.content), caption="Image from URL")
                    return response.content
            except:
                st.error("Failed to fetch the image from the provided URL.")
    return None

# --- Prepare Base64 Image for GPT-4 Vision ---
def encode_image(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

# --- GPT-4 Turbo Vision Call ---
def analyze_medicine_strip(image_data_url):
    openai.api_key = openai_api
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this medicine strip image. Step 1: Extract name, expiry, usage, dosage, food instructions, warnings in a markdown table. Step 2: Write a short, polite bullet-point summary for an elderly person, like a nurse would say. Do NOT include the table in the explanation."},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]}
            ],
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ GPT-4 Vision error: {e}")
        return ""

# --- Clean Text for Audio ---
def clean_audio_text(text):
    text = re.sub(r"(Item\s+Information|Attribute\s+Value|Table\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r'[-•*‣●▪◦]+', '', text)
    text = re.sub(r'[^\w\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return f"Hi there! {text}"

# --- Text-to-Speech ---
def generate_audio(text, lang):
    try:
        cleaned_text = clean_audio_text(text)
        tts = gTTS(text=cleaned_text, lang=lang)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"❌ Error generating audio: {e}")
        return None

# --- Main Pipeline ---
image_bytes = load_image()
if image_bytes and openai_api:
    st.markdown("---")
    st.image(BytesIO(image_bytes), caption="Selected Image", use_column_width=True)

    if st.button("🤖 Analyze Medicine Strip"):
        with st.spinner("Analyzing image using GPT-4 Turbo..."):
            img_data_url = encode_image(image_bytes)
            result = analyze_medicine_strip(img_data_url)

            if not result:
                st.stop()

            if "\n\n" in result:
                table_md, summary = result.split("\n\n", 1)
            else:
                table_md, summary = result, ""

            # Translation
            try:
                translated_summary = GoogleTranslator(source="auto", target=user_lang).translate(summary)
            except Exception as e:
                st.error(f"❌ Translation error: {e}")
                translated_summary = summary

            # Display Results
            st.markdown("### 📋 Extracted Medicine Info")
            st.markdown(table_md, unsafe_allow_html=True)

            st.markdown(f"### 🧑‍⚕️ Advice for Elderly ({user_lang})")
            st.success(translated_summary)

            # Audio
            st.markdown("### 🔊 Audio Output")
            audio_output = generate_audio(translated_summary, user_lang)
            if audio_output:
                st.audio(audio_output, format="audio/mp3")
