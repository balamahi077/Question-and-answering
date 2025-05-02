import streamlit as st
from transformers import pipeline
import openai
import PyPDF2
import io
from docx import Document
import pandas as pd
from datetime import datetime
import os

# Set page configuration
st.set_page_config(page_title="💬 Advanced QA System", layout="wide")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_option = st.selectbox(
        "Select QA Model",
        ["Hugging Face", "GPT-3.5", "Both"]
    )
    
    # Language selection
    language = st.selectbox(
        "Select Language",
        ["English", "Spanish", "French", "German", "Chinese"]
    )
    
    # Question type selection
    question_type = st.selectbox(
        "Question Type",
        ["Factual", "Analytical", "Comparative", "Hypothetical"]
    )
    
    # OpenAI API key input
    api_key = st.text_input("OpenAI API Key (for GPT-3.5)", type="password")
    
    # Export options
    st.header("📤 Export Options")
    export_format = st.selectbox("Export Format", ["PDF", "CSV", "TXT"])
    if st.button("Export Results"):
        st.info("Export functionality will be implemented here")

# Main content
st.title("💬 Advanced Question Answering System")

# File upload option
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Context input
if uploaded_file is not None:
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
    st.write(file_details)
    
    # Read the file content
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        context = ""
        for page in pdf_reader.pages:
            context += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        context = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:  # txt file
        context = uploaded_file.getvalue().decode("utf-8")
    
    st.text_area("Extracted Context:", context, height=150)
else:
    context = st.text_area("Paste context passage (e.g., syllabus):", height=150)

# Question input
question = st.text_input("Ask a question:")

# Load models
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

qa_model = load_qa_model()

# Store session history
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Process question
if st.button("🔍 Answer the Question"):
    if context and question:
        # Create columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        # Hugging Face model
        if model_option in ["Hugging Face", "Both"]:
            with col1:
                st.subheader("🤖 Hugging Face Answer")
                try:
                    result = qa_model(question=question, context=context)
                    answer = result['answer']
                    confidence = result['score']
                    
                    # Display answer with confidence score
                    st.success(f"Answer: {answer}")
                    st.info(f"Confidence Score: {confidence:.2f}")
                    
                    # Highlight the answer in the context
                    st.subheader("Answer Highlighting")
                    highlighted_context = context.replace(answer, f"**{answer}**")
                    st.markdown(highlighted_context)
                    
                    # Store in history
                    st.session_state.qa_history.append({
                        "timestamp": datetime.now(),
                        "question": question,
                        "answer": answer,
                        "confidence": confidence,
                        "model": "Hugging Face",
                        "question_type": question_type
                    })
                except Exception as e:
                    st.error(f"HF Error: {e}")
        
        # GPT-3.5 model
        if model_option in ["GPT-3.5", "Both"] and api_key:
            with col2:
                st.subheader("🤖 GPT-3.5 Answer")
                try:
                    openai.api_key = api_key
                    
                    # Customize prompt based on question type
                    prompt = f"""
                    Context: {context}
                    
                    Question Type: {question_type}
                    Language: {language}
                    
                    Question: {question}
                    
                    Please provide a detailed answer with explanation of your reasoning.
                    """
                    
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500
                    )
                    answer = response.choices[0].message.content.strip()
                    
                    # Display answer
                    st.success(f"Answer: {answer}")
                    
                    # Store in history
                    st.session_state.qa_history.append({
                        "timestamp": datetime.now(),
                        "question": question,
                        "answer": answer,
                        "confidence": "N/A",
                        "model": "GPT-3.5",
                        "question_type": question_type
                    })
                except Exception as e:
                    st.error(f"OpenAI Error: {e}")
        elif model_option in ["GPT-3.5", "Both"] and not api_key:
            with col2:
                st.warning("Please enter your OpenAI API key to use GPT-3.5")

# Display history
if st.session_state.qa_history:
    st.header("📜 Question & Answer History")
    history_df = pd.DataFrame(st.session_state.qa_history)
    st.dataframe(history_df)
