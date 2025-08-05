import streamlit as st
from analyzer.similarity import compare_responses
import pandas as pd
import os

st.set_page_config(page_title="Persona Benchmark", layout="wide")
st.title("🧠 Persona Benchmark: GPT vs Gemini")

persona = st.selectbox("Choose Persona", ["coach-v1"])
model_a = "GPT"
model_b = "Gemini"

input_dir = "data/inputs"
output_dir = "data/outputs"
os.makedirs(output_dir, exist_ok=True)

file_a = os.path.join(input_dir, f"{persona}_gpt.txt")
file_b = os.path.join(input_dir, f"{persona}_gemini.txt")
output_file = os.path.join(output_dir, f"{persona}_comparison.csv")

if st.button("Run Similarity Comparison"):
    compare_responses(file_a, file_b, output_file)
    st.success("Comparison complete.")

    df = pd.read_csv(output_file)
    st.dataframe(df)

    st.line_chart(df.set_index("Example"))
