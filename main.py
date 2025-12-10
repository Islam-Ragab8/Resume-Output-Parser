import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# -----------------------------
# Load env
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# Initialize model
# -----------------------------
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=GROQ_API_KEY,
    temperature=0
)

st.title("CV JSON Parser with LLM")

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")

if uploaded_file is not None:
    # 🔹 Save temp file
    temp_path = "temp_cv.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 🔹 Load PDF
    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    text = " ".join([page.page_content for page in pages])

    # 🔹 Split text to chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # -----------------------------
    # Define output schema
    # -----------------------------
    schemas = [
        ResponseSchema(name="full_name", description="The candidate's full name."),
        ResponseSchema(name="email", description="The candidate's email address."),
        ResponseSchema(name="phone_number", description="The candidate's phone number."),
        ResponseSchema(name="education", description="list of {degree, institution, year}"),
        ResponseSchema(name="skills", description="A list of the candidate's key skills."),
        ResponseSchema(name="experience", description="A list of work experiences, each containing role, company, and years.")
    ]
    out_parser = StructuredOutputParser.from_response_schemas(schemas)
    format_instructions = out_parser.get_format_instructions()

    # -----------------------------
    # Prompt template
    # -----------------------------
    CV_extraction_template = """
You are an intelligent CV parser. Extract the following information from the provided resume text:

- Full name
- Email
- Phone number
- Education (degree, institution, year)
- Skills
- Experience (role, company, years)

Return the output in **JSON format only**, nothing else, no explanation, no markdown.

{format_instructions}

Resume text:
{chunks}
"""
    prompt = PromptTemplate(
        template=CV_extraction_template,
        input_variables=["chunks", "format_instructions"]
    ).format_prompt(chunks=chunks, format_instructions=format_instructions)

    # -----------------------------
    # Get response
    # -----------------------------
    with st.spinner("Parsing CV..."):
        response = model.predict(prompt.to_string())
    
    # -----------------------------
    # Show JSON output
    # -----------------------------
    st.subheader("Parsed CV JSON Output")
    st.json(response)

    # Optional: remove temp file
    os.remove(temp_path)
