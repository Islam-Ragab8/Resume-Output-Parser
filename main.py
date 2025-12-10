# app.py
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load .env
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Initialize model
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=API_KEY,
    temperature=0
)

# Streamlit app
st.title("CV Parser with Llama 4 Scout")

uploaded_file = st.file_uploader("Upload a CV (PDF)", type=["pdf"])

if uploaded_file is not None:
    # 1️⃣ Load PDF
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load()
    text = " ".join([page.page_content for page in pages])

    # 2️⃣ Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # 3️⃣ Define schemas
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

    # 4️⃣ Create prompt
    CV_extraction_template = """
You are an intelligent CV parser. Extract the following information from the provided resume text:

- Full name
- Email
- Education (degree, institution, year)
- Skills
- Experience (role, company, years)

Return the output in JSON format only, nothing else.

{format_instructions}

Resume text:
{text_chunks}
"""
    prompt_template = PromptTemplate(
        template=CV_extraction_template,
        input_variables=["text_chunks", "format_instructions"]
    )
    prompt = prompt_template.format_prompt(
        text_chunks=" ".join(chunks),
        format_instructions=format_instructions
    )

    # 5️⃣ Get response from model
    with st.spinner("Parsing CV..."):
        response = model.predict(prompt.to_string())

    # 6️⃣ Show result
    st.subheader("Parsed CV JSON")
    st.json(response)
