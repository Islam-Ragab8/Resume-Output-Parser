import streamlit as st
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Streamlit UI
st.title("CV JSON Extractor")

uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text = " ".join([page.page_content for page in pages])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

   
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

    
    CV_extraction_template = """
You are an intelligent CV parser. Extract the following information from the provided resume text:

- Full name
- Email
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
    )
    prompt_text = prompt.format_prompt(chunks=chunks, format_instructions=format_instructions).to_string()

    
    with st.spinner("Parsing CV..."):
        response = model.predict(prompt_text)
    
    
    try:
        parsed_output = out_parser.parse(response)
        st.subheader("Extracted JSON")
        st.json(parsed_output)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        st.write("Raw output from model:")
        st.code(response)
