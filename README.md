# 📄 CV JSON Extractor

Extract structured information (Name, Email, Education, Skills, Experience) from any PDF CV using Streamlit + Llama 4 (Groq).

This project parses PDF resumes and converts them into clean, well-formatted JSON output, ready for ATS systems or AI pipelines.

# 🚀 Features

✅ Upload any PDF CV
✅ Extract structured fields:

Full name

Email

Phone number

Education

Skills

Experience

✅ Clean JSON output (no markdown, no noise)
✅ Uses Llama 4 Scout 17B via Groq API
✅ Secure & production-friendly structure


# 📦 Requirements

Install dependencies:
```
pip install -r requirements.txt
```

```
streamlit
langchain
langchain-community
langchain-groq
python-dotenv
pypdf
```

# 📁 Project Structure
```
Resume-Output-Parser/
│── main.py
│── requirements.txt
│── README.md
│── .env
│── assets/
│   └── sample.pdf
```

# 🧠 How It Works

1-User uploads a PDF

2-System saves the file temporarily

3- LangChain loads PDF → splits text

4-StructuredOutputParser forces strict JSON

5-Llama-4 Scout extracts structured fields

6-Output appears as clean JSON

# 🖥️ Screenshot (example output)

```
{
  "full_name": "John Smith",
  "email": "john.smith@email.com",
  "phone_number": "+1 555 123 456",
  "education": [
    {
      "degree": "B.Sc. Computer Science",
      "institution": "MIT",
      "year": "2020"
    }
  ],
  "skills": ["Python", "Machine Learning", "Data Analysis"],
  "experience": [
    {
      "role": "Software Engineer",
      "company": "Google",
      "years": "2020–2023"
    }
  ]
}

```


