from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llm_chains import load_vectordb, create_embeddings
from utils import load_config
import pypdfium2
import openai

import fitz  # PyMuPDF

config = load_config()

# Configure OpenAI API key
openai.api_key = 'openai-api-key'

def get_pdf_texts(pdfs_bytes_list):
    return [extract_text_from_pdf(pdf_bytes.getvalue()) for pdf_bytes in pdfs_bytes_list]


def extract_text_from_pdf(pdf_bytes):
    # Open the PDF with PyMuPDF
    pdf_file = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in pdf_file:
        texts.append(page.get_text())
    pdf_file.close()
    return "\n".join(texts)


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=config["pdf_text_splitter"]["chunk_size"], 
                                              chunk_overlap=config["pdf_text_splitter"]["overlap"],
                                                separators=config["pdf_text_splitter"]["separators"])
    return splitter.split_text(text)

def get_document_chunks(text_list):
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content = chunk))
    return documents

def process_text_with_openai(text):
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use an accessible model
            prompt="Summarize the following text:\n" + text,
            max_tokens=2000,
            temperature=0.3
        )
        summary = response.choices[0].text.strip()
        return summary
    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e)


def add_documents_to_db(uploaded_files):
    # Make sure uploaded_files is a list of UploadedFile objects
    summaries = []
    for uploaded_file in uploaded_files:
        # Read the PDF file into bytes
        pdf_bytes = uploaded_file.getvalue()
        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_bytes)
        # Summarize the extracted text using OpenAI's API
        summary = process_text_with_openai(extracted_text)
        summaries.append(summary)

    # Now, you could add your summaries to a database or otherwise process them
    # For example, printing them:
    for summary in summaries:
        print(summary)
        


    
    