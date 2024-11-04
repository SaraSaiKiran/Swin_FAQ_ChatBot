import os
import pdfplumber
from docx import Document

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(directory, filename)
        try:
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
            elif filename.endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    pdf_text = ' '.join(page.extract_text() for page in pdf.pages if page.extract_text())
                    documents.append(pdf_text)
            elif filename.endswith('.docx'):
                doc = Document(file_path)
                doc_text = '\n'.join([para.text for para in doc.paragraphs])
                documents.append(doc_text)
            else:
                print(f"Unsupported file type: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")
    return documents

# Example usage
if __name__ == "__main__":
    directory = 'data'  # Specify your data directory here
    documents = load_documents(directory)
    print(f"Number of documents loaded: {len(documents)}")
