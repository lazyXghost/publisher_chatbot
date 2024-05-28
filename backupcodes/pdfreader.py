docs = []
metadata = []

# Read PDF documents from the given path
pdf_docs = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pdf')]
for pdf_path in pdf_docs:
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for index, page in enumerate(pdf_reader.pages):
            doc_page = {
                "title": os.path.basename(pdf_path) + " page " + str(index + 1),
                "content": page.extract_text(),
            }
            docs.append(doc_page)

content = [doc["content"] for doc in docs]
metadata = [{"title": doc["title"]} for doc in docs]
print("Content and metadata are extracted from the documents")
