from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"


def build_vector_store(pdf_path, persist_directory):
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    if not documents:
        raise ValueError("No readable text was found in this PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory)
    )
    return vector_store, len(documents), len(chunks)


def answer_question(vector_store, question):
    retrieved_docs = vector_store.similarity_search(question, k=5)
    context_parts = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "Unknown")
        page_number = int(page) + 1 if str(page).isdigit() else page
        context_parts.append(f"[Page {page_number}]\\n{doc.page_content}")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using only the supplied PDF context. If the answer is not "
         "present, say the PDF does not contain enough information. "
         "Do not invent facts."),
        ("human", "PDF context:\\n{context}\\n\\nQuestion: {question}")
    ])
    model = ChatOllama(model=LLM_MODEL, temperature=0)
    response = (prompt | model).invoke({
        "context": "\\n\\n".join(context_parts),
        "question": question
    })
    return response.content, retrieved_docs
