# Streamlit LLM PDF Question Answering App

## Install
```bash
python -m pip install -r requirements.txt
```

## Install Ollama models
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Run
```bash
streamlit run app.py
```

Upload a PDF in the sidebar and ask questions. ChromaDB data is stored in
`chroma_db/`; questions and answers are logged in `dataset/`.
