# 📚 Simple RAG Pipeline with LangChain (PDF → FAISS → OpenAI)

A fully commented, end-to-end **Retrieval-Augmented Generation (RAG)** example built with **LangChain**, **FAISS**, and **OpenAI**.

This project demonstrates how to:

- Load a PDF document
- Split it into semantic chunks
- Generate embeddings
- Store vectors in FAISS
- Build a retriever
- Create a prompt template
- Connect to an OpenAI chat model
- Execute a RAG chain
- (Optionally) Expand retrieval using Multi-Query

The implementation is provided in:

`RAG_LangChain.py` :contentReference[oaicite:0]{index=0}

---

## 🚀 What This Project Does

This script builds a simple legal AI assistant that:

1. Reads a PDF file (`constitution_sample.pdf`)
2. Converts it into searchable vector embeddings
3. Retrieves the most relevant passages
4. Sends them to an OpenAI LLM
5. Generates an answer grounded **only in the document context**

---

## 🏗 Architecture Overview

```text
PDF → Document Loader → Text Splitter → Embeddings → FAISS
                                               ↓
User Question → Retriever → Prompt Template → LLM → Answer
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install langchain langchain-openai langchain-community faiss-cpu pypdf
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## 📄 Step-by-Step Breakdown

### 1️⃣ Load the PDF

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("constitution_sample.pdf")
documents = loader.load()

print(len(documents))
print(documents[0].page_content[:500])
```

✔ Converts each PDF page into a `Document` object  
✔ Includes metadata like page numbers  

---

### 2️⃣ Split Into Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)
print(len(chunks))
```

✔ Keeps chunks manageable for embeddings  
✔ Overlap preserves context across boundaries  

---

### 3️⃣ Generate Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

✔ Converts text into high-dimensional vectors  
✔ Optimized balance between cost and quality  

---

### 4️⃣ Create FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

vectorstore.save_local("faiss_index")
```

✔ Fast similarity search  
✔ Local persistence  
✔ No need to recompute embeddings  

---

### 5️⃣ Create a Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

✔ Returns top 5 most relevant chunks  
✔ Abstracts vector search  

---

### 6️⃣ Define the RAG Prompt

```python
from langchain.prompts import PromptTemplate

rag_prompt = PromptTemplate(
    template="""
You are a legal AI assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say:
"I could not find the answer in the provided documents."

Cite relevant sections when possible.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)
```

✔ Enforces grounded responses  
✔ Reduces hallucination  
✔ Structured input format  

---

### 7️⃣ Initialize the LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
```

✔ Deterministic responses (`temperature=0`)  
✔ Lightweight & cost-efficient  

---

### 8️⃣ Helper Utilities

```python
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

✔ Formats retrieved docs  
✔ Ensures clean string output  

---

### 9️⃣ Build the RAG Chain

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

Pipeline flow:

```text
Question
   ↓
Retriever
   ↓
Format Docs
   ↓
Prompt Template
   ↓
LLM
   ↓
Parsed Answer
```

---

### 🔟 Run a Query

```python
query = "What are the fundamental rights mentioned in the document?"
response = rag_chain.invoke(query)

print(response)
```

✔ Retrieves relevant sections  
✔ Generates grounded legal answer  

---

### 🔁 Optional: Multi-Query Retriever

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
```

✔ Generates alternate search queries using the LLM  
✔ Improves recall  
✔ Better for complex or ambiguous questions  

---

## 🧠 Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Vector embeddings
- Semantic similarity search
- Prompt engineering
- Deterministic LLM configuration
- LangChain runnable pipelines

---

## 📁 Project Structure

```text
.
├── RAG_LangChain.py
├── constitution_sample.pdf
├── faiss_index/
└── README.md
```

---

## 🔐 Why Use RAG?

Without RAG:
> LLM answers from training data (may hallucinate)

With RAG:
> LLM answers grounded in your documents

Benefits:

- ✔ Reduced hallucination  
- ✔ Domain-specific knowledge  
- ✔ Private document QA  
- ✔ Lower cost than fine-tuning  

---

## 🎯 Ideal Use Cases

- Legal document Q&A  
- Internal company knowledge bases  
- Research paper search  
- Policy analysis  
- Compliance documentation  

---

## 📌 Final Notes

This implementation is intentionally simple and educational.  
For production systems, consider:

- Metadata filtering
- Hybrid search (BM25 + vectors)
- Chunk optimization strategies
- Streaming responses
- Caching
- Evaluation pipelines

---

## 🏁 Summary

You now have a complete working example of a PDF-based RAG system using:

- LangChain
- OpenAI embeddings
- FAISS vector search
- Prompt engineering
- Runnable chains

This is the foundation for building intelligent document-based AI systems.

---

## 📜 License

MIT License

---

## 🤝 Contributing

Pull requests are welcome!  
If you find a bug or want to improve retrieval quality, feel free to open an issue.

---

## ⭐ If You Found This Useful

Give it a star and build something awesome with it!
