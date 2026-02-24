"""
This script demonstrates a simple Retrieval-Augmented Generation (RAG) workflow
using the LangChain framework. It loads a PDF document, splits it into smaller
text chunks, embeds those chunks into a vector index, and then builds a
chain that can answer questions by retrieving relevant passages from the
index and passing them to an LLM.

Every line below is commented in plain language. The comments explain what each
import, object, method, and statement does and why it is needed.
"""

# --------------------------------------------------------------------
# 1.  Import helpers for loading PDF files. The langchain_community
#     package contains community-contributed tools; PyPDFLoader is a
#     loader that reads a PDF and produces a list of Document objects.
# --------------------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader

# Create a loader that will open the file "constitution_sample.pdf".
# This is the local path to the PDF we want to index.
loader = PyPDFLoader("constitution_sample.pdf")

# Use the loader to read every page of the PDF and turn it into a list
# of Document objects. Each Document has text and metadata like page
# number. The load() method does the actual file I/O and parsing.
documents = loader.load()

# Print how many pages/documents were returned, just to confirm we
# actually loaded something.
print(len(documents))

# Show the first 500 characters of the first document’s text. This is a
# quick sanity check so we know the loader worked and we can see the
# format of the data.
print(documents[0].page_content[:500])

# --------------------------------------------------------------------
# 2.  Import a text splitter implementation. The splitter will take the
#     long documents produced by the loader and break them into smaller
#     pieces (chunks) suitable for embedding and retrieval.
# --------------------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Instantiate the splitter with a maximum chunk size of 800 characters
# and an overlap of 150 characters between chunks. Overlapping ensures
# that information spanning a boundary isn’t lost.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

# Apply the splitter to the list of Document objects. This returns a new
# list where each element is a chunk (also a Document) with shorter text.
chunks = text_splitter.split_documents(documents)

# Print how many chunks we obtained. A larger number means the original
# content was divided into many pieces, which will affect retrieval
# granularity and index size.
print(len(chunks))

# --------------------------------------------------------------------
# 3.  Import the embeddings class that talks to OpenAI to convert text
#     into vectors. These vectors are what the vector store will index.
# --------------------------------------------------------------------
from langchain_openai import OpenAIEmbeddings

# Create an embeddings generator using the "text-embedding-3-small" model.
# The model choice balances cost and quality; you can swap it out if
# you need higher-dimensional vectors or a different provider.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# --------------------------------------------------------------------
# 4.  Import FAISS, a fast in-memory vector index. We will add our
#     chunks and their embeddings to this store so that we can do
#     similarity searches later.
# --------------------------------------------------------------------
from langchain_community.vectorstores import FAISS

# Build a FAISS index directly from our document chunks, computing
# embeddings on the fly using the embeddings object we created above.
# The `from_documents` helper hides the loop that would otherwise be
# required to embed each chunk manually.
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save the index to disk so we can reload it later without recomputing
# the embeddings. The directory "faiss_index" will be created if needed.
vectorstore.save_local("faiss_index")

# --------------------------------------------------------------------
# 5.  Create a retriever object from the vector store. The retriever
#     abstracts away the details of searching the index.
# --------------------------------------------------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",      # we want nearest-neighbor style search
    search_kwargs={"k": 5}         # return the top 5 closest chunks
)

# --------------------------------------------------------------------
# 6.  Define a prompt template. This is the text that will be sent to
#     the language model. It instructs the model how to behave and
#     where to insert the retrieved context and the user’s question.
# --------------------------------------------------------------------
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
    input_variables=["context", "question"],  # these placeholders must be
                                             # filled in by the chain
)

# --------------------------------------------------------------------
# 7.  Import and configure the chat-capable OpenAI model. We set
#     temperature=0 to make the model deterministic (less creative).
# --------------------------------------------------------------------
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# --------------------------------------------------------------------
# 8.  Define helper utilities used in the chain.
#     - RunnablePassthrough simply passes its input through unchanged.
#     - StrOutputParser ensures the final output is returned as a plain
#       string instead of a more complex object.
# --------------------------------------------------------------------
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def format_docs(docs):
    """
    Convert a list of Document objects into a single string by joining
    their text with blank lines. This formatted string will be placed in
    the {context} variable in the prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# --------------------------------------------------------------------
# 9.  Assemble the RAG chain. The pipeline works as follows:
#     - The retriever fetches relevant documents for the question.
#     - format_docs merges them into one text blob.
#     - rag_prompt formats the full prompt including context and question.
#     - llm invokes the language model with that prompt.
#     - StrOutputParser extracts the raw text response.
# --------------------------------------------------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------------------------
# 10. Run an example query through the chain and print the answer.
# --------------------------------------------------------------------
query = "What are the fundamental rights mentioned in the document?"

response = rag_chain.invoke(query)

print(response)

# --------------------------------------------------------------------
# 11. (Optional) Demonstrate a multi-query retriever. This object
#     generates multiple search queries using the LLM itself to
#     broaden retrieval. We show how to create it but do not run it.
# --------------------------------------------------------------------
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),  # base retriever to wrap
    llm=llm                               # LLM used to generate alternate queries
)