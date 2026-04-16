import pytest
from pydantic import ValidationError
from src.rag import search_documents_keyword, search_documents_hybrid
from src.rag.keyword_retriever import Document, SAMPLE_DOCUMENTS

def test_keyword_retriever_returns_results():
    query = SAMPLE_DOCUMENTS[0].title
    results = search_documents_keyword(query)
    assert len(results) > 0

def test_keyword_retriever_returns_at_most_top_k():
    query = SAMPLE_DOCUMENTS[0].title
    top_k = 2
    results = search_documents_keyword(query, top_k=top_k)
    assert len(results) <= top_k

def test_hybrid_retriever_returns_at_most_top_k():
    query = SAMPLE_DOCUMENTS[0].title
    top_k = 2
    results = search_documents_hybrid(query, top_k=top_k)
    assert len(results) <= top_k

def test_document_pydantic_validation():
    doc = Document(title="Test Document", text="This is a test.")
    assert doc.title == "Test Document"
    assert doc.text == "This is a test."

    with pytest.raises(ValidationError):
        Document(title=123, text="Invalid title type")
    