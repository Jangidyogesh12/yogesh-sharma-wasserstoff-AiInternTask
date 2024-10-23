import pytest
from unittest.mock import patch

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PDF_processor.processor import PDFProcessor


@pytest.fixture
def pdf_processor():
    mongodb_uri = "mongodb://localhost:27017/"
    db_name = "test_pdf_processor"
    return PDFProcessor(mongodb_uri, db_name)


@patch("PDF_processor.processor.MongoClient")
def test_init(mock_mongo_client):
    mongodb_uri = "mongodb://localhost:27017/"
    db_name = "test_pdf_processor"

    # Test successful initialization
    processor = PDFProcessor(mongodb_uri, db_name)
    assert processor is not None

    # Test MongoDB connection error
    mock_mongo_client.side_effect = Exception("Connection failed")
    with pytest.raises(Exception):
        PDFProcessor(mongodb_uri, db_name)


@pytest.mark.asyncio
@patch("PDF_processor.processor.PDFProcessor.extract_text")
@patch("PDF_processor.processor.PDFProcessor.generate_summary")
@patch("PDF_processor.processor.PDFProcessor.extract_keywords")
@patch("PDF_processor.processor.PDFProcessor.store_initial_metadata")
@patch("PDF_processor.processor.PDFProcessor.update_mongodb")
async def test_preprocess_pdf(
    mock_update, mock_store, mock_keywords, mock_summary, mock_extract, pdf_processor
):
    # Mock return values
    mock_extract.return_value = "Sample text"
    mock_summary.return_value = "Summary"
    mock_keywords.return_value = ["keyword1", "keyword2"]
    mock_store.return_value = "doc_id"

    result = await pdf_processor.preprocess_pdf("test.pdf")

    assert result is not None
    assert result["status"] == "completed"
    assert result["summary"] == "Summary"
    assert result["keywords"] == ["keyword1", "keyword2"]

    # Test error handling
    mock_extract.side_effect = Exception("Extraction failed")
    result = await pdf_processor.preprocess_pdf("test.pdf")
    assert result["error"] == "Extraction failed"


def test_categorize_document_length(pdf_processor):
    assert pdf_processor.categorize_document_length(2) == "short"
    assert pdf_processor.categorize_document_length(10) == "medium"
    assert pdf_processor.categorize_document_length(20) == "long"


@pytest.mark.asyncio
@patch("PDF_processor.processor.sent_tokenize")
@patch("PDF_processor.processor.word_tokenize")
async def test_generate_summary(mock_word_tokenize, mock_sent_tokenize, pdf_processor):
    mock_sent_tokenize.return_value = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
    mock_word_tokenize.return_value = ["word1", "word2", "word3"]

    summary = await pdf_processor.generate_summary("Sample text", "short")
    assert isinstance(summary, str)
    assert len(summary) > 0


@pytest.mark.asyncio
@patch("PDF_processor.processor.word_tokenize")
async def test_extract_keywords(mock_word_tokenize, pdf_processor):
    mock_word_tokenize.return_value = ["keyword1", "keyword2", "keyword3"]
    keywords = await pdf_processor.extract_keywords("Sample text", 2)
    assert len(keywords) == 2
    assert isinstance(keywords, list)
