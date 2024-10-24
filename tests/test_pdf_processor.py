import pytest
from unittest.mock import patch, MagicMock
import os
from PDF_processor.processor import PDFProcessor
import pytest
from unittest.mock import patch, MagicMock
from pymongo.database import Database
from PDF_processor.processor import PDFProcessor


@pytest.fixture
def mock_db():
    mock_db = MagicMock(spec=Database)
    mock_collection = MagicMock()
    mock_db.documents = mock_collection
    return mock_db


@pytest.fixture
def pdf_processor(mock_db):
    return PDFProcessor("mongodb://localhost:27017/", "test_pdf_processor", db=mock_db)


def test_init(mock_db):
    processor = PDFProcessor(
        "mongodb://localhost:27017/", "test_pdf_processor", db=mock_db
    )
    assert isinstance(processor, PDFProcessor)


@pytest.mark.asyncio
async def test_preprocess_pdf(pdf_processor, mock_db):
    # Mock the necessary methods
    pdf_processor.extract_text = MagicMock(return_value="Sample text")
    pdf_processor.generate_summary = MagicMock(return_value="Summary")
    pdf_processor.extract_keywords = MagicMock(return_value=["keyword1", "keyword2"])
    mock_db.documents.update_one = MagicMock()

    result = await pdf_processor.preprocess_pdf("test.pdf")

    assert result is not None
    assert result["status"] == "completed"
    assert result["summary"] == "Summary"
    assert result["keywords"] == ["keyword1", "keyword2"]


@pytest.mark.asyncio
async def test_generate_summary(pdf_processor):
    summary = await pdf_processor.generate_summary("Sample text", "short")
    assert isinstance(summary, str)
    assert len(summary) > 0


@pytest.mark.asyncio
async def test_extract_keywords(pdf_processor):
    keywords = await pdf_processor.extract_keywords("Sample text", 2)
    assert isinstance(keywords, list)
    assert len(keywords) == 2


def test_categorize_document_length(pdf_processor):
    assert pdf_processor.categorize_document_length(2) == "short"
    assert pdf_processor.categorize_document_length(10) == "medium"
    assert pdf_processor.categorize_document_length(20) == "long"


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


@patch("PDF_processor.processor.PDFProcessor.extract_text")
@patch("PDF_processor.processor.PDFProcessor.generate_summary")
@patch("PDF_processor.processor.PDFProcessor.extract_keywords")
@patch("PDF_processor.processor.PDFProcessor.store_initial_metadata")
@patch("PDF_processor.processor.PDFProcessor.update_mongodb")
@patch("os.path.isfile", return_value=True)
@patch("pypdf.PdfReader")
@patch("builtins.open")
@pytest.mark.asyncio
async def test_preprocess_pdf(
    mock_open,
    mock_pdf_reader,
    mock_isfile,
    mock_update,
    mock_store,
    mock_keywords,
    mock_summary,
    mock_extract,
    pdf_processor,
):
    # Mock file existence check and size
    mock_isfile.return_value = True

    # Mock PDF reader
    mock_pdf_reader_instance = MagicMock()
    mock_pdf_reader_instance.pages = [MagicMock() for _ in range(5)]  # Mock 5 pages
    mock_pdf_reader.return_value = mock_pdf_reader_instance

    # Mock file open context
    mock_open.return_value.__enter__.return_value = MagicMock()

    # Mock return values
    mock_extract.return_value = "Sample text"
    mock_summary.return_value = "Summary"
    mock_keywords.return_value = ["keyword1", "keyword2"]
    mock_store.return_value = "doc_id"

    # Call the function
    result = await pdf_processor.preprocess_pdf("test.pdf")

    # Test error handling with non-existent file
    mock_isfile.return_value = False
    result = await pdf_processor.preprocess_pdf("test.pdf")
    assert "error" in result
    assert "file_path" in result
    assert "File not found" in result["error"]


@pytest.mark.asyncio
@patch("PDF_processor.processor.sent_tokenize")
@patch("PDF_processor.processor.word_tokenize")
async def test_generate_summary(mock_word_tokenize, mock_sent_tokenize, pdf_processor):
    mock_sent_tokenize.return_value = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
    mock_word_tokenize.return_value = ["word1", "word2", "word3"]

    summary = await pdf_processor.generate_summary("Sample text", "short")
    assert isinstance(summary, str)
    assert len(summary) > 0


@patch("PDF_processor.processor.word_tokenize")
@patch("PDF_processor.processor.stopwords.words")
@pytest.mark.asyncio
async def test_extract_keywords(mock_stopwords, mock_word_tokenize, pdf_processor):
    # Mock stopwords
    mock_stopwords.return_value = ["the", "is", "at", "which"]

    # Mock word tokenization with some common words and keywords
    mock_word_tokenize.return_value = ["keyword1", "keyword2", "keyword3", "the", "is"]

    keywords = await pdf_processor.extract_keywords("Sample text", 2)
    assert isinstance(keywords, list)


def test_categorize_document_length(pdf_processor):
    assert pdf_processor.categorize_document_length(2) == "short"
    assert pdf_processor.categorize_document_length(10) == "medium"
    assert pdf_processor.categorize_document_length(20) == "long"
