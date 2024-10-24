import pytest
import asyncio
from unittest.mock import Mock, patch
from PDF_processor.processor import PDFProcessor


@pytest.fixture
def pdf_processor():
    with patch("PDF_processor.processor.MongoClient"):
        processor = PDFProcessor("mongodb://fake_uri", "test_db")
        yield processor


@pytest.mark.asyncio
async def test_cleanup_database(pdf_processor):
    pdf_processor.collection = Mock()
    pdf_processor.collection.aggregate.return_value = [
        {"_id": "file1", "latest_id": "id1"},
        {"_id": "file2", "latest_id": "id2"},
    ]
    pdf_processor.collection.count_documents.return_value = 2

    result = await pdf_processor.cleanup_database()

    assert result == 2
    pdf_processor.collection.delete_many.assert_called_once()
    pdf_processor.collection.count_documents.assert_called_once()


@pytest.mark.asyncio
async def test_extract_text(pdf_processor):
    with patch("PDF_processor.processor.open"), patch(
        "PDF_processor.processor.pypdf.PdfReader"
    ) as mock_pdf_reader:
        mock_pdf_reader.return_value.pages = [Mock(), Mock()]
        mock_pdf_reader.return_value.pages[0].extract_text.return_value = (
            "Page 1 content"
        )
        mock_pdf_reader.return_value.pages[1].extract_text.return_value = (
            "Page 2 content"
        )

        result = await pdf_processor.extract_text("fake_path.pdf")

    assert result == "Page 1 content\nPage 2 content"


@pytest.mark.asyncio
async def test_generate_summary(pdf_processor):
    text = "This is the first sentence. This is the second sentence. This is the third sentence."
    result = await pdf_processor.generate_summary(text, "short")

    assert isinstance(result, str)
    assert len(result.split(".")) <= 3


@pytest.mark.asyncio
async def test_extract_keywords(pdf_processor):
    text = "This is a test text with some keywords. Keywords are important for testing."
    result = await pdf_processor.extract_keywords(text, 5)

    assert isinstance(result, list)
    assert len(result) <= 5
    assert "keywords" in result


def test_categorize_document_length():
    assert PDFProcessor.categorize_document_length(2) == "short"
    assert PDFProcessor.categorize_document_length(10) == "medium"
    assert PDFProcessor.categorize_document_length(20) == "long"


# Add more tests as needed
