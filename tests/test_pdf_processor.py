import pytest
import os
from PDF_processor.processor import PDFProcessor


@pytest.mark.asyncio
async def test_pdf_processor_initialization():
    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    processor = PDFProcessor(mongodb_uri, "test_db")
    assert processor is not None


@pytest.mark.asyncio
async def test_process_folder():
    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    processor = PDFProcessor(mongodb_uri, "test_db")
    results = await processor.process_folder("Data")
    assert len(results) > 0
