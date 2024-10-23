import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from PDF_processor.PDF_processor import PDFProcessor
import os


class TestPDFProcessor(unittest.IsolatedAsyncioTestCase):
    @patch("pdf_processor.MongoClient")
    def setUp(self, mock_mongo_client):
        # Mock MongoDB connection
        self.processor = PDFProcessor(mongodb_uri="mock_uri", db_name="test_db")
        self.processor.collection = MagicMock()

    async def test_cleanup_database(self):
        # Mock MongoDB aggregate results and delete_many
        self.processor.collection.aggregate.return_value = [
            {"_id": "test_id", "latest_id": "test_id"}
        ]
        self.processor.collection.delete_many = AsyncMock()

        result = await self.processor.cleanup_database()

        self.assertEqual(
            result, 0
        )  # Check if the final document count is returned correctly
        self.processor.collection.delete_many.assert_called_once()

    @patch("pdf_processor.os.path.exists")
    async def test_process_folder_file_not_found(self, mock_exists):
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError):
            await self.processor.process_folder("non_existent_folder")

    @patch("pdf_processor.os.listdir")
    async def test_process_folder_no_pdfs(self, mock_listdir):
        mock_listdir.return_value = []

        result = await self.processor.process_folder("empty_folder")
        self.assertEqual(result, [])

    @patch("pdf_processor.PyPDF2.PdfReader")
    async def test_extract_text_success(self, mock_pdf_reader):
        # Setup mock PDF
        mock_pdf_reader.return_value.pages = [MagicMock(), MagicMock()]
        mock_pdf_reader.return_value.pages[0].extract_text.return_value = "Test page 1"
        mock_pdf_reader.return_value.pages[1].extract_text.return_value = "Test page 2"

        result = await self.processor.extract_text("mock_file.pdf")
        self.assertEqual(result, "Test page 1\nTest page 2\n")

    @patch("pdf_processor.os.path.getsize")
    @patch("pdf_processor.PDFProcessor.extract_text", return_value="Sample text")
    async def test_preprocess_pdf(self, mock_extract_text, mock_getsize):
        mock_getsize.return_value = 1234

        result = await self.processor.preprocess_pdf("mock_file.pdf")
        self.assertEqual(result["metadata"]["filename"], "mock_file.pdf")
        self.assertEqual(result["status"], "completed")
