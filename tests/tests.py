import unittest
from unittest.mock import patch, MagicMock
import asyncio
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PDF_processor.processor import PDFProcessor


class TestPDFProcessor(unittest.TestCase):

    def setUp(self):
        self.mongodb_uri = "mongodb://localhost:27017/"
        self.db_name = "test_pdf_processor"
        self.processor = PDFProcessor(self.mongodb_uri, self.db_name)

    @patch("pdf_processor.MongoClient")
    def test_init(self, mock_mongo_client):
        # Test successful initialization
        processor = PDFProcessor(self.mongodb_uri, self.db_name)
        self.assertIsNotNone(processor)

        # Test MongoDB connection error
        mock_mongo_client.side_effect = Exception("Connection failed")
        with self.assertRaises(Exception):
            PDFProcessor(self.mongodb_uri, self.db_name)

    @patch("pdf_processor.PDFProcessor.extract_text")
    @patch("pdf_processor.PDFProcessor.generate_summary")
    @patch("pdf_processor.PDFProcessor.extract_keywords")
    @patch("pdf_processor.PDFProcessor.store_initial_metadata")
    @patch("pdf_processor.PDFProcessor.update_mongodb")
    async def test_preprocess_pdf(
        self, mock_update, mock_store, mock_keywords, mock_summary, mock_extract
    ):
        # Mock return values
        mock_extract.return_value = "Sample text"
        mock_summary.return_value = "Summary"
        mock_keywords.return_value = ["keyword1", "keyword2"]
        mock_store.return_value = "doc_id"

        result = await self.processor.preprocess_pdf("test.pdf")

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["summary"], "Summary")
        self.assertEqual(result["keywords"], ["keyword1", "keyword2"])

        # Test error handling
        mock_extract.side_effect = Exception("Extraction failed")
        result = await self.processor.preprocess_pdf("test.pdf")
        self.assertIn("error", result)

    def test_categorize_document_length(self):
        self.assertEqual(self.processor.categorize_document_length(2), "short")
        self.assertEqual(self.processor.categorize_document_length(10), "medium")
        self.assertEqual(self.processor.categorize_document_length(20), "long")

    @patch("pdf_processor.sent_tokenize")
    @patch("pdf_processor.word_tokenize")
    async def test_generate_summary(self, mock_word_tokenize, mock_sent_tokenize):
        mock_sent_tokenize.return_value = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
        mock_word_tokenize.return_value = ["word1", "word2", "word3"]

        summary = await self.processor.generate_summary("Sample text", "short")
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)

    @patch("pdf_processor.word_tokenize")
    async def test_extract_keywords(self, mock_word_tokenize):
        mock_word_tokenize.return_value = ["keyword1", "keyword2", "keyword3"]
        keywords = await self.processor.extract_keywords("Sample text", 2)
        self.assertEqual(len(keywords), 2)
        self.assertIsInstance(keywords, list)


if __name__ == "__main__":
    unittest.main()
