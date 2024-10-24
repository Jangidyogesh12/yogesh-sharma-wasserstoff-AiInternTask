import os
import math
import asyncio
from datetime import datetime
import time
import tracemalloc

# import PyPDF2
import pypdf
from loguru import logger
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pymongo import MongoClient, errors as mongo_errors
import pypdf.errors


class PDFProcessor:
    def __init__(self, mongodb_uri: str, db_name: str):
        """Initialize the PDF processor with MongoDB connection."""
        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[db_name]
            self.collection = self.db.documents

            # Add index on filename to ensure uniqueness
            self.collection.create_index([("filename", 1)], unique=True)

            # Test connection
            self.client.server_info()

            # Download required NLTK data with error handling
            nltk_resources = ["punkt", "stopwords", "averaged_perceptron_tagger"]
            for resource in nltk_resources:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.error(
                        f"Failed to download NLTK resource {resource}: {str(e)}"
                    )
                    raise

        # Catch MongoDB connection errors correctly
        except mongo_errors.ServerSelectionTimeoutError as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    async def cleanup_database(self):
        """Clean up duplicate and failed entries from the database."""
        try:
            # Remove duplicate entries keeping only the latest
            pipeline = [
                {"$sort": {"processed_date": -1}},
                {"$group": {"_id": "$filename", "latest_id": {"$first": "$_id"}}},
            ]
            latest_entries = list(self.collection.aggregate(pipeline))
            latest_ids = [entry["latest_id"] for entry in latest_entries]

            # Remove all entries except the latest for each filename
            self.collection.delete_many({"_id": {"$nin": latest_ids}})

            # Get count of remaining documents
            remaining_count = self.collection.count_documents({})
            logger.info(f"Database cleaned up. Remaining documents: {remaining_count}")

            return remaining_count

        except Exception as e:
            logger.error(f"Error cleaning up database: {str(e)}")
            raise

    async def process_folder(
        self, folder_path: str, max_concurrent: int = 3
    ) -> List[Dict]:
        """Process all PDFs in a folder concurrently with duplicate prevention."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Clean up database before processing
        await self.cleanup_database()

        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return []

        # Process PDFs in chunks to control concurrency
        tasks = []
        for pdf_file in pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            tasks.append(self.preprocess_pdf(file_path))

        results = []
        for chunk in [
            tasks[i : i + max_concurrent] for i in range(0, len(tasks), max_concurrent)
        ]:
            chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
            results.extend([r for r in chunk_results if r is not None])

        # Final cleanup and count
        final_count = await self.cleanup_database()
        logger.info(f"Final document count after processing: {final_count}")

        return results

    async def extract_text(self, file_path: str) -> str:
        """Extract text from PDF file. Returns text from first 20 pages if PDF has more than 20 pages."""
        try:
            with ThreadPoolExecutor() as executor:

                def extract():
                    text = ""
                    with open(file_path, "rb") as file:
                        try:
                            pdf = pypdf.PdfReader(file)
                            # Get total number of pages
                            total_pages = len(pdf.pages)
                            # Limit to first 20 pages if total pages exceed 20
                            pages_to_read = min(20, total_pages)

                            for page_num in range(pages_to_read):
                                try:
                                    text += pdf.pages[page_num].extract_text() + "\n"
                                except Exception as e:
                                    logger.warning(
                                        f"Error extracting text from page {page_num} in {file_path}: {str(e)}"
                                    )
                        except pypdf.errors.PdfReadError as e:
                            logger.error(f"Failed to read PDF {file_path}: {str(e)}")
                            raise
                    return text.strip()

                return await asyncio.get_event_loop().run_in_executor(executor, extract)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise

    async def preprocess_pdf(self, file_path: str) -> Optional[Dict]:
        """Extract text from the PDF and return the basic metadata."""
        doc_id = None
        try:
            # check if file Exist
            if not os.path.isfile(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                return {"error": error_msg, "file_path": file_path}

            metadata = {
                "filename": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "processed_date": datetime.now(),
                "status": "processing",
            }

            # Check for existing successful processing
            existing_doc = self.collection.find_one(
                {"filename": metadata["filename"], "status": "completed"}
            )

            if existing_doc:
                logger.info(f"Skipping already processed file: {metadata['filename']}")
                return None

            # Store initial metadata
            doc_id = await self.store_initial_metadata(metadata)
            if not doc_id:
                return None

            # Extract text
            text = await self.extract_text(file_path)
            if not text.strip():
                raise ValueError("No text content extracted from PDF")

            # Get document length
            with open(file_path, "rb") as file:
                pdf = pypdf.PdfReader(file)
                num_pages = len(pdf.pages)

            doc_length = self.categorize_document_length(num_pages)

            # Generate summary and keywords
            summary = await self.generate_summary(text, doc_length)
            keywords = await self.extract_keywords(text, 10)

            result = {
                "metadata": metadata,
                "summary": summary,
                "keywords": keywords,
                "num_pages": num_pages,
                "doc_length": doc_length,
                "status": "completed",
            }

            await self.update_mongodb(doc_id, result)
            return result

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            if doc_id:
                await self.update_mongodb(
                    doc_id, {"status": "failed", "error": error_msg}
                )
            return {"error": error_msg, "file_path": file_path}

    async def generate_summary(self, text: str, doc_length: str) -> str:
        """Generate a summary based on TF-IDF scoring."""
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return "No content available for summarization."

            # Calculate summary length based on document category
            summary_lengths = {"short": 3, "medium": 5, "long": 8}
            num_sentences = summary_lengths.get(doc_length, 3)

            # Ensure we don't try to summarize more sentences than we have
            num_sentences = min(num_sentences, len(sentences))

            sentence_scores = {}
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                word_freq = Counter(words)

                score = 0
                for word, freq in word_freq.items():
                    if word.isalpha():  # Only consider alphabetic words
                        tf = freq / len(words)
                        idf = math.log(len(sentences) / (self.dft(word, sentences) + 1))
                        score += tf * idf

                sentence_scores[sentence] = score

            summary_sentences = sorted(
                sentence_scores.items(), key=lambda x: x[1], reverse=True
            )[:num_sentences]
            summary = " ".join(s[0] for s in summary_sentences)

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary."

    async def extract_keywords(self, text: str, max_keywords: int) -> List[str]:
        """Extract domain-specific keywords from text."""
        try:
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words("english"))

            # Remove stopwords and non-alphabetic tokens
            words = [
                word
                for word in words
                if word.isalpha() and word not in stop_words and len(word) > 3
            ]

            word_freq = Counter(words)
            keywords = [word for word, _ in word_freq.most_common(max_keywords)]

            return keywords

        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    @staticmethod
    def categorize_document_length(num_pages: int) -> str:
        """Categorize document length based on page count."""
        if num_pages <= 3:
            return "short"
        elif num_pages <= 12:
            return "medium"
        else:
            return "long"

    @staticmethod
    def dft(term: str, sentences: List[str]) -> int:
        """Calculate document frequency of a term."""
        return sum(1 for sentence in sentences if term in sentence.lower())

    async def store_initial_metadata(self, metadata: Dict) -> Optional[str]:
        """Store initial metadata in MongoDB with duplicate checking."""
        try:
            # Update existing failed entry or create new one
            result = self.collection.update_one(
                {"filename": metadata["filename"]}, {"$set": metadata}, upsert=True
            )

            return result.upserted_id or result.matched_count

        except mongo_errors.DuplicateKeyError:
            logger.warning(f"Duplicate entry detected for {metadata['filename']}")
            return None
        except Exception as e:
            logger.error(f"Error storing metadata: {str(e)}")
            raise

    async def update_mongodb(self, doc_id: str, data: Dict) -> None:
        """Update MongoDB document."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.collection.update_one, {"_id": doc_id}, {"$set": data}
            )
        except Exception as e:
            logger.error(f"Error updating MongoDB: {str(e)}")
            raise


async def main():
    try:
        # Initialize processor with MongoDB connection
        processor = PDFProcessor(
            mongodb_uri="mongodb://localhost:27017/", db_name="pdf_processor"
        )

        # Clean up any existing duplicates first
        initial_count = await processor.cleanup_database()
        logger.info(f"Initial document count: {initial_count}")

        # Process PDFs in the specified folder
        folder_path = "Data"
        results = await processor.process_folder(folder_path)

        # Print processing results
        for result in results:
            if "error" not in result:
                print(f"Successfully processed: {result['metadata']['filename']}")
                print(f"Summary length: {len(result['summary'])} chars")
                print(f"Keywords: {result['keywords']}")
                print("-" * 50)
            else:
                print(f"Failed to process: {result['file_path']}")
                print(f"Error: {result['error']}")
                print("-" * 50)

    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        raise


if __name__ == "__main__":
    tracemalloc.start()
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"Current memory usage: {current / 10**6} MB")
    logger.info(f"Peak memory usage: {peak / 10**6} MB")
    logger.info(
        f"Total execution time with concurrency: {end_time - start_time} seconds"
    )
