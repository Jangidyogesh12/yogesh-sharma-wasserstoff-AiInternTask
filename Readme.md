# PDF Processor

This project is an asynchronous PDF processing tool that extracts text, generates summaries, and identifies keywords from PDF files. It uses MongoDB for storing processed document metadata and results.

## Features

- Asynchronous processing of multiple PDF files
- Text extraction from PDFs (limited to first 20 pages for large documents)
- Summary generation using TF-IDF scoring
- Keyword extraction
- MongoDB integration for storing and retrieving document metadata
- Duplicate prevention and database cleanup
- Concurrency control for efficient processing
- Memory usage and execution time tracking

## Requirements

- Python 3.7+
- MongoDB

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/pdf-processor.git
   cd pdf-processor
   ```

2. Install the required Python packages:

   ```
   pip install -r requirements.txt
   ```

3. Ensure MongoDB is installed and running on your system.

## Usage

1. Place your PDF files in a folder named `Data` in the project directory.

2. Run the main script:

   ```
   python PDF_processor/processor.py
   ```

3. The script will process all PDF files in the `Data` folder, extract text, generate summaries, and identify keywords.

4. Results will be stored in the MongoDB database and a summary will be printed to the console.

## Configuration

- MongoDB connection: Update the `mongodb_uri` and `db_name` in the `main()` function if your MongoDB setup differs from the default (localhost:27017).
- Concurrency: Adjust the `max_concurrent` parameter in the `process_folder()` method to control the number of PDFs processed simultaneously.

## Output

For each successfully processed PDF, the following information is stored in MongoDB and printed to the console:

- Filename
- File path
- File size
- Processing date
- Number of pages
- Document length category (short, medium, long)
- Summary
- Keywords
- Processing status

## Error Handling

The script includes comprehensive error handling and logging. Failed processing attempts are recorded in the database with an error message.

## Performance

The script tracks and logs:

- Total execution time
- Current and peak memory usage

## Contributing

Contributions to improve the PDF Processor are welcome. Please feel free to submit pull requests or create issues for bugs and feature requests.
