# Use Python official image as base
FROM python:3.9.1

# Set working directory in container
WORKDIR /.
# Copy requirements file first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Create and activate virtual environment
RUN python -m venv myenv
ENV PATH="/app/myenv/bin:$PATH"

# Create necessary directories for __pycache__ and .vscode if needed
RUN mkdir -p __pycache__ .vscode

# Command to run the application
CMD ["python", "-m", "PDF_processor"]