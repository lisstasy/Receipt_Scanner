# Receipt Data Extraction, Parsing, and Expenses Analysis Project

In this project, I developed a comprehensive system for receipt data extraction, parsing, and expense analysis using LLMs capabilities. Here's a breakdown of the key components and technologies involved:

## 1. Optical Character Recognition (OCR) with PaddleOCR:
- Utilized computer vision techniques and the PaddleOCR library for efficient text extraction from receipt images.
- Configured the PaddleOCR model for the Spanish language and enabled GPU acceleration for faster processing.

## 2. Structured Data Extraction using GPT-3.5-turbo API:
- Defined a Pydantic schema to represent the structured data format for receipts.
- Integrated with LangChain's with_structured_output method to pass the schema and leverage the GPT-3.5-turbo API for structured output generation.
- Applied few-shot inference techniques to ensure accurate extraction and categorization of receipt information.

## 3. Data Processing and Analysis:
- Transformed the extracted data into a well-structured JSON object representing various aspects of the receipt, including store details, itemized purchases, dates, and payment methods.
- Utilized Pandas for data manipulation, enabling seamless conversion of structured data into DataFrame format for analysis.

## 4. Interactive Visualization Dashboards with Plotly and Dash:
- Developed interactive visualization dashboards using Plotly for charting and Dash for web application development.
- Enabled users to explore and analyze expense trends, category-wise spending, and budget tracking in a user-friendly interface.
- Implemented features for users to edit and confirm receipt data, enhancing usability and accuracy.

## 5. Technologies Used:
- Python: Leveraged Python programming language for all development tasks.
- PaddleOCR: Integrated PaddleOCR library for OCR functionality, enabling accurate text extraction from receipt images.
- Pydantic: Defined a schema for structured data representation, ensuring data integrity and consistency.
- LangChain with GPT-3.5-turbo API: Employed LangChain's capabilities along with GPT-3.5-turbo API for structured output generation and few-shot inference.
- Pandas: Utilized Pandas library for data manipulation and DataFrame operations.
- Plotly and Dash: Developed interactive visualization dashboards using Plotly for charting and Dash for web application development.
