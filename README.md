# News Research Tool

The News Research Tool is an AI-powered application that allows users to input URLs of news articles and generate detailed answers based on the content of those articles. Built with Python and leveraging the `langchain` library, this tool utilizes OpenAI's GPT-3.5-turbo-1106 model for natural language processing and understanding.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [How It Works](#how-it-works)

## Overview

The News Research Tool processes news articles by extracting text content from provided URLs, creating embeddings for efficient information retrieval, and using a conversational AI model to generate answers based on user queries. The tool is designed to help users quickly gather insights from multiple news sources.

## Features

- **URL Processing**: Input up to three URLs of news articles to extract and analyze content.
- **Text Extraction and Splitting**: Extracts text content and splits it into chunks for better processing.
- **Embeddings Creation**: Generates embeddings for the extracted content using OpenAI's embeddings model and stores them in a FAISS vector store.
- **Query-Based Retrieval**: Allows users to ask questions and retrieves relevant information from the processed content.
- **Conversational AI**: Uses OpenAI's GPT-3.5-turbo-1106 model to provide detailed answers based on the retrieved information.

## Technologies Used

- **Python**: The main programming language used for development.
- **LangChain**: A Python library for building applications with large language models.
- **OpenAI GPT-3.5-turbo-1106**: The language model used for generating answers.
- **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search and clustering of dense vectors.
- **Streamlit**: A framework for building the web interface.
- **OpenAI API**: Used for creating embeddings and for language model interaction.

## Setup and Installation

To run the News Research Tool locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/AyushSojitra/News-Research-tool-using-LLM.git
    cd news-research-tool
    ```

2. **Install Required Dependencies**:  
    Ensure you have Python installed. Then, install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**:  
    Create a `.env` file in the project directory and add your OpenAI API key:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

4. **Run the Application**:  
    Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Input URLs**: In the Streamlit sidebar, enter up to three URLs of news articles you want to process.
2. **Process URLs**: Click the "Process URLs" button to extract and analyze the content from the provided URLs.
3. **Ask Questions**: Type a question related to the content in the "Question" input field, and the tool will provide a detailed answer based on the extracted information.

## How It Works

1. **Application Start**: The app initializes with Streamlit and sets up the OpenAI model.
   
2. **User Inputs URLs**: Users provide up to three URLs in the sidebar.

3. **Process URLs**: When "Process URLs" is clicked:
   - The `UnstructuredURLLoader` loads content from the URLs.
   - The content is split into manageable chunks using `RecursiveCharacterTextSplitter`.
   - Embeddings are created for each chunk using `OpenAIEmbeddings`.
   - The embeddings are stored in a FAISS vector store, which is saved as a pickle file.

4. **Query Input**: When a user inputs a query:
   - The FAISS vector store is loaded from the pickle file.
   - A similarity search is performed on the vector store to find relevant documents.
   - A conversational AI chain (`get_conversational_chain()`) generates a detailed answer using the GPT-3.5-turbo-1106 model.

