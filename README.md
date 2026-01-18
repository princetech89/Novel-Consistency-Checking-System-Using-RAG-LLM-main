# Novel Consistency Checker

This project is a system for checking the consistency of claims against a large text, such as a novel. It uses a Retrieval Augmented Generation (RAG) approach with a Large Language Model (LLM) to achieve this.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set environment variables:**
    You will need to set the following environment variables directly in your shell or CI/CD environment.

    *   `OPENAI_API_KEY`: Your OpenAI API key.
    *   `PINECONE_API_KEY`: Your Pinecone API key.
    *   `INDEX_NAME`: The name of the Pinecone index you want to use.

    For example, in a bash shell, you would do:
    ```bash
    export OPENAI_API_KEY="your-openai-api-key"
    export PINECONE_API_KEY="your-pinecone-api-key"
    export INDEX_NAME="your-index-name"
    ```

    **For GitHub Actions:**

    If you are using GitHub Actions, you should set these as secrets in your repository settings.

## Usage

1.  **Index the novel:**
    ```bash
    python main.py
    ```
    This will chunk the novel, embed the chunks, and store them in a Pinecone index.

2.  **Run the interface:**
    ```bash
    streamlit run interface.py
    ```
    This will start a web interface where you can enter claims and see the consistency results.
