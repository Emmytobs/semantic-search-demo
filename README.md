# Semantic Search Demo

This project demonstrates the power of **Semantic Search** using vector embeddings, compared to traditional keyword-based search. It uses **ChromaDB** as a vector store and **SentenceTransformers** for generating embeddings.

## Features

-   **Semantic Search**: Finds relevant documents based on meaning, not just exact word matches (e.g., "soccer" finds "football").
-   **Keyword Search Baseline**: A traditional keyword matching implementation for comparison.
-   **Stop Word Filtering**: Keyword search effectively filters out common stop words to improve accuracy.
-   **Two User Interfaces**:
    -   **Jupyter Notebook**: Interactive notebook with `ipywidgets` UI.
    -   **Web GUI**: A modern, standalone web interface built with **Gradio**.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Emmytobs/COMP_4750_NLP_Software_Project.git
    cd COMP_4750_NLP_Software_Project
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Option 1: Desktop GUI (Recommended)
Launch the standalone web interface:
```bash
python search_gui.py
```
This will open a local web page in your browser where you can enter queries and see side-by-side results.

### Option 2: Jupyter Notebook
Run the interactive notebook:
```bash
jupyter notebook semantic_search.ipynb
```
Run all cells to initialize the models and database. An interactive search widget will appear at the bottom of the notebook.

### Option 3: Verification Script
Run a headless verification script to see pre-defined test cases:
```bash
python verify_demo.py
```

## Project Structure

-   `documents.json`: The corpus of sample blog articles.
-   `semantic_search.ipynb`: Main project notebook with logic and UI.
-   `search_gui.py`: Gradio-based GUI application.
-   `verify_demo.py`: Python script for automated verification.
-   `requirements.txt`: List of Python dependencies.

## Technologies

-   **Python 3**
-   **ChromaDB**: Open-source embedding database.
-   **SentenceTransformers**: Framework for state-of-the-art text and image embeddings.
-   **Gradio**: For building the web interface.
-   **Pandas**: For data handling.
