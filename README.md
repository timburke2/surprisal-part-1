# Surprise! An Information Theory Approach to IR Evaluation

## Overview

This project explores the application of **surprisal**, a concept from Information Theory, to **Information Retrieval (IR)** systems. The primary goal is to quantify the "difficulty" of retrieving documents in a collection based on the **surprisal** of terms within queries. Using surprisal, the code generates **Exact Queries** that uniquely identify documents, measuring the total amount of information required.

The project introduces key concepts such as:

- **Surprisal**: A measure of how much a term reduces the search space (in bits).
- **Exact Queries**: Boolean queries that return a specific document or set of documents.
- **Minimal Exact Queries (MEQs)**: Queries that retrieve a document using the least possible information.
- **Sufficient Exact Queries (SEQs)**: Queries that are sufficient but not necessarily minimal in information content.

By calculating these metrics, the project aims to classify the difficulty of topics within a document collection and evaluate the effectiveness of IR models.

## Motivation

Traditional IR models like **TF-IDF** and **BM25** infer document relevance based on word similarity, which often ignores semantic meaning. When a ground truth file asserts a set of documents is relevant to a shared topic, it may be the case that there is a high degree of varience between relevant sets in terms of how similar the documents are to eachother. Two topics may differ greatly in how difficult it is to **isolate the exact set of relevant documents**, or at least rank these higher than less relevant results. This project addresses this limitation by using **surprisal as a quantifier** of this prinicple, based on the reduction of uncertainty in a search space. The hypothesis is that **topics with higher surprisal are harder to retrieve** perfect results for, and this difficulty correlates with lower performance on IR evaluation metrics such as **nDCG**, **MAP**, and **MRR**.

## Project Components

### 1. **Tokenization Pipeline**

- **`tokenize_answers()`**: Cleans and tokenizes documents, extracting unique terms. It outputs a JSON file with tokenized answers.

### 2. **Term Indexing**

- **`index_terms()`**: Indexes terms by document frequency and calculates each term's surprisal (in bits). It outputs a JSON file that links terms to the documents containing them and their corresponding surprisal values.

### 3. **Exact Query Generation**

- **`find_sufficient_queries()`**: Generates **Sufficient Exact Queries (SEQs)** for each document, identifying a minimal set of terms that uniquely isolate the document.
- **`compute_min_queries_simple()`**: Attempts to generate **Minimal Exact Queries (MEQs)**, which use the least information possible to uniquely identify a document.
- **`compute_topic_max_queries()`**: For each topic, this function calculates the maximal query, which retrieves all documents relevant to the topic with a boolean OR search of the SEQs for each document.

### 4. **Surprisal-Based Topic Ranking**

- **`rank_topics_to_json()`**: Ranks topics by their total surprisal (in bits) and outputs the ranking to a JSON file, which can be used for performance analysis in IR systems.

### 5. **IR Model Testing**

- **Evaluation Models**: The project includes a basic vector space model and uses external models like PyTerrier's TF-IDF and BM25. Surprisal values are correlated with IR metrics like **nDCG**, **MAP**, and **MRR** to test the hypothesis that higher surprisal topics are harder to retrieve.

## Installation

1. Clone the repository:

    Clone the repository or download the code files, and ensure all the necessary Python files are in your project directory.

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure the necessary data files (e.g., `Answers.json`, `qrel_1.tsv`) are available in the project directory.

## Usage

1. **Full Run**

    ```bash
    python main.py
    
2. **Tokenize Documents**:

    ```bash
    python main.py --init
    ```

    This command tokenizes the documents and generates initial files like `tokenized_answers.json` and `term_index.json`.

3. **Generate Queries and Surprisal**:

    ```bash
    python main.py --queries
    ```

    This command generates **Sufficient Exact Queries (SEQs)**, computes topic surprisal, and ranks the topics based on their surprisal values.

4. **Evaluate Models**:  
   You can run the evaluation models (e.g., PyTerrier's TF-IDF, BM25, or vectimsearch (https://github.com/timburke2/vector-search.git)) and compare their performance with the surprisal-based ranking. The find_correlation() function expects a qrel file, the result binary in TSV, and the ranked topic queries file.

## Key Files

- `tokenized_answers.json`: Contains the tokenized version of the answers (documents), with unique terms extracted for each.
- `term_index.json`: Contains the term index with each term's document frequency and surprisal.
- `sufficient_queries.json`: Contains the sufficient queries (SEQs) for each document.
- `topic_max_queries.json`: Contains maximal queries and surprisal sums for each topic.
- `ranked_topic_queries.json`: Ranks topics by total surprisal and is used for correlation analysis with IR metrics.

## Methodology

1. **Surprisal Calculation**:
   - Surprisal \( S(t) \) for a term \( t \) is calculated as:

        S(t) = - log_2 ( term frequency / collection size )

     Higher surprisal values indicate terms that more strongly reduce the search space.

2. **Exact Queries**:
   - The goal is to find the minimal set of terms (queries) that uniquely identify a document or a set of documents relevant to a topic. Exact queries are classified as:
     - **Maximal Exact Query (MEQ)**: A query using all terms in the document.
     - **Sufficient Exact Query (SEQ)**: Any query that uniquely identifies a document.
     - **Minimal Exact Query (MEQ)**: The query that identifies a document using the least information.

3. **Topic Surprisal**:
   - The total surprisal of a topic is the sum of the surprisal values of all documents relevant to that topic. This provides a metric for topic difficulty, with higher surprisal indicating more complex or harder-to-retrieve topics.

## Conclusion

This project demonstrates the potential of using **surprisal** and **information theory** concepts to improve the evaluation of information retrieval systems. By calculating the exact amount of information required to retrieve documents, we gain insights into the difficulty of topics and the performance of retrieval models. Although early results show weak positive correlations between surprisal and traditional IR metrics, the concept holds promise for further exploration.
