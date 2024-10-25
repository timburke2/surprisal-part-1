from bs4 import BeautifulSoup
from itertools import combinations
from pathlib import Path
from ranx import Qrels, Run, evaluate
from scipy.stats import pearsonr
import re
import json
import tqdm
import math
import argparse
import matplotlib as plt
import pandas as pd

def clean_text(text): 
    """
    Cleans and tokenizes the input text.

    This function removes HTML tags using BeautifulSoup, converts text to lowercase,
    splits it into tokens, removes punctuation from the start and end of tokens, and
    filters out tokens that are not alphabetic or are single characters (except 'a' and 'i').

    Parameters:
        text (str): The raw text string to be cleaned and tokenized.

    Returns:
         List of str: A list of cleaned and tokenized words.
    """

    soup = BeautifulSoup(text, "lxml")
    cleaned_text = soup.get_text(separator=" ")
    tokens = cleaned_text.lower().split()

    clean_tokens = []
    for token in tokens:
        # Remove punctuation at the start and end of the token
        stripped_token = re.sub(r"^[\"']+|[\"']+$", "", token)

        # Check if the token still contains alphabetic characters or an apostrophe
        if re.match(r"^[a-zA-Z']+$", stripped_token):
            # Filter out single characters unless they are "a" or "i"
            if len(stripped_token) > 1 or stripped_token in ('a', 'i'):
                clean_tokens.append(stripped_token)
            #TODO: More token filtering with regex?

    return clean_tokens

def tokenize_answers(answers_file='Answers.json', outfile='tokenized_answers.json'):
    """
    Tokenizes and extracts unique terms from a JSON file of answers.

    This function processes each answer by cleaning its text, extracting unique terms,
    and saving the results in a new JSON file.

    Parameters:
        answers_file (str): The file path of the answers JSON file.
        outfile (str): The file path to save the tokenized answers JSON file.

    Returns:
        str: The path to the tokenized answers JSON file.
    """

    with open(answers_file, 'r', encoding='utf-8') as infile:
        answers = json.load(infile)

        answer_data = {}
        # Iterate over each answer and create a dictionary entry for the cleaned text
        for answer in tqdm.tqdm(answers, desc='Tokenizing documents'):
            id = answer['Id']
            text = clean_text(answer['Text'])
            
            # Ensure unique terms by converting the token list to a set
            unique_terms = set(text)
            
            answer_data[id] = list(unique_terms)
    
    with open(outfile, 'w', encoding='utf-8') as outfile:
        json.dump(answer_data, outfile, ensure_ascii=False, indent=2)

    return outfile

def index_terms(tokenized_answers='tokenized_answers.json', outfile='term_index.json'):
    """
    Indexes terms from tokenized answers and calculates their information content.

    This function processes tokenized answers to create a term index, including 
    the document frequency and information (bits) for each term. The results are 
    saved in a JSON file.

    Parameters:
        tokenized_answers (str): The file path of the tokenized answers JSON file.
        outfile (str): The file path to save the term index JSON file.

    Returns:
        dict: A dictionary with terms as keys and their associated data (bits, docs).
    """
    with open(tokenized_answers, 'r', encoding='utf-8') as infile:
        answers = json.load(infile)
        collection_size = len(answers)

        term_data = {}
        for doc_id, terms in tqdm.tqdm(answers.items(), desc='Indexing terms'):
            for term in terms:
                if term not in term_data:
                    term_data[term] = {'Bits': 0, 'Docs': set([doc_id])}
                else:
                    term_data[term]['Docs'].add(doc_id)
        
        # Convert sets back to lists before saving to JSON
        for term in tqdm.tqdm(term_data, desc='Calculating Information'):
            term_subset = len(term_data[term]['Docs'])  # Calculate based on unique docs
            term_data[term]['Bits'] = -math.log2(term_subset / collection_size)
            term_data[term]['Docs'] = list(term_data[term]['Docs'])
    
    with open(outfile, 'w', encoding='utf-8') as out:
        json.dump(term_data, out, ensure_ascii=False, indent=4)

    return term_data

def get_data_from_json(infile):
    """
    Loads data from a JSON (.json) file.

    Parameters:
        infile (str): The file path to the JSON file.

    Returns:
        dict or list: The data object loaded from the JSON file.
    """

    with open(infile, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def rank_terms(text, term_index):
    """
    Ranks terms by their information content (bits) in descending order.

    Parameters:
        text (list): A list of terms to be ranked.
        term_index (dict): A dictionary with terms as keys and their data (bits, docs) as values.

    Returns:
        list: A list of tuples containing terms and their corresponding bits, sorted in descending order.
    """
    term_data = []
    for term in text:
        term_data.append((term, term_index[term]['Bits']))

    return sorted(term_data, key=lambda x: x[1], reverse=True)

def find_sufficient_queries(tokenized_answers, term_index, output_file='sufficient_queries.json'):
    """
    Finds sufficient queries to uniquely identify each document based on tokenized answers.

    This function calculates the minimal set of terms (queries) necessary to uniquely
    identify each document and stores the results in a JSON file.

    Parameters:
        tokenized_answers (dict): Tokenized answers with document IDs and associated terms.
        term_index (dict): A dictionary with terms as keys and their associated data (bits, docs).
        output_file (str): The file path to save the sufficient queries JSON file.

    Returns:
        str: The path to the sufficient queries JSON file.
    """
    sufficient_queries = {}
    N = len(tokenized_answers)
    max_info = math.log2(N)  # Max information content when a term appears in only one document

    for doc_id, terms in tqdm.tqdm(tokenized_answers.items(), desc='Finding sufficient queries'):
        ranked_terms = rank_terms(terms, term_index)
        query_terms = []
        total_bits = 0
        done = False

        # Handle empty documents
        if not terms:
            sufficient_queries[doc_id] = {
                'terms': query_terms,
                'total_bits': total_bits
            }
            continue

        # Check for a unique term with maximal information
        for term, bits in ranked_terms:
            if bits == max_info:
                query_terms = [term]
                total_bits = bits
                done = True
                break

        if not done:
            # Combine terms to uniquely identify the document
            candidate_terms = []
            union_docs = None
            for term, bits in ranked_terms:
                candidate_terms.append(term)
                term_docs = set(term_index[term]['Docs'])
                if union_docs is None:
                    union_docs = term_docs
                else:
                    union_docs = union_docs.intersection(term_docs)

                if len(union_docs) == 1 and doc_id in union_docs:
                    query_terms = candidate_terms
                    total_bits = sum([term_index[t]['Bits'] for t in query_terms])
                    done = True
                    break

            if not done:
                # Recompute union_docs for negation strategy
                union_docs = set(term_index[terms[0]]['Docs'])
                for term in terms[1:]:
                    union_docs = union_docs.intersection(set(term_index[term]['Docs']))
                union_docs.discard(doc_id)

                # Initialize total_bits with bits from the document's own terms
                query_terms.extend(terms)
                total_bits += sum([term_index[term]['Bits'] for term in terms])

                # Negation strategy: loop until we've isolated the document
                while union_docs:
                    irrelevant_terms = set()

                    # Collect terms from the remaining irrelevant documents
                    for irrelevant_doc_id in union_docs:
                        irrelevant_terms.update(set(tokenized_answers[irrelevant_doc_id]))

                    # Remove terms that are already in the target document
                    irrelevant_terms.difference_update(terms)

                    # If no irrelevant terms remain, stop the process
                    if not irrelevant_terms:
                        break

                    # Rank irrelevant terms by information content (lowest bits first)
                    ranked_irrelevant_terms = sorted(
                        rank_terms(list(irrelevant_terms), term_index), key=lambda x: x[1]
                    )

                    # Negate the lowest-information term
                    lowest_term, bits = ranked_irrelevant_terms[0]
                    P_t = 2 ** (-bits)
                    P_not_t = 1 - P_t

                    # Handle the edge case when P_not_t is zero
                    if P_not_t <= 0:
                        bits_not_t = float('inf')  # Infinite bits, cannot proceed further
                        print(f"Warning: Cannot negate term '{lowest_term}' as P_not_t is zero.")
                        break
                    else:
                        bits_not_t = -math.log2(P_not_t)

                    union_docs -= set(term_index[lowest_term]['Docs'])  # Remove documents using this term
                    query_terms.append(f"NOT {lowest_term}")
                    total_bits += bits_not_t

                    # If union_docs is empty, we've isolated the document
                    if not union_docs:
                        done = True
                        break

        # Store the sufficient query
        sufficient_queries[doc_id] = {
            'terms': query_terms,
            'total_bits': total_bits
        }

    # Save the sufficient queries to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(sufficient_queries, f_out, ensure_ascii=False, indent=2)
    
    return output_file

def get_insufficient_docs(sufficient_queries='sufficient_queries.json'):
    """
    Finds documents that have no sufficient query (i.e., zero total bits).

    Parameters:
        sufficient_queries (str): The file path of the sufficient queries JSON file.

    Returns:
        list: A list of document IDs with no sufficient query (total bits == 0).
    """
    docs = get_data_from_json(sufficient_queries)

    insufficient_docs = []
    for doc_id, _ in docs.items():
        if docs[doc_id]['total_bits'] == 0:
            insufficient_docs.append(doc_id)
    
    return insufficient_docs

def extract_relevant_docs(ground_truth_file='qrel_1.tsv', output_file='relevant_docs.json'):
    """
    Extracts relevant document IDs from a TSV file of ground truth relevance data.

    This function reads the ground truth file, which lists relevant documents by topic,
    and saves the relevant document IDs for each topic in a JSON file.

    Parameters:
        ground_truth_file (str): The file path of the TSV file with ground truth relevance data.
        output_file (str): The file path to save the relevant documents JSON file.

    Returns:
        str: The path to the relevant documents JSON file.
    """
    relevant_docs = {}

    # Open the TSV file
    with open(ground_truth_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            topic_id, _, answer_id, score = line.strip().split()
            score = int(score)

            if score > 0:
                if topic_id not in relevant_docs:
                    relevant_docs[topic_id] = set()  # Use a set to avoid duplicate answer_ids
                relevant_docs[topic_id].add(answer_id)
    
    # Convert sets to lists for JSON serialization
    relevant_docs = {k: list(v) for k, v in relevant_docs.items()}

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(relevant_docs, outfile, ensure_ascii=False, indent=2)

    return output_file

def compute_topic_max_queries(relevant_docs_file='relevant_docs.json',
                              sufficient_queries_file='sufficient_queries.json',
                              output_file='topic_max_queries.json'):
    """
    Computes the maximum queries for each topic based on relevant documents and sufficient queries.

    This function calculates the maximal set of queries needed to cover all relevant documents 
    for each topic and stores the results in a JSON file.

    Parameters:
        relevant_docs_file (str): The file path of the relevant documents JSON file.
        sufficient_queries_file (str): The file path of the sufficient queries JSON file.
        output_file (str): The file path to save the topic maximal queries JSON file.

    Returns:
        str: The path to the topic maximal queries JSON file.
    """
    # Load relevant documents per topic
    with open(relevant_docs_file, 'r', encoding='utf-8') as f:
        relevant_docs = json.load(f)

    # Load sufficient queries per document
    with open(sufficient_queries_file, 'r', encoding='utf-8') as f:
        sufficient_queries = json.load(f)

    topic_max_queries = {}

    for topic_id in tqdm.tqdm(relevant_docs, desc='Processing topics'):
        doc_ids = relevant_docs[topic_id]

        total_bits = 0.0  # Initialize total bits for AND operation
        maximal_query = []

        for doc_id in doc_ids:
            if doc_id not in sufficient_queries:
                continue

            doc_query = sufficient_queries[doc_id]
            terms = doc_query.get('terms', [])
            doc_bits = doc_query.get('total_bits', None)

            # Skip documents with no terms or undefined total bits
            if not terms or doc_bits is None:
                continue

            # Add the sufficient query for this document to the maximal query
            maximal_query.append(terms)

            # Add bits directly for AND operation
            total_bits += doc_bits

        # Store results for the topic
        topic_max_queries[topic_id] = {
            'bits': total_bits, 
            'maximal_query': maximal_query
        }

    # Save to output JSON file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(topic_max_queries, f_out, ensure_ascii=False, indent=2)

    return output_file

def search(query_terms, term_index):
    """
    Searches for documents that match the given query terms.

    Parameters:
        query_terms (list): A list of query terms, including negations (e.g., "NOT term").
        term_index (dict): A dictionary with terms as keys and their associated data (bits, docs).

    Returns:
        set: A set of document IDs that match the query terms.
    """
    union_docs = None
    all_docs = None

    for term in query_terms:
        if term.startswith("NOT "):
            # Negative term
            neg_term = term[4:]  # Remove the "NOT " prefix
            if neg_term in term_index:
                neg_docs = set(term_index[neg_term]['Docs'])
            else:
                neg_docs = set()  # Term not found; no documents to exclude
            if union_docs is None:
                if all_docs is None:
                    # Build the set of all document IDs
                    all_docs = set()
                    for term_info in term_index.values():
                        all_docs.update(term_info['Docs'])
                union_docs = all_docs - neg_docs
            else:
                # Remove neg_docs from the current set of documents
                union_docs -= neg_docs
        else:
            # Positive term
            term_docs = set(term_index[term]['Docs'])
            if union_docs is None:
                # Initialize union_docs with the documents containing the positive term
                union_docs = term_docs.copy()
            else:
                # Intersect the current set with documents containing the positive term
                union_docs &= term_docs

    if union_docs is None:
        # If no terms were processed, return an empty set
        return set()
    return union_docs

def find_common_terms(doc_ids, tokenized_answers):
    """
    Finds common terms across a list of document IDs.

    This function takes a list of document IDs and finds the terms that appear 
    in all of the corresponding tokenized answers.

    Parameters:
        doc_ids (list): A list of document IDs.
        tokenized_answers (dict): A dictionary with document IDs as keys and their associated terms.

    Returns:
        set: A set of terms common to all specified documents.
    """
    # Start with the terms from the first document
    common_terms = set(tokenized_answers[doc_ids[0]])
    for doc_id in doc_ids[1:]:
        # Intersect with terms from each subsequent document
        common_terms.intersection_update(set(tokenized_answers[doc_id]))
    return common_terms

def generate_ordered_subsets(sorted_data, min_bits=0, max_bits=0):
    """
    Generates ordered subsets of terms based on a range of information content.

    Parameters:
        sorted_data (list): A list of tuples containing terms and their bits.
        min_bits (float): The minimum allowed bits for a subset.
        max_bits (float): The maximum allowed bits for a subset.

    Returns:
        list: A list of valid subsets that meet the bit constraints.
    """
    valid_subsets = []

    for r in range(1, len(sorted_data) + 1):
        for subset in combinations(sorted_data, r):
            subset_sum = sum(value for _, value in subset)

            if max_bits != 0 and subset_sum > max_bits:
                continue

            if subset_sum >= min_bits:
                valid_subsets.append(subset)


    # Sort valid subsets by their sum of values
    valid_subsets = sorted(valid_subsets, key=lambda subset: sum(value for _, value in subset))

    return valid_subsets

def create_negated_index(term_index):
    """
    Creates a negated index for terms based on their bits (information content).

    This function computes the negated bits (information content) for each term 
    in the term index and returns a new dictionary with the negated values.

    Parameters:
        term_index (dict): A dictionary with terms as keys and their associated data (bits, docs).

    Returns:
        dict: A dictionary with terms as keys and their negated bits as values.
    """
    negated_index = {}
    
    for term, data in term_index.items():
        surprisal = data['Bits']
        probability = 2 ** -surprisal
        
        if probability == 1:
            negated_bits = float('inf')
        elif probability == 0:
            negated_bits = 0
        else:
            negated_probability = 1 - probability
            negated_bits = -math.log2(negated_probability)
        
        negated_index[term] = {
            'Bits': negated_bits
        }
    
    return negated_index

def compute_min_queries_simple(tokenized_answers, term_index, num_terms=8, output_file='min_queries_simple.json'):
    """
    Computes minimal queries for documents based on tokenized answers and term index.

    This function calculates minimal sets of terms (queries) necessary to uniquely
    identify each document, subject to a constraint on the number of terms, and 
    saves the results in a JSON file.

    Parameters:
        tokenized_answers (dict): Tokenized answers with document IDs and associated terms.
        term_index (dict): A dictionary with terms as keys and their associated data (bits, docs).
        num_terms (int): The maximum number of terms allowed in the minimal query.
        output_file (str): The file path to save the minimal queries JSON file.

    Returns:
        str: The path to the minimal queries JSON file.
    """
    docs = sorted(tokenized_answers.items(), key=lambda x: len(x[1]))
    N = len(docs)
    max_info = math.log2(N)  # Maximal information content
    MIN_INFO = 13.5 # Minimal information content

    trimmed_docs = []
    for doc_id, terms in docs:
        if not terms:
            continue

        query_terms = []
        ranked_terms = rank_terms(terms, term_index)

        # Check if the first term has maximal information content
        if ranked_terms[0][1] == max_info:
            query_terms.append(ranked_terms[0][0])

        for term in ranked_terms[1:]:
            if term[1] == max_info:
                continue
            query_terms.append(term[0])

        trimmed_docs.append((doc_id, query_terms))

    trimmed_docs = sorted(trimmed_docs, key=lambda x: len(x[1]))
    short_docs = [doc for doc in trimmed_docs if 0 < len(doc[1]) <= num_terms]

    queries = {}
    for doc_id, terms in tqdm.tqdm(short_docs, desc="Finding min query"):
        done = False

        ranked_terms = [(term, term_index[term]['Bits']) for term in terms]

        has_unique = ranked_terms[-1] == max_info

        if has_unique:
            # If the doc has a unique term, only try better subsets
            ordered_subsets = generate_ordered_subsets(set(sorted(ranked_terms, key=lambda x: x[1])), MIN_INFO, max_info)
        else:
            ordered_subsets = generate_ordered_subsets(set(sorted(ranked_terms, key=lambda x: x[1])), MIN_INFO)

        # Check each subset, lowest to highest, and return when if it identifies the doc
        for subset in ordered_subsets:
            query = [word for word, _ in subset]
            bits = [val for _, val in subset]
            results = search(query, term_index)
            if len(results) == 1 and doc_id in results:
                queries[doc_id] = {'Query': query, "Bits": sum(bits)}
                done = True
                break

        if not done:
            continue

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(queries, outfile, ensure_ascii=False, indent=2)

    return output_file

def rank_topics_to_json(topic_max_queries='topic_max_queries.json'):
    """
    Ranks topics by the total bits of their maximal query and saves the result to a JSON file.

    This function reads the maximal queries for each topic, sorts them by total bits,
    and saves the ranked results in a JSON file.

    Parameters:
        topic_max_queries (str): The file path of the topic maximal queries JSON file.

    Returns:
        str: The path to the ranked topic queries JSON file.
    """
    topic_queries = get_data_from_json(topic_max_queries)

    topic_queries = sorted(topic_queries.items(), key=lambda x : x[1]['bits'])
    sorted_topics = [(topic[0], topic[1]['bits']) for topic in topic_queries]
    
    with open('ranked_topic_queries.json', 'w', encoding='UTF-8') as outfile:
        topics = {}
        for topic in sorted_topics:
            topics[topic[0]] = topic[1]
        json.dump(topics, outfile, ensure_ascii=False, indent=1)

    return outfile

def find_correlation(qrel_file='qrel_1.tsv', result_binary='result_tfidf_1.tsv', topic_surprisals='ranked_topic_queries.json'):
    """
    Computes and visualizes the correlation between nDCG@5 scores from an IR evaluation and topic surprisal scores.

    This function:
    1. Loads the qrels and system results from files.
    2. Evaluates the system performance using multiple IR metrics.
    3. Extracts nDCG@5 scores and pairs them with topic IDs.
    4. Merges the nDCG@5 scores with the corresponding topic surprisal scores.
    5. Creates a scatter plot to visualize the relationship between nDCG@5 and surprisal scores.
    6. Calculates and prints the Pearson correlation coefficient and p-value between nDCG@5 and surprisal scores.

    Parameters:
        qrel_file (str): The file path to the qrels (ground truth relevance judgments) in TREC format.
        result_binary (str): The file path to the system results in TREC format.
        topic_surprisals (str): The file path to the JSON file containing topic surprisal scores.

    Returns:
        None: Displays a scatter plot and prints the Pearson correlation coefficient and p-value.
    """
    # Load your system results and qrels
    qrels = Qrels.from_file(qrel_file, kind="trec")
    run = Run.from_file(result_binary, kind="trec")


    metrics = ["precision@1", "precision@5", "ndcg@5", "mrr", "map"]
    results = evaluate(qrels, run, metrics, return_mean=False, make_comparable=True)

    # Get topic IDs from the run file
    topic_ids = list(run.run.keys())

    # Sort by a specific metric
    metric = 'ndcg@5'
    metric_scores = results[metric]

    # Pair topic IDs with their corresponding metric score
    topic_scores = list(zip(topic_ids, metric_scores))

    # Sort topics by their nDCG@5 score in descending order
    sorted_topics = sorted(topic_scores, key=lambda x: x[1], reverse=True)
    sorted_topics_df = pd.DataFrame(sorted_topics, columns=['topic_id', 'ndcg@5'])

    surprisal_rankings = json.load(open(topic_surprisals))
    surprisal_rankings_list = [[k, v] for k, v in surprisal_rankings.items()]
    surprisal_rankings = pd.DataFrame(surprisal_rankings_list, columns=['topic_id', 'surprisal_score'])

    merged_df = pd.merge(sorted_topics_df, surprisal_rankings, on='topic_id')

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.scatter(merged_df['surprisal_score'], merged_df['ndcg@5'], alpha=0.7)
    plt.title('Correlation between ndcg@5 and Surprisal Score')
    plt.xlabel('Surprisal Score')
    plt.ylabel('ndcg@5')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    ndcg_scores = merged_df['ndcg@5']
    surprisal_scores = merged_df['surprisal_score']
    correlation, p_value = pearsonr(ndcg_scores, surprisal_scores)
    print(f"Pearson correlation: {correlation}")
    print(f"P-value: {p_value}")

    return

def main(answers_file='Answers.json', topics_file='topics_1.json', qrel_file='qrel_1.tsv'):
    """
    The main function for running the IR system with precomputed or newly generated files.

    This function can tokenize answers, index terms, extract relevant documents, and compute
    sufficient, minimal, and maximal queries based on the specified parameters. It also supports
    loading precomputed files to save time on repeated runs.

    Parameters:
        answers_file (str): The file path of the answers JSON file.
        topics_file (str): The file path of the topics JSON file.
        qrel_file (str): The file path of the QREL file with ground truth relevance data.
    """
    parser = argparse.ArgumentParser(description="Run the IR system with TF-IDF or BM25")
    parser.add_argument('--init', action='store_true', default=False, 
                        help="Set to overwrite precomputed files (default: False)")
    parser.add_argument('--queries', action='store_true', default=False,
                        help="Set to overwrite precomputed files (default: False)")
    args = parser.parse_args()
    
    check_files = [Path('tokenized_answers.json').exists(), 
                   Path('term_index.json').exists()]
    
    if args.init or not all(check_files): # Create initial files from input
        tokenized_answers = tokenize_answers(answers_file)
        # Dict of answers and their terms
        tokenized_answers = get_data_from_json('tokenized_answers.json')

        term_index = index_terms('tokenized_answers.json')
        # Dict of terms and their data
        term_index = get_data_from_json('term_index.json')

    else: # Load preexisting files
        tokenized_answers = get_data_from_json('tokenized_answers.json')
        term_index = get_data_from_json('term_index.json')

    check_files = [Path('relevant_docs.json').exists(),
                   Path('sufficient_queries.json').exists(),
                   Path('topic_max_queries.json').exists(),
                   Path('min_queries_simple.json').exists(),
                   Path('ranked_topic_queries.json').exists()]

    if args.queries or not all(check_files): # Create query analysis files
        # Dict of topic IDs and relevant docs from qrel
        relevant_docs = extract_relevant_docs(qrel_file)

        # Dict of answer IDs and sufficient exact queries
        sufficient_queries = find_sufficient_queries(tokenized_answers, term_index)

        # Dict of topic IDs and sufficient exact queries and surprisal sums
        topic_max_queries = compute_topic_max_queries(relevant_docs, sufficient_queries)

        # Dict of answer IDs and minimal exact queries (WIP, interesting to browse)
        minimal_queries = compute_min_queries_simple(tokenized_answers, term_index)

        # Ranked dict of topic IDs and surprisal sums, for analysis
        ranked_topics = rank_topics_to_json()

if __name__=="__main__":
    main()