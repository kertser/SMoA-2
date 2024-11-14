from flask import jsonify
import logging
import os
import httpx
import configparser
import yaml
from sentence_transformers import SentenceTransformer,util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import warnings

# Ensure NLTK data is downloaded once
nltk.download('punkt')
nltk.download('stopwords')
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Load the multilingual bi-encoder model once globally (will be merged into a function later
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Global variable to cache the semantic model
semantic_model = None

def load_configuration(configFilePath):
    # Load configuration - ini files
    config = configparser.ConfigParser(interpolation=None)
    try:
        with open(configFilePath, 'r', encoding='utf-8') as configfile:
            config.read_file(configfile)
        return config
    except:
        logging.error("Error loading configuration file", exc_info=True)

def load_yaml_config(configFilePath):
    with open(configFilePath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

async def openai_post_request(messages, model_name, max_tokens, temperature):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # Log the URL and headers for debugging
    logging.info(f"Request URL: {url}")
    logging.info(f"Headers: {headers}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Check for successful request
        return response.json()
    except httpx.HTTPStatusError as http_err:
        logging.error(f"HTTP error: {http_err.response.status_code} - {http_err.response.text}")
        raise  # Re-raise the exception after logging
    except Exception as e:
        logging.error(f"Error during OpenAI request: {str(e)}")
        raise

def load_semantic_model():
    """
    Load the semantic model once and cache it for reuse.
    :return: Loaded semantic model.
    """
    global semantic_model
    if semantic_model is None:
        config = load_yaml_config("../config/models_config.yaml")
        semantic_model_name = config['models']['embedding_model_name'].get('model', 'all-MiniLM-L6-v2')  # Default model
        semantic_model = SentenceTransformer(semantic_model_name)
        logging.info(f"Semantic model '{semantic_model_name}' loaded.")
    return semantic_model

def semantic_comparator(prompt, context, threshold=0.7):
    """
    Compare prompt and context based on semantic embeddings.
    Returns similarity score and boolean result based on threshold.

    :param prompt: Incoming prompt for comparison.
    :param context: Incoming context for comparison.
    :param threshold: Threshold for similarity score (default=0.7).
    :return: Tuple containing similarity score and boolean result.
    """
    try:
        # Load or reuse the semantic model
        model = load_semantic_model()

        # Generate embeddings for prompt and context
        prompt_embedding = model.encode(prompt, convert_to_tensor=True)
        context_embedding = model.encode(context, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.cos_sim(prompt_embedding, context_embedding)[0][0].item()

        # Logging similarity score
        logging.info(f"📊 Similarity between prompt and context: {similarity:.2f} (threshold: {threshold})")

        # Determine if similarity meets the threshold
        result = similarity >= threshold
        return similarity, result

    except Exception as e:
        logging.error(f"Error in semantic_comparator: {str(e)}")
        return None, False

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + stopwords.words('russian'))
    return [word for word in tokens if word.isalnum() and word not in stop_words]

def keyword_match(prompt, response):
    prompt_words = set(preprocess_text(prompt))
    response_words = set(preprocess_text(response))
    if not prompt_words:
        return 0
    match_ratio = len(prompt_words.intersection(response_words)) / len(prompt_words)
    if match_ratio < 0.2:  # Penalty for low keyword match
        return match_ratio * 0.5
    return match_ratio

def length_penalty(response, optimal_length=50):
    length_ratio = len(response.split()) / optimal_length
    return 2 / (1 + pow(2.718, -length_ratio))

def evaluate_answer(incoming_prompt, operator_context, outgoing_response):
    """
    The algorithm evaluates the quality of a response based on semantic similarity, keyword matching, and length. It
    generates embeddings for the prompt, context, and response using a pre-trained transformer model and calculates
    cosine similarity scores to measure semantic relevance. A keyword match score is computed by comparing key terms in
    the prompt and response. A length penalty is applied to optimize response length.
    The final quality score is a weighted combination of similarity, keyword match, and length,
    with penalties for poor keyword matches.
    The response is then classified as "Excellent," "Good," "Satisfactory," or "Unsatisfactory."

    """

    # We will store and load the key values into the models_config.yaml later on
    try:
        embeddings = model.encode([incoming_prompt, operator_context, outgoing_response], convert_to_tensor=True)
        prompt_embedding, context_embedding, response_embedding = embeddings

        score_prompt_response = util.cos_sim(prompt_embedding, response_embedding).item()
        score_context_response = util.cos_sim(context_embedding, response_embedding).item()
        keyword_score = keyword_match(incoming_prompt, outgoing_response)

        relevance_score = (0.35 * score_prompt_response + 0.15 * score_context_response + 0.5 * keyword_score)
        length_score = length_penalty(outgoing_response)
        quality_score = relevance_score * length_score

        if keyword_score < 0.3:
            quality_score *= 0.7

        if quality_score > 0.75:
            return "Excellent"
        elif quality_score > 0.6:
            return "Good"
        elif quality_score > 0.4:
            return "Satisfactory"
        else:
            return "Unsatisfactory"
    except Exception as e:
        return None

""" Error Handling """

def handle_openai_error(e):
    logging.error(f"OpenAI API error: {str(e)}", exc_info=True)
    return create_error_response("OpenAI API error", 500)

def create_error_response(message, status_code):
    response = jsonify({"error": {"message": message}})
    response.status_code = status_code
    return response

def handle_request_error(message):
    logging.error(f"Request error: {message}", exc_info=True)
    return create_error_response(message, 400)

def handle_internal_error(e):
    logging.error(f"Internal server error: {str(e)}", exc_info=True)
    return create_error_response("Internal server error", 500)
