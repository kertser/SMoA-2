import logging
from sentence_transformers import SentenceTransformer, util
from api.utility import openai_post_request, load_yaml_config

# Load configuration settings for reasoning tasks
config = load_yaml_config("config/models_config.yaml")
decomposition_model_config = config['models'].get('decomposition_model', {})
semantic_classification_model_config = config['models'].get('semantic_classification_model', {})
stop_classifier_model_config = config['models'].get('stop_classifier', {})

# Initialize the embedding model once
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

async def simplicity_classifier(prompt):
    """Classify if a prompt is simple or complex."""
    messages = [
        {"role": "system", "content": "Determine if the following prompt is simple or complex."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = await openai_post_request(
            messages,
            stop_classifier_model_config.get('model', "gpt-4"),
            stop_classifier_model_config.get('max_tokens', 50),
            stop_classifier_model_config.get('temperature', 0.3)
        )
        result = response['choices'][0]['message']['content'].strip().lower()
        return result in ["simple", "yes"]  # Returns True for simple, False for complex
    except Exception as e:
        logging.error(f"Error in simplicity_classifier: {e}")
        return False  # Default to complex if there's an error

async def splitter(complex_prompt):
    """Decompose a complex prompt into subtasks using GPT-4 decomposition model."""
    messages = [
        {"role": "system", "content": decomposition_model_config.get('context', "Decompose the prompt into subtasks.")},
        {"role": "user", "content": complex_prompt}
    ]
    try:
        response = await openai_post_request(
            messages,
            decomposition_model_config.get('model', "gpt-4"),
            decomposition_model_config.get('max_tokens', 500),
            decomposition_model_config.get('temperature', 0.7)
        )
        split_prompts = response['choices'][0]['message']['content'].split("\n")
        logging.info(f"Splitter result: {split_prompts}")
        return [prompt.strip() for prompt in split_prompts if prompt.strip()]
    except Exception as e:
        logging.error(f"Error in splitter: {e}")
        return [complex_prompt]  # Fallback to original prompt if decomposition fails


async def semantic_classifier(task):
    """Classify the semantic type of a task (e.g., question, command, statement)."""
    messages = [
        {"role": "system",
         "content": semantic_classification_model_config.get('context', "Classify the task's semantic type.")},
        {"role": "user", "content": task}
    ]
    try:
        response = await openai_post_request(
            messages,
            semantic_classification_model_config.get('model', "gpt-4"),
            semantic_classification_model_config.get('max_tokens', 100),
            semantic_classification_model_config.get('temperature', 0.5)
        )
        classification = response['choices'][0]['message']['content'].strip().lower()
        logging.info(f"Semantic classification for '{task}': {classification}")
        return classification
    except Exception as e:
        logging.error(f"Error in semantic classification: {e}")
        return "unknown"  # Fallback classification


def keyword_match(prompt, response):
    """Calculates a simple keyword match score between the prompt and response."""
    prompt_keywords = set(prompt.lower().split())
    response_keywords = set(response.lower().split())
    common_keywords = prompt_keywords & response_keywords
    match_ratio = len(common_keywords) / len(prompt_keywords) if prompt_keywords else 0
    return match_ratio


def length_penalty(response):
    """Applies a penalty based on response length with an optimal range of around 100-150 tokens."""
    optimal_length = 125
    response_length = len(response.split())
    penalty = 1 / (1 + abs(response_length - optimal_length) / optimal_length)
    return penalty


def evaluate_answer(incoming_prompt, operator_context, outgoing_response):
    """Evaluates the quality of an outgoing response based on relevance, keywords, and length."""
    try:
        # Generate embeddings
        embeddings = model.encode([incoming_prompt, operator_context, outgoing_response], convert_to_tensor=True)
        prompt_embedding, context_embedding, response_embedding = embeddings[0], embeddings[1], embeddings[2]

        # Calculate similarity scores
        score_prompt_response = util.cos_sim(prompt_embedding, response_embedding).item()
        score_context_response = util.cos_sim(context_embedding, response_embedding).item()

        # Keyword match score
        keyword_score = keyword_match(incoming_prompt, outgoing_response)

        # Weighted relevance score
        relevance_score = 0.35 * score_prompt_response + 0.15 * score_context_response + 0.5 * keyword_score

        # Length penalty
        length_score = length_penalty(outgoing_response)
        quality_score = relevance_score * length_score

        # Penalty for low keyword match
        if keyword_score < 0.3:
            quality_score *= 0.7

        # Classification
        if quality_score > 0.75:
            classification = "Excellent"
        elif quality_score > 0.6:
            classification = "Good"
        elif quality_score > 0.4:
            classification = "Satisfactory"
        else:
            classification = "Unsatisfactory"

        logging.info(f"Classification: {classification} (Quality Score: {quality_score:.4f})")
        return classification
    except Exception as e:
        logging.error(f"Error in evaluate_answer: {e}")
        return "Error"


async def stop_classifier(outputs, prompt, context):
    """
    Uses the quality score of outputs to determine if further processing is needed.
    An output of "Unsatisfactory" will trigger further processing.
    """
    for response in outputs:
        classification = evaluate_answer(prompt, context, response)
        if classification == "Unsatisfactory":
            logging.info("Further processing required due to unsatisfactory quality.")
            return False  # Indicates further processing is required

    logging.info("Responses meet quality standards, no further processing needed.")
    return True  # Indicates no further processing required


def aggregator(results):
    """Combine multiple outputs from operators into a final cohesive answer."""
    try:
        combined_answer = "\n\n".join(results)
        logging.info("Aggregation complete.")
        return combined_answer
    except Exception as e:
        logging.error(f"Error in aggregator: {e}")
        return "Error in combining results."


async def process_prompt(prompt, context=""):
    """Full reasoning pipeline that applies decomposition, classification, and aggregation to the prompt."""
    tasks = await splitter(prompt)
    results = []

    for task in tasks:
        task_type = await semantic_classifier(task)
        if task_type == "unknown":
            logging.warning(f"Could not classify task: '{task}'.")
        else:
            logging.info(f"Task '{task}' classified as '{task_type}'.")

        # Placeholder result for each task, as operators are not fully implemented
        result = f"Processed {task_type} task: {task}"
        results.append(result)

    # Check if further processing is needed based on quality of results
    if await stop_classifier(results, prompt, context):
        final_result = aggregator(results)
    else:
        final_result = aggregator(results) + " (Further processing applied)"

    return final_result
