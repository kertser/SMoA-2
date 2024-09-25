from api.utility import load_yaml_config, openai_post_request
from api.utility import evaluate_answer

async def aggregator(prompts):
    """
    Combine and summarize multiple input prompts using aggregation_model.

    :param prompts: A list of simple input prompts.
    :return: Summarized output.
    """
    config = load_yaml_config("../config/models_config.yaml")
    aggregation_model = config['models']['aggregation_model'].get('model')
    max_tokens = config['models']['aggregation_model'].get('max_tokens')
    temperature = config['models']['aggregation_model'].get('temperature')
    context = config['models']['aggregation_model'].get('context')

    combined_prompt = "\n".join(prompts)
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": combined_prompt}
    ]

    try:
        response = await openai_post_request(messages, aggregation_model, max_tokens, temperature)
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error in aggregator: {str(e)}")
        return None


async def splitter(complex_prompt):
    """
    Split a complex prompt into simpler lexico-semantic elements using decomposition_model.

    :param complex_prompt: The complex input prompt.
    :return: A list of simpler prompts.
    """

    config = load_yaml_config("../config/models_config.yaml")
    decomposition_model = config['models']['decomposition_model'].get('model')
    max_tokens = config['models']['decomposition_model'].get('max_tokens')
    temperature = config['models']['decomposition_model'].get('temperature')
    context = config['models']['decomposition_model'].get('context')

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": f"Break down this complex task: {complex_prompt}"}
    ]

    try:
        response = await openai_post_request(messages, decomposition_model, max_tokens, temperature)
        split_prompts = response['choices'][0]['message']['content'].split("\n")
        return [prompt.strip() for prompt in split_prompts if prompt.strip()]
    except Exception as e:
        logging.error(f"Error in splitter: {str(e)}")
        return None


async def classify_prompt(prompt):
    """
    Classify the given prompt as either simple or complex using a defined model.

    :param prompt: The input prompt to classify.
    :return: "simple" or "complex".
    """

    config = load_yaml_config("../config/models_config.yaml")
    classification_model = config['models']['classification_model'].get('model')
    max_tokens = config['models']['classification_model'].get('max_tokens')
    temperature = config['models']['classification_model'].get('temperature')
    context = config['models']['classification_model'].get('context')

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": f"Is this task simple or complex: {prompt}"}
    ]

    try:
        response = await openai_post_request(messages, classification_model, max_tokens, temperature)
        classification = response['choices'][0]['message']['content'].strip().lower()
        return "complex" if "complex" in classification else "simple"
    except Exception as e:
        logging.error(f"Error in classify_prompt: {str(e)}")
        return None


async def good_reasoning(incoming_prompt: str, operator_context: str, outgoing_response: str) -> bool:
    """
    Evaluate the quality of reasoning in the operator's response based on the given prompt and context.

    :param incoming_prompt: The user's input prompt.
    :param operator_context: The context provided to the operator.
    :param outgoing_response: The operator's response to evaluate.
    :return: True if the response is classified as 'Good' or 'Excellent', False otherwise.
    """
    classification = evaluate_answer(incoming_prompt, operator_context, outgoing_response)

    if classification in ["Good", "Excellent"]:
        return True

    return False