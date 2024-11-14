import logging
from api.utility import openai_post_request, load_yaml_config


class Operator:
    """Base class for all operators (agents) in the system."""

    def __init__(self, name, config):
        self.name = name
        self.model = config.get('model')
        self.context = config.get('context', "")
        self.max_tokens = config.get('max_tokens', 100)
        self.temperature = config.get('temperature', 0.7)

    async def run(self, task):
        """Execute the operator's task by interacting with the language model."""
        messages = [
            {"role": "system", "content": self.context},
            {"role": "user", "content": task}
        ]
        try:
            response = await openai_post_request(messages, self.model, self.max_tokens, self.temperature)
            logging.info(f"Response for {self.name} task '{task}': {response}")
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error in {self.name} operator for task '{task}': {e}")
            return f"Error occurred in {self.name}."


def load_operators_config(config_path="config/models_config.yaml"):
    """Load the entire YAML config to retrieve 'models' and 'operators' sections."""
    config = load_yaml_config(config_path)
    if 'operators' not in config:
        raise ValueError("No 'operators' section found in config file")
    return config


def load_operators(config_path="config/models_config.yaml"):
    """Load operators dynamically from the 'operators' section in models_config.yaml."""
    config = load_operators_config(config_path)
    operators_config = config.get("operators", {})

    if not operators_config:
        logging.error("No operators found in configuration")
        return {}

    operators = {}
    for name, settings in operators_config.items():
        try:
            operators[name.lower()] = Operator(name, settings)
            logging.info(f"Loaded operator: {name}")
        except Exception as e:
            logging.error(f"Failed to load operator {name}: {e}")
    return operators


async def find_best_operator_with_llm(task_description, operators_config):
    """Uses an LLM to determine the best operator for a given task description."""
    operator_descriptions = "\n".join([
        f"Operator: {name}\nDescription: {details.get('context', 'No description')}\n"
        for name, details in operators_config.get('operators', {}).items()
    ])

    prompt = f"""
Given the task below, select the most appropriate operator from the list. Only return the operator name in lowercase.

Task: {task_description}

Available Operators:
{operator_descriptions}

Response format: [operator_name]
"""

    try:
        response = await openai_post_request(
            messages=[
                {"role": "system", "content": "Select the most appropriate operator for the given task."},
                {"role": "user", "content": prompt}
            ],
            model_name="gpt-4",
            max_tokens=50,
            temperature=0.2
        )

        operator_name = response['choices'][0]['message']['content'].strip().lower()
        operator_name = operator_name.replace('[', '').replace(']', '')

        if operator_name in operators_config.get('operators', {}):
            logging.info(f"Selected operator '{operator_name}' for task '{task_description}'")
            return operator_name
        else:
            logging.warning(f"Selected operator '{operator_name}' not found in config")
            return "base"

    except Exception as e:
        logging.error(f"Error in operator selection: {e}")
        return "base"


# Initialize AGENT_OPERATORS globally, which will store all loaded operators
AGENT_OPERATORS = load_operators()