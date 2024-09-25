import logging
from api.utility import openai_post_request
from api.utility import load_yaml_config, semantic_comparator

class Operator:
    # Base Operator Class
    def __init__(self, name, model, context, max_tokens, temperature):
        """
        Base Class Operator.
        :param name: operator name.
        :param model: LLM model name.
        :param context: incoming context for the model.
        :param max_tokens: max tokens for the model.
        :param temperature: Max temperature for randomness of output.
        """
        self.name = name
        self.model = model
        self.context = context
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def run(self, user_input, context=None):
        """
        Running the operator with the given user input and context.
        :param user_input: Input from the user.
        :param context: Context for the model (optional).
        :return: Response from the model.
        """
        try:
            messages = [
                {"role": "system", "content": context if context else self.context},
                {"role": "user", "content": user_input}
            ]

            response = await openai_post_request(messages, self.model, self.max_tokens, self.temperature)
            return response['choices'][0]['message']['content']
        except Exception as e:
            return self.handle_error(e)

    def handle_error(self, error):
        """
        Error handling for the operator.
        :param error: Error message.
        :return: User-friendly error message.
        """
        logging.error(f"Error in {self.name}: {error}")
        return f"An error occurred while processing your request in {self.name}. Please try again later."

    def set_model(self, model_name):
        """
        Defining the model for the operator.
        :param model_name: Model name.
        :return: None.
        """
        self.model = model_name


class GenericOperator(Operator):
    # Derived from the base class
    def __init__(self, operator_type):
        """
        Initializes the Generic Operator by loading the specific operator type from the config.

        :param operator_type: The type of the operator (e.g., 'comparison', 'definition', 'analysis').
        """
        # Load configuration from the YAML file
        config = load_yaml_config("config/models_config.yaml")

        # Extract settings for the specified operator type
        operator_settings = config['operators'].get(operator_type)

        # Use default settings if specific settings for operator_type are not found
        if not operator_settings:
            raise ValueError(f"Operator type '{operator_type}' not found in the configuration.")

        model = operator_settings.get('model', config['operators']['base']['model'])
        temperature = operator_settings.get('temperature', config['operators']['base']['temperature'])
        max_tokens = operator_settings.get('max_tokens', config['operators']['base']['max_tokens'])
        context = operator_settings.get('context', config['operators']['base']['context'])

        # Initialize the base class with appropriate parameters
        super().__init__(
            name=f"{operator_type.capitalize()}Operator",
            model=model,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature
        )

    async def run(self, user_input, context=None, redundancy=1):
        """
        Run the operator multiple times for redundancy and reliability.

        :param user_input: Input from the user.
        :param context: Optional, specific context for the operator.
        :param redundancy: Number of times to run the operator (default is 1).
        :return: Aggregated response from the redundant runs.
        """
        results = []

        for _ in range(redundancy):
            result = await super().run(user_input, context)
            results.append(result)

        if redundancy > 1:
            return self.aggregate_results(results)
        else:
            return results[0]

    @staticmethod
    def aggregate_results(results, threshold=0.7):
        """
        Aggregate the results based on semantic similarity.
        :param results: List of results to aggregate.
        :param threshold: Similarity threshold to compare results (default is 0.7).
        :return: The most consistent result based on semantic similarity.
        """

        if len(results) == 0:
            return None

        # To store similarity scores for each result
        similarity_matrix = []

        # Compare each result with every other result and calculate similarity
        for i in range(len(results)):
            current_result_similarities = []
            for j in range(len(results)):
                if i != j:
                    similarity, _ = semantic_comparator(results[i], results[j], threshold)
                    current_result_similarities.append(similarity)

            # Calculate the average similarity of the current result with all others
            if len(current_result_similarities) > 0:
                avg_similarity = sum(current_result_similarities) / len(current_result_similarities)
            else:
                avg_similarity = 0  # In case no similarities were calculated

            similarity_matrix.append(avg_similarity)

        # Select the result with the highest average similarity
        best_result_idx = similarity_matrix.index(max(similarity_matrix))

        return results[best_result_idx]


"""
Usage example:
from operators import GenericOperator

# Create an operator dynamically for 'comparison'
operator = GenericOperator('comparison')
user_input = "Compare solar energy and wind energy."

# Run the operator 3 times for redundancy
result = await operator.run(user_input, redundancy=3)

print(result)
"""