import networkx as nx
import logging
import json
import re
import asyncio
import openai
from operators.operators import AGENT_OPERATORS, find_best_operator_with_llm, load_operators_config
from api.utility import openai_post_request, load_yaml_config

class AgentGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.agents = AGENT_OPERATORS
        self.operators_config = load_operators_config()
        self.task_counter = 0

        config = load_yaml_config("config/models_config.yaml")
        if 'models' not in config or 'identification_model' not in config['models']:
            raise KeyError("The 'models -> identification_model' configuration is missing in models_config.yaml.")
        self.identification_config = config['models']['identification_model']

    def reset_graph(self):
        """Clear the graph and reset counters for each new query."""
        self.graph.clear()
        self.task_counter = 0

    def create_node_id(self):
        """Create unique node IDs."""
        self.task_counter += 1
        return f"task_{self.task_counter}"

    async def build_task_graph(self, input_prompt):
        """Construct the task graph for each prompt."""
        self.reset_graph()
        logging.info(f"Processing user input: {input_prompt}")

        # Initialize the graph with start node
        start_node = "start"
        self.graph.add_node(start_node, label="User Input", type="start")

        # Add prompt node
        prompt_node = "prompt"
        self.graph.add_node(prompt_node, label=input_prompt, type="prompt")
        self.graph.add_edge(start_node, prompt_node, label="Initial Input")

        # Decompose tasks
        tasks = await self.decompose_task(input_prompt)
        if not tasks:
            logging.warning("No tasks decomposed from the input.")
            return {}, "No tasks generated from input.", "", "N/A"

        context = f"Initial prompt: {input_prompt}\n\n"
        results = {}
        task_nodes = {}

        # Create nodes for each task
        for task in tasks:
            task_id = task['id']
            task_desc = task['description']
            task_node_id = f"task_{task_id}"
            self.graph.add_node(task_node_id, label=task_desc, type="task")
            task_nodes[task_id] = task_node_id

        # Create edges based on dependencies
        for task in tasks:
            task_id = task['id']
            dependencies = task.get('dependencies', [])
            if not dependencies:
                # If no dependencies, connect from prompt_node
                self.graph.add_edge(prompt_node, task_nodes[task_id], label="Start Task")
            else:
                for dep_id in dependencies:
                    self.graph.add_edge(task_nodes[dep_id], task_nodes[task_id], label="Depends On")

        # Process tasks
        for task in tasks:
            task_id = task['id']
            task_desc = task['description']
            task_node_id = task_nodes[task_id]

            # Process task with appropriate agent
            try:
                agent_type = await find_best_operator_with_llm(task_desc, self.operators_config)
                agent = self.agents.get(agent_type)
                if agent:
                    result = await agent.run(task_desc)
                    results[task_id] = result
                    context += f"Task {task_id}: {task_desc}\nResult: {result}\n\n"
                else:
                    logging.warning(f"No agent found for task: {task_desc}")
            except ValueError as e:
                logging.warning(f"{e}. Skipping task.")

        # Aggregate final results
        final_answer = self.aggregate_results(results, tasks)
        summary_answer = final_answer if final_answer else "(Further processing applied)"
        detailed_path = context + "\nFinal Answer:\n" + final_answer
        quality_score = self.evaluate_quality(final_answer)

        logging.info(f"Final Answer: {final_answer}")

        # Add final output node
        final_node = "final_output"
        self.graph.add_node(final_node, label="Final Output", type="end")

        # Find all sink nodes (nodes with no outgoing edges except the final node)
        sink_nodes = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0 and node != final_node and node != 'start' and node != 'prompt']

        # Connect all sink nodes to the final output node
        for node in sink_nodes:
            self.graph.add_edge(node, final_node, label="Finalize")

        # Now, the graph includes a final output node connected to the last tasks
        return self.get_graph_data(), summary_answer, detailed_path, quality_score

    def aggregate_results(self, results, tasks):
        """Aggregate task results into a final answer."""
        # Sort tasks based on dependencies for proper aggregation
        sorted_tasks = self.topological_sort(tasks)
        final_result = "\n".join([f"Task {task['id']} Result: {results.get(task['id'], 'No result')}" for task in sorted_tasks])
        return final_result if final_result else "No results generated."

    def topological_sort(self, tasks):
        """Perform topological sort on tasks based on dependencies."""
        task_dict = {task['id']: task for task in tasks}
        graph = nx.DiGraph()
        for task in tasks:
            graph.add_node(task['id'])
            for dep in task.get('dependencies', []):
                graph.add_edge(dep, task['id'])
        sorted_task_ids = list(nx.topological_sort(graph))
        return [task_dict[tid] for tid in sorted_task_ids]

    async def decompose_task(self, prompt):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task planner that decomposes user prompts into subtasks with dependencies. "
                    "Provide the output strictly in JSON format without any additional explanation or text. "
                    "Each subtask should include an 'id', 'description', and a list of 'dependencies' (by id)."
                )
            },
            {"role": "user", "content": prompt}
        ]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await openai_post_request(messages, "gpt-4", max_tokens=500, temperature=0.7)
                content = response['choices'][0]['message']['content'].strip()
                # Use regex to extract JSON array
                json_match = re.search(r"\[.*\]", content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
                    tasks = json.loads(json_content)
                    if self.validate_tasks(tasks):
                        return tasks
                    else:
                        logging.error("Tasks validation failed.")
                        logging.debug(f"Invalid tasks data: {tasks}")
                        return []
                else:
                    logging.error("No JSON array found in LLM response.")
                    logging.debug(f"LLM response content: {content}")
                    return []
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {e}")
                logging.debug(f"LLM response content: {content}")
                return []
            except Exception as e:
                logging.error(f"Decomposition error: {e}")
                logging.debug(f"LLM response content: {content}")
                return []
        logging.error("Max retries exceeded. Unable to decompose task.")
        return []

    def validate_tasks(self, tasks):
        """Validate the tasks received from the LLM."""
        if not isinstance(tasks, list):
            logging.error("Tasks data is not a list.")
            return False
        for task in tasks:
            if not isinstance(task, dict):
                logging.error("Task is not a dictionary.")
                return False
            if 'id' not in task or 'description' not in task or 'dependencies' not in task:
                logging.error("Task is missing required keys.")
                return False
            if not isinstance(task['dependencies'], list):
                logging.error("Task dependencies is not a list.")
                return False
        return True

    def get_graph_data(self):
        """Prepare graph data for visualization."""
        nodes = [{'data': {'id': str(node), 'label': self.graph.nodes[node].get('label', str(node)),
                           'type': self.graph.nodes[node].get('type', 'task')}}
                 for node in self.graph.nodes]
        edges = [{'data': {'source': str(source), 'target': str(target),
                           'label': self.graph.edges[source, target].get('label', '')}}
                 for source, target in self.graph.edges]
        return nodes + edges

    def evaluate_quality(self, detailed_path):
        """Scoring based on the length of the detailed path."""
        if len(detailed_path) > 100:
            return "Excellent"
        elif len(detailed_path) > 50:
            return "Good"
        else:
            return "Satisfactory"
