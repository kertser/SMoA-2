import networkx as nx
import logging
#from operators.definition_agent import DefinitionAgent
#from operators.comparison_agent import ComparisonAgent
#from operators.analysis_operator import AnalysisAgent

class AgentGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.agents = {
            #"definition": DefinitionAgent(),
            #"comparison": ComparisonAgent(),
            #"analysis": AnalysisAgent()
        }

    async def run(self, input_prompt):
        logging.info(f"Processing user input: {input_prompt}")
        tasks = await self.decompose_task(input_prompt)

        context = f"Initial prompt: {input_prompt}\n\n"
        previous_task = "START"
        self.graph.add_node(previous_task, label="Input")
        self.graph.add_node(input_prompt, label="User Prompt")
        self.graph.add_edge(previous_task, input_prompt, label="Initial Input")
        previous_task = input_prompt

        for task in tasks:
            agent_type = self.identify_agent_type(task)
            agent = self.agents.get(agent_type)
            if agent:
                logging.info(f"Running task '{task}' with {agent_type} agent.")
                result = await agent.run(task)
                context += f"Task: {task}\nResult: {result}\n\n"
                self.graph.add_node(agent_type, label=f"{agent_type.capitalize()} Agent")
                self.graph.add_edge(previous_task, agent_type, label=task)
                previous_task = agent_type

        result_node = "RESULT"
        self.graph.add_node(result_node, label="Final Output")
        self.graph.add_edge(previous_task, result_node, label="Final Processing")

        return {"graph": self.get_graph_data(), "answer": context}

    def identify_agent_type(self, task):
        task = task.lower()
        if "define" in task:
            return "definition"
        elif "compare" in task:
            return "comparison"
        else:
            return "analysis"

    def get_graph_data(self):
        """Returns graph data in a format suitable for Dash Cytoscape."""
        nodes = [{'data': {'id': str(node), 'label': self.graph.nodes[node].get('label', str(node))}} for node in self.graph.nodes]
        edges = [
            {'data': {'source': str(source), 'target': str(target), 'label': self.graph.edges[source, target].get('label', '')}}
            for source, target in self.graph.edges
        ]
        return nodes + edges

    async def decompose_task(self, input_prompt):
        """Decompose the main task into smaller subtasks."""
        # For simplicity, let's just split by sentences for now
        return [sentence.strip() for sentence in input_prompt.split('.') if sentence.strip()]

