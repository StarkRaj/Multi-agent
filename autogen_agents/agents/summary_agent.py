from factory.agent_factory import AssistantAgent

class SummaryAgent(AssistantAgent):
    def __init__(self, name, description, system_prompt, output_sources, summary_strategy):
        super().__init__(name, description, system_prompt=system_prompt)
        self.output_sources = output_sources
        self.summary_strategy = summary_strategy

    def summarize_outputs(self, agent_outputs):
        # Simple merge and summarize logic
        summary = "\n".join([f"{k}: {v}" for k, v in agent_outputs.items()])
        return f"Summary ({self.summary_strategy}):\n{summary}" 