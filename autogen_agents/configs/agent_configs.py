from base.base_agent_config import AgentConfig

class AudioSyncAgentConfig(AgentConfig):
    def __init__(self):
        super().__init__(
            agent_name="audio_sync_agent",
            description="Detects audio-video sync issues",
            mcp_path="/tools/audio_sync/mcp.py",
            system_prompt="You're an expert in audio-video sync issues."
        )

    def to_dict(self):
        return super().to_dict()

class PixelizationAgentConfig(AgentConfig):
    def __init__(self):
        super().__init__(
            agent_name="pixelization_detection_agent",
            description="Detects pixelation or compression artifacts",
            mcp_path="/tools/pixelization/mcp.py",
            system_prompt="You're an expert in video pixelation analysis."
        )

    def to_dict(self):
        return super().to_dict()

class DeciderAgentConfig(AgentConfig):
    def __init__(self, agent_list, strategy="round_robin"):
        super().__init__(
            agent_name="decider_agent",
            description="Routes tasks to the appropriate analysis agent",
            mcp_path="",
            system_prompt="You're responsible for deciding which agent should handle which task."
        )
        self.agent_list = agent_list
        self.strategy = strategy

    def to_dict(self):
        base = super().to_dict()
        base.update({
            "agent_list": self.agent_list,
            "strategy": self.strategy
        })
        return base

class SummaryAgentConfig(AgentConfig):
    def __init__(self, output_sources, summary_strategy="merge_and_summarize"):
        super().__init__(
            agent_name="summary_agent",
            description="Summarizes the output of all other analysis agents",
            mcp_path="",
            system_prompt="You're responsible for summarizing multi-agent analysis output into a final report."
        )
        self.output_sources = output_sources
        self.summary_strategy = summary_strategy

    def to_dict(self):
        base = super().to_dict()
        base.update({
            "output_sources": self.output_sources,
            "summary_strategy": self.summary_strategy
        })
        return base 