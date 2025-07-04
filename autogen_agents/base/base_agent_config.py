from abc import ABC, abstractmethod

class AgentConfig(ABC):
    def __init__(self, agent_name: str, description: str, mcp_path: str, system_prompt: str):
        self.agent_name = agent_name
        self.description = description
        self.mcp_path = mcp_path
        self.system_prompt = system_prompt

    @abstractmethod
    def to_dict(self):
        return {
            "agent_name": self.agent_name,
            "description": self.description,
            "mcp_path": self.mcp_path,
            "system_prompt": self.system_prompt
        } 