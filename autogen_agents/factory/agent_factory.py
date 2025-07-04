from configs.agent_configs import (
    AudioSyncAgentConfig,
    PixelizationAgentConfig,
    DeciderAgentConfig,
    SummaryAgentConfig,
)
from agents.summary_agent import SummaryAgent

def create_agent(agent_config):
    config = agent_config.to_dict()

    if isinstance(agent_config, DeciderAgentConfig):
        return DeciderAgent(
            name=config["agent_name"],
            description=config["description"],
            agent_list=config["agent_list"],
            strategy=config["strategy"],
            system_prompt=config["system_prompt"]
        )
    elif isinstance(agent_config, SummaryAgentConfig):
        return SummaryAgent(
            name=config["agent_name"],
            description=config["description"],
            system_prompt=config["system_prompt"],
            output_sources=config["output_sources"],
            summary_strategy=config["summary_strategy"]
        )
    else:
        return AssistantAgent(
            name=config["agent_name"],
            description=config["description"],
            mcp_path=config["mcp_path"],
            system_prompt=config["system_prompt"]
        ) 