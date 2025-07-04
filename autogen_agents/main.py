from configs.agent_configs import (
    AudioSyncAgentConfig,
    PixelizationAgentConfig,
    DeciderAgentConfig,
    SummaryAgentConfig,
)
from factory.agent_factory import create_agent

# Create agents
audio_agent = create_agent(AudioSyncAgentConfig())
pixel_agent = create_agent(PixelizationAgentConfig())
decider_agent = create_agent(DeciderAgentConfig(agent_list=["audio_sync_agent", "pixelization_detection_agent"]))
summary_agent = create_agent(SummaryAgentConfig(output_sources=["audio_sync_agent", "pixelization_detection_agent"]))

# Example outputs from agents
outputs = {
    "audio_sync_agent": "Audio is out of sync by 500ms.",
    "pixelization_detection_agent": "Pixelation detected between 00:01:23 - 00:01:40."
}

# Summarize
summary = summary_agent.summarize_outputs(outputs)
print(summary)
