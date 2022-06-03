from .A2C import A2CAgent, A2CRecurrentAgent
from .PPO import PPOAgent, PPORecurrentAgent

from .curriculum_agent_mixin import ExternalCurriculumAgentMixin

class PPORecurrentExternalCurriculumAgent(ExternalCurriculumAgentMixin, PPORecurrentAgent):
    pass