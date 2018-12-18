'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .docker_agent import DockerAgent
from .http_agent import HttpAgent
from .player_agent import PlayerAgent
from .random_agent import RandomAgent
from .simple_agent import SimpleAgent
from .tensorforce_agent import TensorForceAgent
from .pytorch_agent_reinforced import PytorchAgent_Reinforced
from .actor_critic_agent import PytorchAgent_actor_critic