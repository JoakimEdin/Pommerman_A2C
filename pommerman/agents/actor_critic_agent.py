from . import BaseAgent
import torch
import torch.nn as nn
from .. import characters
import features

use_cuda = torch.cuda.is_available()


class PytorchAgent_actor_critic(BaseAgent):
    def __init__(self, character=characters.Bomber, mode='train', model="saved_models/policy_network_power_up3"):
        super(PytorchAgent_actor_critic, self).__init__(character)

        n_inputs = 515
        n_conv_output = 7744
        inputs_other = 3
        n_outputs = 6
        
        self.net = Policy(n_inputs, n_outputs, inputs_other, n_conv_output)
        # self.stack = features.Stack_images()
        if use_cuda:
            self.net = self.net.cuda()  

        if mode is "test":
            self.net.load_state_dict(torch.load(model))
        self.hxs = self.net.init_hidden()

    def act(self, obs, action_space=None):
        obs_im, obs_other = features.features(obs)
        # obs_im = self.stack.add(obs_im)
        self.net.eval()
        with torch.no_grad():
            action_scores, _, hxs = self.net(get_variable(torch.Tensor(obs_im)), get_variable(torch.Tensor(obs_other)), self.hxs)
            self.hxs = hxs
        return action_scores.argmax().item()
    
    
class Policy(nn.Module):
    """Policy-network"""

    def __init__(self, n_inputs, n_outputs, inputs_other, n_conv_output):
        super(Policy, self).__init__()
        # network
        self.CNN = nn.Sequential(
                    nn.Conv2d(9, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    )
        self.CNN_mlp = nn.Sequential(        
                    nn.Linear(n_conv_output, 1024, bias=True),
                    nn.ReLU(),
                    nn.Linear(1024, 512, bias=True),
                    nn.ReLU(),
                    )
                    
        self.fnn_other = nn.Sequential(
                                    nn.Linear(inputs_other, inputs_other, bias=True),
                                    nn.ReLU(),
                                    )
        self.actor = nn.Sequential(
                                    nn.Linear(515, 6, bias=False),
                                    )
        self.state_value = nn.Sequential(
                                    nn.Linear(515, 1, bias=False),
                                    nn.Tanh(),
                                    )
        self.GRU = nn.GRUCell(n_inputs, 515)
        self.rewards = []
        self.values = []
        self.entropies = []
        self.log_prob = []

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return get_variable(torch.zeros(1, 515))

    def forward(self, x_im, x_other, hxs, batch_size=1):
        out = []
        x = self.CNN(x_im)
        x = x.view(batch_size, -1)
        x = self.CNN_mlp(x)
        out.append(x)
        if batch_size > 1:
            x_other = self.input_norm(x_other)
        x = self.fnn_other(x_other)
        out.append(x)
        out = torch.cat(out, dim=1)
        out = hxs = self.GRU(out, hxs)
        action_scores = self.actor(out)
        state_values = self.state_value(out)
        return action_scores, state_values[0], hxs


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()