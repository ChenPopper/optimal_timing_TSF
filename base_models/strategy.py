import torch
import torch.nn as nn
import time
from typing import List, Tuple


class Strategy(nn.Module):
    """
    The optimal time strategy, for minimizing the cost and find the optimal times
    for each product.
    args:
        num_products: int
          the number of products, i.e., the number of time series
        num_samples: int
          the number of samples for each prediction object of the probabilistic forecasting
        prediction_length: int
          the future time steps need to do forecasting, which is the total number of future periods
          for decision making (i.e., to find the best period to trade)

    """
    def __init__(
            self,
            num_product,
            prediction_length,
            device='cpu'
    ) -> None:
        super().__init__()
        self.num_product = num_product
        self.prediction_length = prediction_length
        self.device = device
        self.model_name = str(type(self))

    def forward(self, samples) -> torch.Tensor:
        pass

    @staticmethod
    def objective_function(samples, **kwargs):
        """
        The target functional to be minimized, which can be treated as value function in RL.
        """
        pass

    def optimal_times(self, **kwargs) -> List[int]:
        """
        Get the optimal times for each product.
        """
        pass

    def neural_net(self, x, **kwargs) -> torch.Tensor:
        """
        the neural network part for approximating the indicator function of the stopping time
        """
        pass

    def approx_indicator(self, net_output, samples, **kwargs) -> torch.Tensor:
        """
        calculate the approximated functions of the indicator function of stopping times
        (see, equations (6, 7) in the draft)
        args:
            net_output: torch.Tensor
              the outputs of the neural network for approximating the indicator function of the stopping time
              (see, equation (7)),
              shape = (num_samples, input_dim, prediction_length+1)
            samples: torch.Tensor
              the predicted samples for mini-batches of the time series,
              shape = (num_samples, input_dim, prediction_length+1)
        return:
            approx_d: torch.Tensor
              the approximated stopping time indicator,
              shape = (num_samples, input_dim, prediction_length+1)
        """
        pass

    def load(self, path):
        """
        load data from path
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        save model
        """
        if name is None:
            name = 'checkpoints/' + self.model_name + '_' + time.strftime('%y%m%d_%H%M%S')
        torch.save(self.state_dict(), name)
        return name
