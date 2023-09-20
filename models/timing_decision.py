import torch
import torch.nn as nn
from base_models.strategy import Strategy


class TimingStrategy(Strategy):
    def __init__(self, num_product, prediction_length, hidden_dim=128, rnn_type='rnn'):
        super().__init__(num_product, prediction_length)
        self.model_name = 'TimingStrategy'
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type

        self.BatchNorm1d = nn.BatchNorm1d(self.num_product)
        if rnn_type == 'rnn':
            self.rnn = nn.RNNCell(self.num_product, self.hidden_dim, bias=True, nonlinearity='tanh')
        elif rnn_type == 'gru':
            self.rnn = nn.GRUCell(self.num_product, self.hidden_dim, bias=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTMCell(self.num_product, self.hidden_dim, bias=True)
        else:
            raise ValueError("Unknown rnn type, only support 'rnn', 'gru', and 'lstm'")
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.num_product, bias=True),
            nn.Sigmoid(),
           )

    def forward(self, samples):
        """
        Forward function to get cost value, optimal time distribution in time axis and optimal time
        args:
            samples: torch.Tensor
              the predicted samples for mini-batches of the time series,
              shape = (num_samples, input_dim, prediction_length+1)
        return:
            value: torch.Tensor(1)
              cost value to be minimized.
            optimal_time_distribution: torch.Tensor
              distribution in sampling path, each path has one optimal timing.
              shape = (num_samples, input_dim, 1)
            optimal_time: torch.Tensor
              optimal time for each dimension.
              shape = (input_dim, 1)
        """
        net_output = self.encoder(samples)  # shape -> (num_samples, input_dim, prediction_length + 1)
        indicators = self.approx_indicator(net_output, samples)
        value = self.objective_function(samples, indicators)
        optimal_time_distribution = self.get_optimal_time(indicators)
        optimal_time, _ = torch.mode(optimal_time_distribution, dim=0, keepdim=False)

        return value, optimal_time_distribution, optimal_time

    def encoder(self, samples) -> torch.Tensor:
        """
        shape = (num_samples, input_dim, prediction_length+1)
        hidden:
          shape = (num_samples, hidden_dim)

        return:
            h: ()
        """
        target_length = self.prediction_length + 1
        # hidden = torch.randn(samples.shape[0], self.hidden_dim)
        hidden = torch.zeros(samples.shape[0], self.hidden_dim).to(samples.device)
        if self.rnn_type == 'lstm':
            c_hidden = torch.zeros(samples.shape[0], self.hidden_dim).to(samples.device)
        h = torch.unsqueeze(self.neural_net(hidden), 2)  # shape -> (num_samples, input_dim, 1)
        for i in range(1, target_length):
            if self.rnn_type == 'lstm':
                hidden, c_hidden = self.rnn(self.BatchNorm1d(samples[:, :, i]), (hidden, c_hidden))
            else:
                hidden = self.rnn(samples[:, :, i], hidden)
#                 hidden = self.rnn(self.BatchNorm1d(samples[:, :, i]), hidden)
            h = torch.cat([h, torch.unsqueeze(self.neural_net(hidden), 2)], dim=-1)

        return h  # shape -> (num_samples, input_dim, prediction_length + 1)

    def approx_indicator(self, net_output, samples, **kwargs) -> torch.Tensor:
        """
        Indicator function can be approximated by the following code (see equation (8))
        args:
            net_output: torch.Tensor
              neural network output ht: R_(N, C, P+1) -> (0, 1)
              shape = (num_samples, input_dim, prediction_length+1)
            samples: torch.Tensor
              the predicted samples for mini-batches of the time series,
              shape = (num_samples, input_dim, prediction_length+1)
        return:
            indicators: torch.Tensor
              indicator calculated by approx_indicator function. (see, equation (8))
              shape = (num_samples, input_dim, prediction_length+1)
        """
        approx_d = net_output[:, :, 0:1]
        one_tensor = torch.ones_like(samples[:, :, 0:1], device=samples.device)

        for t in range(1, self.prediction_length + 1):
            main_part = one_tensor - torch.sum(approx_d, dim=-1, keepdim=True)
            time_bound = (t + 1 - self.prediction_length) * one_tensor
            multiplier = torch.max(net_output[:, :, t:t+1], time_bound)
            approx_d_t = torch.mul(multiplier, main_part)
            approx_d = torch.cat([approx_d, approx_d_t], dim=-1)

        return approx_d

    def neural_net(self, x) -> torch.Tensor:
        """
        the neural network part for approximating the indicator function h_t() of the stopping time
        in equation (9)
        args:
            x: torch.Tensor
              the outputs of the rnn encoder,
              shape = (num_samples, hidden_dim, prediction_length+1)

        return:
            out: torch.Tensor
              shape = (num_samples, input_dim, prediction_length+1)
        """

        out = self.out_layer(x)
        return out

    @staticmethod
    def objective_function(samples, indicators) -> torch.Tensor:
        """
        The target functional to be minimized, which can be treated as value function in RL.
        (see, equation (11))
        args:
            samples: torch.Tensor
              the predicted samples for mini-batches of the time series,
              shape = (num_samples, input_dim, prediction_length+1)
            indicators: torch.Tensor
              indicator calculated by approx_indicator function. (see, equation (8))
              shape = (num_samples, input_dim, prediction_length+1)
        return:
            total_cost: torch.Tensor, shape -> (1)
              cost value to be minimized.
        """
        cost_sample = torch.sum(torch.mul(indicators, samples), dim=-1, keepdim=False)
        cost_sample_mean = torch.mean(cost_sample, dim=0, keepdim=False)
        total_cost = torch.sum(cost_sample_mean)
        return total_cost

    def get_optimal_time(self, indicators) -> torch.Tensor:
        """calculate the random variables -- the optimal stopping times,
        which have a value on each sample path
        args:
            indicators: torch.Tensor
              the values of indicator functions,
              shape = (num_samples, input_dim, prediction_length+1)
        return:
            optimal_time: torch.Tensor
              the value of the optimal time at each path,
              shape = (num_samples, input_dim, 1)
        """
        ind_cumsums = indicators.cumsum(dim=-1) + indicators
        idx = torch.arange(self.prediction_length + 1, device=indicators.device)
        optimal_time = self.prediction_length * torch.ones_like(indicators[:, :, -1:], device=indicators.device)
        for i in range(indicators.size(0)):
            for j in range(indicators.size(1)):
                try:
                    optimal_time_ij = idx[torch.where(ind_cumsums[i, j, :] < 1, False, True)][0]
                    optimal_time[i, j, 0] = optimal_time_ij
                except IndexError:
                    pass
        return optimal_time