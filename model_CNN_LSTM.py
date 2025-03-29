import torch
from torch import nn


class HybridCNNandLSTM(nn.Module):
    '''
    https://poloclub.github.io/cnn-explainer/
    '''

    def __init__(self, input_shape_cnn, hidden_units_cnn, hidden_units_rnn, output_shape, num_layers_rnn):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape_cnn,
                out_channels=hidden_units_cnn,
                kernel_size=(3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=hidden_units_cnn),
            nn.Conv2d(
                in_channels=hidden_units_cnn,
                out_channels=hidden_units_cnn,
                kernel_size=(3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=hidden_units_cnn),
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.Conv2d(
                in_channels=hidden_units_cnn,
                out_channels=hidden_units_cnn,
                kernel_size=(3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=hidden_units_cnn),
            nn.Conv2d(
                in_channels=hidden_units_cnn,
                out_channels=hidden_units_cnn,
                kernel_size=(3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=hidden_units_cnn),
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.Conv2d(
                in_channels=hidden_units_cnn,
                out_channels=hidden_units_cnn,
                kernel_size=(3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=hidden_units_cnn),
            nn.Conv2d(
                in_channels=hidden_units_cnn,
                out_channels=hidden_units_cnn,
                kernel_size=(3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=hidden_units_cnn),
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2,
            )
        ) # [1, 350, 2, 2] [batch, hidden_units, H*, W*]

        self.lstm_layer = nn.Sequential( #[batch_size, sequence length,input_size]
            nn.LSTM(
                input_size=hidden_units_cnn * 2 * 2,
                hidden_size=hidden_units_rnn,
                batch_first=True,
                num_layers=num_layers_rnn
            )
        )

        self.seq_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units_rnn,
                out_features=output_shape
            )
        )

    def forward(self, x):
        '''
        x - shape [64, 3, 48, 48],  [batch_size, C, H, W]
        gdzie C, to głębokość obrazu czyli czy jest RGB 3 kolory, czy 1 czarno-biały
        H i W, to są wysokość i szerokość obrazu

        output = cnn_layer(x) - shape [64, 120, 2, 2],  [batch_size, hidden_units_cnn, H*, W*]
        gdzie, H* oraz W* to nowe obrazy przepuszczone przez cnn

        TERAZ WAŻNE zamiana z cnn na rnn
        - ustaliłem obraz cnn jako sekwencję długości 1 a reszte trzeba pomnożyć 120*2*2
        x.view(batch_size, 1, hidden_units * H * W)  # -> [64, 1, 350*4] -> [batch_size, sequence length, input_size]

        '''
        x = self.cnn_layer(x) # [64, hidden_units_cnn, 2, 2]
        batch_size, hidden_units, H, W = x.size()
        x = x.view(batch_size, 1, hidden_units * H * W)  # -> [64, 1, 350*4] -> [batch_size, sequence length, input_size]
        x, _ = self.lstm_layer(x) # [64, 1, 10] -> [batch_size, seq, hidden_units_rnn]
        x = self.seq_linear(x)
        return x
