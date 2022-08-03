import torch
from torch import nn
from transformers import BertModel


class BertLSTMClassifier(nn.Module):

    def __init__(self, network_type="BiLSTM"):
        super().__init__()
        # 基础参数配置
        self.network_type = network_type
        self.hidden_layers_size = 768
        self.hidden_dim = int(self.hidden_layers_size / 2)
        self.n_layers = 2
        self.dropout = 0.5
        self.output_size = 2
        self.bidirectional = True if self.network_type == "BiLSTM" else False

        # Bert Layer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True,
                            bidirectional=self.bidirectional)

        # Dropout Layer
        self.dropout = nn.Dropout(self.dropout)

        # Linear and Sigmoid Layers
        self.function = nn.Linear(self.hidden_dim * 2, self.output_size) \
            if self.bidirectional else nn.Linear(self.hidden_dim, self.output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_x, hidden):
        batch_size = input_x.size(0)
        input_x = self.bert(input_x)[0]  # 生成bert的字向量
        lstm_output, (h_n, h_c) = self.lstm(input_x, hidden)

        if self.bidirectional:
            # BiLSTM: 正向最后一层，最后一个时刻/反向最后一层，最后一个时刻
            hidden_last_left, hidden_last_right = h_n[-2], h_n[-1]
            hidden_last_out = torch.cat([hidden_last_left, hidden_last_right], dim=-1)
        else:
            # LSTM
            hidden_last_out = h_n[-1]
        # dropout and fully connected layer
        out = self.function(self.dropout(hidden_last_out))
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 2 if self.bidirectional else 1
        # 判断是否使用GPU
        cuda_status = torch.cuda.is_available()
        hidden = (
            weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
            weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
        ) if cuda_status else (
            weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
            weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
        )
        return hidden
