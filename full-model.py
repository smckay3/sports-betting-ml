import torch
from torch.optim import Adam
from torch import nn
from tqdm import tqdm

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

torch.set_printoptions(threshold=10000)

games_ten, game_to_player_map, labels, _ = torch.load('game_stats.pt')
mask = (games_ten[:,:,0] > 0).to(torch.float32)

games_ten = torch.nan_to_num(games_ten)
player_games = games_ten.reshape(-1, 32)
player_games[player_games[:, 0] > 0].shape
player_games = player_games[player_games[:, 0] > 0]
std, mean = torch.std_mean(player_games, dim=0)

games_ten = (games_ten - mean) / (std + 1e-5)
# labels = (labels - mean) / (std + 1e-5)

game_to_player_map += mask.long()

rating_size = 32
num_stats = games_ten.shape[-1]


class BertClassifier(nn.Module):

    def __init__(self, rating_size, num_stats, dropout=0.5, hidden_size=256):
        super(BertClassifier, self).__init__()

        config = BertConfig(hidden_size=hidden_size, intermediate_size=hidden_size*4, num_hidden_layers=8, num_attention_heads=8)
        print(config)
        self.linear_in = nn.Linear(rating_size + num_stats, hidden_size)
        self.bert = BertEncoder(config)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_size, rating_size)

    def forward(self, ratings, game_info, mask):
        # ratings: (games_in_batch, players_per_game, rating_size) = (b, p, r)
        # game_info: (games_in_batch, players_per_game, num_player_stats) = (b, p, s)
        # mask: (games_in_batch, players_per_game) = (b, p)

        inputs = torch.cat([ratings, game_info], dim=-1)
        inputs = self.linear_in(inputs)

        # fix mask shape and transform to additive mask
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * torch.finfo(torch.float32).min

        output, = self.bert(inputs, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(output)
        linear_output = self.linear_out(dropout_output)

        return linear_output


def train(model, games_ten, game_to_player_map, labels, mask, learning_rate, epochs):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = Adam(model.parameters(), lr=learning_rate)

    batch_size = 32
    num_games = len(games_ten)
    batches = num_games // batch_size

    if use_cuda:
        model = model.cuda()
        games_ten = games_ten.cuda()
        game_to_player_map = game_to_player_map.cuda()
        labels = labels.cuda()
        mask = mask.cuda()

    num_players = game_to_player_map.max().item()+2

    for epoch_num in range(epochs):
        player_rating = torch.zeros((num_players, rating_size), dtype = torch.float32, device=device)

        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for batch_index in tqdm(range(batches)):
            start_index = batch_index * batch_size
            end_index = (batch_index+1) * batch_size
            train_games = games_ten[start_index:end_index]
            player_indices = game_to_player_map[start_index:end_index]

            ratings = player_rating[player_indices].detach()
            player_mask = mask[start_index:end_index]
            # print(f"{player_indices.unique()=}")

            rating_change = model(ratings, train_games, player_mask)
            new_rating = player_rating[player_indices].detach() + player_mask.unsqueeze(-1) * rating_change
            player_rating[player_indices] = new_rating

            std, mean = torch.std_mean(new_rating, dim=(0,1))
            # print(f"{rating_change[0, 0]=}")
            # print(f"{new_rating[0, 0]=}")
            # print(f"{player_rating[player_indices][0, 0]=}")


            # batch_loss = criterion(output, train_label.long())
            batch_loss = ((1-std)*(1-std) + mean*mean).sum()
            total_loss_train += batch_loss.item()

            # acc = (output.argmax(dim=1) == train_label).sum().item()
            # total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / batches: .3f}"
        )

model = BertClassifier(rating_size, num_stats, hidden_size=256)

EPOCHS = 10
LR = 1e-4
train(model, games_ten, game_to_player_map, labels, mask, LR, EPOCHS)