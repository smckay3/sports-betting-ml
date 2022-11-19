import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
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
labels = labels.long()

game_to_player_map += mask.long()

rating_size = 32
num_stats = games_ten.shape[-1]


class RatingGenerator(nn.Module):

    def __init__(self, rating_size, num_stats, dropout=0.5, hidden_size=256):
        super(RatingGenerator, self).__init__()

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

class BoxScoreGenerator(nn.Module):

    def __init__(self, rating_size, num_stats, dropout=0.5, hidden_size=256):
        super(BoxScoreGenerator, self).__init__()

        config = BertConfig(hidden_size=hidden_size, intermediate_size=hidden_size*4, num_hidden_layers=8, num_attention_heads=8)
        print(config)
        self.linear_in = nn.Linear(rating_size, hidden_size)
        self.bert = BertEncoder(config)
        self.dropout = nn.Dropout(dropout)
        self.linear_out_means = nn.Linear(hidden_size, num_stats)
        self.linear_out_var = nn.Linear(hidden_size, num_stats)

    def forward(self, ratings, mask):
        # ratings: (games_in_batch, players_per_game, rating_size) = (b, p, r)
        # mask: (games_in_batch, players_per_game) = (b, p)


        inputs = ratings
        inputs = self.linear_in(inputs)

        # fix mask shape and transform to additive mask
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * torch.finfo(torch.float32).min

        output, = self.bert(inputs, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(output)
        pred_mean = self.linear_out_means(dropout_output)
        pred_std = self.linear_out_var(dropout_output)

        return pred_mean, pred_std


def train(ratingGenerator, boxScoreGenerator, games_ten, game_to_player_map, labels, mask, learning_rate, epochs):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = Adam(list(ratingGenerator.parameters()) + list(boxScoreGenerator.parameters()), lr=learning_rate)

    rating_batch_size = 32
    box_batch_size = 32
    num_games = len(games_ten)
    batches = num_games // rating_batch_size
    players_per_game = games_ten.shape[1]
    players_per_team = players_per_game // 2

    if use_cuda:
        ratingGenerator = ratingGenerator.cuda()
        boxScoreGenerator = boxScoreGenerator.cuda()
        games_ten = games_ten.cuda()
        game_to_player_map = game_to_player_map.cuda()
        labels = labels.cuda()
        mask = mask.cuda()

    num_players = game_to_player_map.max().item()+2

    for epoch_num in range(epochs):
        player_rating = torch.zeros((num_players, rating_size), dtype = torch.float32, device=device)

        total_acc_train = 0
        total_rating_loss = 0
        total_box_loss = 0

        correct = 0

        ratingGenerator.train()
        boxScoreGenerator.train()
        for batch_index in tqdm(range(batches)):
            player_rating.detach_()
            start_index = batch_index * rating_batch_size
            end_index = (batch_index+1) * rating_batch_size
            train_games = games_ten[start_index:end_index]
            player_indices = game_to_player_map[start_index:end_index]

            # labels: (num_games, players_per_game, lookahead)
            lookahead_game_ids = labels[start_index:end_index].view(-1) # (games in batch * players_per_game * lookahead)
            lookahead_game_id_indicies = torch.randint(0, lookahead_game_ids.numel(), (box_batch_size,), device=device)
            lookahead_game_ids = lookahead_game_ids[lookahead_game_id_indicies] # (games in batch * players_per_game * lookahead)
            box_games = games_ten[lookahead_game_ids] # (games in batch * players_per_game * lookahead, players_per_game, num_stats)
            box_player_indices = game_to_player_map[lookahead_game_ids] # (games in batch * players_per_game * lookahead, players_per_game, num_stats)
            box_masks = mask[lookahead_game_ids]

            ratings = player_rating[player_indices]
            player_mask = mask[start_index:end_index]
            # print(f"{player_indices.unique()=}")

            rating_change = ratingGenerator(ratings, train_games, player_mask)
            new_rating = player_rating[player_indices] + player_mask.unsqueeze(-1) * rating_change
            player_rating[player_indices] = new_rating

            rating_std, rating_mean = torch.std_mean(new_rating, dim=(0,1))
            rating_loss = ((rating_mean * rating_mean + rating_std*rating_std - 1) / 2 - rating_std.log()).sum()
            total_rating_loss += rating_loss.item()

            boxScoreGenRatingsInput = player_rating[box_player_indices]
            pred_mean, pred_std = boxScoreGenerator(boxScoreGenRatingsInput, box_masks)

            box_loss = F.gaussian_nll_loss(pred_mean, box_games, pred_std*pred_std)
            total_box_loss += box_loss.item()

            home_team_score = box_games[:, :players_per_team, 26].sum(-1)
            away_team_score = box_games[:, players_per_team:, 26].sum(-1)
            home_wins = home_team_score > away_team_score
            pred_home_team_score = pred_mean[:, :players_per_team, 26].sum(-1)
            pred_away_team_score = pred_mean[:, players_per_team:, 26].sum(-1)
            pred_home_wins = pred_home_team_score > pred_away_team_score
            correct += pred_home_wins.eq(home_wins).sum().item()

            # print(f"{rating_change[0, 0]=}")
            # print(f"{new_rating[0, 0]=}")
            # print(f"{player_rating[player_indices][0, 0]=}")


            # batch_loss = criterion(output, train_label.long())
            # batch_loss = ((1-std)*(1-std) + mean*mean).sum()
            # { (μ1 - μ2)2 + σ12 - σ22 } / (2.σ22) + ln(σ2/σ1)

            # acc = (output.argmax(dim=1) == train_label).sum().item()
            # total_acc_train += acc

            ratingGenerator.zero_grad()
            boxScoreGenerator.zero_grad()
            # if epoch_num == 0:
            #     rating_loss.backward()
            # else:
            (rating_loss*1 + box_loss * 1).backward()
            optimizer.step()

        print(f"{player_rating[0:32]=}")
        print(
            f"Epochs: {epoch_num + 1} | Rating Loss: {total_rating_loss / batches: .3f} | Box Loss: {total_box_loss / batches: .3f} | Acc: {correct / (batches * box_batch_size)}"
        )

ratingGenerator = RatingGenerator(rating_size, num_stats, hidden_size=256)
boxScoreGenerator = BoxScoreGenerator(rating_size, num_stats, hidden_size=256)

EPOCHS = 10
LR = 1e-4
train(ratingGenerator, boxScoreGenerator, games_ten, game_to_player_map, labels, mask, LR, EPOCHS)