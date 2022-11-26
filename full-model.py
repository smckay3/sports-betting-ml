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

rating_size = 256
num_stats = games_ten.shape[-1]

games_ten_mask = torch.zeros((num_stats, ), dtype = torch.float32)

games_ten_mask[0] = 1 # game.game_year
games_ten_mask[1] = 1 # game.game_month
games_ten_mask[2] = 1 # game.game_day
games_ten_mask[3] = 1 # game.game_days_since_epoch
games_ten_mask[4] = 1 # game.away_team_wins
games_ten_mask[5] = 1 # game.away_team_losses
games_ten_mask[6] = 1 # game.home_team_wins
games_ten_mask[7] = 1 # game.home_team_losses
# games_ten_mask[8] = 1 # player_stats.start_position
# games_ten_mask[9] = 1 # player_stats.seconds_played
# games_ten_mask[10] = 1 # player_stats.fg_made
# games_ten_mask[11] = 1 # player_stats.fg_attempts
# games_ten_mask[12] = 1 # player_stats.fg_made / player_stats.fg_attempts if player_stats.fg_attempts != 0 else 0
# games_ten_mask[13] = 1 # player_stats.fg3_made
# games_ten_mask[14] = 1 # player_stats.fg3_attempts
# games_ten_mask[15] = 1 # player_stats.fg3_made / player_stats.fg3_attempts if player_stats.fg3_attempts != 0 else 0
# games_ten_mask[16] = 1 # player_stats.ft_made
# games_ten_mask[17] = 1 # player_stats.ft_attempts
# games_ten_mask[18] = 1 # player_stats.ft_made / player_stats.ft_attempts if player_stats.ft_attempts != 0 else 0
# games_ten_mask[19] = 1 # player_stats.off_reb
# games_ten_mask[20] = 1 # player_stats.def_reb
# games_ten_mask[21] = 1 # player_stats.assists
# games_ten_mask[22] = 1 # player_stats.steals
# games_ten_mask[23] = 1 # player_stats.blocks
# games_ten_mask[24] = 1 # player_stats.turnovers
# games_ten_mask[25] = 1 # player_stats.fouls
# games_ten_mask[26] = 1 # player_stats.points
# games_ten_mask[27] = 1 # player_stats.plus_minus
# games_ten_mask[28] = 1 # player_stats.previous_game
games_ten_mask[29] = 1 # home / away


class RatingGenerator(nn.Module):

    def __init__(self, rating_size, num_stats, dropout=0.5, hidden_size=128):
        super(RatingGenerator, self).__init__()

        config = BertConfig(hidden_size=hidden_size, intermediate_size=hidden_size*4, num_hidden_layers=4, num_attention_heads=8)
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

    def __init__(self, rating_size, num_stats, dropout=0.5, hidden_size=128):
        super(BoxScoreGenerator, self).__init__()

        config = BertConfig(hidden_size=hidden_size, intermediate_size=hidden_size*4, num_hidden_layers=8, num_attention_heads=8)
        print(config)
        self.linear_in = nn.Linear(rating_size + num_stats, hidden_size)
        self.bert = BertEncoder(config)
        # self.dropout = nn.Dropout(dropout)
        self.linear_out_means = nn.Linear(hidden_size, num_stats)
        self.linear_out_var = nn.Linear(hidden_size, num_stats)

    def forward(self, ratings, games, mask):
        # ratings: (games_in_batch, players_per_game, rating_size) = (b, p, r)
        # mask: (games_in_batch, players_per_game) = (b, p)

        inputs = torch.cat([ratings, games], dim=-1)
        inputs = self.linear_in(inputs)

        # fix mask shape and transform to additive mask
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * torch.finfo(torch.float32).min

        output, = self.bert(inputs, attention_mask=mask,return_dict=False)
        dropout_output = output
        # dropout_output = self.dropout(output)
        # dropout_output = inputs
        pred_mean = self.linear_out_means(dropout_output)
        pred_std = self.linear_out_var(dropout_output)

        return pred_mean, pred_std


def train(ratingGenerator, boxScoreGenerator, games_ten, games_ten_mask, game_to_player_map, labels, mask, learning_rate, epochs, games_ten_mean, games_ten_std):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = Adam(list(ratingGenerator.parameters()) + list(boxScoreGenerator.parameters()), lr=learning_rate)
    if use_cuda:
        ratingGenerator = ratingGenerator.cuda()
        boxScoreGenerator = boxScoreGenerator.cuda()
        games_ten = games_ten.cuda()
        games_ten_mean = games_ten_mean.cuda()
        games_ten_std = games_ten_std.cuda()
        games_ten_mask = games_ten_mask.cuda()
        game_to_player_map = game_to_player_map.cuda()
        labels = labels.cuda()
        mask = mask.cuda()

    rating_batch_size = 32
    box_batch_size = 32*4
    num_games = len(games_ten)
    num_train_games = int(num_games*0.95)
    num_val_games = num_games - num_train_games
    train_games = games_ten[:num_train_games]
    val_games = games_ten[num_train_games:]
    train_mask = mask[:num_train_games]
    val_mask = mask[num_train_games:]
    train_game_to_player_map = game_to_player_map[:num_train_games]
    val_game_to_player_map = game_to_player_map[num_train_games:]
    num_train_batches = num_train_games // rating_batch_size
    num_val_batches = num_val_games // rating_batch_size
    players_per_game = games_ten.shape[1]
    players_per_team = players_per_game // 2
    num_lookahead = labels.shape[-1]

    num_players = game_to_player_map.max().item()+2

    for epoch_num in range(epochs):
        player_rating = torch.zeros((num_players, rating_size), dtype = torch.float32, device=device)

        train_total_rating_loss = 0
        train_total_box_loss = 0

        train_correct = 0

        ratingGenerator.train()
        boxScoreGenerator.train()
        for batch_index in tqdm(range(num_train_batches)):
            player_rating.detach_()
            start_index = batch_index * rating_batch_size
            end_index = (batch_index+1) * rating_batch_size
            rating_games = train_games[start_index:end_index]
            player_indices = train_game_to_player_map[start_index:end_index]
            player_mask = train_mask[start_index:end_index]

            # labels: (num_games, players_per_game, lookahead)
            lookahead_game_ids = labels[start_index:end_index].view(-1) # (games in batch * players_per_game * lookahead)
            lookahead_game_id_weights = (lookahead_game_ids != 0).float()
            lookahead_game_id_indicies = torch.multinomial(lookahead_game_id_weights, box_batch_size)
            # lookahead_game_id_indicies = torch.randint(0, lookahead_game_ids.numel(), (box_batch_size,), device=device)
            lookahead_game_ids = lookahead_game_ids[lookahead_game_id_indicies] # (games in batch * players_per_game * lookahead)
            box_games = games_ten[lookahead_game_ids] # (games in batch * players_per_game * lookahead, players_per_game, num_stats)
            box_player_indices = game_to_player_map[lookahead_game_ids] # (games in batch * players_per_game * lookahead, players_per_game, num_stats)
            box_masks = mask[lookahead_game_ids]

            ratings = player_rating[player_indices]
            # print(f"{player_indices.unique()=}")

            rating_change = ratingGenerator(ratings, rating_games, player_mask)
            # new_rating = player_mask.unsqueeze(-1) * rating_change
            new_rating = player_rating[player_indices] + player_mask.unsqueeze(-1) * rating_change
            player_rating[player_indices] = new_rating

            rating_std, rating_mean = torch.std_mean(new_rating, dim=(0,1))
            rating_loss = ((rating_mean * rating_mean + rating_std*rating_std - 1) / 2 - rating_std.log()).mean()
            train_total_rating_loss += rating_loss.item()

            boxScoreGenRatingsInput = player_rating[box_player_indices]
            masked_box_games = games_ten_mask * box_games
            pred_mean, pred_std = boxScoreGenerator(boxScoreGenRatingsInput, masked_box_games, box_masks)

            # masked_box_games_output = (1 - games_ten_mask) * box_games
            # box_loss = F.gaussian_nll_loss(pred_mean, masked_box_games_output, pred_std*pred_std)
            # box_loss = F.gaussian_nll_loss(pred_mean, box_games, pred_std*pred_std)
            box_loss = F.mse_loss(pred_mean, box_games)
            train_total_box_loss += box_loss.item()

            non_normalized_box_score = (box_games * games_ten_std) + games_ten_mean
            non_normalized_pred_score = (pred_mean * games_ten_std) + games_ten_mean

            home_team_score = non_normalized_box_score[:, :players_per_team, 26].sum(-1)
            away_team_score = non_normalized_box_score[:, players_per_team:, 26].sum(-1)
            home_wins = home_team_score > away_team_score
            pred_home_team_score = non_normalized_pred_score[:, :players_per_team, 26].sum(-1)
            pred_away_team_score = non_normalized_pred_score[:, players_per_team:, 26].sum(-1)
            pred_home_wins = pred_home_team_score > pred_away_team_score
            # print(f"{pred_home_team_score=},\n {pred_away_team_score=},\n {home_team_score=},\n {away_team_score=}")
            # print(f"{non_normalized_box_score[:, players_per_team:, 26]=}")
            # print(f"{non_normalized_box_score[:, players_per_team:, 26].sum(-1)=}")
            train_correct += pred_home_wins.eq(home_wins).sum().item()

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

        # print(f"{player_rating[0:32]=}")
        print(
            f"Epochs: {epoch_num + 1} | Train | Rating Loss: {train_total_rating_loss / num_train_batches: .3f} | Box Loss: {train_total_box_loss / num_train_batches: .3f} | Acc: {train_correct / (num_train_batches * box_batch_size)}"
        )

        val_total_rating_loss = 0
        val_total_box_loss = 0
        val_correct = 0

        ratingGenerator.eval()
        boxScoreGenerator.eval()
        with torch.no_grad():
            for batch_index in tqdm(range(num_val_batches)):
                start_index = batch_index * rating_batch_size
                end_index = (batch_index+1) * rating_batch_size
                batch_games = val_games[start_index:end_index]
                player_indices = val_game_to_player_map[start_index:end_index]
                player_mask = val_mask[start_index:end_index]
                ratings = player_rating[player_indices]


                # predict game winners
                masked_box_games = games_ten_mask * batch_games
                pred_mean, pred_std = boxScoreGenerator(ratings, masked_box_games, player_mask)

                non_normalized_box_score = (batch_games * games_ten_std) + games_ten_mean
                non_normalized_pred_score = (pred_mean * games_ten_std) + games_ten_mean

                home_team_score = non_normalized_box_score[:, :players_per_team, 26].sum(-1)
                away_team_score = non_normalized_box_score[:, players_per_team:, 26].sum(-1)
                home_wins = home_team_score > away_team_score
                pred_home_team_score = non_normalized_pred_score[:, :players_per_team, 26].sum(-1)
                pred_away_team_score = non_normalized_pred_score[:, players_per_team:, 26].sum(-1)
                pred_home_wins = pred_home_team_score > pred_away_team_score
                # print(f"{pred_home_team_score=},\n {pred_away_team_score=},\n {home_team_score=},\n {away_team_score=}")
                # print(f"{non_normalized_box_score[:, players_per_team:, 26]=}")
                val_correct += pred_home_wins.eq(home_wins).sum().item()

                # box_loss = F.gaussian_nll_loss(pred_mean, batch_games, pred_std*pred_std)
                box_loss = F.mse_loss(pred_mean, batch_games)
                val_total_box_loss += box_loss.item()

                # update ratings
                rating_change = ratingGenerator(ratings, batch_games, player_mask)
                # new_rating = player_mask.unsqueeze(-1) * rating_change
                new_rating = player_rating[player_indices] + player_mask.unsqueeze(-1) * rating_change
                player_rating[player_indices] = new_rating
                rating_std, rating_mean = torch.std_mean(new_rating, dim=(0,1))
                rating_loss = ((rating_mean * rating_mean + rating_std*rating_std - 1) / 2 - rating_std.log()).mean()
                val_total_rating_loss += rating_loss.item()

        # print(f"{player_rating[0:32]=}")
        print(
            f"Epochs: {epoch_num + 1} | Val | Rating Loss: {val_total_rating_loss / num_val_batches: .3f} | Box Loss: {val_total_box_loss / num_val_batches: .3f} | Acc: {val_correct / (num_val_batches * rating_batch_size)}"
        )



ratingGenerator = RatingGenerator(rating_size, num_stats, hidden_size=128)
boxScoreGenerator = BoxScoreGenerator(rating_size, num_stats, hidden_size=128)

EPOCHS = 200
LR = 1e-4
train(ratingGenerator, boxScoreGenerator, games_ten, games_ten_mask, game_to_player_map, labels, mask, LR, EPOCHS, mean, std)