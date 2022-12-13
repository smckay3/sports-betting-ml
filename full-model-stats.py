import pandas as pd
from tqdm import tqdm
import torch

import datetime
import math

import pickle

import os

datapath = "data/games.csv"
games_df = pd.read_csv(datapath)
games_df.sort_values(by="GAME_DATE_EST", inplace=True)

datapath = "data/games_details.csv"
details_df = pd.read_csv(datapath)

datapath = "data/ranking.csv"
rankings_df = pd.read_csv(datapath)
rankings_df.sort_values(by="STANDINGSDATE", inplace=True)

lookahead = 8
players_per_game = 48
num_stats = 32
rating_size = 32

class Game:
    def __init__(self, game):
        game_date = game["GAME_DATE_EST"].split("-")
        self.game_year = int(game_date[0])
        self.game_month = int(game_date[1])
        self.game_day = int(game_date[2])
        self.game_days_since_epoch = (datetime.datetime(self.game_year, self.game_month, self.game_day) - datetime.datetime(2000, 1, 1)).days
        self.game_id = game["GAME_ID"]
        self.home_team_id = game["HOME_TEAM_ID"]
        self.away_team_id = game["VISITOR_TEAM_ID"]
        self.home_points = game["PTS_home"]
        self.away_points = game["PTS_away"]
        self.fg_pct_home = game["FG_PCT_home"]
        self.fg_pct_away = game["FG_PCT_away"]
        self.ft_pct_home = game["FT_PCT_home"]
        self.ft_pct_away = game["FT_PCT_away"]
        self.fg3_pct_home = game["FG3_PCT_home"]
        self.fg3_pct_away = game["FG3_PCT_away"]
        self.assist_home = game["AST_home"]
        self.assist_away = game["AST_away"]
        self.rebound_home = game["REB_home"]
        self.rebound_away = game["REB_away"]
        self.home_team_wins = 0
        self.home_team_losses = 0
        self.away_team_wins = 0
        self.away_team_losses = 0
        
        if self.home_points == self.away_points:
            self.home_wins = .5
        else:
            self.home_wins = self.home_points > self.away_points
            
        self.home_players_stats = []
        self.away_players_stats = []
        
    def add_player(self, home_team, stats):
        if home_team == 1:
            self.home_players_stats.append(stats)
        else:
            self.away_players_stats.append(stats)

class Player_Stats:
    def __init__(self, game_stats):
        self.game_id = game_stats["GAME_ID"]
        self.player_id = game_stats["PLAYER_ID"]
        self.team_id = game_stats["TEAM_ID"]
        
        pos = game_stats["START_POSITION"]
        if pd.isnull(pos):
            self.start_position = -1
        elif pos == "G":
            self.start_position = 1
        elif pos == "F":
            self.start_position = 2
        elif pos == "C":
            self.start_position = 3
        else:
            print("error:", pos)
        
        time_played = game_stats["MIN"]
        if isinstance(time_played, float):
            self.seconds_played = 0
        else:
            time_played = time_played.split(":")
            self.seconds_played = int(time_played[0]) * 60
            if len(time_played) == 2:
                self.seconds_played += int(time_played[1])
                
        self.fg_made = game_stats["FGM"]
        self.fg_attempts = game_stats["FGA"]
        self.fg3_made = game_stats["FG3M"]
        self.fg3_attempts = game_stats["FG3A"]
        self.ft_made = game_stats["FTM"]
        self.ft_attempts = game_stats["FTA"]
        self.off_reb = game_stats["OREB"]
        self.def_reb = game_stats["DREB"]
        self.assists = game_stats["AST"]
        self.steals = game_stats["STL"]
        self.blocks = game_stats["BLK"]
        self.turnovers = game_stats["TO"]
        self.fouls = game_stats["PF"]
        self.points = game_stats["PTS"]
        self.plus_minus = game_stats["PLUS_MINUS"]
        self.plus_minus = self.plus_minus if not math.isnan(self.plus_minus) else 0
        self.previous_game = 0
        
class Rating:
    def __init__(self, stats):
        self.date = stats[0]
        self.stat1 = stats[1]
        self.stat2 = stats[2]
        
class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.games = []
        self.lookahead_games = {} # map game id to next N games
        self.ratings = [] # RB - Update this as we train
        
    def add_game(self, game):
        self.games.append(game)
        
def diff_days(game1, game2):
    year1 = game1.game_year
    month1 = game1.game_month
    day1 = game1.game_day
    year2 = game2.game_year
    month2 = game2.game_month
    day2 = game2.game_day
    diff = (datetime.datetime(year2, month2, day2) - datetime.datetime(year1, month1, day1)).days
    return diff

everything_path = "everything.pickle"
rankings = {}
games = {}
players = {}
last_season = 0
season_start = []
team_rankings = {}
if not os.path.exists(everything_path):
    print("parsing rankings")
    for _, ranking_data in tqdm(rankings_df.iterrows()):
        ranking_date = ranking_data["STANDINGSDATE"].split("-")
        year = int(ranking_date[0])
        month = int(ranking_date[1])
        day = int(ranking_date[2])
        
        "RANKINGS: Collect season start dates"
        season = ranking_data["SEASON_ID"]
        season = season - int(season/10000) * 10000
        if season != last_season:
            season_start.append((year, month, day))
        last_season = season
        
        if (year, month, day) in rankings:
            rankings[(year, month, day)].append((ranking_data["TEAM_ID"], ranking_data["W"], ranking_data["L"]))
        else:
            rankings[(year, month, day)] = []
            rankings[(year, month, day)].append((ranking_data["TEAM_ID"], ranking_data["W"], ranking_data["L"]))
    
    "RANKINGS: Simple fix for index out of bounds error"
    season_start.append((2100, 1, 1))

    print("parsing games")
    season_index = 0
    for _, game_data in tqdm(games_df.iterrows()):
        game = Game(game_data)
        "RANKINGS: Set wins/losses"
        if game.home_team_id in team_rankings:
            game.home_team_wins = team_rankings[game.home_team_id][0]
            game.home_team_losses = team_rankings[game.home_team_id][1]
        else:
            game.home_team_wins = 0
            game.home_team_losses = 0
        if game.away_team_id in team_rankings:
            game.away_team_wins = team_rankings[game.away_team_id][0]
            game.away_team_losses = team_rankings[game.away_team_id][1]
        else:
            game.away_team_wins = 0
            game.away_team_losses = 0
        "RANKINGS: Reset rankings for all teams at start of season."
        if (datetime.datetime(game.game_year, game.game_month, game.game_day) - datetime.datetime(season_start[season_index][0], season_start[season_index][1], season_start[season_index][2])).days >= 0:
            team_rankings = {}
            season_index += 1
            if game.home_points > game.away_points:
                team_rankings[game.home_team_id] = [1, 0]
                team_rankings[game.away_team_id] = [0, 1]
            elif game.home_points < game.away_points:
                team_rankings[game.home_team_id] = [0, 1]
                team_rankings[game.away_team_id] = [1, 0]
            else:
                team_rankings[game.home_team_id] = [.5, .5]
                team_rankings[game.away_team_id] = [.5, .5]
        else:
            if game.home_team_id in team_rankings:
                if game.away_team_id in team_rankings:
                    if game.home_points > game.away_points:
                        team_rankings[game.home_team_id][0] += 1
                        team_rankings[game.away_team_id][1] += 1
                    elif game.home_points < game.away_points:
                        team_rankings[game.home_team_id][1] += 1
                        team_rankings[game.away_team_id][0] += 1
                    else:
                        team_rankings[game.home_team_id][0] += .5
                        team_rankings[game.home_team_id][1] += .5
                        team_rankings[game.away_team_id][0] += .5
                        team_rankings[game.away_team_id][1] += .5
                else:
                    if game.home_points > game.away_points:
                        team_rankings[game.home_team_id][0] += 1
                        team_rankings[game.away_team_id] = [0, 1]
                    elif game.home_points < game.away_points:
                        team_rankings[game.home_team_id][1] += 1
                        team_rankings[game.away_team_id] = [1, 0]
                    else:
                        team_rankings[game.home_team_id][0] += .5
                        team_rankings[game.home_team_id][1] += .5
                        team_rankings[game.away_team_id] = [.5, .5]
            else:
                if game.away_team_id in team_rankings:
                    if game.home_points > game.away_points:
                        team_rankings[game.home_team_id] = [1, 0]
                        team_rankings[game.away_team_id][1] += 1
                    elif game.home_points < game.away_points:
                        team_rankings[game.home_team_id] = [0, 1]
                        team_rankings[game.away_team_id][0] += 1
                    else:
                        team_rankings[game.home_team_id] = [.5, .5]
                        team_rankings[game.away_team_id][0] += .5
                        team_rankings[game.away_team_id][1] += .5
                else:
                    if game.home_points > game.away_points:
                        team_rankings[game.home_team_id] = [1, 0]
                        team_rankings[game.away_team_id] = [0, 1]
                    elif game.home_points < game.away_points:
                        team_rankings[game.home_team_id] = [0, 1]
                        team_rankings[game.away_team_id] = [1, 0]
                    else:
                        team_rankings[game.home_team_id] = [.5, .5]
                        team_rankings[game.away_team_id] = [.5, .5]
        games[game.game_id] = game

    print("parsing game details")
    for _, player_info in tqdm(details_df.iterrows()):
        "INJURY: Comment this out to build the dataset without knowledge of injured players"
        if player_info["COMMENT"] == "DNP - Injury/Illness" or player_info["COMMENT"] == "DND - Injury/Illness":
            continue
        player_stats = Player_Stats(player_info)
        if games[player_stats.game_id].home_team_id == player_stats.team_id:
            # if player_stats.player_id in players:
            #     last_game = players[player_stats.player_id].games[len(players[player_stats.player_id].games) - 1]
            #     days = diff_days(last_game, games[player_stats.game_id])
            #     player_stats.previous_game = days
            # else:
            #     player_stats.previous_game = -1
            games[player_stats.game_id].add_player(1, player_stats)
        else:
            games[player_stats.game_id].add_player(0, player_stats)
        
        p = Player(player_stats.player_id)
        if p.player_id in players:
            players[p.player_id].add_game(games[player_stats.game_id])
        else:
            p.add_game(games[player_stats.game_id])
            players[p.player_id] = p

    print("calculating lookahead and days since previous game")
    for player in tqdm(players.values()):
        player.games = sorted(player.games, key=lambda x: x.game_days_since_epoch)
        last_game = None
        for game_index, game in enumerate(player.games):
            game_id = game.game_id
            player.lookahead_games[game_id] = player.games[game_index+1:game_index+1+lookahead]
            for player_stats in game.home_players_stats + game.away_players_stats:
                if player_stats.player_id == player.player_id:
                    if last_game != None:
                        player_stats.previous_game = min(game.game_days_since_epoch - last_game.game_days_since_epoch, 7)
                    else:
                        player_stats.previous_game = 7
            last_game = game

    with open(everything_path, "wb") as file:
        pickle.dump((rankings, games, players), file)
else:
    with open(everything_path, "rb") as file:
        rankings, games, players = pickle.load(file)

num_games = len(games)
num_players = len(players)
games_ten = torch.zeros((num_games, players_per_game, num_stats), dtype = torch.float32)
player_rating = torch.zeros((num_players, rating_size), dtype = torch.float32) # SM - don't need to update this now
game_to_player_map = torch.zeros((num_games, players_per_game), dtype = torch.long) # references player_rating tensor
"labels should use from each player the next 'lookahead' games"
labels = torch.zeros((num_games, players_per_game, lookahead), dtype = torch.long) # references games_ten
masks = torch.zeros((num_games, players_per_game), dtype = torch.float32)

player_id_to_global_player_index_map = {player_id: player_index for player_index, player_id in enumerate(players.keys())}

print("constructing games tensor")
away_team_start = int(players_per_game/2)
for game_index, game in tqdm(enumerate(games.values())):
    for player_index, player_stats in enumerate(game.home_players_stats):
        game_to_player_map[game_index][player_index] = player_id_to_global_player_index_map[player_stats.player_id]
        games_ten[game_index][player_index][0] = game.game_year
        games_ten[game_index][player_index][1] = game.game_month
        games_ten[game_index][player_index][2] = game.game_day
        games_ten[game_index][player_index][3] = game.game_days_since_epoch
        games_ten[game_index][player_index][4] = game.home_team_wins
        games_ten[game_index][player_index][5] = game.home_team_losses
        games_ten[game_index][player_index][6] = game.away_team_wins
        games_ten[game_index][player_index][7] = game.away_team_losses
        games_ten[game_index][player_index][8] = player_stats.start_position
        games_ten[game_index][player_index][9] = player_stats.seconds_played
        games_ten[game_index][player_index][10] = player_stats.fg_made
        games_ten[game_index][player_index][11] = player_stats.fg_attempts
        games_ten[game_index][player_index][12] = player_stats.fg_made / player_stats.fg_attempts if player_stats.fg_attempts != 0 else 0
        games_ten[game_index][player_index][13] = player_stats.fg3_made
        games_ten[game_index][player_index][14] = player_stats.fg3_attempts
        games_ten[game_index][player_index][15] = player_stats.fg3_made / player_stats.fg3_attempts if player_stats.fg3_attempts != 0 else 0
        games_ten[game_index][player_index][16] = player_stats.ft_made
        games_ten[game_index][player_index][17] = player_stats.ft_attempts
        games_ten[game_index][player_index][18] = player_stats.ft_made / player_stats.ft_attempts if player_stats.ft_attempts != 0 else 0
        games_ten[game_index][player_index][19] = player_stats.off_reb
        games_ten[game_index][player_index][20] = player_stats.def_reb
        games_ten[game_index][player_index][21] = player_stats.assists
        games_ten[game_index][player_index][22] = player_stats.steals
        games_ten[game_index][player_index][23] = player_stats.blocks
        games_ten[game_index][player_index][24] = player_stats.turnovers
        games_ten[game_index][player_index][25] = player_stats.fouls
        games_ten[game_index][player_index][26] = player_stats.points
        games_ten[game_index][player_index][27] = player_stats.plus_minus
        games_ten[game_index][player_index][28] = player_stats.previous_game
        games_ten[game_index][player_index][29] = 1
    for player_index, player_stats in enumerate(game.away_players_stats):
        player_index = player_index + away_team_start
        game_to_player_map[game_index][player_index] = player_id_to_global_player_index_map[player_stats.player_id]
        games_ten[game_index][player_index][0] = game.game_year
        games_ten[game_index][player_index][1] = game.game_month
        games_ten[game_index][player_index][2] = game.game_day
        games_ten[game_index][player_index][3] = game.game_days_since_epoch
        games_ten[game_index][player_index][4] = game.away_team_wins
        games_ten[game_index][player_index][5] = game.away_team_losses
        games_ten[game_index][player_index][6] = game.home_team_wins
        games_ten[game_index][player_index][7] = game.home_team_losses
        games_ten[game_index][player_index][8] = player_stats.start_position
        games_ten[game_index][player_index][9] = player_stats.seconds_played
        games_ten[game_index][player_index][10] = player_stats.fg_made
        games_ten[game_index][player_index][11] = player_stats.fg_attempts
        games_ten[game_index][player_index][12] = player_stats.fg_made / player_stats.fg_attempts if player_stats.fg_attempts != 0 else 0
        games_ten[game_index][player_index][13] = player_stats.fg3_made
        games_ten[game_index][player_index][14] = player_stats.fg3_attempts
        games_ten[game_index][player_index][15] = player_stats.fg3_made / player_stats.fg3_attempts if player_stats.fg3_attempts != 0 else 0
        games_ten[game_index][player_index][16] = player_stats.ft_made
        games_ten[game_index][player_index][17] = player_stats.ft_attempts
        games_ten[game_index][player_index][18] = player_stats.ft_made / player_stats.ft_attempts if player_stats.ft_attempts != 0 else 0
        games_ten[game_index][player_index][19] = player_stats.off_reb
        games_ten[game_index][player_index][20] = player_stats.def_reb
        games_ten[game_index][player_index][21] = player_stats.assists
        games_ten[game_index][player_index][22] = player_stats.steals
        games_ten[game_index][player_index][23] = player_stats.blocks
        games_ten[game_index][player_index][24] = player_stats.turnovers
        games_ten[game_index][player_index][25] = player_stats.fouls
        games_ten[game_index][player_index][26] = player_stats.points
        games_ten[game_index][player_index][27] = player_stats.plus_minus
        games_ten[game_index][player_index][28] = player_stats.previous_game
        games_ten[game_index][player_index][29] = -1
    

print("constructing labels")
game_id_to_global_game_index_map = {game_id: game_index for game_index, game_id in enumerate(games.keys())}
for game_index, game in tqdm(enumerate(games.values())):
    for player_index, player_stats in enumerate(game.home_players_stats): # RB - Need to repead this loop for away_player_stats w/ minor differences
        player = players[player_stats.player_id]
        for lookahead_index, lookahead_game in enumerate(player.lookahead_games[game.game_id]):
            labels[game_index][player_index][lookahead_index] = game_id_to_global_game_index_map[lookahead_game.game_id]
    for player_index, player_stats in enumerate(game.away_players_stats): # RB - Need to repead this loop for away_player_stats w/ minor differences
        player_index = player_index + away_team_start
        player = players[player_stats.player_id]
        for lookahead_index, lookahead_game in enumerate(player.lookahead_games[game.game_id]):
            labels[game_index][player_index][lookahead_index] = game_id_to_global_game_index_map[lookahead_game.game_id]
        # game_pos = 0
        # p_id = player_stats.player_id
        # while players[p_id].games[game_pos].game_id != game.game_id:
        #     game_pos += 1
        # for i in range(lookahead):
        #     if game_pos + i < len(players[p_id].games):
        #         flag = False
        #         for g in range(num_games - game_index):
        #             for p_ind in range((int(players_per_game/2))):
        #                 if games_ten[g + game_index][p_ind][7] == players[p_id].games[game_pos].game_id and games_ten[g + game_index][p_ind][8] == p_id:
        #                     labels[game_index][player_index][i] = g + game_index
        #                     game_pos += 1
        #                     flag = True
        #                     break
        #             if flag == True:
        #                 break
                        
torch.save((games_ten, game_to_player_map, labels, masks), "game_stats.pt")