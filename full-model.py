import pandas as pd
from tqdm import tqdm
import torch

datapath = "data/games.csv"
games_df = pd.read_csv(datapath)
games_df.sort_values(by="GAME_DATE_EST", inplace=True)

datapath = "data/games_details.csv"
details_df = pd.read_csv(datapath)

class Game:
    def __init__(self, game):
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
        self.start_position = game_stats["START_POSITION"]
        
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
        
        """
        Need to add the following data values:
        1) Game date
        2) Home/away team playoff buffer (games ahead or back)
        3) Time since last game played
        """
        
class Rating:
    def __init__(self, stats):
        self.date = stats[0]
        self.stat1 = stats[1]
        self.stat2 = stats[2]
        
class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.games = [] # SM - use a sorted map
        self.ratings = [] # RB - Update this as we train
        
    def add_game(self, game):
        self.games.append(game)
        
print("parsing games")
games = {}
for _, game_data in tqdm(games_df.iterrows()):
    game = Game(game_data)
    games[game.game_id] = game

players = {}
print("parsing game details")
for _, player_info in tqdm(details_df.iterrows()):
    player_stats = Player_Stats(player_info)
    if games[player_stats.game_id].home_team_id == player_stats.team_id:
        games[player_stats.game_id].add_player(1, player_stats)
    else:
        games[player_stats.game_id].add_player(0, player_stats)
    
    p = Player(player_stats.player_id)
    if p.player_id in players:
        players[p.player_id].add_game(games[player_stats.game_id])
    else:
        p.add_game(games[player_stats.game_id])
        players[p.player_id] = p

num_games = len(games)
num_players = len(players)
players_per_game = 48
num_stats = 32
rating_size = 32
lookahead = 8
games_ten = torch.zeros((num_games, players_per_game, num_stats), dtype = torch.float32)
player_rating = torch.zeros((num_players, rating_size), dtype = torch.float32) # SM - don't need to update this now
game_to_player_map = torch.zeros((num_games, num_players), dtype = torch.long) # references player_rating tensor
"labels should use from each player the next 'lookahead' games"
labels = torch.zeros((num_games, players_per_game, lookahead), dtype = torch.float32) # references games_ten
masks = torch.zeros((num_games, players_per_game), dtype = torch.float32)
