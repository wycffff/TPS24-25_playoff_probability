import json
import random
import pandas as pd
from collections import defaultdict
import copy

# 1) Read and filter data
with open('liiga_games.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

games_this_season = [
    g for g in data
    if g.get('season') == 2025 
       and g.get('serie') == 'RUNKOSARJA'
]

# 2) Statistical phase: calculate current points, plus wins/losses in regulation and OT/SO
points = defaultdict(int)
games_played = defaultdict(int)

stats_games = defaultdict(int)      # total completed games per team
stats_reg_win = defaultdict(int)    # number of regulation-time wins
stats_reg_loss = defaultdict(int)   # number of regulation-time losses
stats_ot_so_win = defaultdict(int)  # number of OT/SO wins
stats_ot_so_loss = defaultdict(int) # number of OT/SO losses

ended_games = [g for g in games_this_season if g.get('ended') == True]

for game in ended_games:
    home_team = game['homeTeam']['teamName']
    away_team = game['awayTeam']['teamName']
    home_goals = game['homeTeam']['goals']
    away_goals = game['awayTeam']['goals']
    finished_type = game.get('finishedType', '')

    # Record the number of completed games
    stats_games[home_team] += 1
    stats_games[away_team] += 1
    games_played[home_team] += 1
    games_played[away_team] += 1

    # Assign points and update statistics
    if ("OVERTIME" in finished_type 
        or "EXTENDED" in finished_type 
        or "WINNING_SHOT" in finished_type):
        # If the game was decided in overtime or by shootout
        if home_goals > away_goals:
            points[home_team] += 2
            points[away_team] += 1
            stats_ot_so_win[home_team] += 1
            stats_ot_so_loss[away_team] += 1
        else:
            points[away_team] += 2
            points[home_team] += 1
            stats_ot_so_win[away_team] += 1
            stats_ot_so_loss[home_team] += 1
    else:
        # Regulation time
        if home_goals > away_goals:
            points[home_team] += 3
            stats_reg_win[home_team] += 1
            stats_reg_loss[away_team] += 1
        elif home_goals < away_goals:
            points[away_team] += 3
            stats_reg_win[away_team] += 1
            stats_reg_loss[home_team] += 1
        else:
            # Liiga generally doesn't end in a regulation tie, 
            # but handle data if there's an unexpected case
            pass

# Create a dataframe to see the current standings
df_current = pd.DataFrame([
    {
        'team': t,
        'points': points[t],
        'games_played': games_played[t],
        'reg_win': stats_reg_win[t],
        'reg_loss': stats_reg_loss[t],
        'ot_so_win': stats_ot_so_win[t],
        'ot_so_loss': stats_ot_so_loss[t],
        'games_finished': stats_games[t]
    }
    for t in points.keys()
])
df_current.sort_values('points', ascending=False, inplace=True, ignore_index=True)
print("=== Current Standings and Basic Stats ===")
print(df_current)


# 3) Define a probability model based on each team's historical performance
def get_team_stats(team):
    """
    Returns (reg_win_rate, ot_so_win_rate, total_games_for_team).
    Uses 'stats_reg_win' and 'stats_ot_so_win' divided by 'stats_games[team]'.
    Applies basic smoothing if total_games == 0.
    """
    gw = stats_games[team]
    if gw == 0:
        # If no data, default to a neutral assumption
        return (0.5, 0.5, 1)
    
    reg_w = stats_reg_win[team]
    ot_w  = stats_ot_so_win[team]
    
    reg_win_rate = reg_w / gw
    ot_so_win_rate = ot_w / gw
    
    return (reg_win_rate, ot_so_win_rate, gw)

def match_probability(home_team, away_team):
    """
    Returns a tuple (p_home_reg, p_away_reg, p_home_ot, p_away_ot).
    
    Basic approach:
    1) home_reg = reg_win_rate_home / (reg_win_rate_home + reg_win_rate_away)
       away_reg = reg_win_rate_away / (reg_win_rate_home + reg_win_rate_away)
       leftover probability = chance of going to OT/SO
    2) OT/SO is then split proportionally using ot_so_win_rate.
    """
    (reg_home, ot_home, _) = get_team_stats(home_team)
    (reg_away, ot_away, _) = get_team_stats(away_team)
    
    # Calculate regulation win probabilities
    s_reg = reg_home + reg_away
    if s_reg < 1e-9:
        p_home_reg = 0.3
        p_away_reg = 0.3
    else:
        p_home_reg = reg_home / s_reg
        p_away_reg = reg_away / s_reg
    
    # Probability that the game goes to OT/SO
    p_ot = max(0.0, 1 - p_home_reg - p_away_reg)
    
    # OT/SO win probabilities
    s_ot = ot_home + ot_away
    if s_ot < 1e-9:
        p_home_ot = p_ot * 0.5
        p_away_ot = p_ot * 0.5
    else:
        p_home_ot = p_ot * (ot_home / s_ot)
        p_away_ot = p_ot * (ot_away / s_ot)
    
    return p_home_reg, p_away_reg, p_home_ot, p_away_ot

def simulate_single_game(home_team, away_team):
    """
    Uses the probabilities from match_probability() to randomly decide 
    a single game's result. Returns (points_for_home, points_for_away).
    """
    p_home_reg, p_away_reg, p_home_ot, p_away_ot = match_probability(home_team, away_team)
    
    x = random.random()
    if x < p_home_reg:
        # Home team wins in regulation: 3:0
        return 3, 0
    x -= p_home_reg
    
    if x < p_away_reg:
        # Away team wins in regulation: 0:3
        return 0, 3
    x -= p_away_reg
    
    if x < p_home_ot:
        # Home team wins in OT/SO: 2:1
        return 2, 1
    else:
        # Away team wins in OT/SO: 1:2
        return 1, 2

# 4) Collect future (unplayed) games and prepare for Monte Carlo
future_games = [g for g in games_this_season if g.get('ended') == False]
base_points = copy.deepcopy(points)  # current points as a baseline

# 5) Monte Carlo simulation
N = 1000000
TPS_TEAMNAME = "TPS"
PLAYOFFS_CUTOFF = 12
tps_in_top12_count = 0

for _ in range(N):
    sim_points = copy.deepcopy(base_points)
    
    for g in future_games:
        home_team = g['homeTeam']['teamName']
        away_team = g['awayTeam']['teamName']
        
        hp, ap = simulate_single_game(home_team, away_team)
        sim_points[home_team] += hp
        sim_points[away_team] += ap
    
    # Sort the final points to see the ranking
    final_standing = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)
    
    # Determine TPS rank
    for rank, (team, pts) in enumerate(final_standing, start=1):
        if team == TPS_TEAMNAME:
            if rank <= PLAYOFFS_CUTOFF:
                tps_in_top12_count += 1
            break

prob_tps_in_top12 = tps_in_top12_count / N
print(f"\n=== After {N} simulations, TPS finishes in the top {PLAYOFFS_CUTOFF} in {prob_tps_in_top12:.2%} of runs ===")
