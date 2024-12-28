import pandas as pd
import numpy as np
from PIL import Image
from mplsoccer import PyPizza, add_image, FontManager
import urllib.request
import streamlit as st

params = ['Goals_per90', 'G+A_per90', 'G-PK_per90', 'G+A-PK_per90', 'xG_per90', 'xG+xAG_per90', 'npxG_per90',
        'npxG+xAG_per90', 'Take_Ons_Attempted', 'Take_Ons_Succ', 'Take_Ons_Succ%', 'Tackled_Take_Ons', 
        'Tackled_Take_Ons%', 'Touches_per_90', 'Touches_Def_Pen_per_90', 'Touches_Def_3rd_per_90', 
        'Touches_Mid_3rd_per_90', 'Touches_Att_3rd_per_90', 'Touches_Att_Pen_per_90', 'Tocuhes_Live_Balls_per_90',
        'Take_Ons_Attempted_per_90', 'Take_Ons_Succ_per_90', 'Tackled_Take_Ons_per_90', 'Carries_per_90', 
        'Total_Distance_per_90', 'Progressive_Distance_Carried_per_90', 'Progressive_Carries_per_90', 
        '1/3_Carries_per_90', 'Carries_Penalty_Area_per_90', 'Miscontrols_per_90', 'Dispossessed_per_90', 
        'Passes_Received_per_90', 'Progressive_Passes_Received_per_90', 'Shot_Creating_Action_per90', 
        'Goal_Creating_Action_90', 'Pass_Live_Shot_per_90', 'Pass_Dead_Shot_per_90', 'Take_Ons_Shot_per_90', 
        'Shot-Shot_per_90', 'Fouls_drawn_Shot_per_90', 'Defensive_Shot_per_90', 'Pass_Live_Goal_per_90',
        'Pass_Dead_Goal_per_90', 'Take_Ons_Goal_per_90', 'Shot_Goal_per_90', 'Fouls_Drawn_Goal_per_90', 
        'Defensive_Goal_per_90', 'Passes_Total_Cmp', 'Passes_Total_Att', 'Passes_Total_Cmp%', 'Passes_TotDist', 
        'Passes_PrgDist', 'Passes_Short_Cmp', 'Passes_Short_Att', 'Passes_Short_Cmp%', 'Passes_Medium_Cmp', 
        'Passes_Medium_Att', 'Passes_Medium_Cmp%', 'Passes_Long_Cmp', 'Passes_Long_Att', 'Passes_Long_Cmp%', 
        'Assists_per_90', 'xAG_per_90', 'xA_per_90', 'A-xAG_per_90', 'Key_Passes_per_90', 'Passes_1/3_per_90', 
        'Passes_Penalty_Area_per_90', 'Crosses_Penalty_Area_per_90', 'Progressive_Passes_per_90', 
        'Passes_Attempted_per_90', 'Live_Ball_Passes_per_90', 'Dead_Ball_Passes_per_90', 'Free_Kick_Passes_per_90',
        'Through_Balls_per_90', 'Switches_per_90', 'Crosses_per_90', 'Throw_Ins_Taken_per_90', 
        'Corner_Kicks_per_90', 'In_Corner_Kicks_per_90', 'Out_Corner_Kicks_per_90', 'Str_Corner_Kicks_per_90',
        'Passes_Cmp_per_90', 'Passes_Off_per_90', 'Passes_Blocked_per_90', 'Shots_total_per90', 
        'Shots_on_target_per90', 'Goals_per_shot', 'Goals_per_shot_on_target', 'Npxg_per_shot', 'Xg_net', 
        'Npxg_net', 'Percentage_of_Aerials_Won', 'Yellow_Cards_per_90', 'Red_Cards_per_90', 
        'Second_Yellow_Card_per_90', 'Fouls_Committed_per_90', 'Fouls_Drawn_per_90', 'Offsides_per_90', 
        'Interceptions_per_90', 'Tackles_Won_per_90', 'Penalty_Kicks_Won_per_90', 'Penalty_Kicks_Conceded_per_90',
        'Ball_Recoveries_per_90', 'Aerials_Won_per_90', 'Aerials_Lost_per_90']

def load_data(file_path):

    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)
              
    return df
    
def get_player_data(player_name, df):
    
    player_data = df[df['Player'].isin([player_name])]
    return player_data

def filter_player_data(df, position):

    filtered_df = df[(df['Pos'].isin(position.values)) ]
    filtered_df.reset_index(drop=True)
    
    return filtered_df
    
    
def get_player_percentile_ranks(df, player_name = None , param   = params):
    # Calculate percentile ranks for all players
    players_percentile_ranks = df[param].rank(pct=True) * 100
    if player_name :
        if player_name in df['Player'].values:  # Check if player exists
            print(player_name)
            # Find the index of the player in the DataFrame
            player_index = df[df['Player'] == player_name].index[0]
            print(player_index)
            # Get the player's percentile ranks
            player_data_rank = round(players_percentile_ranks.loc[[player_index]], 0)

        
            player_data_rank = player_data_rank.T
        
            player_data_rank = player_data_rank.reset_index()
            player_data_rank.columns = [ 'Parameters', 'Percentile Rank']


            sorted_percentile_ranks = player_data_rank.sort_values(by='Percentile Rank', ascending=False)
        
        
            # Get the top 15 parameters
            top_15_params = sorted_percentile_ranks['Parameters'][:15]

            # Get the player's data
            player_data = df.loc[player_index]
        else:
            print(f"Player '{player_name}' not found in data.")
            return pd.DataFrame(), pd.DataFrame(), []
    else :
        player_data_rank,top_15_params = pd.DataFrame(),[] 
    return players_percentile_ranks, player_data_rank, top_15_params

import matplotlib.pyplot as plt
from mplsoccer import PyPizza
import matplotlib

def plot_player_percentiles(player_name, team_name, competition, player_data_rank, top_15_params,figsize=(8, 8.5)):
  import matplotlib.colors as mcolors
  plt.style.use('bmh')
  cmap = matplotlib.colormaps['jet']
  slice_colors = cmap(np.linspace(0, 1, len(top_15_params))) 
  text_colors = ["#000000"] * 15
  
  split_params = [param.replace('_', '\n') for param in top_15_params]

  # Create the pizza plot object
  baker = PyPizza(
      params = split_params,
      background_color="#F0F0F0",
      straight_line_color="#EBEBE9",
      straight_line_lw=1,
      last_circle_lw=0,
      other_circle_lw=0,
      inner_circle_size=20
  )
   
  player_data_rank = player_data_rank[['Parameters','Percentile Rank']]
  transposed_df = player_data_rank.set_index('Parameters').T
  transposed_df = transposed_df[top_15_params]
  percentile_ranks_list = transposed_df.values.flatten().tolist()

  # Create the pizza plot
  fig, ax = baker.make_pizza(
      percentile_ranks_list,
      figsize=figsize,
      color_blank_space="same",
      slice_colors=slice_colors,
      value_colors=text_colors,
      value_bck_colors=slice_colors,
      blank_alpha=0.4,
      param_location=115,
      kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
      kwargs_params=dict(color="#000000", fontsize=11, va="center"),
      kwargs_values=dict(
          color="#000000", fontsize=11, zorder=3, bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.2", lw=1)
      )
  )

  # Add title and labels
  fig.text(0.515, 0.975, f"{player_name} - {team_name} FC", size=16, ha="center", color="#000000")
  fig.text(
      0.515,
      0.953,
      f"{competition} Stats Percentile Rank vs Top-Five Leagues  Forwards & Midfielders | 2023-24 Season",
      size=13,
      ha="center",
      color="#000000"
  )

  # Add credits
  CREDIT_1 = "@KimMathews data: statsbomb viz fbref"
  CREDIT_2 = "plot: @mplsoccer"
  fig.text(0.99, 0.02, f"{CREDIT_1}\n{CREDIT_2}", size=9, color="#000000", ha="right")

  return fig
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_players(player_name, players_data, param = params):


  # Identify features to minimize (negative impact on performance)
  negative_features = ['Yellow_Cards_per_90', 'Red_Cards_per_90', 'Second_Yellow_Card_per_90', 'Fouls_Committed_per_90',
                       'Aerials_Lost_per_90','Miscontrols_per_90','Dispossessed_per_90','Penalty_Kicks_Conceded_per_90']

  # Reverse the sign of negative features
  for feature in negative_features:
    if feature in players_data.columns:
      players_data[feature] = -players_data[feature]

  # Normalize data
  numerical_data = players_data[params].replace([np.inf, -np.inf], 0)
  normalized_data = (numerical_data - numerical_data.mean()) / numerical_data.std()

  # Apply PCA
  pca = PCA()
  pca.fit(normalized_data)
  explained_variances = np.cumsum(pca.explained_variance_ratio_)
  n_components = np.argmax(explained_variances >= 0.95) + 1
  n_components = 15
  # Apply PCA with the selected number of components
  pca = PCA(n_components=n_components)
  transformed_players = pca.fit_transform(normalized_data)

  # Convert transformed data back to DataFrame
  transformed_df = pd.DataFrame(transformed_players, index=players_data.index)

  # Get player's index and data
  try:
    player_index = players_data.index[players_data['Player'] == player_name][0]
    player_data = transformed_df.loc[player_index].values.reshape(1, -1)
    # Remove player from the dataset for comparison
    transformed_df = transformed_df.drop(index=player_index)
  except IndexError:
    player_index = 0

  

  # Compute cosine similarity
  similarity_scores = cosine_similarity(transformed_df, player_data).flatten()
  transformed_df['similarity_to_player'] = similarity_scores

  # Get top 10 players most similar to the input player
  top_similar_players_indices = transformed_df.sort_values('similarity_to_player', ascending=False).index[:10]
  print(top_similar_players_indices)
  top_similar_players_score = transformed_df.loc[top_similar_players_indices]['similarity_to_player']
  top_similar_players = players_data.loc[top_similar_players_indices]
  top_similar_players['Similarity Score'] = top_similar_players_score
    
  return top_similar_players,transformed_df,top_similar_players_indices
    
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def plot_player_similarity(player_data,  player_name, title_suffix=""):
  """
  This function plots a horizontal bar chart showing the most similar players to a given player.

  Args:
      player_data (pd.DataFrame): DataFrame containing player data with features in 'params'.
      player_name (str): Name of the player for whom to find similar players.
      title_suffix (str, optional): Optional suffix to add to the plot title. Defaults to "".
  """

  # Font properties
  font_path = 'Fonts/Ubuntu/Ubuntu-Medium.ttf'
  font_prop = fm.FontProperties(fname=font_path, size=12)

  # Use dark background style
  plt.style.use('bmh')

  # Sort players by similarity score (descending)
  player_data_sorted = player_data.sort_values(by='Similarity Score', ascending=False)

  # Create figure and axis
  fig, ax = plt.subplots(figsize=(10, 8))

  # Plot the horizontal bar chart
  bars = ax.barh(player_data_sorted['Player'].head(15)[::-1], (player_data_sorted['Similarity Score'].head(15)*100)[::-1],
                 color=plt.cm.viridis(np.linspace(0, 1, 15)))

  # Add labels and title
  ax.set_xlabel('Similarity Percentage', fontproperties=font_prop)
  ax.set_ylabel('Player', fontproperties=font_prop)
  ax.set_title(f'Who is most similar to {player_name}?{title_suffix}', fontproperties=font_prop)

  # Set y-axis tick labels with custom font
  ax.set_yticklabels(player_data_sorted['Player'].head(15)[::-1], fontproperties=font_prop)

  # Remove gridlines
  ax.grid(False)

  # Make top and right spines invisible
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  # Make left and bottom spines white
  ax.spines['left'].set_color('white')
  ax.spines['bottom'].set_color('white')

  # Change tick color to white
  ax.tick_params(colors='white')

  # Change label color to white
  ax.xaxis.label.set_color('white')
  ax.yaxis.label.set_color('white')
  ax.title.set_color('white')

  # Add percentage labels to bars
  for bar in bars:
    width = bar.get_width()
    y_pos = bar.get_y() + bar.get_height() / 2
    ax.text(width, y_pos, f'{width:.1f}%', ha='left', va='center',
            color='white', fontproperties=font_prop)

  # Save the figure
  #plt.savefig(f'Similar_{player_name}.png', dpi=500, bbox_inches='tight', transparent=False)

  #plt.show()
  return fig
    
