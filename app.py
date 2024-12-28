import streamlit as st
import player_analysis as pa

def main():
    st.title("Player Similarity Analysis")
    st.sidebar.info('Welcome to the Football Players Similarity Analysis Tool')
    player_season = st.sidebar.selectbox(
        "Select Season to analyse",
        ["2024", "2023", "2022", "2021"]
    )
    # Text input for player name
    #player_name = st.sidebar.text_input("Enter Player Name")
    
    
    
    if player_season == "2024":
        players_data = pa.load_data('Data Processed/Full_Players_2024.csv')
    elif player_season == "2023":
        players_data = pa.load_data('Data Processed/Full_Players_2023.csv')
    elif player_season == "2022":
        players_data = pa.load_data('Data Processed/Full_Players_2022.csv')
    elif player_season == "2021":
        players_data = pa.load_data('Data Processed/Full_Players_2021.csv')
    else:
        raise ValueError("Invalid season")

    # Player selection using a search box
    player_name = st.sidebar.text_input("Search Player Name", key="player_search")
    
    filtered_players = [p for p in players_data['Player'] if player_name.lower() in p.lower()]

    if filtered_players:
        selected_player = st.sidebar.selectbox("Select Player", filtered_players)
    else:
        selected_player = st.sidebar.text_input("Player not found. Enter name again:", key="player_not_found")
    

    player_name = selected_player
    # Dropdown menu for season
    seasons = st.sidebar.multiselect(
        "Select Seasons to analyse",
        ["2021", "2022", "2023", "2024"]
    )
    
    #final_player = pa.load_data('Data Processed/final_players.csv')
    player_data = pa.get_player_data(player_name,players_data)
   

    player_2025 = pa.load_data('Data Processed/Full_Players_2025.csv')

    player_2024 = pa.load_data('Data Processed/Full_Players_2024.csv')
    player_2023 = pa.load_data('Data Processed/Full_Players_2023.csv')
    player_2022 = pa.load_data('Data Processed/Full_Players_2022.csv')
    player_2021 = pa.load_data('Data Processed/Full_Players_2021.csv')
    newbie_2024 = pa.load_data('Data Processed/newbie_players_2024.csv')
    
    player_data_2024 = pa.get_player_data(player_name,player_2024)
    player_data_2023 = pa.get_player_data(player_name,player_2023)
    player_data_2022 = pa.get_player_data(player_name,player_2022)
    player_data_2021 = pa.get_player_data(player_name,player_2021)

 # Display player details if found
    if not player_data.empty:
        st.write("Player Details:")
        st.table(player_data[['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age']])
        player_pos = player_data['Pos']
    else:
        st.write("Player not found.")

    final_play_pos = pa.filter_player_data(players_data,player_pos)
    newbie_2024_pos = pa.filter_player_data(newbie_2024,player_pos)
    
    for season in seasons:
        if season == "2024":
            player_2024 = pa.filter_player_data(player_2024,player_pos)
        elif season == "2023":
            player_2023 = pa.filter_player_data(player_2023,player_pos)
        elif season == "2022":
            player_2022 = pa.filter_player_data(player_2022,player_pos)
        elif season == "2021":
            player_2021 = pa.filter_player_data(player_2021,player_pos)
        else:
            raise ValueError("Invalid season")
    
    players_percentile_ranks, player_data_rank,top_15_params = pa.get_player_percentile_ranks( final_play_pos,player_name )  

    sorted_percentile_ranks = player_data_rank.sort_values(by='Percentile Rank', ascending=False)
    sorted_percentile_ranks = sorted_percentile_ranks.reset_index()
    st.table(sorted_percentile_ranks[['Parameters' ,'Percentile Rank']][:15])
    
    for season in seasons:
        if season == "2024":
            players_percentile_ranks_2024, player_data_rank_2024,top_15_params_2024 = pa.get_player_percentile_ranks( player_2024,param =  top_15_params)
        elif season == "2023":
            players_percentile_ranks_2023, player_data_rank_2023,top_15_params_2023 = pa.get_player_percentile_ranks( player_2023,param =  top_15_params )
        elif season == "2022":
            players_percentile_ranks_2022, player_data_rank_2022,top_15_params_2022 = pa.get_player_percentile_ranks( player_2022,param =  top_15_params )
        elif season == "2021":
            players_percentile_ranks_2021, player_data_rank_2021,top_15_params_2021 = pa.get_player_percentile_ranks( player_2021,param =  top_15_params ) 
        else:
            raise ValueError("Invalid season")
    
    player_team = player_data['Squad'].str.replace('+', '/', regex=False).values[0]
    player_comp = player_data['Comp'].str.replace('+', '/', regex=False).values[0]
        
    fig = pa.plot_player_percentiles(player_name, player_team, player_comp, player_data_rank,top_15_params)
    st.pyplot(fig)
    
    similar_players,transformed_df,top_similar_players_indices = pa.find_similar_players(player_name, player_2024)
    st.write(similar_players)
    fig = pa.plot_player_similarity(similar_players, player_name, title_suffix="")
    st.pyplot(fig)

if __name__ == "__main__":
    main()