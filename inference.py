import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm

from simulation_utils import (
    build_elo_before_season,
    db_get_data_for_latest_season,
    db_store_results,
    download_best_params_from_s3,
    download_model_and_scaler_from_s3,
    get_elo_dict_from_df,
    simulate_and_get_results,
)


def main():
    # Create an argument parser with the optional argument --num_simulations -n
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_simulations",
        type=int,
        default=10,
        help="Number of simulations to run",
    )
    args = parser.parse_args()

    # Number of simulations to run
    num_simulations = args.num_simulations

    # Get data for the latest season
    df = db_get_data_for_latest_season()

    # Build the elo dataframe before the current season
    elo_df, team_to_points = build_elo_before_season(df)

    # Get the model and scalar from S3
    model_file = Path("random_forest.joblib")
    scaler_file = Path("standard_scaler.joblib")
    download_model_and_scaler_from_s3(model_file, scaler_file)
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # Get adjusted Elo dict
    adjusted_elo = get_elo_dict_from_df(elo_df)

    # Initialize a list to store results
    seasons = []

    # Get the best parameters for the model
    best_params = download_best_params_from_s3()

    # Create a partial function to pass the same arguments to each simulation
    simulate_and_get_results_partial = partial(
        simulate_and_get_results,
        df=df,
        elo=adjusted_elo,
        model=model,
        scaler=scaler,
        k=best_params.k,
        half_life=best_params.half_life,
        decay_method=best_params.decay_method,
    )

    # Initialize a pool of workers
    with ProcessPoolExecutor() as executor:
        seasons = list(
            tqdm(
                executor.map(simulate_and_get_results_partial, range(num_simulations)),
                total=num_simulations,
                desc="Simulating",
                unit="season",
            )
        )
    # Unlink the model and scaler files
    model_file.unlink()
    scaler_file.unlink()

    # Aggregate the results
    results = pd.concat(seasons).groupby("team").agg({"total_outcome": "sum"})

    # Sort results based on total outcome
    results = results.sort_values("total_outcome", ascending=False)

    # Get the place each team finished in the league
    results["place"] = range(1, len(results) + 1)

    # For each season, get the place each team finished in the league
    for index, season_df in enumerate(seasons):
        season_df["place"] = range(1, len(season_df) + 1)

    # Get the average place each team finished in the league
    average_results = (
        pd.concat(seasons).groupby("team").agg({"place": "mean"}).sort_values("place")
    )

    # Get the season from seasons
    list_of_seasons = df["season"].unique()

    assert len(list_of_seasons) == 1, "All seasons should be the same"

    # Get the season
    simulation_season = list_of_seasons[0]

    # Get a mapping of team names to a list places they finished in the league
    team_place_mapping = {}
    for team in average_results.index:
        team_place_mapping[team] = [
            season_df.loc[team, "place"] for season_df in seasons
        ]

    # Get the total number of seasons simulated
    total_seasons = len(seasons)

    # Get a mapping of times each team won the league
    team_win_mapping = {}
    for team in average_results.index:
        team_win_mapping[team] = (
            sum([season_df.loc[team, "place"] == 1 for season_df in seasons])
            / total_seasons
        )

    # Get a mapping of times each team finished in the top 4
    team_top_4_mapping = {}
    for team in average_results.index:
        team_top_4_mapping[team] = (
            sum([season_df.loc[team, "place"] <= 4 for season_df in seasons])
            / total_seasons
        )

    # Get a mapping of times each team finished in the bottom 3
    team_bottom_3_mapping = {}
    for team in average_results.index:
        team_bottom_3_mapping[team] = (
            sum(
                [
                    season_df.loc[team, "place"] > len(season_df) - 3
                    for season_df in seasons
                ]
            )
            / total_seasons
        )

    # Build a dataframe with the average place, times won, times in top 4, and times in bottom 3
    average_results["win_premier_league"] = [
        team_win_mapping[team] for team in average_results.index
    ]
    average_results["top_4"] = [
        team_top_4_mapping[team] for team in average_results.index
    ]
    average_results["bottom_3"] = [
        team_bottom_3_mapping[team] for team in average_results.index
    ]

    team_to_position = {team: {} for team in df["home"].unique()}
    for season in seasons:
        # Assign the position of each team
        for i, team in enumerate(season.index, 1):
            if i not in team_to_position[team]:
                team_to_position[team][i] = 0
            team_to_position[team][i] += 1

    # Create a dataframe where each row is a team and the one column is an array of positions in which the team finished in the league
    team_positions_df = (
        pd.DataFrame.from_dict(team_to_position, orient="index").fillna(0).astype(int)
    )

    # Stack the dataframe to get a row for each team-position pair
    team_positions_df = (
        team_positions_df.stack()
        .reset_index()
        .rename(columns={"level_0": "team", "level_1": "position", 0: "count"})
    )

    # Set the index to the team name-position pair
    team_positions_df.set_index(["team", "position"], inplace=True)

    # Sort the position index
    team_positions_df = team_positions_df.sort_index()

    # Store the results in the database
    db_store_results(
        simulation_season, average_results, team_positions_df, team_to_points
    )


if __name__ == "__main__":
    main()
