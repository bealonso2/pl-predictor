"""
This migration adds the max_points, matches_played, max_possible_position, and min_possible_position columns to the team_to_points table
in the results database.

Adds a version table to the results database to keep track of the version of the database.

matches_played has no intended use at the moment. max_points will be used to calculate
the maximum position possible.
"""

import sqlite3

import pandas as pd

# Connect to the two databases
with sqlite3.connect("results.db") as results_conn, sqlite3.connect(
    "data.db"
) as data_conn:
    results_cursor = results_conn.cursor()

    # Load the data from the two databases
    simulations_df = pd.read_sql_query("SELECT * FROM simulations", results_conn)
    matches_df = pd.read_sql_query(
        "SELECT * FROM football_data_season_results WHERE season = 2024", data_conn
    )

    # Map simulation uuids to dates
    uuids_to_dates = {row["uuid"]: row["date"] for _, row in simulations_df.iterrows()}

    # Get a unique list of teams
    teams = matches_df["home"].unique()

    # Calculate the total matches
    total_matches = (len(teams) - 1) * 2

    # Initialize uuid to updates dictionary
    uuid_to_db_updates = {uuid: [] for uuid in uuids_to_dates.keys()}

    for uuid, date in uuids_to_dates.items():
        # Get matches up to the date of the simulation
        matches_up_to_date: pd.DataFrame = matches_df[matches_df["utc_date"] <= date]

        # Get points for each team
        team_to_points = {team: 0 for team in teams}

        # Get matches played for each team
        team_to_matches_played = {team: 0 for team in teams}

        for _, match in matches_up_to_date.iterrows():
            # Update the points and matches played for each team
            team_to_points[match["home"]] += match["home_outcome"]
            team_to_points[match["away"]] += match["away_outcome"]
            team_to_matches_played[match["home"]] += 1
            team_to_matches_played[match["away"]] += 1

        # Calculate the maximum number of points a team can get
        team_to_max_points = {
            team: ((total_matches - team_to_matches_played[team]) * 3) + points
            for team, points in team_to_points.items()
        }

        # Calculate the lowest position a team can possibly finish
        team_to_min_position = {}
        for team, points in team_to_points.items():
            # Copy the max points dictionary
            team_to_max_points_copy = {
                team: points for team, points in team_to_max_points.items()
            }

            # Replace the team's points with the minimum possible points (current points)
            team_to_max_points_copy[team] = points

            # Get the team's position
            team_position = sorted(
                team_to_max_points_copy.items(), key=lambda x: x[1], reverse=True
            ).index((team, points))

            # Add the team's position to the dictionary
            team_to_min_position[team] = team_position + 1

        # Calculate the highest position a team can possibly finish
        team_to_max_position = {}
        for team, max_points in team_to_max_points.items():
            # Copy the points dictionary
            team_to_points_copy = {
                team: points for team, points in team_to_points.items()
            }

            # Replace the team's points with the maximum possible points
            team_to_points_copy[team] = max_points

            # Get the team's position
            team_position = sorted(
                team_to_points_copy.items(), key=lambda x: x[1], reverse=True
            ).index((team, max_points))

            # Add the team's position to the dictionary
            team_to_max_position[team] = team_position + 1

        # Zip the max points and matches played dictionaries
        team_to_migration_results = {
            team: {
                "max_points": team_to_max_points[team],
                "matches_played": matches_played,
                "max_possible_position": team_to_max_position[team],
                "min_possible_position": team_to_min_position[team],
            }
            for team, matches_played in team_to_matches_played.items()
        }

        # Reformat the data to update the database
        uuid_to_db_updates[uuid] = [
            {team: team_results}
            for team, team_results in team_to_migration_results.items()
        ]

    # Store the results in the database

    # Create a copy of the team_to_points table
    results_cursor.execute(
        "CREATE TABLE team_to_points_copy AS SELECT * FROM team_to_points"
    )

    # Drop the original table
    results_cursor.execute("DROP TABLE team_to_points")

    # Add the new columns to the copy table
    results_cursor.execute(
        "ALTER TABLE team_to_points_copy ADD COLUMN max_points INTEGER"
    )
    results_cursor.execute(
        "ALTER TABLE team_to_points_copy ADD COLUMN matches_played INTEGER"
    )
    results_cursor.execute(
        "ALTER TABLE team_to_points_copy ADD COLUMN max_possible_position INTEGER"
    )
    results_cursor.execute(
        "ALTER TABLE team_to_points_copy ADD COLUMN min_possible_position INTEGER"
    )

    # Update the database with the new values
    for uuid, updates in uuid_to_db_updates.items():
        for update in updates:
            for team, values in update.items():
                results_cursor.execute(
                    "UPDATE team_to_points_copy SET max_points = ?, matches_played = ?, max_possible_position = ?, min_possible_position = ? WHERE team = ? AND simulation_uuid = ?",
                    (
                        values["max_points"],
                        values["matches_played"],
                        values["max_possible_position"],
                        values["min_possible_position"],
                        team,
                        uuid,
                    ),
                )

    # Create a new table with the new columns
    results_cursor.execute(
        "CREATE TABLE team_to_points (simulation_uuid TEXT, team TEXT, points INTEGER, max_points INTEGER, matches_played INTEGER, max_possible_position INTEGER, min_possible_position INTEGER)"
    )

    # Copy the data from the copy table to the new table
    results_cursor.execute(
        "INSERT INTO team_to_points (simulation_uuid, team, points, max_points, matches_played, max_possible_position, min_possible_position) SELECT simulation_uuid, team, points, max_points, matches_played, max_possible_position, min_possible_position FROM team_to_points_copy"
    )

    # Drop the copy table
    results_cursor.execute("DROP TABLE team_to_points_copy")

    # Add a version table to the database
    results_cursor.execute(
        "CREATE TABLE IF NOT EXISTS version (version INTEGER PRIMARY KEY, notes TEXT)"
    )

    # Set the version of the database with an optional note
    results_cursor.execute(
        "INSERT INTO version (version, notes) VALUES (1, 'Initial version')"
    )

    # Commit the changes
    results_conn.commit()
