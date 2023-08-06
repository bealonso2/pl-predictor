import json
from datetime import datetime

from classes.fixture import Fixture
from classes.team import Team

CURRENT_YEAR = datetime.now().year


class PremierLeaguePredictor:
    # the teams for the season
    _teams: list[Team]

    # the fixtures for the season
    _fixtures: list[Fixture]

    # number of seasons to simulate
    _seasons: int

    def __init__(self, seasons: int = 20) -> None:
        # build the teams and fixtures
        teams_dict = build_teams()
        self._fixtures = build_fixtures(teams_dict)
        self._teams = teams_dict.values()
        self._seasons = seasons

    def _simulate_single_season(self) -> None:
        for fixture in self._fixtures:
            # using a random number generator to determine the winner
            fixture.play_match()

            # reset the teams for the next 'season' or simulation
            final_order = sorted(
                self._teams, reverse=True, key=lambda team: team.points
            )

            for team in self._teams:
                # Set the team's final position
                team.result(final_order.index(team) + 1)

    def simulate_all_seasons(self) -> None:
        # simulating the seasons
        for _ in range(self._seasons):
            self._simulate_single_season()

    def to_csv(self) -> None:
        print("Warning: CSV output not implemented yet")
        return

    def to_json(self) -> None:
        print("Warning: JSON output not implemented yet")
        return

    def __str__(self) -> str:
        # output to a table
        sim_str = ""
        # header
        sim_str += (
            f"\n{'': <3} {'Team':<20s} Finish  Points  {'W':>4}  {'D':>4}  {'L':>4}\n"
        )
        sim_str += f"{'':-<57}\n"
        # sort the teams by points and output the teams
        for position, team in enumerate(
            sorted(self._teams, reverse=False, key=lambda team: team.finish), 1
        ):
            avg_finish = team.finish / self._seasons
            avg_points = team.total_points / self._seasons
            avg_wins = team.total_wins / self._seasons
            avg_draws = team.total_draws / self._seasons
            avg_losses = team.total_losses / self._seasons

            # Add the team to the string
            sim_str += f"{str(position) + ')':<3} {team.display_name:<20.20s} {avg_finish:>6.1f}  {avg_points:>6.1f}  {avg_wins:4.1f}  {avg_draws:4.1f}  {avg_losses:4.1f}\n"

        return sim_str + f"{'':-<57}\n"


def build_teams() -> dict[str, Team]:
    """Builds the teams for the season from the teams.json file

    Returns:
        dict[str, Team]: A dictionary of teams for the season
    """

    # open the input file
    with open(f"./{CURRENT_YEAR}-{CURRENT_YEAR + 1}/teams.json", "r") as f:
        teams_raw: dict = json.load(f)

    # fill out the array of teams
    teams: dict[str, Team] = {
        team_name: parse_team(team_name, team_values)
        for team_name, team_values in teams_raw.items()
    }

    return teams


def build_fixtures(teams: dict[str, Team]) -> list[Fixture]:
    """Builds the fixtures for the season

    Args:
        teams (dict[str, Team]): Teams available for the season

    Returns:
        list[Fixture]: A list of fixtures for the season
    """
    # open fixtures file
    with open(f"./{CURRENT_YEAR}-{CURRENT_YEAR + 1}/fixtures.json", "r") as f:
        fixtures_raw = json.load(f)

    # fill out the array of fixtures
    fixtures: list[Fixture] = [
        Fixture(teams.get(fixture["HomeTeam"]), teams.get(fixture["AwayTeam"]))
        for fixture in fixtures_raw
    ]

    return fixtures


# function to parse my input file. Parameter is the line of the file,
# returns a team
def parse_team(team_name: str, team_values: dict) -> Team:
    """Function to parse the team values from the input file"""
    # Display name is optional so use get
    display_name = team_values.get("display name", None)
    home = float(team_values["home"])
    away = float(team_values["away"])
    return Team(team_name, display_name, home, away)
