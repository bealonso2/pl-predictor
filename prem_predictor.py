import json
import random
from datetime import datetime
from typing import Union

CURRENT_YEAR = datetime.now().year

# adding functionality for 'FORM' by using fixtures. Right now fixtures won't change
FORM = 0.005


# this class encapsulates a premier league team
class Team:
    # name field consistent with fixtures
    name = ""

    # optional display name to be consistent with PL website
    _display_name = ""

    # the probability of a win at home and away from home
    home_prob = 0.0
    away_prob = 0.0
    _starting_home_prob = 0.0
    _starting_away_prob = 0.0

    # the teams stats
    points: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    matches_played: int = 0

    # end of season stats
    finish: int = 0
    total_points: int = 0
    total_wins: int = 0
    total_draws: int = 0
    total_losses: int = 0

    # the team can be constructed using these three parameters
    def __init__(
        self,
        name: str,
        display_name: Union[str, None],
        home_prob: float,
        away_prob: float,
    ) -> None:
        self.name = name
        self._display_name = display_name
        self.home_prob = home_prob
        self.away_prob = away_prob
        self._starting_home_prob = home_prob
        self._starting_away_prob = away_prob

    # returns the display name or name if no display name
    @property
    def display_name(self):
        return self._display_name if self._display_name else self.name

    def match_result(self, func: callable) -> None:
        """Function to handle the result of a match

        Args:
            func (callable): The function to call to handle the result
        """
        func()
        self.matches_played += 1

    @match_result
    def win(self):
        """Function to handle wins"""
        self.points += 3
        self.wins += 1
        if self.home_prob < 1:
            self.home_prob += FORM
        if self.away_prob < 1:
            self.away_prob += FORM

    @match_result
    def lose(self):
        """Function to handle losses"""
        self.losses += 1
        self.matches_played += 1
        if self.home_prob > 5 * FORM:
            self.home_prob -= FORM
        if self.away_prob > 5 * FORM:
            self.away_prob -= FORM

    @match_result
    def draw(self):
        """Function to handle draws"""
        self.points += 1
        self.draws += 1
        self.matches_played += 1

    # variable to keep track of table position
    def result(self, finish: int) -> None:
        """Function to reset the teams stats and add previous season stats to
        cumulative stats

        Args:
            finish (int): The position the team finished in the table
        """
        # Set the fields that are cumulative
        self.finish += finish
        self.total_points += self.points
        self.total_wins += self.wins
        self.total_draws += self.draws
        self.total_losses += self.losses

        # Reset the team stats
        self.points = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0

        # Reset the team with the starting probabilities
        self.home_prob = self._starting_home_prob
        self.away_prob = self._starting_away_prob


# this class encapsulates a premier league fixture
class Fixture:
    # the home and away teams
    home: Team
    away: Team

    # the match can be constructed using these three parameters
    def __init__(self, home: Team, away: Team) -> None:
        self.home = home
        self.away = away

    def play_match(self) -> None:
        """Function to play the match"""
        # Get the home and away probabilities
        home = self.home.home_prob
        away = self.away.away_prob

        # computing the draw constant
        # it should be less likely to draw if one team is really good and
        # one team is really bad
        # draw_const = 0.27 + home - away #/ (1 + home - away)
        total = home + away
        draw_const = 0.27 * abs(1 - home - away) / total

        draw = 100 * (draw_const + home + away)
        winner = random.randint(0, int(draw)) / 100.0

        # logic to assign points
        if winner < draw_const:
            self.home.draw()
            self.away.draw()
        elif winner > draw_const and winner < draw_const + home:
            self.home.win()
            self.away.lose()
        else:
            self.home.lose()
            self.away.win()


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
def parse_team(team_name: str, team_values: dict):
    display_name = team_values["display name"]
    home = float(team_values["home"])
    away = float(team_values["away"])
    return Team(team_name, display_name, home, away)


def main():
    # build the teams and fixtures
    teams = build_teams()
    fixtures = build_fixtures(teams)

    # variable to set how many seasons to sim
    sims: int = 20

    # simulating the seasons
    for _ in range(sims):
        for fixture in fixtures:
            # using a random number generator to determine the winner
            fixture.play_match()

        # reset the teams for the next 'season' or simulation
        final_order = sorted(teams.values(), reverse=True, key=lambda team: team.points)

        for team in teams.values():
            # Set the team's final position
            team.result(final_order.index(team) + 1)

    # output to a table
    # header
    print(f"\n{'': <3} {'Team':<20s} Finish  Points  {'W':>4}  {'D':>4}  {'L':>4}")
    print(f"{'':-<57}")
    # sort the teams by points
    final_order = sorted(teams.values(), reverse=False, key=lambda team: team.finish)
    # output the teams
    [
        print(
            f"{str(final_order.index(team) + 1) + ')':<3} {team.display_name:<20.20s} {team.finish/sims:>6.1f}  {team.total_points/sims:>6.1f}  {team.total_wins/sims:4.1f}  {team.total_draws/sims:4.1f}  {team.total_losses/sims:4.1f}"
        )
        for team in final_order
    ]
    print()

    # TODO json output as well, maybe csv too?


if __name__ == "__main__":
    main()
