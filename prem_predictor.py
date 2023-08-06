import json
import random
from datetime import datetime

CURRENT_YEAR = datetime.now().year

# adding functionality for 'FORM' by using fixtures. Right now fixtures won't change
FORM = 0.005


# this class encapsulates a premier league team
class Team:
    # name field consistent with fixtures
    name = ""

    # optional display name to be consistent with PL website
    display_name = ""

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
    def __init__(self, name_in, display_name_in, home_prob_in, away_prob_in):
        self.name = name_in
        self.display_name = display_name_in
        self.home_prob = home_prob_in
        self.away_prob = away_prob_in
        self._starting_home_prob = home_prob_in
        self._starting_away_prob = away_prob_in

    # returns the home win probability
    @property
    def home(self):
        return float(self.home_prob)

    # returns the away win probability
    @property
    def away(self):
        return float(self.away_prob)

    # returns the display name or name if no display name
    @property
    def display_name(self):
        return self.display_name if self.display_name else self.name

    # function to handle wins
    def win(self):
        self.points += 3
        self.wins += 1
        self.matches_played += 1
        if self.home_prob < 1:
            self.home_prob += FORM
        if self.away_prob < 1:
            self.away_prob += FORM

    # function to handle losses
    def lose(self):
        self.losses += 1
        self.matches_played += 1
        if self.home_prob > 5 * FORM:
            self.home_prob -= FORM
        if self.away_prob > 5 * FORM:
            self.away_prob -= FORM

    # function to handle draws
    def draw(self):
        self.points += 1
        self.draws += 1
        self.matches_played += 1

    # reset a teams home and away probability to start a new season
    def reset(self, home_prob, away_prob):
        self.home_prob = home_prob
        self.away_prob = away_prob

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


# function to parse my input file. Parameter is the line of the file,
# returns a team
def parse_team(to_parse: dict):
    name = to_parse.get("name")
    display_name = to_parse.get("display name")
    home = float(to_parse.get("home"))
    away = float(to_parse.get("away"))
    return Team(name, display_name, home, away)


def play_match(home_side: Team, away_side: Team) -> None:
    # Get the home and away probabilities
    home = home_side.home
    away = away_side.away

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
        home_side.draw()
        away_side.draw()
    elif winner > draw_const and winner < draw_const + home:
        home_side.win()
        away_side.lose()
    else:
        home_side.lose()
        away_side.win()


def main():
    # open fixtures file
    with open(f"./{CURRENT_YEAR}-{CURRENT_YEAR + 1}/fixtures.json", "r") as f:
        fixtures = json.load(f)

    # open the input file
    with open(f"./{CURRENT_YEAR}-{CURRENT_YEAR + 1}/teams.json", "r") as f:
        teams_raw: dict = json.load(f)

    # fill out the array of teams
    teams: dict[str, Team] = {}
    for i_team in teams_raw.values():
        new_team = parse_team(i_team)
        teams[new_team.name] = new_team

    # save season starting probabilities
    start_prob = {}
    for team_name, team in teams.items():
        start_prob[team_name] = {"home": team.home, "away": team.away}

    # variable to set how many seasons to sim
    sims: int = 20
    for _ in range(sims):
        for fixture in fixtures:
            home_side: Team = teams.get(fixture["HomeTeam"])
            away_side: Team = teams.get(fixture["AwayTeam"])

            # using a random number generator to determine the winner
            play_match(home_side, away_side)

        for team_name, probs in start_prob.items():
            teams.get(team_name).reset(probs["home"], probs["away"])

        # reset the teams for the next 'season' or simulation
        final_order = sorted(teams.values(), reverse=True, key=lambda team: team.points)
        for team_name, team in teams.items():
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
