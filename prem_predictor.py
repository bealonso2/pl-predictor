import json
import random
from unicodedata import name

# adding functionality for 'FORM' by using fixtures. Right now fixtures won't change
FORM = 0.005

# this class encapsulates a premier league team
class Team():

    # name field consistant with fixtures
    name = ""

    # optional display name to be consistant with PL website
    display_name = ""

    # the probability of a win at home and away from home
    home_prob = 0.0
    away_prob = 0.0

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
    def __init__(self,name_in,display_name_in,home_prob_in,away_prob_in):
        self.name = name_in
        self.display_name = display_name_in
        self.home_prob = home_prob_in
        self.away_prob = away_prob_in

    # returns the home win probability
    def getHome(self):
        return float(self.home_prob)

    # returns the away win probability
    def getAway(self):
        return float(self.away_prob)

    # returns the number of points
    def getPoints(self):
        return self.points

    # returns the number of wins
    def getWs(self):
        return self.wins

    # returns the number of draws
    def getDraws(self):
        return self.draws

    # returns the number of losses
    def getLs(self):
        return self.losses

    # returns the number of matches played
    def getMP(self):
        return self.matches_played

    # returns the display name or name if no display name
    def getDisplayName(self):
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
    def result(self, finish):
        self.finish += finish
        self.total_points += self.points
        self.total_wins += self.wins
        self.total_draws += self.draws
        self.total_losses += self.losses
        self.points = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        

# function to parse my input file. Paraneter is the line of the file,
# returns a team
def parse_team(to_parse: dict):
    name = to_parse.get('name')
    display_name = to_parse.get('display name')
    home = float(to_parse.get('home'))
    away = float(to_parse.get('away'))
    return Team(name, display_name, home, away)

# open fixtures file
with open('./2022-2023/2022-2023_fixtures.json', 'r') as f:
    fixtures = json.load(f)

# open the input file
with open('./2022-2023/2022-2023_teams.json', 'r') as f:
    teams_raw: dict = json.load(f)

# fill out the array of teams
teams = {}
for i_team in teams_raw.values():
    new_team = parse_team(i_team)
    teams[new_team.name] = new_team

# close the file to prevent data leaks
f.close()

# save season starting probablilties
start_prob = {}
for team_name, team in teams.items():
    start_prob[team_name] = {"home" : team.getHome(),
                            "away" : team.getAway()}

# variable to set how many seasons to sim
sims: int = 20
for x in range(sims):
    for fixture in fixtures:
        home_side = teams.get(fixture.get('HomeTeam'))
        away_side = teams.get(fixture.get('AwayTeam'))

        home = home_side.getHome()
        away = away_side.getAway()
        # computing the draw constant
        # it should be less likely to draw if one team is really good and 
        # one team is really bad
        # draw_const = 0.27 + home - away #/ (1 + home - away)
        total = home + away
        draw_const = 0.27 * abs(1 - home - away) / total
        draw = 100*(draw_const + home + away)

        # using a random number generator to determine the winner
        winner = random.randint(0,int(draw)) / 100.0

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

    for team_name, probs in start_prob.items():
        teams.get(team_name).reset(probs.get('home'), probs.get('away'))

    final_order = sorted(teams.values(), reverse=True, key=lambda team: team.points)
    for team_name, team in teams.items():
        team.result(final_order.index(team) + 1)
    

# output into a table
print()
# header
print(f"{'': <3} {'Team':<20s} Finish  Points  {'W':>4}  {'D':>4}  {'L':>4}")
print(f"{'':-<57}")
# sort the teams by points
final_order = sorted(teams.values(), reverse=False, key=lambda team: team.finish)
# output the teams
[print(f"{str(final_order.index(team) + 1) + ')':<3} {team.getDisplayName():<20.20s} {team.finish/sims:>6.1f}  {team.total_points/sims:>6.1f}  {team.total_wins/sims:4.1f}  {team.total_draws/sims:4.1f}  {team.total_losses/sims:4.1f}") for team in final_order]
print()
