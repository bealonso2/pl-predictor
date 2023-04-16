import random
filename = 'teams.csv'

# this class encapsulates a premier league team
class team():

    # the fields that define a team
    name = ""

    # the probability of a win at home and away from home
    home_prob = 0.0
    away_prob = 0.0

    # the teams stats
    points = 0
    wins = 0
    draws = 0
    losses = 0

    # the team can be constructed using these three parameters
    def __init__(self,name_in,home_prob_in,away_prob_in):
        self.name = name_in
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

    # function to handle wins
    def win(self):
        self.points = self.points + 3
        self.wins = self.wins + 1

    # function to handle losses
    def lose(self):
        self.losses = self.losses + 1

    # function to handle draws
    def draw(self):
        self.points = self.points + 1
        self.draws = self.draws + 1

# function to parse my input file. Paraneter is the line of the file,
# returns a team
def parse_team(to_parse):
    name = ""
    home = 0.0
    away = 0.0
    split_str = to_parse.split(',')
    name = split_str[0]
    home = split_str[1]
    away = split_str[2]
    return team(name,home,away)

# open the input file
f = open(filename,'r')
f.readline() # read header
fl = f.readlines() # read the rest of the lines

# fill out the array of teams
teams = []
for i in range(len(fl)): 
    to_parse = fl[i]
    new_team = parse_team(to_parse)
    teams.append(new_team)

# close the file to prevent data leaks
f.close()

# probability of home and away teams
home = 0.0
away = 0.0

# variable to set how many seasons to sim
sims = 1
for x in range(sims):
    # each team playing their home games
    for i in range(len(teams)):
        current = teams[i]
        # playing every other team
        for j in range(len(teams)):
            if (i != j):
                opp = teams[j]
                home = current.getHome()
                away = opp.getAway()
                # computing the draw constant
                # it should be less likely to draw if one team is really good and 
                # one team is really bad
                draw_const = 0.27 + home - away #/ (1 + home - away)
                draw = 100*(draw_const + home + away)

                # using a random number generator to determine the winner
                winner_int = random.randint(0,int(draw))
                winner = winner_int / 100.0

                # logic to assign points
                if (winner > (home + away)):
                    teams[i].draw()
                    teams[j].draw()
                elif (winner > home):
                    teams[j].win()
                    teams[i].lose()
                else:
                    teams[i].win()
                    teams[j].lose()

# output into a table
print()
# header
print("    %-23s Points  W   D   L" %"Team")
print("----------------------------------------------")
# sort the teams by points
final_order = sorted(teams, reverse=True, key=lambda team: team.points)
# output the teams      
for i in range(len(final_order)):
    print("%-2d) %-24s %5d %3d %3d %3d" %(i+1, final_order[i].name, final_order[i].getPoints()/sims,
        final_order[i].getWs()/sims,final_order[i].getDraws()/sims,final_order[i].getLs()/sims))
print()
