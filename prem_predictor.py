import random
filename = 'teams.csv'

class team():
    name = ""
    home_prob = 0.0
    away_prob = 0.0
    points = 0
    wins = 0
    draws = 0
    losses = 0

    def __init__(self,name_in,home_prob_in,away_prob_in):
        self.name = name_in
        self.home_prob = home_prob_in
        self.away_prob = away_prob_in

    def getHome(self):
        return float(self.home_prob)

    def getAway(self):
        return float(self.away_prob)

    def getPoints(self):
        return self.points

    def getWs(self):
        return self.wins

    def getDraws(self):
        return self.draws

    def getLs(self):
        return self.losses

    def win(self):
        self.points = self.points + 3
        self.wins = self.wins + 1

    def lose(self):
        self.losses = self.losses + 1

    def draw(self):
        self.points = self.points + 1
        self.draws = self.draws + 1

def parse_team(to_parse):
    name = ""
    home = 0.0
    away = 0.0
    split_str = to_parse.split(',')
    name = split_str[0]
    home = split_str[1]
    away = split_str[2]
    return team(name,home,away)

f = open(filename,'r')
f.readline()
fl = f.readlines()

teams = []
for i in range(len(fl)): 
    to_parse = fl[i]
    new_team = parse_team(to_parse)
    teams.append(new_team)

f.close()
home = 0.0
away = 0.0
sims = 0
for x in range(1):
    for i in range(len(teams)):
        current = teams[i]
        for j in range(len(teams)):
            if (i != j):
                opp = teams[j]
                home = current.getHome()
                away = current.getAway()
                draw_const = 1.3 / (1 + home-away)
                draw = 100*(draw_const/(home + away))
                winner_int = random.randint(0,int(draw))
                winner = winner_int / 100.0
                if (winner > (home + away)):
                    teams[i].draw()
                    teams[j].draw()
                elif (winner > home):
                    teams[j].win()
                    teams[i].lose()
                else:
                    teams[i].win()
                    teams[j].lose()
    sims = sims + 1

# output
print()
print("    %-23s Points  W   D   L" %"Team")
print("----------------------------------------------")
final_order = sorted(teams, reverse=True, key=lambda team: team.points)            
for i in range(len(final_order)):
    print("%-2d) %-24s %5d %3d %3d %3d" %(i+1, final_order[i].name, final_order[i].getPoints()/sims,
        final_order[i].getWs()/sims,final_order[i].getDraws()/sims,final_order[i].getLs()/sims))

print()