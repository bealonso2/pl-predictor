import json


def split_teams(to_parse: str):
    split_str = to_parse.split(',')
    name = split_str[0]
    home = split_str[1]
    away = split_str[2]
    team = {}
    team['name'] = name
    team['display name'] = ''
    team['home'] = home
    team['away'] = away
    return team


out = {}
with open('./2022-2023/teams 2022 2023.csv','r') as f:
    for team in [split_teams(teams) for teams in f.readlines()[1:]]:
        out[team.get('name')] = team

jstring = json.dumps(out, indent=2)

with open('./2022-2023/2022-2023_teams.json', 'w') as f:
    f.write(jstring)
