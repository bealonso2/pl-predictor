import json

import urllib3

http = urllib3.PoolManager()
url = 'https://fixturedownload.com/feed/json/epl-2022'
response = http.request('GET', url)

fixtures = json.loads(response.data)

jstring = json.dumps(fixtures, indent=2)

with open('./2022-2023_fixtures.json', 'w') as f:
    f.write(jstring)
