import json
from datetime import datetime

import urllib3

# Get current year dynamically

current_year = datetime.now().year

http = urllib3.PoolManager()
url = f"https://fixturedownload.com/feed/json/epl-{current_year}"
response = http.request("GET", url)

fixtures = json.loads(response.data)

jstring = json.dumps(fixtures, indent=2)

with open(f"./{current_year}-{current_year + 1}/fixtures.json", "w") as f:
    f.write(jstring)
