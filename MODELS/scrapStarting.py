import urllib.request
import requests
from bs4 import BeautifulSoup
from MODELS.teamInfo import *
import re
import os

class NBADailyLineups:

    data = []
    
    # Mapping from team name (first word) to NBA abbreviation
    TEAM_ABBREVIATIONS = {
        "Hawks": "ATL", "Celtics": "BOS", "Nets": "BKN", "Hornets": "CHA",
        "Bulls": "CHI", "Cavaliers": "CLE", "Mavericks": "DAL", "Nuggets": "DEN",
        "Pistons": "DET", "Warriors": "GSW", "Rockets": "HOU", "Pacers": "IND",
        "Clippers": "LAC", "Lakers": "LAL", "Grizzlies": "MEM", "Heat": "MIA",
        "Bucks": "MIL", "Timberwolves": "MIN", "Pelicans": "NOP", "Knicks": "NYK",
        "Thunder": "OKC", "Magic": "ORL", "76ers": "PHI", "Suns": "PHX",
        "Trail": "POR", "Kings": "SAC", "Spurs": "SAS", "Raptors": "TOR",
        "Jazz": "UTA", "Wizards": "WAS"
    }

    def __init__(self, url):
        self.url = url
        self.soup = self._getSoup()

    def __str__(self):
        result = ""
        for index, matchup in enumerate(self.data):
            result += "\n\nMatchup {}\n".format(index + 1)
            for team in matchup:
                result += "\n\n{} team: {}\n{}\n".format(team, matchup[team]["team"], '-' * len("team: " + team + matchup[team]["team"]))
                result += "\nConfirmed Playing\n{}\n".format('-' * len("Confirmed Playing"))
                for player in matchup[team]["confirmed"]:
                    result += player + "\n"
                result += "\nGame Time Decision\n{}\n".format('-' * len("Game Time Decision"))
                for player in matchup[team]["gtd"]:
                    result += player + "\n"
                result += "\nConfirmed Out\n{}\n".format('-' * len("Confirmed Out"))
                for player in matchup[team]["out"]:
                    result += player + "\n"
        return result

    def _getSoup(self):
        response = requests.get(self.url, verify=True)  # verify=True uses system certificates
        response.raise_for_status()  # Raise an error for bad status codes
        return BeautifulSoup(response.text, 'html.parser')

    def getDict(self):
        self.data = []
        for matchup in self.soup.find_all("div", {"class": "lineup is-nba"}):
            self.data.append({
                "away": {
                    "team": matchup.find("a", {"class": "lineup__mteam is-visit white"}).text.split(None, 1)[0],
                    "confirmed": set(item.a['title'] for item in matchup.find("ul", {"class": "lineup__list is-visit"}).find_all("li", {"title": "Very Likely To Play"})),
                    "gtd": set(item.a['title'] for item in matchup.find("ul", {"class": "lineup__list is-visit"}).find_all("li", {"title": ["Toss Up To Play", "Likely To Play"]})),
                    "out": set(item.a['title'] for item in matchup.find("ul", {"class": "lineup__list is-visit"}).find_all("li", {"title": "Very Unlikely To Play"})), 
                },
                "home": {
                    "team": matchup.find("a", {"class": "lineup__mteam is-home white"}).text.split(None, 1)[0],
                    "confirmed": set(item.a['title'] for item in matchup.find("ul", {"class": "lineup__list is-home"}).find_all("li", {"title": ["Very Likely To Play", "Likely To Play"]})),
                    "gtd": set(item.a['title'] for item in matchup.find("ul", {"class": "lineup__list is-home"}).find_all("li", {"title": "Toss Up To Play"})),
                    "out": set(item.a['title'] for item in matchup.find("ul", {"class": "lineup__list is-home"}).find_all("li", {"title": "Very Unlikely To Play"})), 
                }
            })
    
    def updateTeamInfo(self, file_path="MODELS/teamInfo.py"):
        """Update teamInfo.py with scraped confirmed playing players"""
        if not self.data:
            print("No data available. Run getDict() first.")
            return
        
        # Get absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        team_info_path = os.path.join(project_root, file_path)
        
        # Read the current file
        with open(team_info_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Build updated projectedStartingFive dictionary
        updated_lineups = {}
        
        # Process each matchup
        for matchup in self.data:
            for team_type in ["away", "home"]:
                team_name = matchup[team_type]["team"]
                team_abbr = self.TEAM_ABBREVIATIONS.get(team_name)
                
                if team_abbr:
                    confirmed_players = list(matchup[team_type]["confirmed"])
                    # Take first 5 confirmed players (or all if less than 5)
                    if confirmed_players:
                        updated_lineups[team_abbr] = confirmed_players[:5] if len(confirmed_players) >= 5 else confirmed_players
                        if len(confirmed_players) < 5:
                            print(f"Warning: {team_abbr} ({team_name}) only has {len(confirmed_players)} confirmed players")
        
        # Update the projectedStartingFive dictionary in the file
        # Find the projectedStartingFive section
        start_match = re.search(r'projectedStartingFive\s*=\s*\{', content)
        if start_match:
            start_pos = start_match.start()
            # Find the closing brace for this dictionary
            brace_count = 0
            end_pos = start_pos
            for i, char in enumerate(content[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            # Build the new dictionary content
            # First, get all existing teams to preserve those not in today's games
            existing_match = re.search(r'projectedStartingFive\s*=\s*\{([^}]+)\}', content, re.DOTALL)
            if existing_match:
                existing_content = existing_match.group(1)
                # Extract existing team abbreviations
                existing_teams = re.findall(r'"([A-Z]{3})"', existing_content)
                
                # Build new content: update teams with new data, keep existing for others
                new_dict_lines = ['projectedStartingFive = {']
                
                # Start with teams that have updates
                for abbr in sorted(set(list(updated_lineups.keys()) + existing_teams)):
                    if abbr in updated_lineups:
                        # Use updated lineup
                        players = updated_lineups[abbr]
                        players_str = ', '.join([f'"{p}"' for p in players])
                        new_dict_lines.append(f'    "{abbr}": [{players_str}],')
                    else:
                        # Keep existing lineup - extract from original content
                        team_match = re.search(rf'"{abbr}":\s*\[(.*?)\](?:,|$)', existing_content, re.DOTALL)
                        if team_match:
                            new_dict_lines.append(f'    "{abbr}": [{team_match.group(1)}],')
                
                new_dict_lines.append('}')
                
                # Replace the section
                before = content[:start_pos]
                after = content[end_pos:]
                new_content = before + '\n'.join(new_dict_lines) + '\n\n' + after.lstrip()
                
                # Write back to file
                with open(team_info_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"Successfully updated {team_info_path}")
                print(f"Updated {len(updated_lineups)} teams with confirmed lineups")
            else:
                print("Error: Could not parse existing projectedStartingFive")
        else:
            print("Error: Could not find projectedStartingFive in file")

# if __name__ == "__main__":
#     scraper = NBADailyLineups("https://www.rotowire.com/basketball/nba-lineups.php")
#     scraper.getDict()
#     print(scraper)
#     scraper.updateTeamInfo()