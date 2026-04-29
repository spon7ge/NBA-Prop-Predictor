import urllib.request
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from src.utils.team_info import *
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
                result += "\nQuestionable (May Not Play)\n{}\n".format('-' * len("Questionable (May Not Play)"))
                for player in matchup[team]["questionable"]:
                    result += player + "\n"
                result += "\nConfirmed Out\n{}\n".format('-' * len("Confirmed Out"))
                for player in matchup[team]["out"]:
                    result += player + "\n"
        return result

    def _getSoup(self):
        response = requests.get(self.url, verify=True)  # verify=True uses system certificates
        response.raise_for_status()  # Raise an error for bad status codes
        return BeautifulSoup(response.text, 'html.parser')

    def _get_injured_players(self, lineup_list):
        """
        Extract injured players from the lineup list.
        
        Players in "MAY NOT PLAY" section have:
        - class="lineup__player has-injury-status"
        - Status in <span class="lineup__inj">Out</span> or <span class="lineup__inj">Ques</span>
        
        Args:
            lineup_list: The BeautifulSoup ul element for the team's lineup
        
        Returns:
            dict with 'questionable' and 'out' player sets
        """
        questionable = set()
        out = set()
        
        # Find all players with injury status (has-injury-status class)
        injured_items = lineup_list.find_all("li", class_=lambda x: x and "has-injury-status" in x)
        
        for item in injured_items:
            player_elem = item.find('a')
            if not player_elem:
                continue
            
            player_name = player_elem.get('title') or player_elem.get_text(strip=True)
            
            # Get the injury status from <span class="lineup__inj">
            status_span = item.find('span', class_='lineup__inj')
            if status_span:
                status_text = status_span.get_text(strip=True).lower()
                
                if status_text == 'out':
                    out.add(player_name)
                elif status_text in ['ques', 'gtd', 'questionable', 'doubtful']:
                    questionable.add(player_name)
            else:
                # Fallback: check the title attribute or item text
                title = item.get('title', '').lower()
                if 'unlikely' in title:
                    out.add(player_name)
        
        return {'questionable': questionable, 'out': out}

    def getDict(self):
        self.data = []
        for matchup in self.soup.find_all("div", {"class": "lineup is-nba"}):
            # Get team names
            away_team = matchup.find("a", {"class": "lineup__mteam is-visit white"}).text.split(None, 1)[0]
            home_team = matchup.find("a", {"class": "lineup__mteam is-home white"}).text.split(None, 1)[0]
            
            # Get lineup lists
            away_list = matchup.find("ul", {"class": "lineup__list is-visit"})
            home_list = matchup.find("ul", {"class": "lineup__list is-home"})
            
            # Extract confirmed starters (Very Likely To Play, not has-injury-status)
            away_confirmed = set()
            away_gtd = set()
            for item in away_list.find_all("li", {"title": "Very Likely To Play"}):
                if item.a and "has-injury-status" not in (item.get('class') or []):
                    away_confirmed.add(item.a['title'])
            
            for item in away_list.find_all("li", {"title": ["Toss Up To Play", "Likely To Play"]}):
                if item.a and "has-injury-status" not in (item.get('class') or []):
                    away_gtd.add(item.a['title'])
            
            home_confirmed = set()
            home_gtd = set()
            for item in home_list.find_all("li", {"title": ["Very Likely To Play", "Likely To Play"]}):
                if item.a and "has-injury-status" not in (item.get('class') or []):
                    home_confirmed.add(item.a['title'])
            
            for item in home_list.find_all("li", {"title": "Toss Up To Play"}):
                if item.a and "has-injury-status" not in (item.get('class') or []):
                    home_gtd.add(item.a['title'])
            
            # Get injured/questionable players from the same list
            away_injured = self._get_injured_players(away_list)
            home_injured = self._get_injured_players(home_list)
            
            self.data.append({
                "away": {
                    "team": away_team,
                    "confirmed": away_confirmed,
                    "gtd": away_gtd,
                    "questionable": away_injured['questionable'],
                    "out": away_injured['out'],
                },
                "home": {
                    "team": home_team,
                    "confirmed": home_confirmed,
                    "gtd": home_gtd,
                    "questionable": home_injured['questionable'],
                    "out": home_injured['out'],
                }
            })
    
    def updateTeamInfo(self, file_path=None):
        """Update team_info.py with scraped confirmed playing players and questionable players"""
        if not self.data:
            print("No data available. Run getDict() first.")
            return
        
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        
        if file_path is None:
            # Default to src/utils/team_info.py
            team_info_path = project_root / 'src' / 'utils' / 'team_info.py'
        else:
            team_info_path = project_root / file_path
        
        team_info_path = str(team_info_path)
        
        # Read the current file
        with open(team_info_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Build updated projectedStartingFive dictionary
        updated_lineups = {}
        # Build questionable players dictionary
        questionable_players = {}
        
        # Process each matchup
        for matchup in self.data:
            for team_type in ["away", "home"]:
                team_name = matchup[team_type]["team"]
                team_abbr = self.TEAM_ABBREVIATIONS.get(team_name)
                
                if team_abbr:
                    # Get confirmed players for lineup
                    confirmed_players = list(matchup[team_type]["confirmed"])
                    
                    # Get questionable players
                    team_questionable = list(matchup[team_type]["questionable"])
                    if team_questionable:
                        questionable_players[team_abbr] = team_questionable
                    
                    # Update lineup if we have ANY confirmed players
                    # This will update even if team doesn't have full starting lineup
                    if confirmed_players:
                        updated_lineups[team_abbr] = confirmed_players[:5] if len(confirmed_players) >= 5 else confirmed_players
                        if len(confirmed_players) < 5:
                            print(f"Note: {team_abbr} ({team_name}) has {len(confirmed_players)} confirmed players - lineup will still be updated")
        
        # Update the projectedStartingFive dictionary in the file
        content = self._update_dict_in_file(content, 'projectedStartingFive', updated_lineups)
        
        # Update or add the questionablePlayers dictionary in the file
        content = self._update_or_add_dict_in_file(content, 'questionablePlayers', questionable_players)
        
        # Write back to file
        with open(team_info_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Successfully updated {team_info_path}")
        print(f"Updated {len(updated_lineups)} teams with confirmed lineups")
        print(f"Updated {len(questionable_players)} teams with questionable players")
    
    def _update_dict_in_file(self, content, dict_name, updated_data):
        """Update an existing dictionary in the file content"""
        start_match = re.search(rf'{dict_name}\s*=\s*\{{', content)
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
            existing_content = content[start_pos:end_pos]
            existing_teams = re.findall(r'"([A-Z]{3})"', existing_content)
            
            # Build new content: update teams with new data, keep existing for others
            new_dict_lines = [f'{dict_name} = {{']
            
            # Start with teams that have updates
            for abbr in sorted(set(list(updated_data.keys()) + existing_teams)):
                if abbr in updated_data:
                    # Use updated lineup
                    players = updated_data[abbr]
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
            return before + '\n'.join(new_dict_lines) + '\n\n' + after.lstrip()
        else:
            print(f"Error: Could not find {dict_name} in file")
            return content
    
    def _update_or_add_dict_in_file(self, content, dict_name, data):
        """Update existing dictionary or add new one if it doesn't exist"""
        start_match = re.search(rf'{dict_name}\s*=\s*\{{', content)
        
        if start_match:
            # Dictionary exists, update it
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
            new_dict_lines = [f'{dict_name} = {{']
            for abbr in sorted(data.keys()):
                players = data[abbr]
                players_str = ', '.join([f'"{p}"' for p in players])
                new_dict_lines.append(f'    "{abbr}": [{players_str}],')
            new_dict_lines.append('}')
            
            # Replace the section
            before = content[:start_pos]
            after = content[end_pos:]
            return before + '\n'.join(new_dict_lines) + '\n\n' + after.lstrip()
        else:
            # Dictionary doesn't exist, add it at the end
            new_dict_lines = [f'\n# Players listed as questionable/may not play\n{dict_name} = {{']
            for abbr in sorted(data.keys()):
                players = data[abbr]
                players_str = ', '.join([f'"{p}"' for p in players])
                new_dict_lines.append(f'    "{abbr}": [{players_str}],')
            new_dict_lines.append('}\n')
            
            return content.rstrip() + '\n' + '\n'.join(new_dict_lines)

    def getQuestionablePlayers(self):
        """Return a dictionary of questionable players by team"""
        if not self.data:
            print("No data available. Run getDict() first.")
            return {}
        
        questionable = {}
        for matchup in self.data:
            for team_type in ["away", "home"]:
                team_name = matchup[team_type]["team"]
                team_abbr = self.TEAM_ABBREVIATIONS.get(team_name)
                
                if team_abbr:
                    team_questionable = list(matchup[team_type]["questionable"])
                    if team_questionable:
                        questionable[team_abbr] = team_questionable
        
        return questionable
    
    def getOutPlayers(self):
        """Return a dictionary of confirmed out players by team"""
        if not self.data:
            print("No data available. Run getDict() first.")
            return {}
        
        out_players = {}
        for matchup in self.data:
            for team_type in ["away", "home"]:
                team_name = matchup[team_type]["team"]
                team_abbr = self.TEAM_ABBREVIATIONS.get(team_name)
                
                if team_abbr:
                    team_out = list(matchup[team_type]["out"])
                    if team_out:
                        out_players[team_abbr] = team_out
        
        return out_players

    def debugPrintMatchupStructure(self):
        """Debug helper to print HTML structure of first matchup"""
        matchups = self.soup.find_all("div", {"class": "lineup is-nba"})
        if matchups:
            print("=== First Matchup HTML Structure ===")
            print(matchups[0].prettify()[:5000])
            print("\n=== All class names found ===")
            for elem in matchups[0].find_all(True):
                if elem.get('class'):
                    print(f"  {elem.name}: {elem.get('class')}")

# if __name__ == "__main__":
#     scraper = NBADailyLineups("https://www.rotowire.com/basketball/nba-lineups.php")
#     scraper.getDict()
#     print(scraper)
#     print("\nQuestionable Players:")
#     print(scraper.getQuestionablePlayers())
#     print("\nOut Players:")
#     print(scraper.getOutPlayers())
#     scraper.updateTeamInfo()