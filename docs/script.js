const prizepicksSinglesData = [
    {"name": "Ivica Zubac", "bookmaker": "BetRivers", "line": 15.5, "prediction": 11.88, "side": "Under", "odds": 100, "recommendation": 0, "ev": 49.67, "kelly": 0.497, "sigma": "Med"},
    {"name": "Rui Hachimura", "bookmaker": "BetRivers", "line": 12.5, "prediction": 15.43, "side": "Over", "odds": 120, "recommendation": 0, "ev": 48.47, "kelly": 0.404, "sigma": "High"},
    {"name": "Dillon Brooks", "bookmaker": "BetRivers", "line": 20.5, "prediction": 23.81, "side": "Over", "odds": 114, "recommendation": 0, "ev": 44.91, "kelly": 0.394, "sigma": "High"},
    {"name": "Dyson Daniels", "bookmaker": "FanDuel", "line": 12.5, "prediction": 9.64, "side": "Under", "odds": -106, "recommendation": 0, "ev": 41.63, "kelly": 0.441, "sigma": "Low"},
    {"name": "Keyonte George", "bookmaker": "FanDuel", "line": 20.5, "prediction": 24.8, "side": "Over", "odds": -106, "recommendation": 1, "ev": 41.46, "kelly": 0.439, "sigma": "High"},
    {"name": "James Harden", "bookmaker": "DraftKings", "line": 24.5, "prediction": 20.91, "side": "Under", "odds": -109, "recommendation": 0, "ev": 39.45, "kelly": 0.43, "sigma": "Med"},
    {"name": "Austin Reaves", "bookmaker": "FanDuel", "line": 23.5, "prediction": 26.89, "side": "Over", "odds": 102, "recommendation": 0, "ev": 37.72, "kelly": 0.37, "sigma": "High"},
    {"name": "Jake LaRavia", "bookmaker": "DraftKings", "line": 7.5, "prediction": 11.15, "side": "Over", "odds": -115, "recommendation": 0, "ev": 34.47, "kelly": 0.396, "sigma": "High"},
    {"name": "Donovan Clingan", "bookmaker": "BetRivers", "line": 10.5, "prediction": 12.47, "side": "Over", "odds": 112, "recommendation": 0, "ev": 33.71, "kelly": 0.301, "sigma": "Med"},
    {"name": "Kevin Love", "bookmaker": "BetMGM", "line": 4.5, "prediction": 7.45, "side": "Over", "odds": -120, "recommendation": 0, "ev": 33.12, "kelly": 0.397, "sigma": "Low"},
];const prizepicksPairsData = [
    {"name1": "Jordan Poole", "name2": "Justin Champagnie", "line1": 12.5, "line2": 9.5, "odds1": -115, "odds2": -118, "prediction1": 20.83, "prediction2": 1.64, "modelProb1": 0.926, "modelProb2": 0.954, "side1": "over", "side2": "under", "parlayProb": 0.883, "parlayOdds": 245, "evPercent": 204.79, "kellyQuarter": 0.209},
    {"name1": "Amen Thompson", "name2": "Maxime Raynaud", "line1": 16.5, "line2": 11.5, "odds1": -111, "odds2": -101, "prediction1": 22.75, "prediction2": 5.31, "modelProb1": 0.866, "modelProb2": 0.83, "side1": "over", "side2": "under", "parlayProb": 0.719, "parlayOdds": 278, "evPercent": 171.71, "kellyQuarter": 0.1544},
    {"name1": "Anfernee Simons", "name2": "Reed Sheppard", "line1": 11.5, "line2": 10.5, "odds1": -118, "odds2": -115, "prediction1": 16.62, "prediction2": 16.25, "modelProb1": 0.829, "modelProb2": 0.86, "side1": "over", "side2": "over", "parlayProb": 0.713, "parlayOdds": 245, "evPercent": 145.82, "kellyQuarter": 0.1488},
    {"name1": "Shaedon Sharpe", "name2": "Nique Clifford", "line1": 22.5, "line2": 8.5, "odds1": -121, "odds2": -123, "prediction1": 28.95, "prediction2": 2.67, "modelProb1": 0.828, "modelProb2": 0.829, "side1": "over", "side2": "under", "parlayProb": 0.687, "parlayOdds": 231, "evPercent": 127.24, "kellyQuarter": 0.1377},
    {"name1": "Jabari Smith Jr.", "name2": "Russell Westbrook", "line1": 14.0, "line2": 17.5, "odds1": -137, "odds2": -115, "prediction1": 20.02, "prediction2": 10.85, "modelProb1": 0.854, "modelProb2": 0.792, "side1": "over", "side2": "under", "parlayProb": 0.676, "parlayOdds": 223, "evPercent": 118.35, "kellyQuarter": 0.1327},
    {"name1": "Jamal Murray", "name2": "Paolo Banchero", "line1": 24.5, "line2": 21.5, "odds1": -105, "odds2": -110, "prediction1": 29.23, "prediction2": 15.31, "modelProb1": 0.754, "modelProb2": 0.773, "side1": "over", "side2": "under", "parlayProb": 0.583, "parlayOdds": 273, "evPercent": 117.41, "kellyQuarter": 0.1075},
    {"name1": "Kawhi Leonard", "name2": "Stephen Curry", "line1": 22.5, "line2": 24.5, "odds1": -122, "odds2": -106, "prediction1": 27.68, "prediction2": 29.99, "modelProb1": 0.809, "modelProb2": 0.749, "side1": "over", "side2": "over", "parlayProb": 0.606, "parlayOdds": 254, "evPercent": 114.51, "kellyQuarter": 0.1127},
    {"name1": "Payton Pritchard", "name2": "Brandon Miller", "line1": 16.5, "line2": 21.5, "odds1": -110, "odds2": -118, "prediction1": 20.87, "prediction2": 14.74, "modelProb1": 0.763, "modelProb2": 0.786, "side1": "over", "side2": "under", "parlayProb": 0.599, "parlayOdds": 253, "evPercent": 111.46, "kellyQuarter": 0.1101},
    {"name1": "Josh Minott", "name2": "Devin Vassell", "line1": 5.5, "line2": 11.5, "odds1": -129, "odds2": -127, "prediction1": 9.19, "prediction2": 16.36, "modelProb1": 0.816, "modelProb2": 0.81, "side1": "over", "side2": "over", "parlayProb": 0.662, "parlayOdds": 217, "evPercent": 109.7, "kellyQuarter": 0.1264},
    {"name1": "Jaylen Brown", "name2": "Moses Moody", "line1": 29.5, "line2": 8.5, "odds1": 100, "odds2": -118, "prediction1": 34.59, "prediction2": 11.95, "modelProb1": 0.716, "modelProb2": 0.781, "side1": "over", "side2": "over", "parlayProb": 0.559, "parlayOdds": 269, "evPercent": 106.2, "kellyQuarter": 0.0987},
];const prizepicksTriosData = [
    {"name1": "Jordan Poole", "name2": "Maxime Raynaud", "name3": "Justin Champagnie", "line1": 12.5, "line2": 11.5, "line3": 9.5, "odds1": -115, "odds2": -101, "odds3": -118, "prediction1": 20.83, "prediction2": 5.31, "prediction3": 1.64, "modelProb1": 0.926, "modelProb2": 0.83, "modelProb3": 0.954, "side1": "over", "side2": "under", "side3": "under", "parlayProb": 0.733, "parlayOdds": 587, "evPercent": 403.7, "kellyQuarter": 0.1719},
    {"name1": "Anfernee Simons", "name2": "Amen Thompson", "name3": "Shaedon Sharpe", "line1": 11.5, "line2": 16.5, "line3": 22.5, "odds1": -118, "odds2": -111, "odds3": -121, "prediction1": 16.62, "prediction2": 22.75, "prediction3": 28.95, "modelProb1": 0.829, "modelProb2": 0.866, "modelProb3": 0.828, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.594, "parlayOdds": 541, "evPercent": 280.99, "kellyQuarter": 0.1298},
    {"name1": "Reed Sheppard", "name2": "Nique Clifford", "name3": "Paolo Banchero", "line1": 10.5, "line2": 8.5, "line3": 21.5, "odds1": -115, "odds2": -123, "odds3": -110, "prediction1": 16.25, "prediction2": 2.67, "prediction3": 15.31, "modelProb1": 0.86, "modelProb2": 0.829, "modelProb3": 0.773, "side1": "over", "side2": "under", "side3": "under", "parlayProb": 0.551, "parlayOdds": 547, "evPercent": 256.33, "kellyQuarter": 0.1172},
    {"name1": "Jabari Smith Jr.", "name2": "Russell Westbrook", "name3": "Stephen Curry", "line1": 14.0, "line2": 17.5, "line3": 24.5, "odds1": -137, "odds2": -115, "odds3": -106, "prediction1": 20.02, "prediction2": 10.85, "prediction3": 29.99, "modelProb1": 0.854, "modelProb2": 0.792, "modelProb3": 0.749, "side1": "over", "side2": "under", "side3": "over", "parlayProb": 0.506, "parlayOdds": 529, "evPercent": 218.45, "kellyQuarter": 0.1032},
    {"name1": "Payton Pritchard", "name2": "Kawhi Leonard", "name3": "Jamal Murray", "line1": 16.5, "line2": 22.5, "line3": 24.5, "odds1": -110, "odds2": -122, "odds3": -105, "prediction1": 20.87, "prediction2": 27.68, "prediction3": 29.23, "modelProb1": 0.763, "modelProb2": 0.809, "modelProb3": 0.754, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.465, "parlayOdds": 578, "evPercent": 215.58, "kellyQuarter": 0.0932},
    {"name1": "Josh Minott", "name2": "Brandon Miller", "name3": "Devin Vassell", "line1": 5.5, "line2": 21.5, "line3": 11.5, "odds1": -129, "odds2": -118, "odds3": -127, "prediction1": 9.19, "prediction2": 14.74, "prediction3": 16.36, "modelProb1": 0.816, "modelProb2": 0.786, "modelProb3": 0.81, "side1": "over", "side2": "under", "side3": "over", "parlayProb": 0.52, "parlayOdds": 486, "evPercent": 204.54, "kellyQuarter": 0.1052},
    {"name1": "Jaylen Brown", "name2": "Steven Adams", "name3": "Moses Moody", "line1": 29.5, "line2": 5.5, "line3": 8.5, "odds1": 100, "odds2": -113, "odds3": -118, "prediction1": 34.59, "prediction2": 7.7, "prediction3": 11.95, "modelProb1": 0.716, "modelProb2": 0.753, "modelProb3": 0.781, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.421, "parlayOdds": 596, "evPercent": 192.78, "kellyQuarter": 0.0809},
    {"name1": "Alperen Sengun", "name2": "Tobias Harris", "name3": "Donovan Mitchell", "line1": 20.5, "line2": 13.5, "line3": 30.5, "odds1": -141, "odds2": -108, "odds3": -104, "prediction1": 25.59, "prediction2": 16.88, "prediction3": 35.14, "modelProb1": 0.804, "modelProb2": 0.726, "modelProb3": 0.718, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.419, "parlayOdds": 546, "evPercent": 170.77, "kellyQuarter": 0.0782},
    {"name1": "Zaccharie Risacher", "name2": "Lonzo Ball", "name3": "Harrison Barnes", "line1": 10.5, "line2": 6.5, "line3": 10.5, "odds1": 100, "odds2": -118, "odds3": -108, "prediction1": 12.56, "prediction2": 9.53, "prediction3": 12.99, "modelProb1": 0.684, "modelProb2": 0.744, "modelProb3": 0.707, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.36, "parlayOdds": 612, "evPercent": 156.22, "kellyQuarter": 0.0638},
    {"name1": "Derrick White", "name2": "Jonas Valan\u010di\u016bnas", "name3": "De'Anthony Melton", "line1": 18.5, "line2": 8.5, "line3": 8.5, "odds1": -113, "odds2": -110, "odds3": -137, "prediction1": 23.09, "prediction2": 10.9, "prediction3": 11.79, "modelProb1": 0.719, "modelProb2": 0.703, "modelProb3": 0.787, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.398, "parlayOdds": 523, "evPercent": 147.89, "kellyQuarter": 0.0707},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Jabari Smith Jr.", "name2": "Maxime Raynaud", "line1": 14.5, "line2": 11.5, "odds1": 100, "odds2": -101, "prediction1": 20.02, "prediction2": 5.31, "modelProb1": 0.831, "modelProb2": 0.83, "side1": "over", "side2": "under", "parlayProb": 0.689, "parlayOdds": 298, "evPercent": 174.4, "kellyQuarter": 0.1463},
    {"name1": "Anfernee Simons", "name2": "Amen Thompson", "line1": 11.5, "line2": 16.5, "odds1": -118, "odds2": -111, "prediction1": 16.62, "prediction2": 22.75, "modelProb1": 0.829, "modelProb2": 0.866, "side1": "over", "side2": "over", "parlayProb": 0.718, "parlayOdds": 251, "evPercent": 151.9, "kellyQuarter": 0.1513},
    {"name1": "Shaedon Sharpe", "name2": "Nique Clifford", "line1": 22.5, "line2": 8.5, "odds1": -121, "odds2": -123, "prediction1": 28.95, "prediction2": 2.67, "modelProb1": 0.828, "modelProb2": 0.829, "side1": "over", "side2": "under", "parlayProb": 0.687, "parlayOdds": 231, "evPercent": 127.24, "kellyQuarter": 0.1377},
    {"name1": "Russell Westbrook", "name2": "Paolo Banchero", "line1": 17.5, "line2": 21.5, "odds1": -115, "odds2": -110, "prediction1": 10.85, "prediction2": 15.31, "modelProb1": 0.792, "modelProb2": 0.773, "side1": "under", "side2": "under", "parlayProb": 0.612, "parlayOdds": 257, "evPercent": 118.34, "kellyQuarter": 0.1151},
    {"name1": "Kawhi Leonard", "name2": "Jamal Murray", "line1": 22.5, "line2": 24.5, "odds1": -122, "odds2": -105, "prediction1": 27.68, "prediction2": 29.23, "modelProb1": 0.809, "modelProb2": 0.754, "side1": "over", "side2": "over", "parlayProb": 0.61, "parlayOdds": 255, "evPercent": 116.7, "kellyQuarter": 0.1144},
    {"name1": "Payton Pritchard", "name2": "Stephen Curry", "line1": 16.5, "line2": 24.5, "odds1": -110, "odds2": -106, "prediction1": 20.87, "prediction2": 29.99, "modelProb1": 0.763, "modelProb2": 0.749, "side1": "over", "side2": "over", "parlayProb": 0.571, "parlayOdds": 271, "evPercent": 111.87, "kellyQuarter": 0.1032},
    {"name1": "Josh Minott", "name2": "Tobias Harris", "line1": 5.5, "line2": 13.5, "odds1": -129, "odds2": -108, "prediction1": 9.19, "prediction2": 16.88, "modelProb1": 0.816, "modelProb2": 0.726, "side1": "over", "side2": "over", "parlayProb": 0.593, "parlayOdds": 242, "evPercent": 102.76, "kellyQuarter": 0.1062},
    {"name1": "Jaylen Brown", "name2": "De'Anthony Melton", "line1": 29.5, "line2": 8.5, "odds1": 100, "odds2": -137, "prediction1": 34.59, "prediction2": 11.79, "modelProb1": 0.716, "modelProb2": 0.787, "side1": "over", "side2": "over", "parlayProb": 0.564, "parlayOdds": 246, "evPercent": 94.97, "kellyQuarter": 0.0965},
    {"name1": "Jonas Valan\u010di\u016bnas", "name2": "Jalen Suggs", "line1": 8.5, "line2": 17.5, "odds1": -110, "odds2": -110, "prediction1": 10.9, "prediction2": 12.4, "modelProb1": 0.703, "modelProb2": 0.696, "side1": "over", "side2": "under", "parlayProb": 0.49, "parlayOdds": 264, "evPercent": 78.19, "kellyQuarter": 0.074},
    {"name1": "Ivica Zubac", "name2": "Saddiq Bey", "line1": 14.5, "line2": 16.5, "odds1": -115, "odds2": -113, "prediction1": 16.83, "prediction2": 12.11, "modelProb1": 0.695, "modelProb2": 0.688, "side1": "over", "side2": "under", "parlayProb": 0.478, "parlayOdds": 252, "evPercent": 68.24, "kellyQuarter": 0.0677},
];const underdogTriosData = [
    {"name1": "Anfernee Simons", "name2": "Jabari Smith Jr.", "name3": "Maxime Raynaud", "line1": 11.5, "line2": 14.5, "line3": 11.5, "odds1": -118, "odds2": 100, "odds3": -101, "prediction1": 16.62, "prediction2": 20.02, "prediction3": 5.31, "modelProb1": 0.829, "modelProb2": 0.831, "modelProb3": 0.83, "side1": "over", "side2": "over", "side3": "under", "parlayProb": 0.571, "parlayOdds": 635, "evPercent": 319.88, "kellyQuarter": 0.1259},
    {"name1": "Amen Thompson", "name2": "Shaedon Sharpe", "name3": "Nique Clifford", "line1": 16.5, "line2": 22.5, "line3": 8.5, "odds1": -111, "odds2": -121, "odds3": -123, "prediction1": 22.75, "prediction2": 28.95, "prediction3": 2.67, "modelProb1": 0.866, "modelProb2": 0.828, "modelProb3": 0.829, "side1": "over", "side2": "over", "side3": "under", "parlayProb": 0.595, "parlayOdds": 529, "evPercent": 274.03, "kellyQuarter": 0.1295},
    {"name1": "Kawhi Leonard", "name2": "Russell Westbrook", "name3": "Paolo Banchero", "line1": 22.5, "line2": 17.5, "line3": 21.5, "odds1": -122, "odds2": -115, "odds3": -110, "prediction1": 27.68, "prediction2": 10.85, "prediction3": 15.31, "modelProb1": 0.809, "modelProb2": 0.792, "modelProb3": 0.773, "side1": "over", "side2": "under", "side3": "under", "parlayProb": 0.495, "parlayOdds": 549, "evPercent": 221.16, "kellyQuarter": 0.1007},
    {"name1": "Payton Pritchard", "name2": "Jamal Murray", "name3": "Stephen Curry", "line1": 16.5, "line2": 24.5, "line3": 24.5, "odds1": -110, "odds2": -105, "odds3": -106, "prediction1": 20.87, "prediction2": 29.23, "prediction3": 29.99, "modelProb1": 0.763, "modelProb2": 0.754, "modelProb3": 0.749, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.431, "parlayOdds": 624, "evPercent": 211.93, "kellyQuarter": 0.0849},
    {"name1": "Josh Minott", "name2": "Tobias Harris", "name3": "De'Anthony Melton", "line1": 5.5, "line2": 13.5, "line3": 8.5, "odds1": -129, "odds2": -108, "odds3": -137, "prediction1": 9.19, "prediction2": 16.88, "prediction3": 11.79, "modelProb1": 0.816, "modelProb2": 0.726, "modelProb3": 0.787, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.467, "parlayOdds": 491, "evPercent": 175.77, "kellyQuarter": 0.0895},
    {"name1": "Jaylen Brown", "name2": "Jonas Valan\u010di\u016bnas", "name3": "Jalen Suggs", "line1": 29.5, "line2": 8.5, "line3": 17.5, "odds1": 100, "odds2": -110, "odds3": -110, "prediction1": 34.59, "prediction2": 10.9, "prediction3": 12.4, "modelProb1": 0.716, "modelProb2": 0.703, "modelProb3": 0.696, "side1": "over", "side2": "over", "side3": "under", "parlayProb": 0.35, "parlayOdds": 629, "evPercent": 155.5, "kellyQuarter": 0.0618},
    {"name1": "Ivica Zubac", "name2": "Saddiq Bey", "name3": "Cameron Johnson", "line1": 14.5, "line2": 16.5, "line3": 14.5, "odds1": -115, "odds2": -113, "odds3": -119, "prediction1": 16.83, "prediction2": 12.11, "prediction3": 9.79, "modelProb1": 0.695, "modelProb2": 0.688, "modelProb3": 0.696, "side1": "over", "side2": "under", "side3": "under", "parlayProb": 0.333, "parlayOdds": 549, "evPercent": 115.89, "kellyQuarter": 0.0528},
    {"name1": "Sam Hauser", "name2": "Kobe Sanders", "name3": "Buddy Hield", "line1": 7.5, "line2": 4.5, "line3": 8.5, "odds1": -110, "odds2": -137, "odds3": -122, "prediction1": 9.4, "prediction2": 6.99, "prediction3": 4.5, "modelProb1": 0.661, "modelProb2": 0.726, "modelProb3": 0.692, "side1": "over", "side2": "over", "side3": "under", "parlayProb": 0.332, "parlayOdds": 501, "evPercent": 99.49, "kellyQuarter": 0.0496},
    {"name1": "Jerami Grant", "name2": "Jalen Duren", "name3": "De'Aaron Fox", "line1": 21.5, "line2": 18.5, "line3": 20.5, "odds1": -118, "odds2": -110, "odds3": -112, "prediction1": 16.89, "prediction2": 14.65, "prediction3": 22.51, "modelProb1": 0.667, "modelProb2": 0.653, "modelProb3": 0.65, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.283, "parlayOdds": 568, "evPercent": 88.97, "kellyQuarter": 0.0392},
    {"name1": "Toumani Camara", "name2": "Tim Hardaway Jr.", "name3": "Chet Holmgren", "line1": 13.5, "line2": 12.5, "line3": 16.5, "odds1": -105, "odds2": -105, "odds3": -112, "prediction1": 15.27, "prediction2": 13.85, "prediction3": 13.36, "modelProb1": 0.625, "modelProb2": 0.61, "modelProb3": 0.629, "side1": "over", "side2": "over", "side3": "under", "parlayProb": 0.24, "parlayOdds": 622, "evPercent": 73.25, "kellyQuarter": 0.0294},
];const prizepicksPointsHitRates = [
    {"name": "Devin Vassell", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.942, "underPct": 0.058},
    {"name": "De'Anthony Melton", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.879, "underPct": 0.121},
    {"name": "Rudy Gobert", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.868, "underPct": 0.132},
    {"name": "Jordan Poole", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.866, "underPct": 0.134},
    {"name": "Aaron Holiday", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.864, "underPct": 0.136},
    {"name": "Stephen Curry", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.847, "underPct": 0.153},
    {"name": "Harrison Barnes", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.831, "underPct": 0.169},
    {"name": "Kawhi Leonard", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.813, "underPct": 0.187},
    {"name": "Jordan Walsh", "line": 7.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.807, "underPct": 0.193},
    {"name": "Chet Holmgren", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.79, "underPct": 0.21},
    {"name": "Jose Alvarado", "line": 8.0, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.783, "underPct": 0.217},
    {"name": "Jalen Johnson", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.775, "underPct": 0.225},
    {"name": "Jamal Murray", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.775, "underPct": 0.225},
    {"name": "Quinten Post", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.774, "underPct": 0.226},
    {"name": "Josh Hart", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.771, "underPct": 0.229},
    {"name": "Kobe Sanders", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.767, "underPct": 0.233},
    {"name": "John Collins", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.763, "underPct": 0.237},
    {"name": "Derrick White", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.751, "underPct": 0.249},
    {"name": "Saddiq Bey", "line": 16.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.741, "underPct": 0.259},
    {"name": "Alperen Sengun", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.74, "underPct": 0.26},
    {"name": "Kris Dunn", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.738, "underPct": 0.262},
    {"name": "De'Aaron Fox", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.736, "underPct": 0.264},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.733, "underPct": 0.267},
    {"name": "Anthony Edwards", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.732, "underPct": 0.268},
    {"name": "Stephon Castle", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.729, "underPct": 0.271},
    {"name": "James Harden", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.729, "underPct": 0.271},
    {"name": "Buddy Hield", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.728, "underPct": 0.272},
    {"name": "Nickeil Alexander-Walker", "line": 21.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.728, "underPct": 0.272},
    {"name": "Jaylen Brown", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.712, "underPct": 0.288},
    {"name": "Moses Moody", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.699, "underPct": 0.301},
    {"name": "Will Richard", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.686, "underPct": 0.314},
    {"name": "Brandin Podziemski", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.682, "underPct": 0.318},
    {"name": "Mikal Bridges", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.681, "underPct": 0.319},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.671, "underPct": 0.329},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.661, "underPct": 0.339},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.658, "underPct": 0.342},
    {"name": "Amen Thompson", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Isaac Okoro", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.649, "underPct": 0.351},
    {"name": "Steven Adams", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.647, "underPct": 0.353},
    {"name": "Deni Avdija", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.645, "underPct": 0.355},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.631, "underPct": 0.369},
    {"name": "Reed Sheppard", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.63, "underPct": 0.37},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.627, "underPct": 0.373},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.62, "underPct": 0.38},
    {"name": "Luguentz Dort", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.618, "underPct": 0.382},
    {"name": "Cade Cunningham", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.617, "underPct": 0.383},
    {"name": "Josh Minott", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.613, "underPct": 0.387},
    {"name": "Tyrese Maxey", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.605, "underPct": 0.395},
    {"name": "Donte DiVincenzo", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.602, "underPct": 0.398},
    {"name": "Cason Wallace", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.601, "underPct": 0.399},
    {"name": "Myles Turner", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.599, "underPct": 0.401},
    {"name": "Tobias Harris", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.597, "underPct": 0.403},
    {"name": "Jalen Brunson", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.589, "underPct": 0.411},
    {"name": "Kevin Porter Jr.", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.577, "underPct": 0.423},
    {"name": "Julius Randle", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.567, "underPct": 0.433},
    {"name": "Anthony Black", "line": 15.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.554, "underPct": 0.446},
    {"name": "Peyton Watson", "line": 12.0, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.542, "underPct": 0.458},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.539, "underPct": 0.461},
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.535, "underPct": 0.465},
    {"name": "Sam Hauser", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.524, "underPct": 0.476},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.519, "underPct": 0.481},
    {"name": "Jeremiah Fears", "line": 15.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.508, "underPct": 0.492},
    {"name": "Zaccharie Risacher", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.497, "underPct": 0.503},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.492, "underPct": 0.508},
    {"name": "Jaylon Tyson", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.492, "underPct": 0.508},
    {"name": "Coby White", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.49, "underPct": 0.51},
    {"name": "Cameron Johnson", "line": 14.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.486, "underPct": 0.514},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.482, "underPct": 0.518},
    {"name": "Pascal Siakam", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.478, "underPct": 0.522},
    {"name": "Kevin Durant", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Ivica Zubac", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.469, "underPct": 0.531},
    {"name": "Luke Kennard", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.468, "underPct": 0.532},
    {"name": "Tre Johnson", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.464, "underPct": 0.536},
    {"name": "Sidy Cissoko", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.44, "underPct": 0.56},
    {"name": "Keegan Murray", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.439, "underPct": 0.561},
    {"name": "Jalen Duren", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.435, "underPct": 0.565},
    {"name": "Anfernee Simons", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.421, "underPct": 0.579},
    {"name": "Jalen Suggs", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.416, "underPct": 0.584},
    {"name": "Jerami Grant", "line": 21.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.404, "underPct": 0.596},
    {"name": "Donovan Mitchell", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.403, "underPct": 0.597},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.389, "underPct": 0.611},
    {"name": "Ausar Thompson", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.388, "underPct": 0.612},
    {"name": "OG Anunoby", "line": 14.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.388, "underPct": 0.612},
    {"name": "Nicolas Batum", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.377, "underPct": 0.623},
    {"name": "Maxime Raynaud", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.375, "underPct": 0.625},
    {"name": "Karl-Anthony Towns", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.357, "underPct": 0.643},
    {"name": "Ryan Rollins", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.339, "underPct": 0.661},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.334, "underPct": 0.666},
    {"name": "Josh Okogie", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.331, "underPct": 0.669},
    {"name": "Jaden Ivey", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.327, "underPct": 0.673},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.323, "underPct": 0.677},
    {"name": "Derik Queen", "line": 16.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.321, "underPct": 0.679},
    {"name": "Bobby Portis", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.301, "underPct": 0.699},
    {"name": "Miles Bridges", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.293, "underPct": 0.707},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.277, "underPct": 0.723},
    {"name": "Darius Garland", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.274, "underPct": 0.726},
    {"name": "Jalen Williams", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.273, "underPct": 0.727},
    {"name": "Marvin Bagley III", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.244, "underPct": 0.756},
    {"name": "Shaedon Sharpe", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.229, "underPct": 0.771},
    {"name": "Kon Knueppel", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.223, "underPct": 0.777},
    {"name": "Justin Champagnie", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.208, "underPct": 0.792},
    {"name": "Russell Westbrook", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.205, "underPct": 0.795},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 22.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.183, "underPct": 0.817},
    {"name": "Paolo Banchero", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.164, "underPct": 0.836},
    {"name": "Lonzo Ball", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.128, "underPct": 0.872},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.12, "underPct": 0.88},
    {"name": "Brandon Miller", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.081, "underPct": 0.919},
    {"name": "Nique Clifford", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.04, "underPct": 0.96},
    {"name": "Keon Ellis", "line": 6.5, "l5": 0.0, "l10": 0.0, "l15": 0.13, "overPct": 0.023, "underPct": 0.977},
];const prizepicksAssistsHitRates = [
    {"name": "Trey Murphy III", "line": 3.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.664, "underPct": 0.336},
    {"name": "Rudy Gobert", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.648, "underPct": 0.352},
    {"name": "De'Anthony Melton", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.638, "underPct": 0.362},
    {"name": "Jaylen Brown", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.619, "underPct": 0.381},
    {"name": "Julius Randle", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.616, "underPct": 0.384},
    {"name": "Buddy Hield", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.59, "underPct": 0.41},
    {"name": "Jalen Williams", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.57, "underPct": 0.43},
    {"name": "Desmond Bane", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.542, "underPct": 0.458},
    {"name": "Cade Cunningham", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.536, "underPct": 0.464},
    {"name": "Luguentz Dort", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.535, "underPct": 0.465},
    {"name": "Jalen Suggs", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.509, "underPct": 0.491},
    {"name": "Payton Pritchard", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.508, "underPct": 0.492},
    {"name": "Jamal Murray", "line": 7.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.502, "underPct": 0.498},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.496, "underPct": 0.504},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.491, "underPct": 0.509},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.485, "underPct": 0.515},
    {"name": "Alperen Sengun", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.482, "underPct": 0.518},
    {"name": "Ryan Rollins", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.479, "underPct": 0.521},
    {"name": "Russell Westbrook", "line": 9.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.475, "underPct": 0.525},
    {"name": "Anthony Black", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.473, "underPct": 0.527},
    {"name": "Tyrese Maxey", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.464, "underPct": 0.536},
    {"name": "Aaron Holiday", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.463, "underPct": 0.537},
    {"name": "Derrick White", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.461, "underPct": 0.539},
    {"name": "Mike Conley", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Josh Hart", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.417, "underPct": 0.583},
    {"name": "Amen Thompson", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Maxime Raynaud", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.406, "underPct": 0.594},
    {"name": "James Harden", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.404, "underPct": 0.596},
    {"name": "Derik Queen", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.385, "underPct": 0.615},
    {"name": "Kevin Porter Jr.", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.378, "underPct": 0.622},
    {"name": "Pascal Siakam", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.348, "underPct": 0.652},
    {"name": "Malik Monk", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.302, "underPct": 0.698},
    {"name": "DeMar DeRozan", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.254, "underPct": 0.746},
    {"name": "Deni Avdija", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.224, "underPct": 0.776},
];const prizepicksReboundsHitRates = [
    {"name": "Amen Thompson", "line": 6.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.754, "underPct": 0.246},
    {"name": "Duncan Robinson", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.695, "underPct": 0.305},
    {"name": "Ausar Thompson", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.652, "underPct": 0.348},
    {"name": "Jaylen Brown", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.643, "underPct": 0.357},
    {"name": "Lonzo Ball", "line": 4.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.626, "underPct": 0.374},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.619, "underPct": 0.381},
    {"name": "Kevin Durant", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.604, "underPct": 0.396},
    {"name": "Jalen Johnson", "line": 10.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.594, "underPct": 0.406},
    {"name": "Rudy Gobert", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.576, "underPct": 0.424},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.572, "underPct": 0.428},
    {"name": "Spencer Jones", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.561, "underPct": 0.439},
    {"name": "Ivica Zubac", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.553, "underPct": 0.447},
    {"name": "Jaylon Tyson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.552, "underPct": 0.448},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.55, "underPct": 0.45},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.55, "underPct": 0.45},
    {"name": "Derrick White", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.543, "underPct": 0.457},
    {"name": "Saddiq Bey", "line": 6.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.541, "underPct": 0.459},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.537, "underPct": 0.463},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.537, "underPct": 0.463},
    {"name": "Josh Giddey", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.532, "underPct": 0.468},
    {"name": "Miles Bridges", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.52, "underPct": 0.48},
    {"name": "Josh Hart", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.518, "underPct": 0.482},
    {"name": "OG Anunoby", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.518, "underPct": 0.482},
    {"name": "Julius Randle", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.506, "underPct": 0.494},
    {"name": "Isaac Okoro", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.506, "underPct": 0.494},
    {"name": "Anthony Edwards", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.491, "underPct": 0.509},
    {"name": "Desmond Bane", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.486, "underPct": 0.514},
    {"name": "Tobias Harris", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
    {"name": "Mike Conley", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.479, "underPct": 0.521},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.476, "underPct": 0.524},
    {"name": "Ryan Rollins", "line": 4.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.46, "underPct": 0.54},
    {"name": "Russell Westbrook", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.455, "underPct": 0.545},
    {"name": "Chet Holmgren", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.453, "underPct": 0.547},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.448, "underPct": 0.552},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.441, "underPct": 0.559},
    {"name": "Naz Reid", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Kyle Kuzma", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Cameron Johnson", "line": 4.0, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.426, "underPct": 0.574},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.423, "underPct": 0.577},
    {"name": "Bruce Brown", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.422, "underPct": 0.578},
    {"name": "Jordan Walsh", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.41, "underPct": 0.59},
    {"name": "Peyton Watson", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.398, "underPct": 0.602},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.359, "underPct": 0.641},
    {"name": "John Collins", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.334, "underPct": 0.666},
    {"name": "Toumani Camara", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.331, "underPct": 0.669},
    {"name": "Marvin Bagley III", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.314, "underPct": 0.686},
    {"name": "Justin Champagnie", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.291, "underPct": 0.709},
    {"name": "Onyeka Okongwu", "line": 8.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.289, "underPct": 0.711},
    {"name": "Maxime Raynaud", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.254, "underPct": 0.746},
    {"name": "Bobby Portis", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.249, "underPct": 0.751},
    {"name": "Shaedon Sharpe", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.221, "underPct": 0.779},
    {"name": "Jericho Sims", "line": 7.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.219, "underPct": 0.781},
    {"name": "Myles Turner", "line": 5.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.201, "underPct": 0.799},
    {"name": "Kevin Porter Jr.", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.134, "underPct": 0.866},
];const prizepicksBlocksHitRates = [
    {"name": "John Collins", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.541, "underPct": 0.459},
    {"name": "Deni Avdija", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.359, "underPct": 0.641},
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.548, "underPct": 0.452},
    {"name": "Anthony Edwards", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.627, "underPct": 0.373},
    {"name": "Quinten Post", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.509, "underPct": 0.491},
    {"name": "Jalen Suggs", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.582, "underPct": 0.418},
    {"name": "Victor Wembanyama", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.625, "underPct": 0.375},
    {"name": "Shai Gilgeous-Alexander", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.349, "underPct": 0.651},
];const prizepicksStealsHitRates = [
    {"name": "Josh Minott", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.307, "underPct": 0.693},
    {"name": "Sam Hauser", "line": 0.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.558, "underPct": 0.442},
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.425, "underPct": 0.575},
    {"name": "Reed Sheppard", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.433, "underPct": 0.567},
    {"name": "Sidy Cissoko", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.385, "underPct": 0.615},
    {"name": "Maxime Raynaud", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.27, "underPct": 0.73},
    {"name": "Spencer Jones", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.393, "underPct": 0.607},
    {"name": "Caris LeVert", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.407, "underPct": 0.593},
    {"name": "Julius Randle", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.481, "underPct": 0.519},
    {"name": "Quinten Post", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.573, "underPct": 0.427},
    {"name": "Paolo Banchero", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.443, "underPct": 0.557},
    {"name": "Jordan Clarkson", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.308, "underPct": 0.692},
    {"name": "Cason Wallace", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.674, "underPct": 0.326},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Jaylen Brown", "line": 41.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Victor Wembanyama", "line": 30.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jose Alvarado", "line": 13.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jalen Johnson", "line": 43.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Buddy Hield", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Maxime Raynaud", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Steven Adams", "line": 14.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Kevin Porter Jr.", "line": 33.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 31.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Stephon Castle", "line": 26.5, "l5": 0.8, "l10": 0.9, "l15": 0.87, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Tyrese Maxey", "line": 39.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Coby White", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quinten Post", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandin Podziemski", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 37.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bruce Brown", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Onyeka Okongwu", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Poole", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Reed Sheppard", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Minott", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jericho Sims", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 35.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 44.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 25.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Toumani Camara", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 22.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jordan Walsh", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Anthony Melton", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Darius Garland", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 40.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sam Hauser", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gary Harris", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 12.0, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 31.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 12.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sidy Cissoko", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 33.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cameron Johnson", "line": 21.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 34.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Duncan Robinson", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kennard", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 19.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 26.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Miller", "line": 29.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 19.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lonzo Ball", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "OG Anunoby", "line": 21.0, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 38.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paolo Banchero", "line": 32.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nique Clifford", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "De'Andre Hunter", "line": 22.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 20.5, "l5": 0.0, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keon Ellis", "line": 9.0, "l5": 0.0, "l10": 0.0, "l15": 0.2, "overPct": 0.0, "underPct": 1.0},
    {"name": "DeMar DeRozan", "line": 29.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPRHitRates = [
    {"name": "Jalen Johnson", "line": 34.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Victor Wembanyama", "line": 27.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jordan Walsh", "line": 13.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 22.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Precious Achiuwa", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Holiday", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylon Tyson", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Buddy Hield", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Maxime Raynaud", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Stephon Castle", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Harrison Barnes", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Porter Jr.", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tobias Harris", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Marvin Bagley III", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Justin Champagnie", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jericho Sims", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephen Curry", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandin Podziemski", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quinten Post", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Moses Moody", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Brunson", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 15.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Rudy Gobert", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ausar Thompson", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Okogie", "line": 9.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sam Hauser", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Yves Missi", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Steven Adams", "line": 13.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Spencer Jones", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Anthony Melton", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Minott", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Williams", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bobby Portis", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 33.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 17.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Caris LeVert", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kennard", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ryan Kalkbrenner", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 34.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Poole", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Miller", "line": 25.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Myles Turner", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nique Clifford", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Peyton Watson", "line": 18.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "OG Anunoby", "line": 19.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Will Richard", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keegan Murray", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lonzo Ball", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paolo Banchero", "line": 28.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Keon Ellis", "line": 7.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPAHitRates = [
    {"name": "Jose Alvarado", "line": 10.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jalen Johnson", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Saddiq Bey", "line": 18.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tre Jones", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 16.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Porter Jr.", "line": 28.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quinten Post", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Steven Adams", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Holiday", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 35.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Victor Wembanyama", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Sam Hauser", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Maxime Raynaud", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Precious Achiuwa", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandin Podziemski", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tobias Harris", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Harrison Barnes", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cason Wallace", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephon Castle", "line": 21.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "James Harden", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Dunn", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Minott", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kobe Sanders", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 25.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Neemias Queta", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Poole", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shai Gilgeous-Alexander", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bruce Brown", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "John Collins", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "De'Anthony Melton", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Karl-Anthony Towns", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 19.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Malik Monk", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cameron Johnson", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sidy Cissoko", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kennard", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paolo Banchero", "line": 25.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "OG Anunoby", "line": 16.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keegan Murray", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nique Clifford", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Miller", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Lonzo Ball", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keon Ellis", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "De'Andre Hunter", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
];const prizepicksRAHitRates = [
    {"name": "Jalen Johnson", "line": 18.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Black", "line": 9.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 12.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Maxime Raynaud", "line": 8.0, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Okogie", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 6.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ivica Zubac", "line": 13.5, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Nicolas Batum", "line": 3.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Victor Wembanyama", "line": 12.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ausar Thompson", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keegan Murray", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derik Queen", "line": 13.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 11.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 5.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Minott", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 10.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Harrison Barnes", "line": 4.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Stephon Castle", "line": 11.0, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Desmond Bane", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quinten Post", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Caris LeVert", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Poole", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 10.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 12.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Saddiq Bey", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bobby Portis", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Walsh", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 16.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cameron Johnson", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 15.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Alperen Sengun", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paolo Banchero", "line": 11.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Porter Jr.", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jalen Suggs", "line": 8.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 7.0, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nique Clifford", "line": 6.0, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
];const prizepicksTurnoversHitRates = [
    {"name": "Kyle Kuzma", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jericho Sims", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Okogie", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Minott", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 2.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Black", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Kevin Porter Jr.", "line": 1.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Okogie", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "OG Anunoby", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Victor Wembanyama", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jericho Sims", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mike Conley", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
];const underdogPointsHitRates = [
    {"name": "De'Anthony Melton", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.879, "underPct": 0.121},
    {"name": "Stephen Curry", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.847, "underPct": 0.153},
    {"name": "Kawhi Leonard", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.813, "underPct": 0.187},
    {"name": "Jordan Walsh", "line": 7.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.807, "underPct": 0.193},
    {"name": "Chet Holmgren", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.79, "underPct": 0.21},
    {"name": "Rudy Gobert", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.789, "underPct": 0.211},
    {"name": "Jamal Murray", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.775, "underPct": 0.225},
    {"name": "Quinten Post", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.774, "underPct": 0.226},
    {"name": "Kobe Sanders", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.767, "underPct": 0.233},
    {"name": "Saddiq Bey", "line": 16.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.741, "underPct": 0.259},
    {"name": "Kris Dunn", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.738, "underPct": 0.262},
    {"name": "De'Aaron Fox", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.736, "underPct": 0.264},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.733, "underPct": 0.267},
    {"name": "Anthony Edwards", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.732, "underPct": 0.268},
    {"name": "Stephon Castle", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.729, "underPct": 0.271},
    {"name": "Buddy Hield", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.728, "underPct": 0.272},
    {"name": "Jaylen Brown", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.712, "underPct": 0.288},
    {"name": "Jalen Johnson", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.711, "underPct": 0.289},
    {"name": "Will Richard", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.686, "underPct": 0.314},
    {"name": "Brandin Podziemski", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.682, "underPct": 0.318},
    {"name": "Mikal Bridges", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.681, "underPct": 0.319},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.661, "underPct": 0.339},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.658, "underPct": 0.342},
    {"name": "Amen Thompson", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Deni Avdija", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.645, "underPct": 0.355},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.62, "underPct": 0.38},
    {"name": "Luguentz Dort", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.618, "underPct": 0.382},
    {"name": "Cade Cunningham", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.617, "underPct": 0.383},
    {"name": "Josh Minott", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.613, "underPct": 0.387},
    {"name": "Cason Wallace", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.601, "underPct": 0.399},
    {"name": "Myles Turner", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.599, "underPct": 0.401},
    {"name": "Tobias Harris", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.597, "underPct": 0.403},
    {"name": "Jalen Brunson", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Nicolas Batum", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.572, "underPct": 0.428},
    {"name": "Julius Randle", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.567, "underPct": 0.433},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.563, "underPct": 0.437},
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.535, "underPct": 0.465},
    {"name": "Sam Hauser", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.524, "underPct": 0.476},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.519, "underPct": 0.481},
    {"name": "Jeremiah Fears", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.508, "underPct": 0.492},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.492, "underPct": 0.508},
    {"name": "Kevin Porter Jr.", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.491, "underPct": 0.509},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.486, "underPct": 0.514},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.482, "underPct": 0.518},
    {"name": "Ivica Zubac", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.469, "underPct": 0.531},
    {"name": "Luke Kennard", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.468, "underPct": 0.532},
    {"name": "Jalen Duren", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.435, "underPct": 0.565},
    {"name": "Anfernee Simons", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.421, "underPct": 0.579},
    {"name": "Jalen Suggs", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.416, "underPct": 0.584},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.404, "underPct": 0.596},
    {"name": "OG Anunoby", "line": 14.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.388, "underPct": 0.612},
    {"name": "Maxime Raynaud", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.375, "underPct": 0.625},
    {"name": "Karl-Anthony Towns", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.357, "underPct": 0.643},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.334, "underPct": 0.666},
    {"name": "Jaden Ivey", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.327, "underPct": 0.673},
    {"name": "Bobby Portis", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.301, "underPct": 0.699},
    {"name": "Jalen Williams", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.273, "underPct": 0.727},
    {"name": "Shaedon Sharpe", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.229, "underPct": 0.771},
    {"name": "Russell Westbrook", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.205, "underPct": 0.795},
    {"name": "Paolo Banchero", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.164, "underPct": 0.836},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.111, "underPct": 0.889},
    {"name": "Nique Clifford", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.04, "underPct": 0.96},
];const underdogAssistsHitRates = [
    {"name": "Trey Murphy III", "line": 3.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.664, "underPct": 0.336},
    {"name": "Rudy Gobert", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.648, "underPct": 0.352},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.607, "underPct": 0.393},
    {"name": "Kyle Kuzma", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.595, "underPct": 0.405},
    {"name": "Buddy Hield", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.59, "underPct": 0.41},
    {"name": "Bobby Portis", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Jalen Suggs", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.509, "underPct": 0.491},
    {"name": "Payton Pritchard", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.508, "underPct": 0.492},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.491, "underPct": 0.509},
    {"name": "Aaron Holiday", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.463, "underPct": 0.537},
    {"name": "Mike Conley", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Josh Hart", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.417, "underPct": 0.583},
];const underdogReboundsHitRates = [
    {"name": "Keegan Murray", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.725, "underPct": 0.275},
    {"name": "Rudy Gobert", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.576, "underPct": 0.424},
    {"name": "Spencer Jones", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.561, "underPct": 0.439},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.555, "underPct": 0.445},
    {"name": "Ivica Zubac", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.553, "underPct": 0.447},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.55, "underPct": 0.45},
    {"name": "Derrick White", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.543, "underPct": 0.457},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.537, "underPct": 0.463},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.537, "underPct": 0.463},
    {"name": "De'Aaron Fox", "line": 3.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.521, "underPct": 0.479},
    {"name": "OG Anunoby", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.518, "underPct": 0.482},
    {"name": "Luke Kennard", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.509, "underPct": 0.491},
    {"name": "Julius Randle", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.506, "underPct": 0.494},
    {"name": "Anthony Edwards", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.491, "underPct": 0.509},
    {"name": "Tobias Harris", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
    {"name": "Aaron Holiday", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.471, "underPct": 0.529},
    {"name": "Ryan Rollins", "line": 4.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.46, "underPct": 0.54},
    {"name": "Naz Reid", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Alperen Sengun", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.398, "underPct": 0.602},
    {"name": "Shai Gilgeous-Alexander", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.355, "underPct": 0.645},
    {"name": "Derik Queen", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.304, "underPct": 0.696},
    {"name": "Maxime Raynaud", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.254, "underPct": 0.746},
    {"name": "Nique Clifford", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.22, "underPct": 0.78},
    {"name": "Myles Turner", "line": 5.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.201, "underPct": 0.799},
];const underdogBlocksHitRates = [
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.405, "underPct": 0.595},
    {"name": "Victor Wembanyama", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.625, "underPct": 0.375},
];const underdogStealsHitRates = [
    {"name": "Reed Sheppard", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.433, "underPct": 0.567},
    {"name": "Cason Wallace", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.674, "underPct": 0.326},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Jalen Johnson", "line": 42.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 41.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Victor Wembanyama", "line": 30.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Trey Murphy III", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Buddy Hield", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "John Collins", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 31.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Maxime Raynaud", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kevin Porter Jr.", "line": 33.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephon Castle", "line": 26.5, "l5": 0.8, "l10": 0.9, "l15": 0.87, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Derrick White", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 37.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandin Podziemski", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quinten Post", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Devin Vassell", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tobias Harris", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden Ivey", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 44.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Poole", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 35.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Minott", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Williams", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sam Hauser", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Anthony Melton", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Caris LeVert", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kennard", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 33.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sidy Cissoko", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keegan Murray", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "OG Anunoby", "line": 20.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nique Clifford", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Paolo Banchero", "line": 32.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mike Conley", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 41.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 29.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const underdogPRHitRates = [
    {"name": "Victor Wembanyama", "line": 27.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Johnson", "line": 34.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 22.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jaylen Brown", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Maxime Raynaud", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephon Castle", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Porter Jr.", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Toumani Camara", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 33.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Williams", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Paolo Banchero", "line": 28.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keegan Murray", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const underdogPAHitRates = [
    {"name": "Jalen Johnson", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylen Brown", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Victor Wembanyama", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 25.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Amen Thompson", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 31.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Onyeka Okongwu", "line": 19.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paolo Banchero", "line": 25.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
];const underdogRAHitRates = [
    {"name": "Anthony Black", "line": 9.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Steven Adams", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ausar Thompson", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Saddiq Bey", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shai Gilgeous-Alexander", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 8.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const underdogTurnoversHitRates = [
    {"name": "Ryan Rollins", "line": 2.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const underdogBlocksStealsHitRates = [
    {"name": "OG Anunoby", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Victor Wembanyama", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
];let currentPlatform = 'prizepicks';
let currentType = 'pairs';
let currentPropType = 'points';
let searchFilter = '';

function getEVClass(ev) {
    if (ev >= 10) return 'ev-high';
    if (ev >= 7) return 'ev-medium';
    return 'ev-low';
}

function getSigmaClass(sigma) {
    if (sigma === 'High') return 'sigma-high';
    if (sigma === 'Med') return 'sigma-med';
    return 'sigma-low';
}

function getHitRateClass(hitRate) {
    if (hitRate >= 70) return 'hit-rate-high';
    if (hitRate >= 60) return 'hit-rate-medium';
    return 'hit-rate-low';
}

function getTrendArrow(l5, l15) {
    const diff = l5 - l15;
    if (diff > 0.15) return '<span class="trend-arrow trend-up">↑</span>';
    if (diff < -0.15) return '<span class="trend-arrow trend-down">↓</span>';
    return '<span class="trend-arrow trend-stable">→</span>';
}

function formatOdds(odds) {
    return odds > 0 ? `+${odds}` : odds;
}

function renderSinglesTable(data) {
    const thead = `
        <tr>
            <th style="width: 5%">#</th>
            <th style="width: 18%">Player</th>
            <th style="width: 12%">Bookmaker</th>
            <th style="width: 10%">Line</th>
            <th style="width: 10%">Proj.</th>
            <th style="width: 10%">Side</th>
            <th style="width: 8%">Odds</th>
            <th style="width: 9%">EV%</th>
            <th style="width: 9%">Kelly</th>
            <th style="width: 9%">Odds</th>
        </tr>
    `;

    const tbody = data.map((row, index) => `
        <tr>
            <td style="color: #667eea; font-weight: 700;">${index + 1}</td>
            <td class="player-name">${row.name}</td>
            <td style="color: #a0a0a0;">${row.bookmaker}</td>
            <td class="line-value">${row.line}</td>
            <td style="color: #9ca3af;">${row.prediction}</td>
            <td>
                <span class="side-badge side-${row.side.toLowerCase()}">${row.side}</span>
            </td>
            <td style="font-weight: 600; color: ${row.odds > 0 ? '#34d399' : '#f87171'};">${formatOdds(row.odds)}</td>
            <td class="ev-cell ${getEVClass(row.ev)}">${row.ev.toFixed(2)}%</td>
            <td class="kelly-cell">${(row.kelly * 100).toFixed(1)}%</td>
            <td style="font-weight: 600; color: ${row.odds > 0 ? '#f87171' : '#34d399'};" title="American odds format. Positive odds show potential profit on $100 bet. Negative odds show amount needed to bet to win $100.">
                ${formatOdds(row.odds)}
            </td>
        </tr>
    `).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

function renderPairsTable(data) {
    const thead = `
        <tr>
            <th style="width: 2%">#</th>
            <th style="width: 12%">Player 1</th>
            <th style="width: 5%">Line</th>
            <th style="width: 5%">Proj.</th>
            <th style="width: 5%">Odds</th>
            <th style="width: 6%">Model Prob</th>
            <th style="width: 12%">Player 2</th>
            <th style="width: 5%">Line</th>
            <th style="width: 5%">Proj.</th>
            <th style="width: 5%">Odds</th>
            <th style="width: 6%">Model Prob</th>
            <th style="width: 6%">Parlay Prob</th>
            <th style="width: 6%">Parlay Odds</th>
            <th style="width: 6%">EV%</th>
            <th style="width: 6%">Kelly</th>
        </tr>
    `;

    const tbody = data.map((row, index) => {
        const evPercent = row.evPercent || 0;
        const kellyQuarter = row.kellyQuarter || 0;
        
        return `
        <tr>
            <td style="color: #667eea; font-weight: 700;">${index + 1}</td>
            <td>
                <div class="player-cell">
                    <span class="player-name">${row.name1}</span>
                    <div class="player-stats">
                        <span class="side-badge side-${row.side1}">${row.side1}</span>
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line1}</td>
            <td class="prediction-value" style="color: ${row.prediction1 > row.line1 ? '#10b981' : '#f59e0b'}; font-weight: 600;">${row.prediction1.toFixed(1)}</td>
            <td style="font-weight: 600; color: ${row.odds1 > 0 ? '#f87171' : '#34d399'}; font-size: 0.9rem;">
                ${formatOdds(row.odds1 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.9rem;">
                ${((row.modelProb1 || 0) * 100).toFixed(1)}%
            </td>
            <td>
                <div class="player-cell">
                    <span class="player-name">${row.name2}</span>
                    <div class="player-stats">
                        <span class="side-badge side-${row.side2}">${row.side2}</span>
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line2}</td>
            <td class="prediction-value" style="color: ${row.prediction2 > row.line2 ? '#10b981' : '#f59e0b'}; font-weight: 600;">${row.prediction2.toFixed(1)}</td>
            <td style="font-weight: 600; color: ${row.odds2 > 0 ? '#f87171' : '#34d399'}; font-size: 0.9rem;">
                ${formatOdds(row.odds2 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.9rem;">
                ${((row.modelProb2 || 0) * 100).toFixed(1)}%
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.9rem;">
                ${((row.parlayProb || 0) * 100).toFixed(1)}%
            </td>
            <td style="font-weight: 600; color: ${row.parlayOdds > 0 ? '#f87171' : '#34d399'}; font-size: 0.9rem;">
                ${formatOdds(row.parlayOdds || -137)}
            </td>
            <td class="ev-cell ${getEVClass(evPercent)}">${evPercent.toFixed(2)}%</td>
            <td class="kelly-cell">${(kellyQuarter * 100).toFixed(1)}%</td>
        </tr>
    `}).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

function renderTriosTable(data) {
    const thead = `
        <tr>
            <th style="width: 2%">#</th>
            <th style="width: 9%">Player 1</th>
            <th style="width: 4%">Line</th>
            <th style="width: 4%">Proj.</th>
            <th style="width: 4%">Odds</th>
            <th style="width: 5%">Model Prob</th>
            <th style="width: 9%">Player 2</th>
            <th style="width: 4%">Line</th>
            <th style="width: 4%">Proj.</th>
            <th style="width: 4%">Odds</th>
            <th style="width: 5%">Model Prob</th>
            <th style="width: 9%">Player 3</th>
            <th style="width: 4%">Line</th>
            <th style="width: 4%">Proj.</th>
            <th style="width: 4%">Odds</th>
            <th style="width: 5%">Model Prob</th>
            <th style="width: 5%">Parlay Prob</th>
            <th style="width: 5%">Parlay Odds</th>
            <th style="width: 5%">EV%</th>
            <th style="width: 5%">Kelly</th>
        </tr>
    `;

    const tbody = data.map((row, index) => {
        const evPercent = row.evPercent || 0;
        const kellyQuarter = row.kellyQuarter || 0;
        
        return `
        <tr>
            <td style="color: #667eea; font-weight: 700;">${index + 1}</td>
            <td>
                <div class="player-cell">
                    <span class="player-name">${row.name1}</span>
                    <div class="player-stats">
                        <span class="side-badge side-${row.side1}">${row.side1}</span>
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line1}</td>
            <td class="prediction-value" style="color: ${row.prediction1 > row.line1 ? '#10b981' : '#f59e0b'}; font-weight: 600;">${row.prediction1.toFixed(1)}</td>
            <td style="font-weight: 600; color: ${row.odds1 > 0 ? '#f87171' : '#34d399'}; font-size: 0.85rem;">
                ${formatOdds(row.odds1 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.modelProb1 || 0) * 100).toFixed(1)}%
            </td>
            <td>
                <div class="player-cell">
                    <span class="player-name">${row.name2}</span>
                    <div class="player-stats">
                        <span class="side-badge side-${row.side2}">${row.side2}</span>
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line2}</td>
            <td class="prediction-value" style="color: ${row.prediction2 > row.line2 ? '#10b981' : '#f59e0b'}; font-weight: 600;">${row.prediction2.toFixed(1)}</td>
            <td style="font-weight: 600; color: ${row.odds2 > 0 ? '#f87171' : '#34d399'}; font-size: 0.85rem;">
                ${formatOdds(row.odds2 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.modelProb2 || 0) * 100).toFixed(1)}%
            </td>
            <td>
                <div class="player-cell">
                    <span class="player-name">${row.name3}</span>
                    <div class="player-stats">
                        <span class="side-badge side-${row.side3}">${row.side3}</span>
                    </div>
                </div>
            </td>
            <td class="line-value">${row.line3}</td>
            <td class="prediction-value" style="color: ${row.prediction3 > row.line3 ? '#10b981' : '#f59e0b'}; font-weight: 600;">${row.prediction3.toFixed(1)}</td>
            <td style="font-weight: 600; color: ${row.odds3 > 0 ? '#f87171' : '#34d399'}; font-size: 0.85rem;">
                ${formatOdds(row.odds3 || -137)}
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.modelProb3 || 0) * 100).toFixed(1)}%
            </td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.parlayProb || 0) * 100).toFixed(1)}%
            </td>
            <td style="font-weight: 600; color: ${row.parlayOdds > 0 ? '#f87171' : '#34d399'}; font-size: 0.85rem;">
                ${formatOdds(row.parlayOdds || -137)}
            </td>
            <td class="ev-cell ${getEVClass(evPercent)}">${evPercent.toFixed(2)}%</td>
            <td class="kelly-cell">${(kellyQuarter * 100).toFixed(1)}%</td>
        </tr>
    `}).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

function renderHitRatesTable(data) {
    const thead = `
        <tr>
            <th style="width: 5%">#</th>
            <th style="width: 25%">Player</th>
            <th style="width: 10%">Line</th>
            <th style="width: 10%">L-5</th>
            <th style="width: 10%">L-10</th>
            <th style="width: 10%">L-15</th>
            <th style="width: 12%">Over %</th>
            <th style="width: 12%">Under %</th>
            <th style="width: 6%">Trend</th>
        </tr>
    `;

    const tbody = data.map((row, index) => `
        <tr>
            <td style="color: #667eea; font-weight: 700;">${index + 1}</td>
            <td class="player-name">${row.name}</td>
            <td class="line-value">${row.line}</td>
            <td style="color: ${row.l5 >= 0.7 ? '#34d399' : row.l5 >= 0.5 ? '#fbbf24' : '#f87171'}; font-weight: 600;">${(row.l5 * 100).toFixed(0)}%</td>
            <td style="color: ${row.l10 >= 0.7 ? '#34d399' : row.l10 >= 0.5 ? '#fbbf24' : '#f87171'}; font-weight: 600;">${(row.l10 * 100).toFixed(0)}%</td>
            <td style="color: ${row.l15 >= 0.7 ? '#34d399' : row.l15 >= 0.5 ? '#fbbf24' : '#f87171'}; font-weight: 600;">${(row.l15 * 100).toFixed(0)}%</td>
            <td>
                <span class="hit-rate ${getHitRateClass(row.overPct * 100)}">${(row.overPct * 100).toFixed(1)}%</span>
            </td>
            <td>
                <span class="hit-rate ${getHitRateClass(row.underPct * 100)}">${(row.underPct * 100).toFixed(1)}%</span>
            </td>
            <td>${getTrendArrow(row.l5, row.l15)}</td>
        </tr>
    `).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

function updateStats(data) {
    let statsHTML = '';
    
    if (currentType === 'hitrates') {
        statsHTML = `
            <div class="stat-card">
                <div class="stat-label">L-5 / L-10 / L-15</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Hit rate over last 5, 10, or 15 games</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Over % / Under %</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Probability of hitting the line from Poisson model for PTS, REB, AST, BLK, and STL, other prop types use L-10 as a proxy for over percentage</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Trend</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">↑ Improving ↓ Declining → Stable</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Color Coding</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;"><span style="color: #34d399;">Green ≥70%</span> <span style="color: #fbbf24;">Yellow ≥50%</span> <span style="color: #f87171;">Red <50%</span></div>
            </div>
        `;
    } else {
        statsHTML = `
            <div class="stat-card">
                <div class="stat-label">Probability</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">How confident the model is that the player will hit their prop line.</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Projection</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Model's projected value given the context of the game and player performance</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Expected Value %</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value as a percentage of your stake (Ex. If EV% is 5%, you can expect to profit 5% of whatever stake you place on that bet on average.)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Odds</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">American odds format. Positive odds (red) show potential profit on $100 bet. Negative odds (green) show amount needed to bet to win $100.</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Kelly Criterion</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Optimal bet sizing percentage to maximize long-term bankroll growth while managing risk (Ex. If bankroll is $10 per bet, and kelly is 25%, bet only $2.50)</div>
            </div>
        `;
    }
    
    document.getElementById('statsContainer').innerHTML = statsHTML;
}

function getData() {
    if (currentType === 'hitrates') {
        // Get hit rate data based on platform and prop type
        if (currentPlatform === 'prizepicks') {
            if (currentPropType === 'points') return prizepicksPointsHitRates;
            if (currentPropType === 'assists') return prizepicksAssistsHitRates;
            if (currentPropType === 'rebounds') return prizepicksReboundsHitRates;
            if (currentPropType === 'blocks') return prizepicksBlocksHitRates;
            if (currentPropType === 'steals') return prizepicksStealsHitRates;
            // Combo props
            if (currentPropType === 'PRA') return prizepicksPRAHitRates;
            if (currentPropType === 'PR') return prizepicksPRHitRates;
            if (currentPropType === 'PA') return prizepicksPAHitRates;
            if (currentPropType === 'RA') return prizepicksRAHitRates;
            if (currentPropType === 'Turnovers') return prizepicksTurnoversHitRates;
            if (currentPropType === 'BlocksSteals') return prizepicksBlocksStealsHitRates;
        } else {
            if (currentPropType === 'points') return underdogPointsHitRates;
            if (currentPropType === 'assists') return underdogAssistsHitRates;
            if (currentPropType === 'rebounds') return underdogReboundsHitRates;
            if (currentPropType === 'blocks') return underdogBlocksHitRates;
            if (currentPropType === 'steals') return underdogStealsHitRates;
            // Combo props
            if (currentPropType === 'PRA') return underdogPRAHitRates;
            if (currentPropType === 'PR') return underdogPRHitRates;
            if (currentPropType === 'PA') return underdogPAHitRates;
            if (currentPropType === 'RA') return underdogRAHitRates;
            if (currentPropType === 'Turnovers') return underdogTurnoversHitRates;
            if (currentPropType === 'BlocksSteals') return underdogBlocksStealsHitRates;
        }
    }
    
    if (currentPlatform === 'prizepicks') {
        if (currentType === 'pairs') return prizepicksPairsData;
        return prizepicksTriosData;
    } else {
        if (currentType === 'pairs') return underdogPairsData;
        return underdogTriosData;
    }
}

function render() {
    let data = getData();
    
    // Apply search filter if on hit rates view
    if (currentType === 'hitrates' && searchFilter) {
        data = data.filter(row => 
            row.name.toLowerCase().includes(searchFilter.toLowerCase())
        );
    }
    
    // Show/hide platform toggle and prop type selector based on bet type
    const platformToggle = document.getElementById('platformToggle');
    const propTypeGroup = document.getElementById('propTypeGroup');
    const searchGroup = document.getElementById('searchGroup');
    
    if (currentType === 'hitrates') {
        platformToggle.style.display = 'flex';
        propTypeGroup.style.display = 'flex';
        searchGroup.style.display = 'flex';
    } else {
        platformToggle.style.display = 'flex';
        propTypeGroup.style.display = 'none';
        searchGroup.style.display = 'none';
    }
    
    if (currentType === 'pairs') {
        renderPairsTable(data);
    } else if (currentType === 'trios') {
        renderTriosTable(data);
    } else if (currentType === 'hitrates') {
        renderHitRatesTable(data);
    }
    
    updateStats(data);
    
    const betTypeLabel = currentType === 'hitrates' ? 'Hit Rates' : currentType.charAt(0).toUpperCase() + currentType.slice(1);
    const platformLabel = currentPlatform === 'prizepicks' ? 'PrizePicks' : 'Underdog';
    const propTypeLabel = currentPropType.charAt(0).toUpperCase() + currentPropType.slice(1);
    
    if (currentType === 'hitrates') {
        const totalCount = getData().length;
        const filteredText = searchFilter ? ` (filtered from ${totalCount})` : '';
        document.getElementById('picksCount').textContent = `Showing ${data.length} ${propTypeLabel} ${betTypeLabel} for ${platformLabel}${filteredText}`;
    } else {
        document.getElementById('picksCount').textContent = `Showing top ${data.length} ${betTypeLabel} for ${platformLabel}`;
    }
}

// Event listeners
document.querySelectorAll('[data-platform]').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('[data-platform]').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        currentPlatform = this.dataset.platform;
        render();
    });
});

document.getElementById('betTypeSelect').addEventListener('change', function() {
    currentType = this.value;
    render();
});

document.getElementById('propTypeSelect').addEventListener('change', function() {
    currentPropType = this.value;
    render();
});

document.getElementById('playerSearch').addEventListener('input', function() {
    searchFilter = this.value;
    render();
});

// Update last updated timestamp
function updateLastUpdated() {
    // Get timestamp from meta tag (set during GitHub Pages deployment)
    const metaTag = document.querySelector('meta[name="last-updated"]');
    let timestamp;
    
    if (metaTag && metaTag.content && metaTag.content !== 'BUILD_TIMESTAMP') {
        // Use the deployment timestamp from GitHub Actions
        timestamp = new Date(metaTag.content);
    } else {
        // Fallback to current time if meta tag not set (for local development)
        timestamp = new Date();
    }
    
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    };
    const formattedDate = timestamp.toLocaleString('en-US', options);
    const timeElement = document.getElementById('lastUpdatedTime');
    if (timeElement) {
        timeElement.textContent = formattedDate;
    }
}

// Initial render
updateLastUpdated();
render();

