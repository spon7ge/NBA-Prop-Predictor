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
    {"name1": "Jaime Jaquez Jr.", "name2": "Isaiah Joe", "line1": 10.0, "line2": 7.5, "odds1": -137, "odds2": -105, "prediction1": 20.78, "prediction2": 14.38, "modelProb1": 0.978, "modelProb2": 0.891, "side1": "over", "side2": "over", "parlayProb": 0.871, "parlayOdds": 238, "evPercent": 194.44, "kellyQuarter": 0.2042},
    {"name1": "Kel'el Ware", "name2": "Aaron Wiggins", "line1": 8.5, "line2": 9.0, "odds1": -102, "odds2": -137, "prediction1": 12.99, "prediction2": 15.88, "modelProb1": 0.85, "modelProb2": 0.881, "side1": "over", "side2": "over", "parlayProb": 0.749, "parlayOdds": 243, "evPercent": 156.98, "kellyQuarter": 0.1615},
    {"name1": "Mark Williams", "name2": "Victor Wembanyama", "line1": 9.5, "line2": 20.5, "odds1": -122, "odds2": -109, "prediction1": 13.29, "prediction2": 24.97, "modelProb1": 0.809, "modelProb2": 0.779, "side1": "over", "side2": "over", "parlayProb": 0.63, "parlayOdds": 249, "evPercent": 119.79, "kellyQuarter": 0.1203},
    {"name1": "Anthony Black", "name2": "Devin Booker", "line1": 15.5, "line2": 22.5, "odds1": -110, "odds2": -115, "prediction1": 10.22, "prediction2": 26.48, "modelProb1": 0.756, "modelProb2": 0.745, "side1": "under", "side2": "over", "parlayProb": 0.563, "parlayOdds": 257, "evPercent": 101.09, "kellyQuarter": 0.0983},
    {"name1": "Tristan da Silva", "name2": "Dillon Brooks", "line1": 8.5, "line2": 19.0, "odds1": -105, "odds2": -137, "prediction1": 10.97, "prediction2": 22.74, "modelProb1": 0.712, "modelProb2": 0.76, "side1": "over", "side2": "over", "parlayProb": 0.541, "parlayOdds": 238, "evPercent": 82.91, "kellyQuarter": 0.0871},
    {"name1": "Dru Smith", "name2": "Deandre Ayton", "line1": 4.5, "line2": 13.5, "odds1": -110, "odds2": -110, "prediction1": 6.64, "prediction2": 15.72, "modelProb1": 0.716, "modelProb2": 0.683, "side1": "over", "side2": "over", "parlayProb": 0.489, "parlayOdds": 264, "evPercent": 77.97, "kellyQuarter": 0.0738},
    {"name1": "Jalen Suggs", "name2": "Ja'Kobe Walter", "line1": 18.5, "line2": 7.0, "odds1": -110, "odds2": -137, "prediction1": 12.2, "prediction2": 2.68, "modelProb1": 0.709, "modelProb2": 0.74, "side1": "under", "side2": "under", "parlayProb": 0.525, "parlayOdds": 230, "evPercent": 73.32, "kellyQuarter": 0.0797},
    {"name1": "Tyus Jones", "name2": "Devin Vassell", "line1": 4.0, "line2": 13.5, "odds1": -137, "odds2": -104, "prediction1": 0.87, "prediction2": 15.52, "modelProb1": 0.754, "modelProb2": 0.653, "side1": "under", "side2": "over", "parlayProb": 0.492, "parlayOdds": 239, "evPercent": 66.74, "kellyQuarter": 0.0698},
    {"name1": "Desmond Bane", "name2": "Rui Hachimura", "line1": 22.0, "line2": 11.5, "odds1": -137, "odds2": -102, "prediction1": 16.32, "prediction2": 12.84, "modelProb1": 0.729, "modelProb2": 0.629, "side1": "under", "side2": "over", "parlayProb": 0.459, "parlayOdds": 243, "evPercent": 57.47, "kellyQuarter": 0.0591},
    {"name1": "Jalen Brunson", "name2": "Chet Holmgren", "line1": 27.5, "line2": 18.0, "odds1": 105, "odds2": -137, "prediction1": 28.93, "prediction2": 13.51, "modelProb1": 0.595, "modelProb2": 0.706, "side1": "over", "side2": "under", "parlayProb": 0.42, "parlayOdds": 255, "evPercent": 49.05, "kellyQuarter": 0.0481},
];const prizepicksTriosData = [
    {"name1": "Jaime Jaquez Jr.", "name2": "Isaiah Joe", "name3": "Victor Wembanyama", "line1": 10.0, "line2": 7.5, "line3": 20.5, "odds1": -137, "odds2": -105, "odds3": -109, "prediction1": 20.78, "prediction2": 14.38, "prediction3": 24.97, "modelProb1": 0.978, "modelProb2": 0.891, "modelProb3": 0.779, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.678, "parlayOdds": 548, "evPercent": 339.47, "kellyQuarter": 0.1549},
    {"name1": "Kel'el Ware", "name2": "Aaron Wiggins", "name3": "Deandre Ayton", "line1": 8.5, "line2": 9.0, "line3": 13.5, "odds1": -102, "odds2": -137, "odds3": -110, "prediction1": 12.99, "prediction2": 15.88, "prediction3": 15.72, "modelProb1": 0.85, "modelProb2": 0.881, "modelProb3": 0.683, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.511, "parlayOdds": 554, "evPercent": 234.47, "kellyQuarter": 0.1058},
    {"name1": "Anthony Black", "name2": "Ja'Kobe Walter", "name3": "Mark Williams", "line1": 15.5, "line2": 7.0, "line3": 9.5, "odds1": -110, "odds2": -137, "odds3": -122, "prediction1": 10.22, "prediction2": 2.68, "prediction3": 13.29, "modelProb1": 0.756, "modelProb2": 0.74, "modelProb3": 0.809, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.453, "parlayOdds": 501, "evPercent": 172.02, "kellyQuarter": 0.0858},
    {"name1": "Tristan da Silva", "name2": "Devin Booker", "name3": "Devin Vassell", "line1": 8.5, "line2": 22.5, "line3": 13.5, "odds1": -105, "odds2": -115, "odds3": -104, "prediction1": 10.97, "prediction2": 26.48, "prediction3": 15.52, "modelProb1": 0.712, "modelProb2": 0.745, "modelProb3": 0.653, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.346, "parlayOdds": 616, "evPercent": 147.96, "kellyQuarter": 0.06},
    {"name1": "Dru Smith", "name2": "Dillon Brooks", "name3": "Rui Hachimura", "line1": 4.5, "line2": 19.0, "line3": 11.5, "odds1": -110, "odds2": -137, "odds3": -102, "prediction1": 6.64, "prediction2": 22.74, "prediction3": 12.84, "modelProb1": 0.716, "modelProb2": 0.76, "modelProb3": 0.629, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.343, "parlayOdds": 554, "evPercent": 124.05, "kellyQuarter": 0.056},
    {"name1": "Jalen Suggs", "name2": "Jalen Brunson", "name3": "Chet Holmgren", "line1": 18.5, "line2": 27.5, "line3": 18.0, "odds1": -110, "odds2": 105, "odds3": -137, "prediction1": 12.2, "prediction2": 28.93, "prediction3": 13.51, "modelProb1": 0.709, "modelProb2": 0.595, "modelProb3": 0.706, "side1": "under", "side2": "over", "side3": "under", "parlayProb": 0.298, "parlayOdds": 577, "evPercent": 101.66, "kellyQuarter": 0.044},
    {"name1": "Tyus Jones", "name2": "Josh Hart", "name3": "Grayson Allen", "line1": 4.0, "line2": 13.0, "line3": 15.5, "odds1": -137, "odds2": -137, "odds3": -115, "prediction1": 0.87, "prediction2": 8.84, "prediction3": 17.27, "modelProb1": 0.754, "modelProb2": 0.681, "modelProb3": 0.646, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.332, "parlayOdds": 459, "evPercent": 85.52, "kellyQuarter": 0.0466},
    {"name1": "Desmond Bane", "name2": "Sandro Mamukelashvili", "name3": "Luguentz Dort", "line1": 22.0, "line2": 9.5, "line3": 7.5, "odds1": -137, "odds2": -115, "odds3": -102, "prediction1": 16.32, "prediction2": 6.67, "prediction3": 8.18, "modelProb1": 0.729, "modelProb2": 0.602, "modelProb3": 0.593, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.26, "parlayOdds": 541, "evPercent": 66.91, "kellyQuarter": 0.0309},
    {"name1": "Paolo Banchero", "name2": "Mikal Bridges", "name3": "Royce O'Neale", "line1": 20.0, "line2": 15.5, "line3": 8.5, "odds1": -137, "odds2": -102, "odds3": -114, "prediction1": 15.75, "prediction2": 16.03, "prediction3": 9.51, "modelProb1": 0.675, "modelProb2": 0.564, "modelProb3": 0.616, "side1": "under", "side2": "over", "side3": "over", "parlayProb": 0.234, "parlayOdds": 543, "evPercent": 50.75, "kellyQuarter": 0.0234},
    {"name1": "Tyler Herro", "name2": "Ochai Agbaji", "name3": "Alex Caruso", "line1": 22.0, "line2": 6.0, "line3": 4.5, "odds1": -137, "odds2": -137, "odds3": -114, "prediction1": 24.51, "prediction2": 7.23, "prediction3": 5.13, "modelProb1": 0.67, "modelProb2": 0.635, "modelProb3": 0.603, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.257, "parlayOdds": 462, "evPercent": 44.19, "kellyQuarter": 0.0239},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Jaime Jaquez Jr.", "name2": "Isaiah Joe", "line1": 10.5, "line2": 7.5, "odds1": 100, "odds2": -105, "prediction1": 20.78, "prediction2": 14.38, "modelProb1": 0.97, "modelProb2": 0.891, "side1": "over", "side2": "over", "parlayProb": 0.864, "parlayOdds": 290, "evPercent": 237.04, "kellyQuarter": 0.2043},
    {"name1": "Kel'el Ware", "name2": "Mark Williams", "line1": 8.5, "line2": 9.5, "odds1": -102, "odds2": -122, "prediction1": 12.99, "prediction2": 13.29, "modelProb1": 0.85, "modelProb2": 0.809, "side1": "over", "side2": "over", "parlayProb": 0.688, "parlayOdds": 260, "evPercent": 147.56, "kellyQuarter": 0.1419},
    {"name1": "Anthony Black", "name2": "Devin Booker", "line1": 15.5, "line2": 22.5, "odds1": -110, "odds2": -115, "prediction1": 10.22, "prediction2": 26.48, "modelProb1": 0.756, "modelProb2": 0.745, "side1": "under", "side2": "over", "parlayProb": 0.563, "parlayOdds": 257, "evPercent": 101.09, "kellyQuarter": 0.0983},
    {"name1": "Dru Smith", "name2": "Dillon Brooks", "line1": 4.5, "line2": 18.5, "odds1": -110, "odds2": -137, "prediction1": 6.64, "prediction2": 22.74, "modelProb1": 0.716, "modelProb2": 0.788, "side1": "over", "side2": "over", "parlayProb": 0.564, "parlayOdds": 230, "evPercent": 86.24, "kellyQuarter": 0.0937},
    {"name1": "Jalen Suggs", "name2": "Jalen Brunson", "line1": 18.5, "line2": 27.5, "odds1": -110, "odds2": 105, "prediction1": 12.2, "prediction2": 28.93, "modelProb1": 0.709, "modelProb2": 0.595, "side1": "under", "side2": "over", "parlayProb": 0.422, "parlayOdds": 291, "evPercent": 64.99, "kellyQuarter": 0.0558},
    {"name1": "Tyler Herro", "name2": "Grayson Allen", "line1": 22.5, "line2": 15.5, "odds1": 100, "odds2": -115, "prediction1": 24.51, "prediction2": 17.27, "modelProb1": 0.642, "modelProb2": 0.646, "side1": "over", "side2": "over", "parlayProb": 0.415, "parlayOdds": 274, "evPercent": 55.04, "kellyQuarter": 0.0502},
    {"name1": "Paolo Banchero", "name2": "Luguentz Dort", "line1": 20.5, "line2": 7.5, "odds1": -136, "odds2": -102, "prediction1": 15.75, "prediction2": 8.18, "modelProb1": 0.702, "modelProb2": 0.593, "side1": "under", "side2": "over", "parlayProb": 0.416, "parlayOdds": 244, "evPercent": 43.26, "kellyQuarter": 0.0443},
    {"name1": "Davion Mitchell", "name2": "Royce O'Neale", "line1": 8.5, "line2": 8.5, "odds1": 105, "odds2": -114, "prediction1": 8.84, "prediction2": 9.51, "modelProb1": 0.568, "modelProb2": 0.616, "side1": "over", "side2": "over", "parlayProb": 0.35, "parlayOdds": 285, "evPercent": 34.74, "kellyQuarter": 0.0305},
    {"name1": "Sandro Mamukelashvili", "name2": "Alex Caruso", "line1": 9.5, "line2": 4.5, "odds1": -115, "odds2": -114, "prediction1": 6.67, "prediction2": 5.13, "modelProb1": 0.602, "modelProb2": 0.603, "side1": "under", "side2": "over", "parlayProb": 0.363, "parlayOdds": 251, "evPercent": 27.27, "kellyQuarter": 0.0272},
    {"name1": "Mikal Bridges", "name2": "Jalen Williams", "line1": 15.5, "line2": 18.5, "odds1": -102, "odds2": -105, "prediction1": 16.03, "prediction2": 18.83, "modelProb1": 0.564, "modelProb2": 0.556, "side1": "over", "side2": "over", "parlayProb": 0.314, "parlayOdds": 287, "evPercent": 21.42, "kellyQuarter": 0.0187},
];const underdogTriosData = [
    {"name1": "Jaime Jaquez Jr.", "name2": "Jalen Brunson", "name3": "Isaiah Joe", "line1": 10.5, "line2": 27.5, "line3": 7.5, "odds1": 100, "odds2": 105, "odds3": -105, "prediction1": 20.78, "prediction2": 28.93, "prediction3": 14.38, "modelProb1": 0.97, "modelProb2": 0.595, "modelProb3": 0.891, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.514, "parlayOdds": 700, "evPercent": 311.19, "kellyQuarter": 0.1111},
    {"name1": "Kel'el Ware", "name2": "Sandro Mamukelashvili", "name3": "Mark Williams", "line1": 8.5, "line2": 9.5, "line3": 9.5, "odds1": -102, "odds2": -115, "odds3": -122, "prediction1": 12.99, "prediction2": 6.67, "prediction3": 13.29, "modelProb1": 0.85, "modelProb2": 0.602, "modelProb3": 0.809, "side1": "over", "side2": "under", "side3": "over", "parlayProb": 0.414, "parlayOdds": 574, "evPercent": 178.88, "kellyQuarter": 0.0779},
    {"name1": "Anthony Black", "name2": "Mikal Bridges", "name3": "Devin Booker", "line1": 15.5, "line2": 15.5, "line3": 22.5, "odds1": -110, "odds2": -102, "odds3": -115, "prediction1": 10.22, "prediction2": 16.03, "prediction3": 26.48, "modelProb1": 0.756, "modelProb2": 0.564, "modelProb3": 0.745, "side1": "under", "side2": "over", "side3": "over", "parlayProb": 0.318, "parlayOdds": 607, "evPercent": 124.51, "kellyQuarter": 0.0513},
    {"name1": "Dru Smith", "name2": "Brandon Ingram", "name3": "Dillon Brooks", "line1": 4.5, "line2": 23.5, "line3": 18.5, "odds1": -110, "odds2": -116, "odds3": -137, "prediction1": 6.64, "prediction2": 20.65, "prediction3": 22.74, "modelProb1": 0.716, "modelProb2": 0.587, "modelProb3": 0.788, "side1": "over", "side2": "under", "side3": "over", "parlayProb": 0.331, "parlayOdds": 515, "evPercent": 103.79, "kellyQuarter": 0.0504},
    {"name1": "Jalen Suggs", "name2": "Immanuel Quickley", "name3": "Grayson Allen", "line1": 18.5, "line2": 17.5, "line3": 15.5, "odds1": -110, "odds2": -104, "odds3": -115, "prediction1": 12.2, "prediction2": 17.54, "prediction3": 17.27, "modelProb1": 0.709, "modelProb2": 0.53, "modelProb3": 0.646, "side1": "under", "side2": "over", "side3": "over", "parlayProb": 0.243, "parlayOdds": 600, "evPercent": 69.93, "kellyQuarter": 0.0291},
    {"name1": "Tyler Herro", "name2": "Scottie Barnes", "name3": "Luguentz Dort", "line1": 22.5, "line2": 20.5, "line3": 7.5, "odds1": 100, "odds2": 102, "odds3": -102, "prediction1": 24.51, "prediction2": 19.25, "prediction3": 8.18, "modelProb1": 0.642, "modelProb2": 0.485, "modelProb3": 0.593, "side1": "over", "side2": "under", "side3": "over", "parlayProb": 0.185, "parlayOdds": 700, "evPercent": 47.66, "kellyQuarter": 0.017},
    {"name1": "Paolo Banchero", "name2": "Karl-Anthony Towns", "name3": "Royce O'Neale", "line1": 20.5, "line2": 22.5, "line3": 8.5, "odds1": -136, "odds2": -120, "odds3": -114, "prediction1": 15.75, "prediction2": 22.19, "prediction3": 9.51, "modelProb1": 0.702, "modelProb2": 0.437, "modelProb3": 0.616, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.189, "parlayOdds": 497, "evPercent": 12.82, "kellyQuarter": 0.0064},
];const prizepicksPointsHitRates = [
    {"name": "Mark Williams", "line": 9.5, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.926, "underPct": 0.074},
    {"name": "Austin Reaves", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.907, "underPct": 0.093},
    {"name": "Devin Vassell", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.896, "underPct": 0.104},
    {"name": "Aaron Wiggins", "line": 9.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.892, "underPct": 0.108},
    {"name": "Isaiah Joe", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.798, "underPct": 0.202},
    {"name": "Josh Hart", "line": 13.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.792, "underPct": 0.208},
    {"name": "Norman Powell", "line": 21.0, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.791, "underPct": 0.209},
    {"name": "De'Aaron Fox", "line": 21.0, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.758, "underPct": 0.242},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.754, "underPct": 0.246},
    {"name": "Dillon Brooks", "line": 19.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.746, "underPct": 0.254},
    {"name": "Andrew Wiggins", "line": 14.0, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.732, "underPct": 0.268},
    {"name": "Victor Wembanyama", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.717, "underPct": 0.283},
    {"name": "Kel'el Ware", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.679, "underPct": 0.321},
    {"name": "Luguentz Dort", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.676, "underPct": 0.324},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.674, "underPct": 0.326},
    {"name": "Royce O'Neale", "line": 8.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.661, "underPct": 0.339},
    {"name": "Cason Wallace", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.642, "underPct": 0.358},
    {"name": "Grayson Allen", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.615, "underPct": 0.385},
    {"name": "Tyler Herro", "line": 22.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.614, "underPct": 0.386},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.611, "underPct": 0.389},
    {"name": "Deandre Ayton", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.608, "underPct": 0.392},
    {"name": "Dru Smith", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.603, "underPct": 0.397},
    {"name": "Stephon Castle", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.581, "underPct": 0.419},
    {"name": "Alex Caruso", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.58, "underPct": 0.42},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.578, "underPct": 0.422},
    {"name": "Bam Adebayo", "line": 18.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.571, "underPct": 0.429},
    {"name": "Chet Holmgren", "line": 18.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.539, "underPct": 0.461},
    {"name": "Karl-Anthony Towns", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.535, "underPct": 0.465},
    {"name": "Anthony Black", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.528, "underPct": 0.472},
    {"name": "Scottie Barnes", "line": 21.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.503, "underPct": 0.497},
    {"name": "Goga Bitadze", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.491, "underPct": 0.509},
    {"name": "Jalen Brunson", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
    {"name": "OG Anunoby", "line": 15.0, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.453, "underPct": 0.547},
    {"name": "Sandro Mamukelashvili", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.404, "underPct": 0.596},
    {"name": "Jalen Suggs", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.395, "underPct": 0.605},
    {"name": "Jalen Williams", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.386, "underPct": 0.614},
    {"name": "LeBron James", "line": 17.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.38, "underPct": 0.62},
    {"name": "Ja'Kobe Walter", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.366, "underPct": 0.634},
    {"name": "Tyus Jones", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.365, "underPct": 0.635},
    {"name": "Jakob Poeltl", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.326, "underPct": 0.674},
    {"name": "Desmond Bane", "line": 22.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.276, "underPct": 0.724},
    {"name": "Paolo Banchero", "line": 20.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.274, "underPct": 0.726},
    {"name": "Tristan da Silva", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.235, "underPct": 0.765},
    {"name": "Brandon Ingram", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.228, "underPct": 0.772},
    {"name": "Devin Booker", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.206, "underPct": 0.794},
    {"name": "Ochai Agbaji", "line": 6.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.138, "underPct": 0.862},
    {"name": "Jordan Clarkson", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.124, "underPct": 0.876},
    {"name": "Jamal Shead", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.088, "underPct": 0.912},
];const prizepicksAssistsHitRates = [
    {"name": "Davion Mitchell", "line": 7.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.744, "underPct": 0.256},
    {"name": "De'Aaron Fox", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.677, "underPct": 0.323},
    {"name": "Andrew Wiggins", "line": 2.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.589, "underPct": 0.411},
    {"name": "Scottie Barnes", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.561, "underPct": 0.439},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 4.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.469, "underPct": 0.531},
    {"name": "Mikal Bridges", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.461, "underPct": 0.539},
    {"name": "OG Anunoby", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.421, "underPct": 0.579},
    {"name": "Sandro Mamukelashvili", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.398, "underPct": 0.602},
    {"name": "Grayson Allen", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.343, "underPct": 0.657},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.342, "underPct": 0.658},
    {"name": "Devin Booker", "line": 7.0, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.301, "underPct": 0.699},
    {"name": "Jalen Suggs", "line": 5.0, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.284, "underPct": 0.716},
    {"name": "Jamal Shead", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.225, "underPct": 0.775},
    {"name": "Tyler Herro", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.19, "underPct": 0.81},
];const prizepicksReboundsHitRates = [
    {"name": "Royce O'Neale", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.724, "underPct": 0.276},
    {"name": "Mark Williams", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.717, "underPct": 0.283},
    {"name": "Josh Hart", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.652, "underPct": 0.348},
    {"name": "Ja'Kobe Walter", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.613, "underPct": 0.387},
    {"name": "Davion Mitchell", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.608, "underPct": 0.392},
    {"name": "Andrew Wiggins", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.596, "underPct": 0.404},
    {"name": "Deandre Ayton", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.582, "underPct": 0.418},
    {"name": "Mitchell Robinson", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.551, "underPct": 0.449},
    {"name": "Sandro Mamukelashvili", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.549, "underPct": 0.451},
    {"name": "Scottie Barnes", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.537, "underPct": 0.463},
    {"name": "Kel'el Ware", "line": 8.5, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.533, "underPct": 0.467},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.498, "underPct": 0.502},
    {"name": "Tyler Herro", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.484, "underPct": 0.516},
    {"name": "Brandon Ingram", "line": 5.0, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.484, "underPct": 0.516},
    {"name": "Chet Holmgren", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.468, "underPct": 0.532},
    {"name": "Paolo Banchero", "line": 7.0, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.461, "underPct": 0.539},
    {"name": "Jakob Poeltl", "line": 8.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.461, "underPct": 0.539},
    {"name": "Bam Adebayo", "line": 10.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.425, "underPct": 0.575},
    {"name": "Jalen Williams", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.392, "underPct": 0.608},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.359, "underPct": 0.641},
    {"name": "LeBron James", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.34, "underPct": 0.66},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.337, "underPct": 0.663},
    {"name": "Immanuel Quickley", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.316, "underPct": 0.684},
];const prizepicksBlocksHitRates = [
    {"name": "Anthony Black", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.406, "underPct": 0.594},
    {"name": "Mark Williams", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.48, "underPct": 0.52},
    {"name": "Victor Wembanyama", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.602, "underPct": 0.398},
];const prizepicksStealsHitRates = [
    {"name": "Kel'el Ware", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.602, "underPct": 0.398},
    {"name": "Paolo Banchero", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.492, "underPct": 0.508},
    {"name": "Cason Wallace", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.642, "underPct": 0.358},
    {"name": "Devin Booker", "line": 0.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.605, "underPct": 0.395},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Victor Wembanyama", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Devin Vassell", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Mark Williams", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Royce O'Neale", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Wiggins", "line": 13.5, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Isaiah Joe", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 41.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Aaron Fox", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephon Castle", "line": 23.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Andrew Wiggins", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 12.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sandro Mamukelashvili", "line": 16.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 29.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 25.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dru Smith", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Oso Ighodaro", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Scottie Barnes", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cason Wallace", "line": 13.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 30.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ochai Agbaji", "line": 10.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Williams", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ja'Kobe Walter", "line": 11.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyler Herro", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Suggs", "line": 28.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 18.0, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 17.5, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jonathan Isaac", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 37.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 23.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "OG Anunoby", "line": 21.0, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jakob Poeltl", "line": 20.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Paolo Banchero", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 15.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luguentz Dort", "line": 12.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Grayson Allen", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
];const prizepicksPRHitRates = [
    {"name": "Victor Wembanyama", "line": 29.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Aaron Wiggins", "line": 12.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Stephon Castle", "line": 18.5, "l5": 0.8, "l10": 0.9, "l15": 0.87, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Mark Williams", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Joe", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 22.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mitchell Robinson", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Royce O'Neale", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Oso Ighodaro", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 28.0, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deandre Ayton", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 24.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyler Herro", "line": 27.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Caruso", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 23.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 28.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ochai Agbaji", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Shead", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 21.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 19.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 17.0, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyus Jones", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 29.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "OG Anunoby", "line": 19.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 19.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paolo Banchero", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ja'Kobe Walter", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 18.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 28.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Grayson Allen", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
];const prizepicksPAHitRates = [
    {"name": "Dillon Brooks", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alex Caruso", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Victor Wembanyama", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Wiggins", "line": 10.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "De'Aaron Fox", "line": 27.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cason Wallace", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephon Castle", "line": 19.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Oso Ighodaro", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Davion Mitchell", "line": 15.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 21.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Norman Powell", "line": 23.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "LeBron James", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Williams", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "OG Anunoby", "line": 17.0, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Karl-Anthony Towns", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 34.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyus Jones", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paolo Banchero", "line": 24.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 12.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ja'Kobe Walter", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Grayson Allen", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyler Herro", "line": 26.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Mikal Bridges", "line": 19.0, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Black", "line": 20.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const prizepicksRAHitRates = [
    {"name": "Victor Wembanyama", "line": 12.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 9.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mitchell Robinson", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephon Castle", "line": 9.5, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 10.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Goga Bitadze", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Williams", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Aaron Fox", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "LeBron James", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Grayson Allen", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 11.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 14.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paolo Banchero", "line": 11.0, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyler Herro", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jalen Suggs", "line": 9.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const prizepicksTurnoversHitRates = [
    {"name": "Norman Powell", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyler Kolek", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Caruso", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Jalen Williams", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogPointsHitRates = [
    {"name": "Mark Williams", "line": 9.5, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.926, "underPct": 0.074},
    {"name": "Norman Powell", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.847, "underPct": 0.153},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.812, "underPct": 0.188},
    {"name": "Isaiah Joe", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.798, "underPct": 0.202},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.754, "underPct": 0.246},
    {"name": "Kel'el Ware", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.679, "underPct": 0.321},
    {"name": "Luguentz Dort", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.676, "underPct": 0.324},
    {"name": "Royce O'Neale", "line": 8.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.661, "underPct": 0.339},
    {"name": "Bam Adebayo", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.659, "underPct": 0.341},
    {"name": "Oso Ighodaro", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.655, "underPct": 0.345},
    {"name": "Cason Wallace", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.642, "underPct": 0.358},
    {"name": "Grayson Allen", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.615, "underPct": 0.385},
    {"name": "Tyler Herro", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.614, "underPct": 0.386},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.611, "underPct": 0.389},
    {"name": "Dru Smith", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.603, "underPct": 0.397},
    {"name": "Scottie Barnes", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "Alex Caruso", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.58, "underPct": 0.42},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.578, "underPct": 0.422},
    {"name": "Davion Mitchell", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.565, "underPct": 0.435},
    {"name": "Karl-Anthony Towns", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.535, "underPct": 0.465},
    {"name": "Anthony Black", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.528, "underPct": 0.472},
    {"name": "Jalen Brunson", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.485, "underPct": 0.515},
    {"name": "Sandro Mamukelashvili", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.404, "underPct": 0.596},
    {"name": "Jalen Suggs", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.395, "underPct": 0.605},
    {"name": "Jalen Williams", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.386, "underPct": 0.614},
    {"name": "Paolo Banchero", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.274, "underPct": 0.726},
    {"name": "Brandon Ingram", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.228, "underPct": 0.772},
    {"name": "Devin Booker", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.206, "underPct": 0.794},
];const underdogAssistsHitRates = [
    {"name": "Alex Caruso", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.7, "underPct": 0.3},
    {"name": "OG Anunoby", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.421, "underPct": 0.579},
    {"name": "Sandro Mamukelashvili", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.398, "underPct": 0.602},
    {"name": "Jamal Shead", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.225, "underPct": 0.775},
];const underdogReboundsHitRates = [
    {"name": "Royce O'Neale", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.724, "underPct": 0.276},
    {"name": "Andrew Wiggins", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.596, "underPct": 0.404},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.337, "underPct": 0.663},
    {"name": "Immanuel Quickley", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.316, "underPct": 0.684},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
    {"name": "OG Anunoby", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.522, "underPct": 0.478},
    {"name": "Cason Wallace", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.642, "underPct": 0.358},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Shai Gilgeous-Alexander", "line": 41.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Joe", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Wiggins", "line": 13.5, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Royce O'Neale", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dru Smith", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Oso Ighodaro", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Black", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ochai Agbaji", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ja'Kobe Walter", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paolo Banchero", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 17.5, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyler Herro", "line": 30.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Suggs", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jakob Poeltl", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Grayson Allen", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
];const underdogPRHitRates = [
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyler Herro", "line": 27.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Norman Powell", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Williams", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "OG Anunoby", "line": 19.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Paolo Banchero", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 28.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
];const underdogPAHitRates = [
    {"name": "Dillon Brooks", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Norman Powell", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paolo Banchero", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyler Herro", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Scottie Barnes", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 34.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Williams", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Black", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
];const underdogRAHitRates = [
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Devin Booker", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyler Herro", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const underdogTurnoversHitRates = [
    {"name": "Karl-Anthony Towns", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
];const underdogBlocksStealsHitRates = [
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
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

