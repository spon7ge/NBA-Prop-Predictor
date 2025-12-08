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
    {"name1": "Maxime Raynaud", "name2": "Dylan Harper", "line1": 12.0, "line2": 12.5, "odds1": -137, "odds2": 102, "prediction1": 4.63, "prediction2": 16.96, "modelProb1": 0.882, "modelProb2": 0.807, "side1": "under", "side2": "over", "parlayProb": 0.713, "parlayOdds": 249, "evPercent": 148.68, "kellyQuarter": 0.1493},
    {"name1": "Pascal Siakam", "name2": "De'Aaron Fox", "line1": 24.5, "line2": 24.5, "odds1": -104, "odds2": 108, "prediction1": 27.85, "prediction2": 29.46, "modelProb1": 0.702, "modelProb2": 0.778, "side1": "over", "side2": "over", "parlayProb": 0.546, "parlayOdds": 308, "evPercent": 122.84, "kellyQuarter": 0.0997},
    {"name1": "Collin Gillespie", "name2": "Stephon Castle", "line1": 17.0, "line2": 14.5, "odds1": -137, "odds2": 120, "prediction1": 9.73, "prediction2": 16.48, "modelProb1": 0.757, "modelProb2": 0.656, "side1": "under", "side2": "over", "parlayProb": 0.497, "parlayOdds": 281, "evPercent": 89.36, "kellyQuarter": 0.0795},
    {"name1": "Grayson Allen", "name2": "Saddiq Bey", "line1": 18.5, "line2": 16.5, "odds1": -104, "odds2": -107, "prediction1": 14.24, "prediction2": 12.29, "modelProb1": 0.665, "modelProb2": 0.675, "side1": "under", "side2": "under", "parlayProb": 0.449, "parlayOdds": 279, "evPercent": 70.05, "kellyQuarter": 0.0628},
    {"name1": "Ryan Dunn", "name2": "Devin Vassell", "line1": 6.5, "line2": 15.5, "odds1": -110, "odds2": -115, "prediction1": 7.94, "prediction2": 18.59, "modelProb1": 0.672, "modelProb2": 0.697, "side1": "over", "side2": "over", "parlayProb": 0.468, "parlayOdds": 257, "evPercent": 67.2, "kellyQuarter": 0.0654},
    {"name1": "Malik Monk", "name2": "Jordan Hawkins", "line1": 13.5, "line2": 6.5, "odds1": 102, "odds2": -108, "prediction1": 15.0, "prediction2": 8.14, "modelProb1": 0.621, "modelProb2": 0.66, "side1": "over", "side2": "over", "parlayProb": 0.41, "parlayOdds": 289, "evPercent": 59.57, "kellyQuarter": 0.0515},
    {"name1": "Dillon Brooks", "name2": "Keldon Johnson", "line1": 22.5, "line2": 12.5, "odds1": -108, "odds2": -105, "prediction1": 24.68, "prediction2": 13.7, "modelProb1": 0.648, "modelProb2": 0.62, "side1": "over", "side2": "over", "parlayProb": 0.402, "parlayOdds": 276, "evPercent": 51.26, "kellyQuarter": 0.0464},
    {"name1": "Julius Randle", "name2": "Derik Queen", "line1": 22.5, "line2": 13.5, "odds1": -109, "odds2": -110, "prediction1": 19.15, "prediction2": 10.32, "modelProb1": 0.618, "modelProb2": 0.623, "side1": "under", "side2": "under", "parlayProb": 0.385, "parlayOdds": 266, "evPercent": 41.02, "kellyQuarter": 0.0385},
    {"name1": "Rudy Gobert", "name2": "Jeremy Sochan", "line1": 10.5, "line2": 6.5, "odds1": -103, "odds2": -110, "prediction1": 7.92, "prediction2": 7.42, "modelProb1": 0.594, "modelProb2": 0.608, "side1": "under", "side2": "over", "parlayProb": 0.361, "parlayOdds": 276, "evPercent": 35.73, "kellyQuarter": 0.0324},
    {"name1": "Andrew Nembhard", "name2": "Trey Murphy III", "line1": 16.5, "line2": 21.5, "odds1": -103, "odds2": -105, "prediction1": 16.78, "prediction2": 19.27, "modelProb1": 0.554, "modelProb2": 0.54, "side1": "over", "side2": "under", "parlayProb": 0.299, "parlayOdds": 285, "evPercent": 15.11, "kellyQuarter": 0.0133},
];const prizepicksTriosData = [
    {"name1": "Maxime Raynaud", "name2": "Collin Gillespie", "name3": "Dylan Harper", "line1": 12.0, "line2": 17.0, "line3": 12.5, "odds1": -137, "odds2": -137, "odds3": 102, "prediction1": 4.63, "prediction2": 9.73, "prediction3": 16.96, "modelProb1": 0.882, "modelProb2": 0.757, "modelProb3": 0.807, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.54, "parlayOdds": 505, "evPercent": 226.44, "kellyQuarter": 0.1121},
    {"name1": "Pascal Siakam", "name2": "Grayson Allen", "name3": "De'Aaron Fox", "line1": 24.5, "line2": 18.5, "line3": 24.5, "odds1": -104, "odds2": -104, "odds3": 108, "prediction1": 27.85, "prediction2": 14.24, "prediction3": 29.46, "modelProb1": 0.702, "modelProb2": 0.665, "modelProb3": 0.778, "side1": "over", "side2": "under", "side3": "over", "parlayProb": 0.363, "parlayOdds": 700, "evPercent": 190.59, "kellyQuarter": 0.0681},
    {"name1": "Malik Monk", "name2": "Ryan Dunn", "name3": "Stephon Castle", "line1": 13.5, "line2": 6.5, "line3": 14.5, "odds1": 102, "odds2": -110, "odds3": 120, "prediction1": 15.0, "prediction2": 7.94, "prediction3": 16.48, "modelProb1": 0.621, "modelProb2": 0.672, "modelProb3": 0.656, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.274, "parlayOdds": 748, "evPercent": 132.43, "kellyQuarter": 0.0443},
    {"name1": "Andrew Nembhard", "name2": "Dillon Brooks", "name3": "Saddiq Bey", "line1": 16.5, "line2": 22.5, "line3": 16.5, "odds1": -103, "odds2": -108, "odds3": -107, "prediction1": 16.78, "prediction2": 24.68, "prediction3": 12.29, "modelProb1": 0.554, "modelProb2": 0.648, "modelProb3": 0.675, "side1": "over", "side2": "over", "side3": "under", "parlayProb": 0.242, "parlayOdds": 634, "evPercent": 77.82, "kellyQuarter": 0.0307},
    {"name1": "Zach LaVine", "name2": "Julius Randle", "name3": "Devin Vassell", "line1": 21.5, "line2": 22.5, "line3": 15.5, "odds1": -120, "odds2": -109, "odds3": -115, "prediction1": 18.57, "prediction2": 19.15, "prediction3": 18.59, "modelProb1": 0.58, "modelProb2": 0.618, "modelProb3": 0.697, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.25, "parlayOdds": 557, "evPercent": 64.06, "kellyQuarter": 0.0288},
    {"name1": "Bennedict Mathurin", "name2": "Rudy Gobert", "name3": "Jordan Hawkins", "line1": 21.5, "line2": 10.5, "line3": 6.5, "odds1": -102, "odds2": -103, "odds3": -108, "prediction1": 19.51, "prediction2": 7.92, "prediction3": 8.14, "modelProb1": 0.527, "modelProb2": 0.594, "modelProb3": 0.66, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.207, "parlayOdds": 652, "evPercent": 55.52, "kellyQuarter": 0.0213},
    {"name1": "DeMar DeRozan", "name2": "Oso Ighodaro", "name3": "Keldon Johnson", "line1": 18.5, "line2": 4.5, "line3": 12.5, "odds1": 105, "odds2": 100, "odds3": -105, "prediction1": 17.13, "prediction2": 2.82, "prediction3": 13.7, "modelProb1": 0.492, "modelProb2": 0.525, "modelProb3": 0.62, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.16, "parlayOdds": 700, "evPercent": 28.15, "kellyQuarter": 0.0101},
    {"name1": "Russell Westbrook", "name2": "Donte DiVincenzo", "name3": "Derik Queen", "line1": 13.5, "line2": 14.5, "line3": 13.5, "odds1": -117, "odds2": -120, "odds3": -110, "prediction1": 11.57, "prediction2": 12.29, "prediction3": 10.32, "modelProb1": 0.53, "modelProb2": 0.543, "modelProb3": 0.623, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.18, "parlayOdds": 549, "evPercent": 16.61, "kellyQuarter": 0.0076},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Pascal Siakam", "name2": "Grayson Allen", "line1": 24.5, "line2": 18.5, "odds1": -104, "odds2": -104, "prediction1": 27.85, "prediction2": 14.24, "modelProb1": 0.702, "modelProb2": 0.665, "side1": "over", "side2": "under", "parlayProb": 0.467, "parlayOdds": 285, "evPercent": 79.65, "kellyQuarter": 0.0699},
    {"name1": "Ryan Dunn", "name2": "Saddiq Bey", "line1": 6.5, "line2": 16.5, "odds1": -110, "odds2": -107, "prediction1": 7.94, "prediction2": 12.29, "modelProb1": 0.672, "modelProb2": 0.675, "side1": "over", "side2": "under", "parlayProb": 0.454, "parlayOdds": 269, "evPercent": 67.39, "kellyQuarter": 0.0626},
    {"name1": "Malik Monk", "name2": "Jordan Hawkins", "line1": 13.5, "line2": 6.5, "odds1": 102, "odds2": -108, "prediction1": 15.0, "prediction2": 8.14, "modelProb1": 0.621, "modelProb2": 0.66, "side1": "over", "side2": "over", "parlayProb": 0.41, "parlayOdds": 289, "evPercent": 59.57, "kellyQuarter": 0.0515},
    {"name1": "Dillon Brooks", "name2": "Derik Queen", "line1": 22.5, "line2": 13.5, "odds1": -108, "odds2": -110, "prediction1": 24.68, "prediction2": 10.32, "modelProb1": 0.648, "modelProb2": 0.623, "side1": "over", "side2": "under", "parlayProb": 0.404, "parlayOdds": 268, "evPercent": 48.78, "kellyQuarter": 0.0455},
    {"name1": "Julius Randle", "name2": "Jeremy Sochan", "line1": 22.5, "line2": 6.5, "odds1": -109, "odds2": -110, "prediction1": 19.15, "prediction2": 7.42, "modelProb1": 0.618, "modelProb2": 0.608, "side1": "under", "side2": "over", "parlayProb": 0.376, "parlayOdds": 266, "evPercent": 37.44, "kellyQuarter": 0.0352},
    {"name1": "Andrew Nembhard", "name2": "Trey Murphy III", "line1": 16.5, "line2": 21.5, "odds1": -103, "odds2": -105, "prediction1": 16.78, "prediction2": 19.27, "modelProb1": 0.554, "modelProb2": 0.54, "side1": "over", "side2": "under", "parlayProb": 0.299, "parlayOdds": 285, "evPercent": 15.11, "kellyQuarter": 0.0133},
    {"name1": "Zach LaVine", "name2": "Oso Ighodaro", "line1": 21.5, "line2": 4.5, "odds1": -120, "odds2": 100, "prediction1": 18.57, "prediction2": 2.82, "modelProb1": 0.58, "modelProb2": 0.525, "side1": "under", "side2": "under", "parlayProb": 0.304, "parlayOdds": 267, "evPercent": 11.74, "kellyQuarter": 0.011},
    {"name1": "Bennedict Mathurin", "name2": "Jaden McDaniels", "line1": 21.5, "line2": 15.5, "odds1": -102, "odds2": -107, "prediction1": 19.51, "prediction2": 13.99, "modelProb1": 0.527, "modelProb2": 0.501, "side1": "under", "side2": "under", "parlayProb": 0.264, "parlayOdds": 283, "evPercent": 1.16, "kellyQuarter": 0.001},
    {"name1": "Russell Westbrook", "name2": "Donte DiVincenzo", "line1": 12.5, "line2": 13.5, "odds1": 112, "odds2": 100, "prediction1": 11.57, "prediction2": 12.29, "modelProb1": 0.458, "modelProb2": 0.482, "side1": "under", "side2": "under", "parlayProb": 0.221, "parlayOdds": 324, "evPercent": -6.45, "kellyQuarter": 0.0},
    {"name1": "Anthony Edwards", "name2": "Jeremiah Fears", "line1": 28.5, "line2": 16.5, "odds1": -105, "odds2": -107, "prediction1": 27.34, "prediction2": 15.5, "modelProb1": 0.483, "modelProb2": 0.467, "side1": "under", "side2": "under", "parlayProb": 0.226, "parlayOdds": 278, "evPercent": -14.62, "kellyQuarter": 0.0},
];const underdogTriosData = [
    {"name1": "Pascal Siakam", "name2": "Grayson Allen", "name3": "Saddiq Bey", "line1": 24.5, "line2": 18.5, "line3": 16.5, "odds1": -104, "odds2": -104, "odds3": -107, "prediction1": 27.85, "prediction2": 14.24, "prediction3": 12.29, "modelProb1": 0.702, "modelProb2": 0.665, "modelProb3": 0.675, "side1": "over", "side2": "under", "side3": "under", "parlayProb": 0.315, "parlayOdds": 644, "evPercent": 134.22, "kellyQuarter": 0.0521},
    {"name1": "Malik Monk", "name2": "Ryan Dunn", "name3": "Jordan Hawkins", "line1": 13.5, "line2": 6.5, "line3": 6.5, "odds1": 102, "odds2": -110, "odds3": -108, "prediction1": 15.0, "prediction2": 7.94, "prediction3": 8.14, "modelProb1": 0.621, "modelProb2": 0.672, "modelProb3": 0.66, "side1": "over", "side2": "over", "side3": "over", "parlayProb": 0.276, "parlayOdds": 643, "evPercent": 104.93, "kellyQuarter": 0.0408},
    {"name1": "Andrew Nembhard", "name2": "Dillon Brooks", "name3": "Derik Queen", "line1": 16.5, "line2": 22.5, "line3": 13.5, "odds1": -103, "odds2": -108, "odds3": -110, "prediction1": 16.78, "prediction2": 24.68, "prediction3": 10.32, "modelProb1": 0.554, "modelProb2": 0.648, "modelProb3": 0.623, "side1": "over", "side2": "over", "side3": "under", "parlayProb": 0.224, "parlayOdds": 625, "evPercent": 62.32, "kellyQuarter": 0.0249},
    {"name1": "Zach LaVine", "name2": "Julius Randle", "name3": "Jeremy Sochan", "line1": 21.5, "line2": 22.5, "line3": 6.5, "odds1": -120, "odds2": -109, "odds3": -110, "prediction1": 18.57, "prediction2": 19.15, "prediction3": 7.42, "modelProb1": 0.58, "modelProb2": 0.618, "modelProb3": 0.608, "side1": "under", "side2": "under", "side3": "over", "parlayProb": 0.218, "parlayOdds": 571, "evPercent": 46.18, "kellyQuarter": 0.0202},
    {"name1": "Bennedict Mathurin", "name2": "Oso Ighodaro", "name3": "Trey Murphy III", "line1": 21.5, "line2": 4.5, "line3": 21.5, "odds1": -102, "odds2": 100, "odds3": -105, "prediction1": 19.51, "prediction2": 2.82, "prediction3": 19.27, "modelProb1": 0.527, "modelProb2": 0.525, "modelProb3": 0.54, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.149, "parlayOdds": 673, "evPercent": 15.46, "kellyQuarter": 0.0057},
    {"name1": "Russell Westbrook", "name2": "Jaden McDaniels", "name3": "Jeremiah Fears", "line1": 12.5, "line2": 15.5, "line3": 16.5, "odds1": 112, "odds2": -107, "odds3": -107, "prediction1": 11.57, "prediction2": 13.99, "prediction3": 15.5, "modelProb1": 0.458, "modelProb2": 0.501, "modelProb3": 0.467, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.107, "parlayOdds": 693, "evPercent": -15.01, "kellyQuarter": 0.0},
    {"name1": "Precious Achiuwa", "name2": "Donte DiVincenzo", "name3": "Harrison Barnes", "line1": 6.5, "line2": 13.5, "line3": 12.5, "odds1": -130, "odds2": 100, "odds3": -113, "prediction1": 6.2, "prediction2": 12.29, "prediction3": 11.53, "modelProb1": 0.395, "modelProb2": 0.482, "modelProb3": 0.464, "side1": "under", "side2": "under", "side3": "under", "parlayProb": 0.088, "parlayOdds": 567, "evPercent": -41.15, "kellyQuarter": 0.0},
];const prizepicksPointsHitRates = [
    {"name": "Devin Vassell", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.902, "underPct": 0.098},
    {"name": "Jose Alvarado", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.833, "underPct": 0.167},
    {"name": "Mark Williams", "line": 12.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.758, "underPct": 0.242},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.68, "underPct": 0.32},
    {"name": "Saddiq Bey", "line": 16.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.666, "underPct": 0.334},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.8, "l10": 0.9, "l15": 0.6, "overPct": 0.655, "underPct": 0.345},
    {"name": "Stephon Castle", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.56, "underPct": 0.44},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.559, "underPct": 0.441},
    {"name": "Royce O'Neale", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.557, "underPct": 0.443},
    {"name": "Anthony Edwards", "line": 29.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.554, "underPct": 0.446},
    {"name": "Collin Gillespie", "line": 17.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.545, "underPct": 0.455},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.536, "underPct": 0.464},
    {"name": "Keldon Johnson", "line": 12.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.529, "underPct": 0.471},
    {"name": "Jordan Hawkins", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.519, "underPct": 0.481},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.508, "underPct": 0.492},
    {"name": "Pascal Siakam", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.494, "underPct": 0.506},
    {"name": "Grayson Allen", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.483, "underPct": 0.517},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Ryan Dunn", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.46, "underPct": 0.54},
    {"name": "Oso Ighodaro", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.447, "underPct": 0.553},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.437, "underPct": 0.563},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.426, "underPct": 0.574},
    {"name": "Donte DiVincenzo", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.404, "underPct": 0.596},
    {"name": "Dylan Harper", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.36, "underPct": 0.64},
    {"name": "Jeremy Sochan", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.357, "underPct": 0.643},
    {"name": "Naz Reid", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.355, "underPct": 0.645},
    {"name": "Jaden McDaniels", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.354, "underPct": 0.646},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.311, "underPct": 0.689},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.298, "underPct": 0.702},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.297, "underPct": 0.703},
    {"name": "Luke Kornet", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.23, "underPct": 0.77},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.216, "underPct": 0.784},
    {"name": "Maxime Raynaud", "line": 12.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.198, "underPct": 0.802},
];const prizepicksAssistsHitRates = [
    {"name": "Stephon Castle", "line": 5.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.902, "underPct": 0.098},
    {"name": "Russell Westbrook", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.613, "underPct": 0.387},
    {"name": "Trey Murphy III", "line": 3.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.599, "underPct": 0.401},
    {"name": "Keldon Johnson", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.543, "underPct": 0.457},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Jaden McDaniels", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.509, "underPct": 0.491},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.497, "underPct": 0.503},
    {"name": "Derik Queen", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Isaiah Jackson", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.461, "underPct": 0.539},
    {"name": "De'Aaron Fox", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.444, "underPct": 0.556},
    {"name": "Bennedict Mathurin", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.443, "underPct": 0.557},
    {"name": "Anthony Edwards", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.433, "underPct": 0.567},
    {"name": "T.J. McConnell", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.373, "underPct": 0.627},
    {"name": "Dylan Harper", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.369, "underPct": 0.631},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.316, "underPct": 0.684},
    {"name": "Andrew Nembhard", "line": 7.0, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.311, "underPct": 0.689},
    {"name": "Collin Gillespie", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.282, "underPct": 0.718},
];const prizepicksReboundsHitRates = [
    {"name": "Jarace Walker", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.72, "underPct": 0.28},
    {"name": "Stephon Castle", "line": 4.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.65, "underPct": 0.35},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.649, "underPct": 0.351},
    {"name": "Saddiq Bey", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.628, "underPct": 0.372},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.565, "underPct": 0.435},
    {"name": "Mark Williams", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.562, "underPct": 0.438},
    {"name": "Pascal Siakam", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.546, "underPct": 0.454},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.51, "underPct": 0.49},
    {"name": "Jeremiah Fears", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.507, "underPct": 0.493},
    {"name": "Yves Missi", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.45, "underPct": 0.55},
    {"name": "Royce O'Neale", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.406, "underPct": 0.594},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.387, "underPct": 0.613},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.386, "underPct": 0.614},
    {"name": "Ryan Dunn", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.386, "underPct": 0.614},
    {"name": "Harrison Barnes", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.382, "underPct": 0.618},
    {"name": "Bennedict Mathurin", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.367, "underPct": 0.633},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.363, "underPct": 0.637},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.362, "underPct": 0.638},
    {"name": "Isaiah Jackson", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.357, "underPct": 0.643},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.357, "underPct": 0.643},
    {"name": "Naz Reid", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.332, "underPct": 0.668},
    {"name": "Jeremy Sochan", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.327, "underPct": 0.673},
    {"name": "Oso Ighodaro", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.312, "underPct": 0.688},
    {"name": "Jay Huff", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.302, "underPct": 0.698},
    {"name": "Luke Kornet", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.246, "underPct": 0.754},
    {"name": "Derik Queen", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.193, "underPct": 0.807},
    {"name": "Maxime Raynaud", "line": 7.0, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.099, "underPct": 0.901},
];const prizepicksBlocksHitRates = [
    {"name": "Maxime Raynaud", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.497, "underPct": 0.503},
    {"name": "Oso Ighodaro", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.379, "underPct": 0.621},
];const prizepicksStealsHitRates = [
    {"name": "Isaiah Jackson", "line": 0.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.598, "underPct": 0.402},
    {"name": "Rudy Gobert", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.272, "underPct": 0.728},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Julian Champagnie", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keldon Johnson", "line": 20.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jay Huff", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Stephon Castle", "line": 24.0, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Saddiq Bey", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Royce O'Neale", "line": 18.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Hawkins", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 14.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Dunn", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Micah Peavy", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Gillespie", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Maxime Raynaud", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 23.0, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "T.J. McConnell", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremy Sochan", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dylan Harper", "line": 19.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Oso Ighodaro", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Grayson Allen", "line": 25.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bennedict Mathurin", "line": 30.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Jackson", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
];const prizepicksPRHitRates = [
    {"name": "Pascal Siakam", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 23.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephon Castle", "line": 18.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Devin Vassell", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Maxime Raynaud", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden McDaniels", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Hawkins", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julian Champagnie", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dylan Harper", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 19.0, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jay Huff", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Oso Ighodaro", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 22.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarace Walker", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 26.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Micah Peavy", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Grayson Allen", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rudy Gobert", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Royce O'Neale", "line": 15.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremy Sochan", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Harrison Barnes", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Jackson", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
];const prizepicksPAHitRates = [
    {"name": "Keldon Johnson", "line": 13.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Saddiq Bey", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jose Alvarado", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jay Huff", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Yves Missi", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dylan Harper", "line": 16.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephon Castle", "line": 19.0, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Royce O'Neale", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Micah Peavy", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 29.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jarace Walker", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Grayson Allen", "line": 22.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Malik Monk", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bennedict Mathurin", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 22.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 28.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
];const prizepicksRAHitRates = [
    {"name": "Stephon Castle", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Harrison Barnes", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 9.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jose Alvarado", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mark Williams", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Saddiq Bey", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Micah Peavy", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 13.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 6.0, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Royce O'Neale", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Andrew Nembhard", "line": 10.0, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.0, "underPct": 1.0},
];const prizepicksTurnoversHitRates = [
    {"name": "Jose Alvarado", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Hawkins", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 0.5, "l5": 0.6, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Donte DiVincenzo", "line": 1.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Ryan Dunn", "line": 1.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Jackson", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Gillespie", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
];const underdogPointsHitRates = [
    {"name": "Jose Alvarado", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.833, "underPct": 0.167},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.707, "underPct": 0.293},
    {"name": "Saddiq Bey", "line": 16.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.666, "underPct": 0.334},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.626, "underPct": 0.374},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.559, "underPct": 0.441},
    {"name": "Royce O'Neale", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.557, "underPct": 0.443},
    {"name": "Russell Westbrook", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.547, "underPct": 0.453},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.536, "underPct": 0.464},
    {"name": "Jordan Hawkins", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.519, "underPct": 0.481},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.51, "underPct": 0.49},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.508, "underPct": 0.492},
    {"name": "Pascal Siakam", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.494, "underPct": 0.506},
    {"name": "Grayson Allen", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.483, "underPct": 0.517},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Ryan Dunn", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.46, "underPct": 0.54},
    {"name": "Oso Ighodaro", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.447, "underPct": 0.553},
    {"name": "Jeremy Sochan", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.357, "underPct": 0.643},
    {"name": "Naz Reid", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.355, "underPct": 0.645},
    {"name": "Jaden McDaniels", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.354, "underPct": 0.646},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.311, "underPct": 0.689},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.298, "underPct": 0.702},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.297, "underPct": 0.703},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.216, "underPct": 0.784},
];const underdogAssistsHitRates = [
    {"name": "Naz Reid", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.554, "underPct": 0.446},
    {"name": "Keldon Johnson", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.543, "underPct": 0.457},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Bennedict Mathurin", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.443, "underPct": 0.557},
    {"name": "Dylan Harper", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.369, "underPct": 0.631},
    {"name": "Grayson Allen", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.345, "underPct": 0.655},
    {"name": "Malik Monk", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.31, "underPct": 0.69},
    {"name": "Collin Gillespie", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.282, "underPct": 0.718},
];const underdogReboundsHitRates = [
    {"name": "Stephon Castle", "line": 4.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.65, "underPct": 0.35},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.649, "underPct": 0.351},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.595, "underPct": 0.405},
    {"name": "Harrison Barnes", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.577, "underPct": 0.423},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.51, "underPct": 0.49},
    {"name": "Jeremiah Fears", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.507, "underPct": 0.493},
    {"name": "Ryan Dunn", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.386, "underPct": 0.614},
    {"name": "Bennedict Mathurin", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.367, "underPct": 0.633},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.363, "underPct": 0.637},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.359, "underPct": 0.641},
    {"name": "Isaiah Jackson", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.357, "underPct": 0.643},
    {"name": "Jeremy Sochan", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.327, "underPct": 0.673},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
    {"name": "Jalen Williams", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.263, "underPct": 0.737},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Saddiq Bey", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jay Huff", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Precious Achiuwa", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keldon Johnson", "line": 20.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Hawkins", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Oso Ighodaro", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Grayson Allen", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden McDaniels", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dylan Harper", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 30.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Nembhard", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogPRHitRates = [
    {"name": "Saddiq Bey", "line": 22.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Pascal Siakam", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephon Castle", "line": 18.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Devin Vassell", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Grayson Allen", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Russell Westbrook", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogPAHitRates = [
    {"name": "Stephon Castle", "line": 18.5, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Grayson Allen", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 28.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogRAHitRates = [
    {"name": "Maxime Raynaud", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Saddiq Bey", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "T.J. McConnell", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
];const underdogTurnoversHitRates = [
    {"name": "Draymond Green", "line": 2.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Coby White", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Miller", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
];const underdogBlocksStealsHitRates = [
    {"name": "Peyton Watson", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
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

