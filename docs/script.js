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
    {"name1": "Jaden McDaniels", "name2": "Jaime Jaquez Jr.", "line1": 12.5, "line2": 13.5, "prediction1": 19.39, "prediction2": 22.29, "side1": "over", "side2": "over", "edge1": 6.89, "edge2": 8.79, "impliedProb1": 0.53, "impliedProb2": 0.52},
    {"name1": "Marvin Bagley III", "name2": "Tyler Herro", "line1": 13.5, "line2": 23.5, "prediction1": 7.07, "prediction2": 30.0, "side1": "under", "side2": "over", "edge1": 6.43, "edge2": 6.5, "impliedProb1": 0.52, "impliedProb2": 0.51},
    {"name1": "Danny Wolf", "name2": "Kevin Porter Jr.", "line1": 11.5, "line2": 20.5, "prediction1": 5.5, "prediction2": 14.42, "side1": "under", "side2": "under", "edge1": 6.0, "edge2": 6.08, "impliedProb1": 0.52, "impliedProb2": 0.5},
    {"name1": "Noah Clowney", "name2": "Julius Randle", "line1": 16.5, "line2": 20.5, "prediction1": 10.74, "prediction2": 26.11, "side1": "under", "side2": "over", "edge1": 5.76, "edge2": 5.61, "impliedProb1": 0.51, "impliedProb2": 0.51},
    {"name1": "Maxime Raynaud", "name2": "Aaron Holiday", "line1": 10.5, "line2": 9.5, "prediction1": 5.19, "prediction2": 4.21, "side1": "under", "side2": "under", "edge1": 5.31, "edge2": 5.29, "impliedProb1": 0.5, "impliedProb2": 0.51},
    {"name1": "Nickeil Alexander-Walker", "name2": "Ryan Nembhard", "line1": 20.5, "line2": 9.5, "prediction1": 25.3, "prediction2": 4.54, "side1": "over", "side2": "under", "edge1": 4.8, "edge2": 4.96, "impliedProb1": 0.53, "impliedProb2": 0.56},
    {"name1": "Kyshawn George", "name2": "Jalen Duren", "line1": 15.5, "line2": 18.5, "prediction1": 10.77, "prediction2": 13.9, "side1": "under", "side2": "under", "edge1": 4.73, "edge2": 4.6, "impliedProb1": 0.53, "impliedProb2": 0.49},
    {"name1": "Trey Murphy III", "name2": "Ryan Rollins", "line1": 20.5, "line2": 20.5, "prediction1": 25.03, "prediction2": 15.91, "side1": "over", "side2": "under", "edge1": 4.53, "edge2": 4.59, "impliedProb1": 0.56, "impliedProb2": 0.5},
    {"name1": "Saddiq Bey", "name2": "Jimmy Butler III", "line1": 17.5, "line2": 20.5, "prediction1": 13.26, "prediction2": 16.22, "side1": "under", "side2": "under", "edge1": 4.24, "edge2": 4.28, "impliedProb1": 0.53, "impliedProb2": 0.46},
    {"name1": "Andrew Wiggins", "name2": "Kevin Durant", "line1": 16.5, "line2": 25.5, "prediction1": 20.63, "prediction2": 21.32, "side1": "over", "side2": "under", "edge1": 4.13, "edge2": 4.18, "impliedProb1": 0.5, "impliedProb2": 0.51},
];const prizepicksTriosData = [
    {"name1": "Marvin Bagley III", "name2": "Jaden McDaniels", "name3": "Jaime Jaquez Jr.", "line1": 13.5, "line2": 12.5, "line3": 13.5, "prediction1": 7.07, "prediction2": 19.39, "prediction3": 22.29, "side1": "under", "side2": "over", "side3": "over", "edge1": 6.43, "edge2": 6.89, "edge3": 8.79, "impliedProb1": 0.52, "impliedProb2": 0.53, "impliedProb3": 0.52},
    {"name1": "Danny Wolf", "name2": "Kevin Porter Jr.", "name3": "Jaime Jaquez Jr.", "line1": 11.5, "line2": 20.5, "line3": 13.5, "prediction1": 5.5, "prediction2": 14.42, "prediction3": 22.29, "side1": "under", "side2": "under", "side3": "over", "edge1": 6.0, "edge2": 6.08, "edge3": 8.79, "impliedProb1": 0.52, "impliedProb2": 0.5, "impliedProb3": 0.52},
    {"name1": "Noah Clowney", "name2": "Julius Randle", "name3": "Jaime Jaquez Jr.", "line1": 16.5, "line2": 20.5, "line3": 13.5, "prediction1": 10.74, "prediction2": 26.11, "prediction3": 22.29, "side1": "under", "side2": "over", "side3": "over", "edge1": 5.76, "edge2": 5.61, "edge3": 8.79, "impliedProb1": 0.51, "impliedProb2": 0.51, "impliedProb3": 0.52},
    {"name1": "Nickeil Alexander-Walker", "name2": "Jaime Jaquez Jr.", "name3": "Aaron Holiday", "line1": 20.5, "line2": 13.5, "line3": 9.5, "prediction1": 25.3, "prediction2": 22.29, "prediction3": 4.21, "side1": "over", "side2": "over", "side3": "under", "edge1": 4.8, "edge2": 8.79, "edge3": 5.29, "impliedProb1": 0.53, "impliedProb2": 0.52, "impliedProb3": 0.51},
    {"name1": "Kyshawn George", "name2": "Jalen Duren", "name3": "Jaime Jaquez Jr.", "line1": 15.5, "line2": 18.5, "line3": 13.5, "prediction1": 10.77, "prediction2": 13.9, "prediction3": 22.29, "side1": "under", "side2": "under", "side3": "over", "edge1": 4.73, "edge2": 4.6, "edge3": 8.79, "impliedProb1": 0.53, "impliedProb2": 0.49, "impliedProb3": 0.52},
    {"name1": "Trey Murphy III", "name2": "Ryan Rollins", "name3": "Jaime Jaquez Jr.", "line1": 20.5, "line2": 20.5, "line3": 13.5, "prediction1": 25.03, "prediction2": 15.91, "prediction3": 22.29, "side1": "over", "side2": "under", "side3": "over", "edge1": 4.53, "edge2": 4.59, "edge3": 8.79, "impliedProb1": 0.56, "impliedProb2": 0.5, "impliedProb3": 0.52},
    {"name1": "Saddiq Bey", "name2": "Jimmy Butler III", "name3": "Jaime Jaquez Jr.", "line1": 17.5, "line2": 20.5, "line3": 13.5, "prediction1": 13.26, "prediction2": 16.22, "prediction3": 22.29, "side1": "under", "side2": "under", "side3": "over", "edge1": 4.24, "edge2": 4.28, "edge3": 8.79, "impliedProb1": 0.53, "impliedProb2": 0.46, "impliedProb3": 0.52},
    {"name1": "Khris Middleton", "name2": "Jonathan Kuminga", "name3": "Jaime Jaquez Jr.", "line1": 12.5, "line2": 14.5, "line3": 13.5, "prediction1": 9.06, "prediction2": 10.56, "prediction3": 22.29, "side1": "under", "side2": "under", "side3": "over", "edge1": 3.44, "edge2": 3.94, "edge3": 8.79, "impliedProb1": 0.5, "impliedProb2": 0.53, "impliedProb3": 0.52},
    {"name1": "Darius Garland", "name2": "Myles Turner", "name3": "Jaime Jaquez Jr.", "line1": 16.5, "line2": 12.5, "line3": 13.5, "prediction1": 12.62, "prediction2": 9.13, "prediction3": 22.29, "side1": "under", "side2": "under", "side3": "over", "edge1": 3.88, "edge2": 3.37, "edge3": 8.79, "impliedProb1": 0.48, "impliedProb2": 0.49, "impliedProb3": 0.52},
    {"name1": "Brandin Podziemski", "name2": "Naz Reid", "name3": "Jaime Jaquez Jr.", "line1": 13.5, "line2": 13.5, "line3": 13.5, "prediction1": 9.72, "prediction2": 16.63, "prediction3": 22.29, "side1": "under", "side2": "over", "side3": "over", "edge1": 3.78, "edge2": 3.13, "edge3": 8.79, "impliedProb1": 0.51, "impliedProb2": 0.5, "impliedProb3": 0.52},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Marvin Bagley III", "name2": "Tyler Herro", "line1": 13.5, "line2": 23.5, "prediction1": 7.07, "prediction2": 30.0, "side1": "under", "side2": "over", "edge1": 6.43, "edge2": 6.5, "impliedProb1": 0.52, "impliedProb2": 0.51},
    {"name1": "Danny Wolf", "name2": "Kevin Porter Jr.", "line1": 11.5, "line2": 20.5, "prediction1": 5.5, "prediction2": 14.42, "side1": "under", "side2": "under", "edge1": 6.0, "edge2": 6.08, "impliedProb1": 0.52, "impliedProb2": 0.5},
    {"name1": "Noah Clowney", "name2": "Jaden McDaniels", "line1": 16.5, "line2": 13.5, "prediction1": 10.74, "prediction2": 19.39, "side1": "under", "side2": "over", "edge1": 5.76, "edge2": 5.89, "impliedProb1": 0.51, "impliedProb2": 0.46},
    {"name1": "Julius Randle", "name2": "Maxime Raynaud", "line1": 20.5, "line2": 10.5, "prediction1": 26.11, "prediction2": 5.19, "side1": "over", "side2": "under", "edge1": 5.61, "edge2": 5.31, "impliedProb1": 0.51, "impliedProb2": 0.5},
    {"name1": "Gary Harris", "name2": "Aaron Holiday", "line1": 5.5, "line2": 9.5, "prediction1": 0.49, "prediction2": 4.21, "side1": "under", "side2": "under", "edge1": 5.01, "edge2": 5.29, "impliedProb1": 0.55, "impliedProb2": 0.51},
    {"name1": "Nickeil Alexander-Walker", "name2": "Ryan Nembhard", "line1": 20.5, "line2": 9.5, "prediction1": 25.3, "prediction2": 4.54, "side1": "over", "side2": "under", "edge1": 4.8, "edge2": 4.96, "impliedProb1": 0.53, "impliedProb2": 0.56},
    {"name1": "Kyshawn George", "name2": "Ryan Rollins", "line1": 15.5, "line2": 20.5, "prediction1": 10.77, "prediction2": 15.91, "side1": "under", "side2": "under", "edge1": 4.73, "edge2": 4.59, "impliedProb1": 0.53, "impliedProb2": 0.5},
    {"name1": "Saddiq Bey", "name2": "Jimmy Butler III", "line1": 17.5, "line2": 20.5, "prediction1": 13.26, "prediction2": 16.22, "side1": "under", "side2": "under", "edge1": 4.24, "edge2": 4.28, "impliedProb1": 0.53, "impliedProb2": 0.46},
    {"name1": "Andrew Wiggins", "name2": "Kevin Durant", "line1": 16.5, "line2": 25.5, "prediction1": 20.63, "prediction2": 21.32, "side1": "over", "side2": "under", "edge1": 4.13, "edge2": 4.18, "impliedProb1": 0.5, "impliedProb2": 0.51},
    {"name1": "Jonathan Kuminga", "name2": "P.J. Washington", "line1": 14.5, "line2": 13.5, "prediction1": 10.56, "prediction2": 17.49, "side1": "under", "side2": "over", "edge1": 3.94, "edge2": 3.99, "impliedProb1": 0.53, "impliedProb2": 0.5},
];const underdogTriosData = [
    {"name1": "Marvin Bagley III", "name2": "Kevin Porter Jr.", "name3": "Tyler Herro", "line1": 13.5, "line2": 20.5, "line3": 23.5, "prediction1": 7.07, "prediction2": 14.42, "prediction3": 30.0, "side1": "under", "side2": "under", "side3": "over", "edge1": 6.43, "edge2": 6.08, "edge3": 6.5, "impliedProb1": 0.52, "impliedProb2": 0.5, "impliedProb3": 0.51},
    {"name1": "Danny Wolf", "name2": "Jaden McDaniels", "name3": "Tyler Herro", "line1": 11.5, "line2": 13.5, "line3": 23.5, "prediction1": 5.5, "prediction2": 19.39, "prediction3": 30.0, "side1": "under", "side2": "over", "side3": "over", "edge1": 6.0, "edge2": 5.89, "edge3": 6.5, "impliedProb1": 0.52, "impliedProb2": 0.46, "impliedProb3": 0.51},
    {"name1": "Noah Clowney", "name2": "Julius Randle", "name3": "Tyler Herro", "line1": 16.5, "line2": 20.5, "line3": 23.5, "prediction1": 10.74, "prediction2": 26.11, "prediction3": 30.0, "side1": "under", "side2": "over", "side3": "over", "edge1": 5.76, "edge2": 5.61, "edge3": 6.5, "impliedProb1": 0.51, "impliedProb2": 0.51, "impliedProb3": 0.51},
    {"name1": "Gary Harris", "name2": "Tyler Herro", "name3": "Aaron Holiday", "line1": 5.5, "line2": 23.5, "line3": 9.5, "prediction1": 0.49, "prediction2": 30.0, "prediction3": 4.21, "side1": "under", "side2": "over", "side3": "under", "edge1": 5.01, "edge2": 6.5, "edge3": 5.29, "impliedProb1": 0.55, "impliedProb2": 0.51, "impliedProb3": 0.51},
    {"name1": "Nickeil Alexander-Walker", "name2": "Ryan Rollins", "name3": "Tyler Herro", "line1": 20.5, "line2": 20.5, "line3": 23.5, "prediction1": 25.3, "prediction2": 15.91, "prediction3": 30.0, "side1": "over", "side2": "under", "side3": "over", "edge1": 4.8, "edge2": 4.59, "edge3": 6.5, "impliedProb1": 0.53, "impliedProb2": 0.5, "impliedProb3": 0.51},
    {"name1": "Kyshawn George", "name2": "Jimmy Butler III", "name3": "Tyler Herro", "line1": 15.5, "line2": 20.5, "line3": 23.5, "prediction1": 10.77, "prediction2": 16.22, "prediction3": 30.0, "side1": "under", "side2": "under", "side3": "over", "edge1": 4.73, "edge2": 4.28, "edge3": 6.5, "impliedProb1": 0.53, "impliedProb2": 0.46, "impliedProb3": 0.51},
    {"name1": "Saddiq Bey", "name2": "Maxime Raynaud", "name3": "Aaron Holiday", "line1": 17.5, "line2": 10.5, "line3": 9.5, "prediction1": 13.26, "prediction2": 5.19, "prediction3": 4.21, "side1": "under", "side2": "under", "side3": "under", "edge1": 4.24, "edge2": 5.31, "edge3": 5.29, "impliedProb1": 0.53, "impliedProb2": 0.5, "impliedProb3": 0.51},
    {"name1": "Trey Murphy III", "name2": "Jonathan Kuminga", "name3": "Tyler Herro", "line1": 21.5, "line2": 14.5, "line3": 23.5, "prediction1": 25.03, "prediction2": 10.56, "prediction3": 30.0, "side1": "over", "side2": "under", "side3": "over", "edge1": 3.53, "edge2": 3.94, "edge3": 6.5, "impliedProb1": 0.49, "impliedProb2": 0.53, "impliedProb3": 0.51},
    {"name1": "Khris Middleton", "name2": "Darius Garland", "name3": "Tyler Herro", "line1": 12.5, "line2": 16.5, "line3": 23.5, "prediction1": 9.06, "prediction2": 12.62, "prediction3": 30.0, "side1": "under", "side2": "under", "side3": "over", "edge1": 3.44, "edge2": 3.88, "edge3": 6.5, "impliedProb1": 0.5, "impliedProb2": 0.48, "impliedProb3": 0.51},
    {"name1": "Brandin Podziemski", "name2": "Myles Turner", "name3": "Tyler Herro", "line1": 13.5, "line2": 12.5, "line3": 23.5, "prediction1": 9.72, "prediction2": 9.13, "prediction3": 30.0, "side1": "under", "side2": "under", "side3": "over", "edge1": 3.78, "edge2": 3.37, "edge3": 6.5, "impliedProb1": 0.51, "impliedProb2": 0.49, "impliedProb3": 0.51},
];const prizepicksPointsHitRates = [
    {"name": "D'Angelo Russell", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.902, "underPct": 0.098},
    {"name": "Jaylen Wells", "line": 11.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.878, "underPct": 0.122},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.843, "underPct": 0.157},
    {"name": "Paolo Banchero", "line": 16.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.833, "underPct": 0.167},
    {"name": "Max Christie", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.814, "underPct": 0.186},
    {"name": "P.J. Washington", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.804, "underPct": 0.196},
    {"name": "Aaron Holiday", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.798, "underPct": 0.202},
    {"name": "Mark Williams", "line": 11.0, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.796, "underPct": 0.204},
    {"name": "Kel'el Ware", "line": 10.0, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.795, "underPct": 0.205},
    {"name": "Josh Hart", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.792, "underPct": 0.208},
    {"name": "Reed Sheppard", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.785, "underPct": 0.215},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.784, "underPct": 0.216},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.781, "underPct": 0.219},
    {"name": "Jordan Goodwin", "line": 7.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.778, "underPct": 0.222},
    {"name": "Aaron Wiggins", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.777, "underPct": 0.223},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.776, "underPct": 0.224},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.776, "underPct": 0.224},
    {"name": "Julian Champagnie", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.754, "underPct": 0.246},
    {"name": "Dillon Brooks", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.752, "underPct": 0.248},
    {"name": "Tyrese Maxey", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.75, "underPct": 0.25},
    {"name": "Norman Powell", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.741, "underPct": 0.259},
    {"name": "OG Anunoby", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.738, "underPct": 0.262},
    {"name": "Royce O'Neale", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.737, "underPct": 0.263},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.735, "underPct": 0.265},
    {"name": "Cam Spencer", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.734, "underPct": 0.266},
    {"name": "Grayson Allen", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.734, "underPct": 0.266},
    {"name": "VJ Edgecombe", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.732, "underPct": 0.268},
    {"name": "Donovan Mitchell", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.71, "underPct": 0.29},
    {"name": "Jalen Johnson", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.709, "underPct": 0.291},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.709, "underPct": 0.291},
    {"name": "Collin Gillespie", "line": 16.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.707, "underPct": 0.293},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.705, "underPct": 0.295},
    {"name": "Keldon Johnson", "line": 12.0, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.703, "underPct": 0.297},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.702, "underPct": 0.298},
    {"name": "Nickeil Alexander-Walker", "line": 19.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.699, "underPct": 0.301},
    {"name": "Desmond Bane", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.698, "underPct": 0.302},
    {"name": "Kon Knueppel", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.693, "underPct": 0.307},
    {"name": "Tobias Harris", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.688, "underPct": 0.312},
    {"name": "Duncan Robinson", "line": 10.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.686, "underPct": 0.314},
    {"name": "Santi Aldama", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.679, "underPct": 0.321},
    {"name": "Kris Dunn", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.673, "underPct": 0.327},
    {"name": "Jared McCain", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.672, "underPct": 0.328},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.671, "underPct": 0.329},
    {"name": "Steven Adams", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.664, "underPct": 0.336},
    {"name": "Nicolas Batum", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.662, "underPct": 0.338},
    {"name": "Peyton Watson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.66, "underPct": 0.34},
    {"name": "Mouhamed Gueye", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.656, "underPct": 0.344},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.654, "underPct": 0.346},
    {"name": "Harrison Barnes", "line": 12.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Tyler Herro", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.652, "underPct": 0.348},
    {"name": "Miles McBride", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.646, "underPct": 0.354},
    {"name": "Kevin Durant", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.643, "underPct": 0.357},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.641, "underPct": 0.359},
    {"name": "Davion Mitchell", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.634, "underPct": 0.366},
    {"name": "Justin Edwards", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.632, "underPct": 0.368},
    {"name": "Ryan Nembhard", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.628, "underPct": 0.372},
    {"name": "Isaiah Joe", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.625, "underPct": 0.375},
    {"name": "Cason Wallace", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.624, "underPct": 0.376},
    {"name": "Bam Adebayo", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.624, "underPct": 0.376},
    {"name": "Anfernee Simons", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.622, "underPct": 0.378},
    {"name": "Anthony Black", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.617, "underPct": 0.383},
    {"name": "Jaylon Tyson", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.611, "underPct": 0.389},
    {"name": "Cade Cunningham", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.609, "underPct": 0.391},
    {"name": "Kyle Anderson", "line": 6.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.609, "underPct": 0.391},
    {"name": "Derrick White", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.603, "underPct": 0.397},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.601, "underPct": 0.399},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.601, "underPct": 0.399},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.597, "underPct": 0.403},
    {"name": "Goga Bitadze", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.597, "underPct": 0.403},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.59, "underPct": 0.41},
    {"name": "Jaylen Brown", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.583, "underPct": 0.417},
    {"name": "Jalen Suggs", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.581, "underPct": 0.419},
    {"name": "Ajay Mitchell", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.576, "underPct": 0.424},
    {"name": "Ivica Zubac", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.576, "underPct": 0.424},
    {"name": "Myles Turner", "line": 12.0, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.563, "underPct": 0.437},
    {"name": "Deni Avdija", "line": 25.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.562, "underPct": 0.438},
    {"name": "Isaiah Collier", "line": 4.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.559, "underPct": 0.441},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.555, "underPct": 0.445},
    {"name": "Alperen Sengun", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.553, "underPct": 0.447},
    {"name": "Sam Hauser", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.552, "underPct": 0.448},
    {"name": "Shaedon Sharpe", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.545, "underPct": 0.455},
    {"name": "Andrew Wiggins", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.545, "underPct": 0.455},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.544, "underPct": 0.456},
    {"name": "Kyle Kuzma", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.542, "underPct": 0.458},
    {"name": "De'Aaron Fox", "line": 26.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.541, "underPct": 0.459},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.54, "underPct": 0.46},
    {"name": "T.J. McConnell", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.538, "underPct": 0.462},
    {"name": "Jarace Walker", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.531, "underPct": 0.469},
    {"name": "Naji Marshall", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.525, "underPct": 0.475},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.523, "underPct": 0.477},
    {"name": "Kris Murray", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.523, "underPct": 0.477},
    {"name": "Jalen Duren", "line": 17.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.521, "underPct": 0.479},
    {"name": "Scottie Barnes", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.521, "underPct": 0.479},
    {"name": "Luke Kennard", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.518, "underPct": 0.482},
    {"name": "Toumani Camara", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.515, "underPct": 0.485},
    {"name": "Coby White", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.515, "underPct": 0.485},
    {"name": "James Harden", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.513, "underPct": 0.487},
    {"name": "Andrew Nembhard", "line": 16.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.513, "underPct": 0.487},
    {"name": "Miles Bridges", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.482, "underPct": 0.518},
    {"name": "De'Andre Hunter", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.481, "underPct": 0.519},
    {"name": "Ayo Dosunmu", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.478, "underPct": 0.522},
    {"name": "Karl-Anthony Towns", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.475, "underPct": 0.525},
    {"name": "Paul George", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.474, "underPct": 0.526},
    {"name": "Bobby Portis", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.474, "underPct": 0.526},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.473, "underPct": 0.527},
    {"name": "Ja'Kobe Walter", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.468, "underPct": 0.532},
    {"name": "Sidy Cissoko", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.463, "underPct": 0.537},
    {"name": "Keaton Wallace", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.462, "underPct": 0.538},
    {"name": "Caris LeVert", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.461, "underPct": 0.539},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.46, "underPct": 0.54},
    {"name": "Anthony Davis", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.45, "underPct": 0.55},
    {"name": "Brandon Ingram", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.435, "underPct": 0.565},
    {"name": "Oso Ighodaro", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.434, "underPct": 0.566},
    {"name": "Marcus Smart", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.43, "underPct": 0.57},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.423, "underPct": 0.577},
    {"name": "Deandre Ayton", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.414, "underPct": 0.586},
    {"name": "Dylan Harper", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.408, "underPct": 0.592},
    {"name": "Brice Sensabaugh", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Ausar Thompson", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Jaden Ivey", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.396, "underPct": 0.604},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.395, "underPct": 0.605},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.394, "underPct": 0.606},
    {"name": "Brandon Miller", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.39, "underPct": 0.61},
    {"name": "Jalen Brunson", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.388, "underPct": 0.612},
    {"name": "Lauri Markkanen", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.383, "underPct": 0.617},
    {"name": "Jay Huff", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.383, "underPct": 0.617},
    {"name": "Sandro Mamukelashvili", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.382, "underPct": 0.618},
    {"name": "Austin Reaves", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.373, "underPct": 0.627},
    {"name": "Quentin Grimes", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.371, "underPct": 0.629},
    {"name": "Pascal Siakam", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.368, "underPct": 0.632},
    {"name": "John Collins", "line": 13.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.347, "underPct": 0.653},
    {"name": "Luke Kornet", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.337, "underPct": 0.663},
    {"name": "Ryan Rollins", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.324, "underPct": 0.676},
    {"name": "Kevin Porter Jr.", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.312, "underPct": 0.688},
    {"name": "Jamal Shead", "line": 5.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.311, "underPct": 0.689},
    {"name": "Jordan Clarkson", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.304, "underPct": 0.696},
    {"name": "Josh Giddey", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.297, "underPct": 0.703},
    {"name": "Walter Clayton Jr.", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.291, "underPct": 0.709},
    {"name": "Onyeka Okongwu", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.273, "underPct": 0.727},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.272, "underPct": 0.728},
    {"name": "Cedric Coward", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.267, "underPct": 0.733},
    {"name": "Josh Okogie", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.253, "underPct": 0.747},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.244, "underPct": 0.756},
    {"name": "Dean Wade", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.194, "underPct": 0.806},
    {"name": "LaMelo Ball", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.192, "underPct": 0.808},
    {"name": "Patrick Williams", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.177, "underPct": 0.823},
    {"name": "Jalen Williams", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.147, "underPct": 0.853},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.13, "underPct": 0.87},
    {"name": "Dalton Knecht", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.124, "underPct": 0.876},
    {"name": "Andre Drummond", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.118, "underPct": 0.882},
    {"name": "Gabe Vincent", "line": 6.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.112, "underPct": 0.888},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Lonzo Ball", "line": 8.0, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.078, "underPct": 0.922},
    {"name": "Matas Buzelis", "line": 16.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.059, "underPct": 0.941},
    {"name": "Julian Phillips", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.05, "underPct": 0.95},
    {"name": "Jake LaRavia", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.034, "underPct": 0.966},
];const prizepicksAssistsHitRates = [
    {"name": "Coby White", "line": 4.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.74, "underPct": 0.26},
    {"name": "Davion Mitchell", "line": 7.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.725, "underPct": 0.275},
    {"name": "Oso Ighodaro", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.703, "underPct": 0.297},
    {"name": "Cam Spencer", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.678, "underPct": 0.322},
    {"name": "P.J. Washington", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.661, "underPct": 0.339},
    {"name": "Jaylen Wells", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.63, "underPct": 0.37},
    {"name": "Jakob Poeltl", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.612, "underPct": 0.388},
    {"name": "Jamal Murray", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.612, "underPct": 0.388},
    {"name": "Naji Marshall", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.611, "underPct": 0.389},
    {"name": "Jalen Johnson", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.608, "underPct": 0.392},
    {"name": "Jalen Suggs", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.579, "underPct": 0.421},
    {"name": "Tobias Harris", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.575, "underPct": 0.425},
    {"name": "Shaedon Sharpe", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.568, "underPct": 0.432},
    {"name": "Isaiah Collier", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.551, "underPct": 0.449},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.547, "underPct": 0.453},
    {"name": "Bobby Portis", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.529, "underPct": 0.471},
    {"name": "Jalen Williams", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.526, "underPct": 0.474},
    {"name": "Deni Avdija", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.507, "underPct": 0.493},
    {"name": "Jaylen Brown", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.486, "underPct": 0.514},
    {"name": "De'Aaron Fox", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.485, "underPct": 0.515},
    {"name": "Tyrese Maxey", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.482, "underPct": 0.518},
    {"name": "Toumani Camara", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.479, "underPct": 0.521},
    {"name": "Grayson Allen", "line": 3.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.477, "underPct": 0.523},
    {"name": "Amen Thompson", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.476, "underPct": 0.524},
    {"name": "Ryan Rollins", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.475, "underPct": 0.525},
    {"name": "Shai Gilgeous-Alexander", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.474, "underPct": 0.526},
    {"name": "Evan Mobley", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.469, "underPct": 0.531},
    {"name": "Alperen Sengun", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.462, "underPct": 0.538},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.453, "underPct": 0.547},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.449, "underPct": 0.551},
    {"name": "Keyonte George", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.448, "underPct": 0.552},
    {"name": "Lonzo Ball", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.436, "underPct": 0.564},
    {"name": "Anthony Davis", "line": 3.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.433, "underPct": 0.567},
    {"name": "Cade Cunningham", "line": 8.0, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.428, "underPct": 0.572},
    {"name": "Andrew Nembhard", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.427, "underPct": 0.573},
    {"name": "Kevin Durant", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Jarace Walker", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.42, "underPct": 0.58},
    {"name": "Franz Wagner", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.419, "underPct": 0.581},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.417, "underPct": 0.583},
    {"name": "Immanuel Quickley", "line": 7.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.401, "underPct": 0.599},
    {"name": "Aaron Holiday", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.398, "underPct": 0.602},
    {"name": "Max Christie", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.382, "underPct": 0.618},
    {"name": "T.J. McConnell", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.378, "underPct": 0.622},
    {"name": "Josh Giddey", "line": 9.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.376, "underPct": 0.624},
    {"name": "Derrick White", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.373, "underPct": 0.627},
    {"name": "Jamal Shead", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.372, "underPct": 0.628},
    {"name": "LaMelo Ball", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.371, "underPct": 0.629},
    {"name": "VJ Edgecombe", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.37, "underPct": 0.63},
    {"name": "Mikal Bridges", "line": 3.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.37, "underPct": 0.63},
    {"name": "Josh Hart", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.367, "underPct": 0.633},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.359, "underPct": 0.641},
    {"name": "Bam Adebayo", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.357, "underPct": 0.643},
    {"name": "Ayo Dosunmu", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.352, "underPct": 0.648},
    {"name": "Cedric Coward", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.346, "underPct": 0.654},
    {"name": "James Harden", "line": 9.0, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.343, "underPct": 0.657},
    {"name": "Scottie Barnes", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.336, "underPct": 0.664},
    {"name": "Jalen Brunson", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.321, "underPct": 0.679},
    {"name": "Julian Phillips", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.31, "underPct": 0.69},
    {"name": "Collin Gillespie", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.303, "underPct": 0.697},
    {"name": "Payton Pritchard", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.299, "underPct": 0.701},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.275, "underPct": 0.725},
    {"name": "Kevin Porter Jr.", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.204, "underPct": 0.796},
    {"name": "Pascal Siakam", "line": 4.0, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.201, "underPct": 0.799},
    {"name": "Jordan Goodwin", "line": 3.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.187, "underPct": 0.813},
    {"name": "Tyler Herro", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.181, "underPct": 0.819},
    {"name": "Austin Reaves", "line": 9.0, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.043, "underPct": 0.957},
];const prizepicksReboundsHitRates = [
    {"name": "VJ Edgecombe", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.787, "underPct": 0.213},
    {"name": "Paolo Banchero", "line": 6.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.769, "underPct": 0.231},
    {"name": "Kel'el Ware", "line": 8.5, "l5": 0.8, "l10": 0.9, "l15": 0.87, "overPct": 0.742, "underPct": 0.258},
    {"name": "Steven Adams", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.724, "underPct": 0.276},
    {"name": "Julian Champagnie", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.713, "underPct": 0.287},
    {"name": "Cade Cunningham", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.694, "underPct": 0.306},
    {"name": "Amen Thompson", "line": 7.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.669, "underPct": 0.331},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.662, "underPct": 0.338},
    {"name": "Dominick Barlow", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.661, "underPct": 0.339},
    {"name": "Mitchell Robinson", "line": 6.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.64, "underPct": 0.36},
    {"name": "Bam Adebayo", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.638, "underPct": 0.362},
    {"name": "Lonzo Ball", "line": 4.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.619, "underPct": 0.381},
    {"name": "Austin Reaves", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.608, "underPct": 0.392},
    {"name": "P.J. Washington", "line": 7.0, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.601, "underPct": 0.399},
    {"name": "Jarace Walker", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.588, "underPct": 0.412},
    {"name": "Josh Hart", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.585, "underPct": 0.415},
    {"name": "Jalen Johnson", "line": 9.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.583, "underPct": 0.417},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.57, "underPct": 0.43},
    {"name": "Donovan Clingan", "line": 10.0, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.564, "underPct": 0.436},
    {"name": "Spencer Jones", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.561, "underPct": 0.439},
    {"name": "Davion Mitchell", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.559, "underPct": 0.441},
    {"name": "Santi Aldama", "line": 5.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.556, "underPct": 0.444},
    {"name": "Dyson Daniels", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.554, "underPct": 0.446},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.554, "underPct": 0.446},
    {"name": "OG Anunoby", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.553, "underPct": 0.447},
    {"name": "Anthony Black", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.552, "underPct": 0.448},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jakob Poeltl", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.55, "underPct": 0.45},
    {"name": "Paul George", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.541, "underPct": 0.459},
    {"name": "Jeremy Sochan", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.538, "underPct": 0.462},
    {"name": "Mark Williams", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.534, "underPct": 0.466},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.531, "underPct": 0.469},
    {"name": "Donovan Mitchell", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.529, "underPct": 0.471},
    {"name": "Lauri Markkanen", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.525, "underPct": 0.475},
    {"name": "Jerami Grant", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.523, "underPct": 0.477},
    {"name": "Pascal Siakam", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.523, "underPct": 0.477},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.516, "underPct": 0.484},
    {"name": "Josh Giddey", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.512, "underPct": 0.488},
    {"name": "Zach Edey", "line": 12.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.511, "underPct": 0.489},
    {"name": "Scottie Barnes", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.51, "underPct": 0.49},
    {"name": "Keldon Johnson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.509, "underPct": 0.491},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.508, "underPct": 0.492},
    {"name": "Kevin Durant", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.502, "underPct": 0.498},
    {"name": "Ivica Zubac", "line": 12.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.496, "underPct": 0.504},
    {"name": "Jordan Walsh", "line": 5.0, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.495, "underPct": 0.505},
    {"name": "Matas Buzelis", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.494, "underPct": 0.506},
    {"name": "Jordan Goodwin", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.49, "underPct": 0.51},
    {"name": "Goga Bitadze", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.478, "underPct": 0.522},
    {"name": "Jaylen Brown", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.475, "underPct": 0.525},
    {"name": "Peyton Watson", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.472, "underPct": 0.528},
    {"name": "Tyler Herro", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.47, "underPct": 0.53},
    {"name": "LaMelo Ball", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.464, "underPct": 0.536},
    {"name": "Cameron Johnson", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.459, "underPct": 0.541},
    {"name": "Andrew Wiggins", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.455, "underPct": 0.545},
    {"name": "Shaedon Sharpe", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.454, "underPct": 0.546},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.453, "underPct": 0.547},
    {"name": "Ryan Kalkbrenner", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.453, "underPct": 0.547},
    {"name": "Kyle Kuzma", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.447, "underPct": 0.553},
    {"name": "Ryan Rollins", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.446, "underPct": 0.554},
    {"name": "Jaylon Tyson", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.444, "underPct": 0.556},
    {"name": "Franz Wagner", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.44, "underPct": 0.56},
    {"name": "Kris Murray", "line": 4.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.437, "underPct": 0.563},
    {"name": "Kon Knueppel", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.431, "underPct": 0.569},
    {"name": "Harrison Barnes", "line": 4.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.429, "underPct": 0.571},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.425, "underPct": 0.575},
    {"name": "Devin Vassell", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.424, "underPct": 0.576},
    {"name": "Toumani Camara", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.423, "underPct": 0.577},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.423, "underPct": 0.577},
    {"name": "Derrick White", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.417, "underPct": 0.583},
    {"name": "Immanuel Quickley", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.416, "underPct": 0.584},
    {"name": "Bruce Brown", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.411, "underPct": 0.589},
    {"name": "Nickeil Alexander-Walker", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.407, "underPct": 0.593},
    {"name": "De'Andre Hunter", "line": 4.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.403, "underPct": 0.597},
    {"name": "Grayson Allen", "line": 2.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.403, "underPct": 0.597},
    {"name": "Sandro Mamukelashvili", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Davis", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.393, "underPct": 0.607},
    {"name": "Jalen Williams", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.39, "underPct": 0.61},
    {"name": "Royce O'Neale", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.388, "underPct": 0.612},
    {"name": "Jared McCain", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.386, "underPct": 0.614},
    {"name": "Collin Gillespie", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.386, "underPct": 0.614},
    {"name": "Ausar Thompson", "line": 6.0, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.383, "underPct": 0.617},
    {"name": "Marcus Smart", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.381, "underPct": 0.619},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.378, "underPct": 0.622},
    {"name": "Deni Avdija", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.376, "underPct": 0.624},
    {"name": "Cedric Coward", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.375, "underPct": 0.625},
    {"name": "Kawhi Leonard", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.372, "underPct": 0.628},
    {"name": "John Collins", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.369, "underPct": 0.631},
    {"name": "James Harden", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.364, "underPct": 0.636},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.363, "underPct": 0.637},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.361, "underPct": 0.639},
    {"name": "Luke Kornet", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.359, "underPct": 0.641},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.351, "underPct": 0.649},
    {"name": "Chet Holmgren", "line": 9.0, "l5": 0.0, "l10": 0.0, "l15": 0.13, "overPct": 0.345, "underPct": 0.655},
    {"name": "Jay Huff", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.32, "underPct": 0.68},
    {"name": "Shai Gilgeous-Alexander", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.319, "underPct": 0.681},
    {"name": "Ryan Nembhard", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.319, "underPct": 0.681},
    {"name": "Neemias Queta", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Myles Turner", "line": 6.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.316, "underPct": 0.684},
    {"name": "Tobias Harris", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.308, "underPct": 0.692},
    {"name": "Deandre Ayton", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.289, "underPct": 0.711},
    {"name": "Dylan Harper", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.282, "underPct": 0.718},
    {"name": "Oso Ighodaro", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.278, "underPct": 0.722},
    {"name": "Onyeka Okongwu", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.276, "underPct": 0.724},
    {"name": "Dean Wade", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.257, "underPct": 0.743},
    {"name": "Bennedict Mathurin", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.245, "underPct": 0.755},
    {"name": "Andre Drummond", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.244, "underPct": 0.756},
    {"name": "Rui Hachimura", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.243, "underPct": 0.757},
    {"name": "Ajay Mitchell", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.242, "underPct": 0.758},
    {"name": "Isaiah Jackson", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.234, "underPct": 0.766},
    {"name": "Miles Bridges", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.222, "underPct": 0.778},
    {"name": "Alperen Sengun", "line": 9.0, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.203, "underPct": 0.797},
    {"name": "Bobby Portis", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.191, "underPct": 0.809},
    {"name": "Jake LaRavia", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.099, "underPct": 0.901},
    {"name": "Julian Phillips", "line": 3.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.094, "underPct": 0.906},
    {"name": "Kevin Porter Jr.", "line": 5.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.055, "underPct": 0.945},
];const prizepicksBlocksHitRates = [
    {"name": "Anthony Black", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.41, "underPct": 0.59},
    {"name": "Jalen Suggs", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.568, "underPct": 0.432},
    {"name": "Nickeil Alexander-Walker", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.488, "underPct": 0.512},
    {"name": "Ryan Kalkbrenner", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.422, "underPct": 0.578},
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.614, "underPct": 0.386},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.391, "underPct": 0.609},
    {"name": "Karl-Anthony Towns", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.439, "underPct": 0.561},
    {"name": "OG Anunoby", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.492, "underPct": 0.508},
    {"name": "Alperen Sengun", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.695, "underPct": 0.305},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.446, "underPct": 0.554},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.501, "underPct": 0.499},
];const prizepicksStealsHitRates = [
    {"name": "Neemias Queta", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.311, "underPct": 0.689},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.381, "underPct": 0.619},
    {"name": "Paolo Banchero", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.544, "underPct": 0.456},
    {"name": "Luke Kornet", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.311, "underPct": 0.689},
    {"name": "OG Anunoby", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Brice Sensabaugh", "line": 0.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.559, "underPct": 0.441},
    {"name": "Oso Ighodaro", "line": 0.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.693, "underPct": 0.307},
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.482, "underPct": 0.518},
    {"name": "Zach Edey", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.54, "underPct": 0.46},
    {"name": "Cedric Coward", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.183, "underPct": 0.817},
    {"name": "Paul George", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.541, "underPct": 0.459},
    {"name": "Andre Drummond", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.45, "underPct": 0.55},
    {"name": "Jared McCain", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.53, "underPct": 0.47},
    {"name": "Justin Edwards", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.635, "underPct": 0.365},
    {"name": "Kyle Kuzma", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.615, "underPct": 0.385},
    {"name": "Myles Turner", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.481, "underPct": 0.519},
    {"name": "Shai Gilgeous-Alexander", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.513, "underPct": 0.487},
    {"name": "Ryan Nembhard", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.18, "underPct": 0.82},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Josh Hart", "line": 27.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylen Brown", "line": 41.0, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Royce O'Neale", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 36.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Aaron Holiday", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brice Sensabaugh", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 38.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Black", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 44.0, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Steven Adams", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Paolo Banchero", "line": 27.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Toumani Camara", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Mark Williams", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 41.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach Edey", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "John Collins", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kawhi Leonard", "line": 34.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Duncan Robinson", "line": 14.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Caris LeVert", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shaedon Sharpe", "line": 26.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tobias Harris", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "OG Anunoby", "line": 19.0, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Goodwin", "line": 14.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Walter Clayton Jr.", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "T.J. McConnell", "line": 20.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jay Huff", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Devin Vassell", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jared McCain", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 14.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cameron Johnson", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Onyeka Okongwu", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 34.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Justin Edwards", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Davis", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyler Herro", "line": 29.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Chet Holmgren", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Nembhard", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Neemias Queta", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bruce Brown", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 36.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 41.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 34.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Miller", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Porter Jr.", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Okogie", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Oso Ighodaro", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "James Harden", "line": 41.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Grayson Allen", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 41.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 31.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 47.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Patrick Williams", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keaton Wallace", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dylan Harper", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarace Walker", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kennard", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 29.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 16.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 44.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 12.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 26.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jake LaRavia", "line": 20.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Marcus Smart", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gabe Vincent", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Franz Wagner", "line": 34.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 24.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Williams", "line": 30.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deandre Ayton", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 34.5, "l5": 0.2, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "Karl-Anthony Towns", "line": 39.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 40.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bobby Portis", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sidy Cissoko", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 18.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lonzo Ball", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 22.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 30.0, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Rui Hachimura", "line": 21.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPRHitRates = [
    {"name": "Jaylen Wells", "line": 15.5, "l5": 1.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 20.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 37.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 30.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 20.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Duncan Robinson", "line": 12.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jaylen Brown", "line": 35.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jay Huff", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Mark Williams", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Holiday", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dominick Barlow", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Joe", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Keyonte George", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 21.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Paolo Banchero", "line": 23.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Walter Clayton Jr.", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 25.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 31.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brice Sensabaugh", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Steven Adams", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "OG Anunoby", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tyler Herro", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Royce O'Neale", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Tobias Harris", "line": 19.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jared McCain", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Nembhard", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Naji Marshall", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 27.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Davis", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Dunn", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 11.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "John Collins", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 26.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Caris LeVert", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "D'Angelo Russell", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ja'Kobe Walter", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Miller", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 16.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dylan Harper", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 17.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mouhamed Gueye", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Wiggins", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cameron Johnson", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Sandro Mamukelashvili", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 35.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Okogie", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Porter Jr.", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paul George", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyle Kuzma", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kennard", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zaccharie Risacher", "line": 15.0, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Wiggins", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Grayson Allen", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 20.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 34.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 19.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 16.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 34.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 27.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mitchell Robinson", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Patrick Williams", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alperen Sengun", "line": 32.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 30.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden Ivey", "line": 11.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deandre Ayton", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sidy Cissoko", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 29.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bruce Brown", "line": 12.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lonzo Ball", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ausar Thompson", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julian Phillips", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Isaiah Jackson", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Marcus Smart", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 19.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 25.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jake LaRavia", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 30.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rui Hachimura", "line": 19.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 21.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
];const prizepicksPAHitRates = [
    {"name": "Josh Hart", "line": 18.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Miles McBride", "line": 13.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Wells", "line": 13.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Royce O'Neale", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 34.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brice Sensabaugh", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Holiday", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 28.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cam Spencer", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dominick Barlow", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "D'Angelo Russell", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Goga Bitadze", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 17.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tobias Harris", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 12.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "John Collins", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach Edey", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kawhi Leonard", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 36.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "OG Anunoby", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Oso Ighodaro", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Paolo Banchero", "line": 20.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Walter Clayton Jr.", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Norman Powell", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Gillespie", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 23.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jay Huff", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Miller", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cameron Johnson", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 18.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dylan Harper", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylon Tyson", "line": 16.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jared McCain", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Grayson Allen", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyler Herro", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Reed Sheppard", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cason Wallace", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Goodwin", "line": 10.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Davis", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paul George", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kevin Porter Jr.", "line": 27.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyrese Maxey", "line": 37.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 33.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Davion Mitchell", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 13.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 18.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "De'Andre Hunter", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "Myles Turner", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marcus Smart", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Patrick Williams", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Lonzo Ball", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Brunson", "line": 36.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 9.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sidy Cissoko", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Austin Reaves", "line": 41.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jamal Shead", "line": 11.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rui Hachimura", "line": 17.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jake LaRavia", "line": 14.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 28.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
];const prizepicksRAHitRates = [
    {"name": "P.J. Washington", "line": 8.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Reed Sheppard", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 16.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Amen Thompson", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ausar Thompson", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dominick Barlow", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "VJ Edgecombe", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 9.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Svi Mykhailiuk", "line": 4.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 9.0, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Paolo Banchero", "line": 9.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jared McCain", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "OG Anunoby", "line": 5.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 8.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Dunn", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keaton Wallace", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 13.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "John Collins", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Wells", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kris Dunn", "line": 5.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Oso Ighodaro", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ivica Zubac", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jay Huff", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mikal Bridges", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Davis", "line": 14.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bruce Brown", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lonzo Ball", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 14.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tobias Harris", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 10.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Chet Holmgren", "line": 11.0, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Austin Reaves", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Alperen Sengun", "line": 15.0, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Jackson", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deandre Ayton", "line": 12.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 14.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyler Herro", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jalen Suggs", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 9.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Porter Jr.", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksTurnoversHitRates = [
    {"name": "Josh Hart", "line": 1.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Paul George", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 3.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Isaiah Joe", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keaton Wallace", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dylan Harper", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyler Herro", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksBlocksStealsHitRates = [
    {"name": "P.J. Washington", "line": 1.5, "l5": 1.0, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Gillespie", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Walter Clayton Jr.", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Davis", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach Edey", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Porter Jr.", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dominick Barlow", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derrick White", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Adem Bona", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Garrison Mathews", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Aaron Holiday", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Brown", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Joe", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
];const underdogPointsHitRates = [
    {"name": "D'Angelo Russell", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.902, "underPct": 0.098},
    {"name": "Max Christie", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.87, "overPct": 0.886, "underPct": 0.114},
    {"name": "Mark Williams", "line": 10.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.868, "underPct": 0.132},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.843, "underPct": 0.157},
    {"name": "P.J. Washington", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.804, "underPct": 0.196},
    {"name": "Collin Gillespie", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.785, "underPct": 0.215},
    {"name": "Reed Sheppard", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.785, "underPct": 0.215},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.784, "underPct": 0.216},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.781, "underPct": 0.219},
    {"name": "Paolo Banchero", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.768, "underPct": 0.232},
    {"name": "Dillon Brooks", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.752, "underPct": 0.248},
    {"name": "Tyrese Maxey", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.75, "underPct": 0.25},
    {"name": "Norman Powell", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.741, "underPct": 0.259},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.735, "underPct": 0.265},
    {"name": "Grayson Allen", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.734, "underPct": 0.266},
    {"name": "VJ Edgecombe", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.732, "underPct": 0.268},
    {"name": "Jalen Johnson", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.709, "underPct": 0.291},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.709, "underPct": 0.291},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.705, "underPct": 0.295},
    {"name": "Keldon Johnson", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.703, "underPct": 0.297},
    {"name": "Nickeil Alexander-Walker", "line": 19.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.699, "underPct": 0.301},
    {"name": "Desmond Bane", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.698, "underPct": 0.302},
    {"name": "Tobias Harris", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.688, "underPct": 0.312},
    {"name": "Duncan Robinson", "line": 10.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.686, "underPct": 0.314},
    {"name": "Svi Mykhailiuk", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.676, "underPct": 0.324},
    {"name": "Jared McCain", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.672, "underPct": 0.328},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.671, "underPct": 0.329},
    {"name": "Peyton Watson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.66, "underPct": 0.34},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.654, "underPct": 0.346},
    {"name": "Tyler Herro", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.652, "underPct": 0.348},
    {"name": "Kevin Durant", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.643, "underPct": 0.357},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.641, "underPct": 0.359},
    {"name": "Davion Mitchell", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.634, "underPct": 0.366},
    {"name": "Justin Edwards", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.632, "underPct": 0.368},
    {"name": "Ryan Nembhard", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.628, "underPct": 0.372},
    {"name": "Isaiah Joe", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.625, "underPct": 0.375},
    {"name": "Anfernee Simons", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.622, "underPct": 0.378},
    {"name": "De'Aaron Fox", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.617, "underPct": 0.383},
    {"name": "Anthony Black", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.617, "underPct": 0.383},
    {"name": "Jaylon Tyson", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.611, "underPct": 0.389},
    {"name": "Andrew Nembhard", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.61, "underPct": 0.39},
    {"name": "Cade Cunningham", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.609, "underPct": 0.391},
    {"name": "Derrick White", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.603, "underPct": 0.397},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.601, "underPct": 0.399},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.601, "underPct": 0.399},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.597, "underPct": 0.403},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.59, "underPct": 0.41},
    {"name": "Jaylen Brown", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.583, "underPct": 0.417},
    {"name": "Jalen Suggs", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.581, "underPct": 0.419},
    {"name": "Ivica Zubac", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.576, "underPct": 0.424},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.562, "underPct": 0.438},
    {"name": "Isaiah Collier", "line": 4.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.559, "underPct": 0.441},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.555, "underPct": 0.445},
    {"name": "Karl-Anthony Towns", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.554, "underPct": 0.446},
    {"name": "Alperen Sengun", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.553, "underPct": 0.447},
    {"name": "Andrew Wiggins", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.545, "underPct": 0.455},
    {"name": "Shaedon Sharpe", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.545, "underPct": 0.455},
    {"name": "Kyle Kuzma", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.542, "underPct": 0.458},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.54, "underPct": 0.46},
    {"name": "Bam Adebayo", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.533, "underPct": 0.467},
    {"name": "Jarace Walker", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.531, "underPct": 0.469},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.523, "underPct": 0.477},
    {"name": "Jalen Duren", "line": 17.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.521, "underPct": 0.479},
    {"name": "Luke Kennard", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.518, "underPct": 0.482},
    {"name": "Toumani Camara", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.515, "underPct": 0.485},
    {"name": "James Harden", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.513, "underPct": 0.487},
    {"name": "De'Andre Hunter", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.481, "underPct": 0.519},
    {"name": "Ayo Dosunmu", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.478, "underPct": 0.522},
    {"name": "Paul George", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.474, "underPct": 0.526},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.473, "underPct": 0.527},
    {"name": "Spencer Jones", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.465, "underPct": 0.535},
    {"name": "Sidy Cissoko", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.463, "underPct": 0.537},
    {"name": "Caris LeVert", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.461, "underPct": 0.539},
    {"name": "Anthony Davis", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.45, "underPct": 0.55},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.423, "underPct": 0.577},
    {"name": "T.J. McConnell", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.419, "underPct": 0.581},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.409, "underPct": 0.591},
    {"name": "Dylan Harper", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.408, "underPct": 0.592},
    {"name": "Ausar Thompson", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Brice Sensabaugh", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.394, "underPct": 0.606},
    {"name": "Lauri Markkanen", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "Jay Huff", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.383, "underPct": 0.617},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.383, "underPct": 0.617},
    {"name": "Austin Reaves", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.373, "underPct": 0.627},
    {"name": "Quentin Grimes", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.371, "underPct": 0.629},
    {"name": "Pascal Siakam", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.368, "underPct": 0.632},
    {"name": "Jalen Brunson", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.319, "underPct": 0.681},
    {"name": "Kevin Porter Jr.", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.312, "underPct": 0.688},
    {"name": "Jordan Clarkson", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.304, "underPct": 0.696},
    {"name": "Josh Giddey", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.297, "underPct": 0.703},
    {"name": "Cedric Coward", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.267, "underPct": 0.733},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.244, "underPct": 0.756},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.183, "underPct": 0.817},
    {"name": "Patrick Williams", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.177, "underPct": 0.823},
    {"name": "Jalen Williams", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.147, "underPct": 0.853},
    {"name": "Andre Drummond", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.118, "underPct": 0.882},
    {"name": "Matas Buzelis", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.097, "underPct": 0.903},
    {"name": "Julian Phillips", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.05, "underPct": 0.95},
];const underdogAssistsHitRates = [
    {"name": "Coby White", "line": 4.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.74, "underPct": 0.26},
    {"name": "Oso Ighodaro", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.703, "underPct": 0.297},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.663, "underPct": 0.337},
    {"name": "P.J. Washington", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.661, "underPct": 0.339},
    {"name": "Dominick Barlow", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.64, "underPct": 0.36},
    {"name": "Jaylen Wells", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.63, "underPct": 0.37},
    {"name": "Jamal Murray", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.612, "underPct": 0.388},
    {"name": "Tobias Harris", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.575, "underPct": 0.425},
    {"name": "Shaedon Sharpe", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.568, "underPct": 0.432},
    {"name": "Isaiah Collier", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.551, "underPct": 0.449},
    {"name": "Bobby Portis", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.529, "underPct": 0.471},
    {"name": "Toumani Camara", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.479, "underPct": 0.521},
    {"name": "Grayson Allen", "line": 3.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.477, "underPct": 0.523},
    {"name": "Shai Gilgeous-Alexander", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.474, "underPct": 0.526},
    {"name": "Andrew Nembhard", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.427, "underPct": 0.573},
    {"name": "Jarace Walker", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.42, "underPct": 0.58},
    {"name": "Max Christie", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.382, "underPct": 0.618},
    {"name": "Bam Adebayo", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.357, "underPct": 0.643},
    {"name": "Ayo Dosunmu", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.352, "underPct": 0.648},
    {"name": "Cedric Coward", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.346, "underPct": 0.654},
    {"name": "Kevin Porter Jr.", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.204, "underPct": 0.796},
];const underdogReboundsHitRates = [
    {"name": "Steven Adams", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.724, "underPct": 0.276},
    {"name": "Cade Cunningham", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.694, "underPct": 0.306},
    {"name": "Dominick Barlow", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.661, "underPct": 0.339},
    {"name": "Mitchell Robinson", "line": 6.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.64, "underPct": 0.36},
    {"name": "Bam Adebayo", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.638, "underPct": 0.362},
    {"name": "Austin Reaves", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.608, "underPct": 0.392},
    {"name": "Jalen Johnson", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.583, "underPct": 0.417},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.57, "underPct": 0.43},
    {"name": "Spencer Jones", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.561, "underPct": 0.439},
    {"name": "Davion Mitchell", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.559, "underPct": 0.441},
    {"name": "Dyson Daniels", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.554, "underPct": 0.446},
    {"name": "Anthony Black", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.552, "underPct": 0.448},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.552, "underPct": 0.448},
    {"name": "Mark Williams", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.534, "underPct": 0.466},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.531, "underPct": 0.469},
    {"name": "Max Christie", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.529, "underPct": 0.471},
    {"name": "Jerami Grant", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.523, "underPct": 0.477},
    {"name": "Josh Giddey", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.512, "underPct": 0.488},
    {"name": "Matas Buzelis", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.494, "underPct": 0.506},
    {"name": "Goga Bitadze", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.478, "underPct": 0.522},
    {"name": "Kyle Kuzma", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.447, "underPct": 0.553},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.423, "underPct": 0.577},
    {"name": "De'Andre Hunter", "line": 4.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.403, "underPct": 0.597},
    {"name": "Jared McCain", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.386, "underPct": 0.614},
    {"name": "Ryan Nembhard", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.319, "underPct": 0.681},
    {"name": "Dylan Harper", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.282, "underPct": 0.718},
    {"name": "Julian Phillips", "line": 3.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.094, "underPct": 0.906},
];const underdogBlocksHitRates = [
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.446, "underPct": 0.554},
];const underdogStealsHitRates = [
    {"name": "OG Anunoby", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Jaylen Brown", "line": 41.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 27.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Royce O'Neale", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brice Sensabaugh", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Nickeil Alexander-Walker", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylon Tyson", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Holiday", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 38.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Goga Bitadze", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 44.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 40.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "P.J. Washington", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Paolo Banchero", "line": 26.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Dominick Barlow", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Toumani Camara", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Steven Adams", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "John Collins", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Duncan Robinson", "line": 14.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Lauri Markkanen", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Santi Aldama", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "OG Anunoby", "line": 19.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Zach Edey", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Walter Clayton Jr.", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kawhi Leonard", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 26.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Gillespie", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ayo Dosunmu", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "T.J. McConnell", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jay Huff", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Dunn", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Joe", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Caris LeVert", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cameron Johnson", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyler Herro", "line": 29.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Davis", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Spencer Jones", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Mitchell", "line": 41.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Justin Edwards", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keldon Johnson", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jared McCain", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Porter Jr.", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Okogie", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Oso Ighodaro", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "James Harden", "line": 41.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 41.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Reed Sheppard", "line": 15.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 47.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 29.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keaton Wallace", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dylan Harper", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dean Wade", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Grayson Allen", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarace Walker", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Patrick Williams", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 43.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mitchell Robinson", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andre Drummond", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 40.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Davion Mitchell", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Williams", "line": 29.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Franz Wagner", "line": 34.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 39.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sidy Cissoko", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 29.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const underdogPRHitRates = [
    {"name": "Saddiq Bey", "line": 21.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Joel Embiid", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Scottie Barnes", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 33.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Immanuel Quickley", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyle Filipowski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Anfernee Simons", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jonathan Kuminga", "line": 20.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogPAHitRates = [
    {"name": "Anthony Edwards", "line": 34.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Joel Embiid", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 30.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Ingram", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julius Randle", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
];const underdogRAHitRates = [
    {"name": "P.J. Washington", "line": 8.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylen Brown", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Santi Aldama", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Paolo Banchero", "line": 9.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Ausar Thompson", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 16.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Dominick Barlow", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Steven Adams", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Nembhard", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 7.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 14.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Myles Turner", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Suggs", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Porter Jr.", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogTurnoversHitRates = [
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
];const underdogBlocksStealsHitRates = [
    {"name": "Derrick White", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
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
            <th style="width: 9%">Sigma</th>
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
            <td>
                <span class="sigma-badge ${getSigmaClass(row.sigma)}">${row.sigma}</span>
            </td>
        </tr>
    `).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

function renderPairsTable(data) {
    const thead = `
        <tr>
            <th style="width: 3%">#</th>
            <th style="width: 15%">Player 1</th>
            <th style="width: 7%">Line</th>
            <th style="width: 7%">Proj.</th>
            <th style="width: 8%">Impl. Prob.</th>
            <th style="width: 7%">Edge</th>
            <th style="width: 15%">Player 2</th>
            <th style="width: 7%">Line</th>
            <th style="width: 7%">Proj.</th>
            <th style="width: 8%">Impl. Prob.</th>
            <th style="width: 7%">Edge</th>
            <th style="width: 9%">Total Edge</th>
        </tr>
    `;

    const tbody = data.map((row, index) => {
        const edge1 = row.edge1 || 0;
        const edge2 = row.edge2 || 0;
        const totalEdge = edge1 + edge2;
        
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
            <td style="font-weight: 600; color: #667eea; font-size: 0.9rem;">
                ${((row.impliedProb1 || 0) * 100).toFixed(1)}%
            </td>
            <td class="prediction-value" style="color: ${edge1 > 0 ? '#10b981' : '#ef4444'}; font-weight: 600;">${edge1.toFixed(1)}</td>
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
            <td style="font-weight: 600; color: #667eea; font-size: 0.9rem;">
                ${((row.impliedProb2 || 0) * 100).toFixed(1)}%
            </td>
            <td class="prediction-value" style="color: ${edge2 > 0 ? '#10b981' : '#ef4444'}; font-weight: 600;">${edge2.toFixed(1)}</td>
            <td class="prediction-value" style="color: ${totalEdge > 0 ? '#10b981' : '#ef4444'}; font-weight: 700; font-size: 1rem;">${totalEdge.toFixed(1)}</td>
        </tr>
    `}).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

function renderTriosTable(data) {
    const thead = `
        <tr>
            <th style="width: 2%">#</th>
            <th style="width: 11%">Player 1</th>
            <th style="width: 5%">Line</th>
            <th style="width: 5%">Proj.</th>
            <th style="width: 6%">Impl. Prob.</th>
            <th style="width: 5%">Edge</th>
            <th style="width: 11%">Player 2</th>
            <th style="width: 5%">Line</th>
            <th style="width: 5%">Proj.</th>
            <th style="width: 6%">Impl. Prob.</th>
            <th style="width: 5%">Edge</th>
            <th style="width: 11%">Player 3</th>
            <th style="width: 5%">Line</th>
            <th style="width: 5%">Proj.</th>
            <th style="width: 6%">Impl. Prob.</th>
            <th style="width: 5%">Edge</th>
            <th style="width: 7%">Total Edge</th>
        </tr>
    `;

    const tbody = data.map((row, index) => {
        const edge1 = row.edge1 || 0;
        const edge2 = row.edge2 || 0;
        const edge3 = row.edge3 || 0;
        const totalEdge = edge1 + edge2 + edge3;
        
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
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.impliedProb1 || 0) * 100).toFixed(1)}%
            </td>
            <td class="prediction-value" style="color: ${edge1 > 0 ? '#10b981' : '#ef4444'}; font-weight: 600;">${edge1.toFixed(1)}</td>
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
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.impliedProb2 || 0) * 100).toFixed(1)}%
            </td>
            <td class="prediction-value" style="color: ${edge2 > 0 ? '#10b981' : '#ef4444'}; font-weight: 600;">${edge2.toFixed(1)}</td>
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
            <td style="font-weight: 600; color: #667eea; font-size: 0.85rem;">
                ${((row.impliedProb3 || 0) * 100).toFixed(1)}%
            </td>
            <td class="prediction-value" style="color: ${edge3 > 0 ? '#10b981' : '#ef4444'}; font-weight: 600;">${edge3.toFixed(1)}</td>
            <td class="prediction-value" style="color: ${totalEdge > 0 ? '#10b981' : '#ef4444'}; font-weight: 700; font-size: 0.95rem;">${totalEdge.toFixed(1)}</td>
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
                <div class="stat-label">Sigma</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (volatile, less reliable projections), Med, Low (consistent, more reliable projections)</div>
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

