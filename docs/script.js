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
    {"name1": "Marvin Bagley III", "name2": "Jaime Jaquez Jr.", "line1": 13.5, "line2": 14.5, "prediction1": 7.05, "prediction2": 22.29, "side1": "under", "side2": "over", "edge1": 6.45, "edge2": 7.79, "impliedProb1": 0.49, "impliedProb2": 0.51},
    {"name1": "Kevin Porter Jr.", "name2": "Aaron Holiday", "line1": 20.5, "line2": 10.5, "prediction1": 14.42, "prediction2": 4.21, "side1": "under", "side2": "under", "edge1": 6.08, "edge2": 6.29, "impliedProb1": 0.51, "impliedProb2": 0.5},
    {"name1": "Jaden McDaniels", "name2": "Maxime Raynaud", "line1": 13.5, "line2": 10.5, "prediction1": 19.39, "prediction2": 5.19, "side1": "over", "side2": "under", "edge1": 5.89, "edge2": 5.31, "impliedProb1": 0.5, "impliedProb2": 0.51},
    {"name1": "Julius Randle", "name2": "Kevin Durant", "line1": 20.5, "line2": 26.5, "prediction1": 26.11, "prediction2": 21.32, "side1": "over", "side2": "under", "edge1": 5.61, "edge2": 5.18, "impliedProb1": 0.49, "impliedProb2": 0.52},
    {"name1": "Danny Wolf", "name2": "Ryan Nembhard", "line1": 10.5, "line2": 9.5, "prediction1": 5.5, "prediction2": 4.54, "side1": "under", "side2": "under", "edge1": 5.0, "edge2": 4.96, "impliedProb1": 0.51, "impliedProb2": 0.52},
    {"name1": "Noah Clowney", "name2": "Anthony Davis", "line1": 15.5, "line2": 20.5, "prediction1": 10.74, "prediction2": 15.56, "side1": "under", "side2": "under", "edge1": 4.76, "edge2": 4.94, "impliedProb1": 0.49, "impliedProb2": 0.51},
    {"name1": "Trey Murphy III", "name2": "Jalen Duren", "line1": 20.5, "line2": 18.5, "prediction1": 25.03, "prediction2": 13.9, "side1": "over", "side2": "under", "edge1": 4.53, "edge2": 4.6, "impliedProb1": 0.52, "impliedProb2": 0.53},
    {"name1": "Nickeil Alexander-Walker", "name2": "Andrew Wiggins", "line1": 21.0, "line2": 16.5, "prediction1": 25.3, "prediction2": 20.63, "side1": "over", "side2": "over", "edge1": 4.3, "edge2": 4.13, "impliedProb1": 0.56, "impliedProb2": 0.51},
    {"name1": "Cam Whitmore", "name2": "P.J. Washington", "line1": 10.5, "line2": 13.5, "prediction1": 6.6, "prediction2": 17.49, "side1": "under", "side2": "over", "edge1": 3.9, "edge2": 3.99, "impliedProb1": 0.5, "impliedProb2": 0.5},
    {"name1": "Darius Garland", "name2": "Bam Adebayo", "line1": 16.5, "line2": 20.5, "prediction1": 12.62, "prediction2": 24.4, "side1": "under", "side2": "over", "edge1": 3.88, "edge2": 3.9, "impliedProb1": 0.5, "impliedProb2": 0.51},
];const prizepicksTriosData = [
    {"name1": "Marvin Bagley III", "name2": "Jaime Jaquez Jr.", "name3": "Aaron Holiday", "line1": 13.5, "line2": 14.5, "line3": 10.5, "prediction1": 7.05, "prediction2": 22.29, "prediction3": 4.21, "side1": "under", "side2": "over", "side3": "under", "edge1": 6.45, "edge2": 7.79, "edge3": 6.29, "impliedProb1": 0.49, "impliedProb2": 0.51, "impliedProb3": 0.5},
    {"name1": "Kevin Porter Jr.", "name2": "Jaden McDaniels", "name3": "Maxime Raynaud", "line1": 20.5, "line2": 13.5, "line3": 10.5, "prediction1": 14.42, "prediction2": 19.39, "prediction3": 5.19, "side1": "under", "side2": "over", "side3": "under", "edge1": 6.08, "edge2": 5.89, "edge3": 5.31, "impliedProb1": 0.51, "impliedProb2": 0.5, "impliedProb3": 0.51},
    {"name1": "Danny Wolf", "name2": "Julius Randle", "name3": "Kevin Durant", "line1": 10.5, "line2": 20.5, "line3": 26.5, "prediction1": 5.5, "prediction2": 26.11, "prediction3": 21.32, "side1": "under", "side2": "over", "side3": "under", "edge1": 5.0, "edge2": 5.61, "edge3": 5.18, "impliedProb1": 0.51, "impliedProb2": 0.49, "impliedProb3": 0.52},
    {"name1": "Noah Clowney", "name2": "Jalen Duren", "name3": "Ryan Nembhard", "line1": 15.5, "line2": 18.5, "line3": 9.5, "prediction1": 10.74, "prediction2": 13.9, "prediction3": 4.54, "side1": "under", "side2": "under", "side3": "under", "edge1": 4.76, "edge2": 4.6, "edge3": 4.96, "impliedProb1": 0.49, "impliedProb2": 0.53, "impliedProb3": 0.52},
    {"name1": "Trey Murphy III", "name2": "Nickeil Alexander-Walker", "name3": "Anthony Davis", "line1": 20.5, "line2": 21.0, "line3": 20.5, "prediction1": 25.03, "prediction2": 25.3, "prediction3": 15.56, "side1": "over", "side2": "over", "side3": "under", "edge1": 4.53, "edge2": 4.3, "edge3": 4.94, "impliedProb1": 0.52, "impliedProb2": 0.56, "impliedProb3": 0.51},
    {"name1": "Cam Whitmore", "name2": "Andrew Wiggins", "name3": "P.J. Washington", "line1": 10.5, "line2": 16.5, "line3": 13.5, "prediction1": 6.6, "prediction2": 20.63, "prediction3": 17.49, "side1": "under", "side2": "over", "side3": "over", "edge1": 3.9, "edge2": 4.13, "edge3": 3.99, "impliedProb1": 0.5, "impliedProb2": 0.51, "impliedProb3": 0.5},
    {"name1": "Kyshawn George", "name2": "Darius Garland", "name3": "Bam Adebayo", "line1": 14.5, "line2": 16.5, "line3": 20.5, "prediction1": 10.77, "prediction2": 12.62, "prediction3": 24.4, "side1": "under", "side2": "under", "side3": "over", "edge1": 3.73, "edge2": 3.88, "edge3": 3.9, "impliedProb1": 0.5, "impliedProb2": 0.5, "impliedProb3": 0.51},
    {"name1": "Saddiq Bey", "name2": "Ryan Rollins", "name3": "Ivica Zubac", "line1": 16.5, "line2": 19.5, "line3": 14.5, "prediction1": 13.26, "prediction2": 15.91, "prediction3": 17.94, "side1": "under", "side2": "under", "side3": "over", "edge1": 3.24, "edge2": 3.59, "edge3": 3.44, "impliedProb1": 0.5, "impliedProb2": 0.5, "impliedProb3": 0.5},
    {"name1": "Myles Turner", "name2": "Naz Reid", "name3": "Josh Okogie", "line1": 12.5, "line2": 13.5, "line3": 6.5, "prediction1": 9.13, "prediction2": 16.63, "prediction3": 3.59, "side1": "under", "side2": "over", "side3": "under", "edge1": 3.37, "edge2": 3.13, "edge3": 2.91, "impliedProb1": 0.51, "impliedProb2": 0.5, "impliedProb3": 0.49},
    {"name1": "Mouhamed Gueye", "name2": "Caris LeVert", "name3": "Russell Westbrook", "line1": 7.5, "line2": 8.5, "line3": 13.5, "prediction1": 4.99, "prediction2": 5.34, "prediction3": 10.73, "side1": "under", "side2": "under", "side3": "under", "edge1": 2.51, "edge2": 3.16, "edge3": 2.77, "impliedProb1": 0.52, "impliedProb2": 0.52, "impliedProb3": 0.5},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Marvin Bagley III", "name2": "Jaime Jaquez Jr.", "line1": 13.5, "line2": 14.5, "prediction1": 7.05, "prediction2": 22.29, "side1": "under", "side2": "over", "edge1": 6.45, "edge2": 7.79, "impliedProb1": 0.49, "impliedProb2": 0.51},
    {"name1": "Kevin Porter Jr.", "name2": "Jaden McDaniels", "line1": 20.5, "line2": 13.5, "prediction1": 14.42, "prediction2": 19.39, "side1": "under", "side2": "over", "edge1": 6.08, "edge2": 5.89, "impliedProb1": 0.51, "impliedProb2": 0.5},
    {"name1": "Noah Clowney", "name2": "Julius Randle", "line1": 16.5, "line2": 20.5, "prediction1": 10.74, "prediction2": 26.11, "side1": "under", "side2": "over", "edge1": 5.76, "edge2": 5.61, "impliedProb1": 0.55, "impliedProb2": 0.49},
    {"name1": "Maxime Raynaud", "name2": "Kevin Durant", "line1": 10.5, "line2": 26.5, "prediction1": 5.19, "prediction2": 21.32, "side1": "under", "side2": "under", "edge1": 5.31, "edge2": 5.18, "impliedProb1": 0.51, "impliedProb2": 0.52},
    {"name1": "Danny Wolf", "name2": "Ryan Nembhard", "line1": 10.5, "line2": 9.5, "prediction1": 5.5, "prediction2": 4.54, "side1": "under", "side2": "under", "edge1": 5.0, "edge2": 4.96, "impliedProb1": 0.51, "impliedProb2": 0.52},
    {"name1": "Saddiq Bey", "name2": "Nickeil Alexander-Walker", "line1": 17.5, "line2": 20.5, "prediction1": 13.26, "prediction2": 25.3, "side1": "under", "side2": "over", "edge1": 4.24, "edge2": 4.8, "impliedProb1": 0.53, "impliedProb2": 0.53},
    {"name1": "Kyshawn George", "name2": "P.J. Washington", "line1": 15.5, "line2": 13.5, "prediction1": 10.77, "prediction2": 17.49, "side1": "under", "side2": "over", "edge1": 4.73, "edge2": 3.99, "impliedProb1": 0.55, "impliedProb2": 0.5},
    {"name1": "Darius Garland", "name2": "Bam Adebayo", "line1": 16.5, "line2": 20.5, "prediction1": 12.62, "prediction2": 24.4, "side1": "under", "side2": "over", "edge1": 3.88, "edge2": 3.9, "impliedProb1": 0.5, "impliedProb2": 0.51},
    {"name1": "Bryce McGowens", "name2": "Ivica Zubac", "line1": 8.5, "line2": 14.5, "prediction1": 4.84, "prediction2": 17.94, "side1": "under", "side2": "over", "edge1": 3.66, "edge2": 3.44, "impliedProb1": 0.48, "impliedProb2": 0.5},
    {"name1": "Myles Turner", "name2": "Naz Reid", "line1": 12.5, "line2": 13.5, "prediction1": 9.13, "prediction2": 16.63, "side1": "under", "side2": "over", "edge1": 3.37, "edge2": 3.13, "impliedProb1": 0.51, "impliedProb2": 0.5},
];const underdogTriosData = [
    {"name1": "Marvin Bagley III", "name2": "Kevin Porter Jr.", "name3": "Jaime Jaquez Jr.", "line1": 13.5, "line2": 20.5, "line3": 14.5, "prediction1": 7.05, "prediction2": 14.42, "prediction3": 22.29, "side1": "under", "side2": "under", "side3": "over", "edge1": 6.45, "edge2": 6.08, "edge3": 7.79, "impliedProb1": 0.49, "impliedProb2": 0.51, "impliedProb3": 0.51},
    {"name1": "Noah Clowney", "name2": "Jaden McDaniels", "name3": "Maxime Raynaud", "line1": 16.5, "line2": 13.5, "line3": 10.5, "prediction1": 10.74, "prediction2": 19.39, "prediction3": 5.19, "side1": "under", "side2": "over", "side3": "under", "edge1": 5.76, "edge2": 5.89, "edge3": 5.31, "impliedProb1": 0.55, "impliedProb2": 0.5, "impliedProb3": 0.51},
    {"name1": "Danny Wolf", "name2": "Julius Randle", "name3": "Kevin Durant", "line1": 10.5, "line2": 20.5, "line3": 26.5, "prediction1": 5.5, "prediction2": 26.11, "prediction3": 21.32, "side1": "under", "side2": "over", "side3": "under", "edge1": 5.0, "edge2": 5.61, "edge3": 5.18, "impliedProb1": 0.51, "impliedProb2": 0.49, "impliedProb3": 0.52},
    {"name1": "Saddiq Bey", "name2": "Nickeil Alexander-Walker", "name3": "Ryan Nembhard", "line1": 17.5, "line2": 20.5, "line3": 9.5, "prediction1": 13.26, "prediction2": 25.3, "prediction3": 4.54, "side1": "under", "side2": "over", "side3": "under", "edge1": 4.24, "edge2": 4.8, "edge3": 4.96, "impliedProb1": 0.53, "impliedProb2": 0.53, "impliedProb3": 0.52},
    {"name1": "Kyshawn George", "name2": "Bam Adebayo", "name3": "P.J. Washington", "line1": 15.5, "line2": 20.5, "line3": 13.5, "prediction1": 10.77, "prediction2": 24.4, "prediction3": 17.49, "side1": "under", "side2": "over", "side3": "over", "edge1": 4.73, "edge2": 3.9, "edge3": 3.99, "impliedProb1": 0.55, "impliedProb2": 0.51, "impliedProb3": 0.5},
    {"name1": "Bryce McGowens", "name2": "Darius Garland", "name3": "Ivica Zubac", "line1": 8.5, "line2": 16.5, "line3": 14.5, "prediction1": 4.84, "prediction2": 12.62, "prediction3": 17.94, "side1": "under", "side2": "under", "side3": "over", "edge1": 3.66, "edge2": 3.88, "edge3": 3.44, "impliedProb1": 0.48, "impliedProb2": 0.5, "impliedProb3": 0.5},
    {"name1": "Myles Turner", "name2": "Naz Reid", "name3": "Amen Thompson", "line1": 12.5, "line2": 13.5, "line3": 19.5, "prediction1": 9.13, "prediction2": 16.63, "prediction3": 16.51, "side1": "under", "side2": "over", "side3": "under", "edge1": 3.37, "edge2": 3.13, "edge3": 2.99, "impliedProb1": 0.51, "impliedProb2": 0.5, "impliedProb3": 0.52},
    {"name1": "Mouhamed Gueye", "name2": "Caris LeVert", "name3": "Russell Westbrook", "line1": 7.5, "line2": 8.5, "line3": 13.5, "prediction1": 4.99, "prediction2": 5.34, "prediction3": 10.73, "side1": "under", "side2": "under", "side3": "under", "edge1": 2.51, "edge2": 3.16, "edge3": 2.77, "impliedProb1": 0.52, "impliedProb2": 0.52, "impliedProb3": 0.5},
    {"name1": "Luke Kennard", "name2": "Kyle Kuzma", "name3": "Nicolas Batum", "line1": 7.5, "line2": 13.5, "line3": 4.5, "prediction1": 5.9, "prediction2": 11.41, "prediction3": 2.41, "side1": "under", "side2": "under", "side3": "under", "edge1": 1.6, "edge2": 2.09, "edge3": 2.09, "impliedProb1": 0.52, "impliedProb2": 0.5, "impliedProb3": 0.51},
    {"name1": "Donovan Mitchell", "name2": "Jaden Ivey", "name3": "James Harden", "line1": 29.5, "line2": 8.5, "line3": 24.5, "prediction1": 27.92, "prediction2": 6.72, "prediction3": 22.6, "side1": "under", "side2": "under", "side3": "under", "edge1": 1.58, "edge2": 1.78, "edge3": 1.9, "impliedProb1": 0.53, "impliedProb2": 0.5, "impliedProb3": 0.5},
];const prizepicksPointsHitRates = [
    {"name": "P.J. Washington", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.84, "underPct": 0.16},
    {"name": "Rudy Gobert", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.815, "underPct": 0.185},
    {"name": "Micah Peavy", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.804, "underPct": 0.196},
    {"name": "Jaylon Tyson", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.796, "underPct": 0.204},
    {"name": "Anthony Edwards", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.779, "underPct": 0.221},
    {"name": "Noah Clowney", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.775, "underPct": 0.225},
    {"name": "Jose Alvarado", "line": 9.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.742, "underPct": 0.258},
    {"name": "Kawhi Leonard", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.731, "underPct": 0.269},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.721, "underPct": 0.279},
    {"name": "Naji Marshall", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.72, "underPct": 0.28},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.694, "underPct": 0.306},
    {"name": "Max Christie", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.688, "underPct": 0.312},
    {"name": "Saddiq Bey", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.676, "underPct": 0.324},
    {"name": "Ryan Nembhard", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.672, "underPct": 0.328},
    {"name": "Kris Dunn", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.661, "underPct": 0.339},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.642, "underPct": 0.358},
    {"name": "Reed Sheppard", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.639, "underPct": 0.361},
    {"name": "Tyrese Martin", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.614, "underPct": 0.386},
    {"name": "Nickeil Alexander-Walker", "line": 21.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.587, "underPct": 0.413},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.572, "underPct": 0.428},
    {"name": "Cade Cunningham", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.57, "underPct": 0.43},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.569, "underPct": 0.431},
    {"name": "Amen Thompson", "line": 19.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.567, "underPct": 0.433},
    {"name": "John Collins", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.562, "underPct": 0.438},
    {"name": "Norman Powell", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.538, "underPct": 0.462},
    {"name": "Tobias Harris", "line": 14.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.537, "underPct": 0.463},
    {"name": "Ivica Zubac", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.526, "underPct": 0.474},
    {"name": "Myles Turner", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.52, "underPct": 0.48},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.504, "underPct": 0.496},
    {"name": "Kel'el Ware", "line": 12.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.503, "underPct": 0.497},
    {"name": "De'Andre Hunter", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.493, "underPct": 0.507},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.475, "underPct": 0.525},
    {"name": "Caris LeVert", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.475, "underPct": 0.525},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.469, "underPct": 0.531},
    {"name": "Andrew Wiggins", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.468, "underPct": 0.532},
    {"name": "Ausar Thompson", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.462, "underPct": 0.538},
    {"name": "Julius Randle", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.461, "underPct": 0.539},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.456, "underPct": 0.544},
    {"name": "Bam Adebayo", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.443, "underPct": 0.557},
    {"name": "Nicolas Batum", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.425, "underPct": 0.575},
    {"name": "Jordan Hawkins", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.425, "underPct": 0.575},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.424, "underPct": 0.576},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.422, "underPct": 0.578},
    {"name": "Kevin Porter Jr.", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.408, "underPct": 0.592},
    {"name": "DeMar DeRozan", "line": 18.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.404, "underPct": 0.596},
    {"name": "Aaron Holiday", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.397, "underPct": 0.603},
    {"name": "Jalen Johnson", "line": 24.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.389, "underPct": 0.611},
    {"name": "James Harden", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.386, "underPct": 0.614},
    {"name": "Yves Missi", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.383, "underPct": 0.617},
    {"name": "Maxime Raynaud", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.37, "underPct": 0.63},
    {"name": "Mike Conley", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.364, "underPct": 0.636},
    {"name": "Kobe Sanders", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.361, "underPct": 0.639},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.361, "underPct": 0.639},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.33, "underPct": 0.67},
    {"name": "Darius Garland", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.305, "underPct": 0.695},
    {"name": "Cam Whitmore", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.299, "underPct": 0.701},
    {"name": "Zaccharie Risacher", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.284, "underPct": 0.716},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.283, "underPct": 0.717},
    {"name": "Danny Wolf", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.278, "underPct": 0.722},
    {"name": "Jalen Duren", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.271, "underPct": 0.729},
    {"name": "Derik Queen", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.269, "underPct": 0.731},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.266, "underPct": 0.734},
    {"name": "Josh Okogie", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.255, "underPct": 0.745},
    {"name": "Kyshawn George", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.255, "underPct": 0.745},
    {"name": "Jaden Ivey", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.251, "underPct": 0.749},
    {"name": "Onyeka Okongwu", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.247, "underPct": 0.753},
    {"name": "Luke Kennard", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.227, "underPct": 0.773},
    {"name": "Anthony Davis", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.205, "underPct": 0.795},
    {"name": "Mouhamed Gueye", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.19, "underPct": 0.81},
    {"name": "Terance Mann", "line": 8.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.186, "underPct": 0.814},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.147, "underPct": 0.853},
    {"name": "Marvin Bagley III", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.139, "underPct": 0.861},
];const prizepicksAssistsHitRates = [
    {"name": "Rudy Gobert", "line": 1.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.695, "underPct": 0.305},
    {"name": "Jalen Johnson", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.57, "underPct": 0.43},
    {"name": "Russell Westbrook", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.554, "underPct": 0.446},
    {"name": "Darius Garland", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.535, "underPct": 0.465},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.505, "underPct": 0.495},
    {"name": "Davion Mitchell", "line": 8.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.496, "underPct": 0.504},
    {"name": "Saddiq Bey", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.487, "underPct": 0.513},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.483, "underPct": 0.517},
    {"name": "Mike Conley", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.48, "underPct": 0.52},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.475, "underPct": 0.525},
    {"name": "Evan Mobley", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.466, "underPct": 0.534},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.455, "underPct": 0.545},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.417, "underPct": 0.583},
    {"name": "Donovan Mitchell", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.417, "underPct": 0.583},
    {"name": "Cade Cunningham", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.39, "underPct": 0.61},
    {"name": "Kevin Durant", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.378, "underPct": 0.622},
    {"name": "Derik Queen", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.365, "underPct": 0.635},
    {"name": "Kevin Porter Jr.", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.356, "underPct": 0.644},
    {"name": "Caris LeVert", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.354, "underPct": 0.646},
    {"name": "Marvin Bagley III", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.313, "underPct": 0.687},
    {"name": "Amen Thompson", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.177, "underPct": 0.823},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.162, "underPct": 0.838},
];const prizepicksReboundsHitRates = [
    {"name": "Cade Cunningham", "line": 6.0, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.724, "underPct": 0.276},
    {"name": "P.J. Washington", "line": 6.0, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.653, "underPct": 0.347},
    {"name": "Jaden McDaniels", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.646, "underPct": 0.354},
    {"name": "Kel'el Ware", "line": 10.0, "l5": 0.0, "l10": 0.5, "l15": 0.53, "overPct": 0.612, "underPct": 0.388},
    {"name": "Rudy Gobert", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Darius Garland", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.555, "underPct": 0.445},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.548, "underPct": 0.452},
    {"name": "Kevin Durant", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.545, "underPct": 0.455},
    {"name": "Bam Adebayo", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.544, "underPct": 0.456},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.518, "underPct": 0.482},
    {"name": "Keegan Murray", "line": 7.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.506, "underPct": 0.494},
    {"name": "Kawhi Leonard", "line": 5.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.503, "underPct": 0.497},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.493, "underPct": 0.507},
    {"name": "Ivica Zubac", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.489, "underPct": 0.511},
    {"name": "Max Christie", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.489, "underPct": 0.511},
    {"name": "Jalen Johnson", "line": 11.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.475, "underPct": 0.525},
    {"name": "Russell Westbrook", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.467, "underPct": 0.533},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.464, "underPct": 0.536},
    {"name": "Amen Thompson", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.461, "underPct": 0.539},
    {"name": "Kyle Kuzma", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Naz Reid", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.424, "underPct": 0.576},
    {"name": "Trey Murphy III", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.419, "underPct": 0.581},
    {"name": "Danny Wolf", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.406, "underPct": 0.594},
    {"name": "Donovan Mitchell", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.395, "underPct": 0.605},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.394, "underPct": 0.606},
    {"name": "Evan Mobley", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.391, "underPct": 0.609},
    {"name": "De'Andre Hunter", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.387, "underPct": 0.613},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "Cooper Flagg", "line": 6.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.369, "underPct": 0.631},
    {"name": "Anthony Davis", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.354, "underPct": 0.646},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.322, "underPct": 0.678},
    {"name": "Onyeka Okongwu", "line": 8.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.305, "underPct": 0.695},
    {"name": "James Harden", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.302, "underPct": 0.698},
    {"name": "Tobias Harris", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.292, "underPct": 0.708},
    {"name": "Jericho Sims", "line": 7.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.249, "underPct": 0.751},
    {"name": "Kevin Porter Jr.", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.23, "underPct": 0.77},
    {"name": "Derik Queen", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.219, "underPct": 0.781},
    {"name": "Caris LeVert", "line": 2.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.206, "underPct": 0.794},
    {"name": "Marvin Bagley III", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.183, "underPct": 0.817},
    {"name": "Myles Turner", "line": 6.0, "l5": 0.0, "l10": 0.4, "l15": 0.4, "overPct": 0.177, "underPct": 0.823},
];const prizepicksBlocksHitRates = [
    {"name": "Nickeil Alexander-Walker", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.459, "underPct": 0.541},
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.564, "underPct": 0.436},
    {"name": "Ausar Thompson", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.45, "underPct": 0.55},
    {"name": "Jericho Sims", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.324, "underPct": 0.676},
    {"name": "John Collins", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.539, "underPct": 0.461},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.323, "underPct": 0.677},
    {"name": "Anthony Davis", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
];const prizepicksStealsHitRates = [
    {"name": "Noah Clowney", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.502, "underPct": 0.498},
    {"name": "Terance Mann", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.324, "underPct": 0.676},
    {"name": "Jalen Johnson", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.382, "underPct": 0.618},
    {"name": "Dean Wade", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.535, "underPct": 0.465},
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.505, "underPct": 0.495},
    {"name": "Bobby Portis", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Caris LeVert", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.431, "underPct": 0.569},
    {"name": "Jericho Sims", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.249, "underPct": 0.751},
    {"name": "Ivica Zubac", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.324, "underPct": 0.676},
    {"name": "Rudy Gobert", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.288, "underPct": 0.712},
    {"name": "Maxime Raynaud", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.257, "underPct": 0.743},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Saddiq Bey", "line": 25.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 31.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 17.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Nickeil Alexander-Walker", "line": 28.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Micah Peavy", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylon Tyson", "line": 21.0, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 44.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ziaire Williams", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 39.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Maxime Raynaud", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Duncan Robinson", "line": 15.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Rudy Gobert", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ausar Thompson", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tobias Harris", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 38.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kobe Sanders", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Porter Jr.", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Nembhard", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 15.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keegan Murray", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Hawkins", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dyson Daniels", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 39.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mike Conley", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 34.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Nicolas Batum", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Wiggins", "line": 25.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Davis", "line": 34.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Caris LeVert", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bobby Portis", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 34.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Reed Sheppard", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mouhamed Gueye", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyrese Martin", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Danny Wolf", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kel'el Ware", "line": 23.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Holiday", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden Ivey", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jericho Sims", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 20.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 29.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kennard", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zaccharie Risacher", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 15.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Okogie", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Duren", "line": 32.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Onyeka Okongwu", "line": 30.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPRHitRates = [
    {"name": "Naji Marshall", "line": 15.5, "l5": 1.0, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kawhi Leonard", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 28.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 19.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jose Alvarado", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 35.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Precious Achiuwa", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 26.0, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donte DiVincenzo", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Malik Monk", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kobe Sanders", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mike Conley", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keegan Murray", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Porter Jr.", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bobby Portis", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Danny Wolf", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 21.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Hawkins", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 19.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Caris LeVert", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 10.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nicolas Batum", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 10.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mouhamed Gueye", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Davis", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Wiggins", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Maxime Raynaud", "line": 16.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rudy Gobert", "line": 20.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Terance Mann", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jericho Sims", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Duncan Robinson", "line": 13.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ausar Thompson", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dean Wade", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Marvin Bagley III", "line": 21.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Drew Eubanks", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 30.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kel'el Ware", "line": 22.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kennard", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Andre Hunter", "line": 19.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden Ivey", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 33.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 27.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPAHitRates = [
    {"name": "Saddiq Bey", "line": 19.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 13.0, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cooper Flagg", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 35.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jose Alvarado", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nicolas Batum", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tobias Harris", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Porter Jr.", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Duncan Robinson", "line": 13.0, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Darius Garland", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Hawkins", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Nembhard", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Wiggins", "line": 20.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 10.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 25.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Caris LeVert", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 13.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Danny Wolf", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dean Wade", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mouhamed Gueye", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zaccharie Risacher", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 22.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Davis", "line": 23.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kennard", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kel'el Ware", "line": 13.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 18.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden Ivey", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Okogie", "line": 8.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
];const prizepicksRAHitRates = [
    {"name": "P.J. Washington", "line": 7.5, "l5": 1.0, "l10": 0.9, "l15": 0.87, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Evan Mobley", "line": 14.0, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Yves Missi", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 19.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keegan Murray", "line": 9.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylon Tyson", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Davis", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 10.0, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Saddiq Bey", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Danny Wolf", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Darius Garland", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Onyeka Okongwu", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bam Adebayo", "line": 13.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kennard", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 15.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 11.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "James Harden", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bobby Portis", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zaccharie Risacher", "line": 4.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Porter Jr.", "line": 11.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Russell Westbrook", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 5.5, "l5": 0.2, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 11.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 7.0, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const prizepicksTurnoversHitRates = [
    {"name": "Darius Garland", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drew Eubanks", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Okogie", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mouhamed Gueye", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Micah Peavy", "line": 0.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "P.J. Washington", "line": 1.5, "l5": 1.0, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Trey Murphy III", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Martin", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mouhamed Gueye", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Davis", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Okogie", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zaccharie Risacher", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogPointsHitRates = [
    {"name": "P.J. Washington", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.84, "underPct": 0.16},
    {"name": "Rudy Gobert", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.815, "underPct": 0.185},
    {"name": "Micah Peavy", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.804, "underPct": 0.196},
    {"name": "Jaylon Tyson", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.796, "underPct": 0.204},
    {"name": "Naji Marshall", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.72, "underPct": 0.28},
    {"name": "Noah Clowney", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.695, "underPct": 0.305},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.694, "underPct": 0.306},
    {"name": "Bryce McGowens", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.682, "underPct": 0.318},
    {"name": "Ryan Nembhard", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.672, "underPct": 0.328},
    {"name": "Nickeil Alexander-Walker", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.668, "underPct": 0.332},
    {"name": "Evan Mobley", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.661, "underPct": 0.339},
    {"name": "Donovan Mitchell", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.657, "underPct": 0.343},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.642, "underPct": 0.358},
    {"name": "Reed Sheppard", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.639, "underPct": 0.361},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.615, "underPct": 0.385},
    {"name": "Tyrese Martin", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.614, "underPct": 0.386},
    {"name": "Saddiq Bey", "line": 17.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.586, "underPct": 0.414},
    {"name": "Cade Cunningham", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.57, "underPct": 0.43},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.569, "underPct": 0.431},
    {"name": "Amen Thompson", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.567, "underPct": 0.433},
    {"name": "John Collins", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.562, "underPct": 0.438},
    {"name": "Norman Powell", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.538, "underPct": 0.462},
    {"name": "Ivica Zubac", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.526, "underPct": 0.474},
    {"name": "Myles Turner", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.52, "underPct": 0.48},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.504, "underPct": 0.496},
    {"name": "Donte DiVincenzo", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.497, "underPct": 0.503},
    {"name": "Brandon Williams", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.493, "underPct": 0.507},
    {"name": "Caris LeVert", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.475, "underPct": 0.525},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.475, "underPct": 0.525},
    {"name": "Ausar Thompson", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.462, "underPct": 0.538},
    {"name": "Julius Randle", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.461, "underPct": 0.539},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.456, "underPct": 0.544},
    {"name": "Bam Adebayo", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.443, "underPct": 0.557},
    {"name": "Nicolas Batum", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.425, "underPct": 0.575},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.424, "underPct": 0.576},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.422, "underPct": 0.578},
    {"name": "Kevin Porter Jr.", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.408, "underPct": 0.592},
    {"name": "James Harden", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.386, "underPct": 0.614},
    {"name": "Maxime Raynaud", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.37, "underPct": 0.63},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.365, "underPct": 0.635},
    {"name": "Mike Conley", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.364, "underPct": 0.636},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.361, "underPct": 0.639},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.33, "underPct": 0.67},
    {"name": "Darius Garland", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.305, "underPct": 0.695},
    {"name": "Zaccharie Risacher", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.284, "underPct": 0.716},
    {"name": "Danny Wolf", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.278, "underPct": 0.722},
    {"name": "Jaden Ivey", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.251, "underPct": 0.749},
    {"name": "Onyeka Okongwu", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.247, "underPct": 0.753},
    {"name": "Luke Kennard", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.227, "underPct": 0.773},
    {"name": "Mouhamed Gueye", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.19, "underPct": 0.81},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.177, "underPct": 0.823},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.147, "underPct": 0.853},
    {"name": "Marvin Bagley III", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.139, "underPct": 0.861},
];const underdogAssistsHitRates = [
    {"name": "Rudy Gobert", "line": 1.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.695, "underPct": 0.305},
    {"name": "Naz Reid", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.541, "underPct": 0.459},
    {"name": "Darius Garland", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.535, "underPct": 0.465},
    {"name": "Keegan Murray", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.524, "underPct": 0.476},
    {"name": "Mike Conley", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.48, "underPct": 0.52},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.471, "underPct": 0.529},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.455, "underPct": 0.545},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.417, "underPct": 0.583},
    {"name": "Kevin Porter Jr.", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.356, "underPct": 0.644},
    {"name": "Caris LeVert", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.354, "underPct": 0.646},
    {"name": "Marvin Bagley III", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.313, "underPct": 0.687},
    {"name": "Amen Thompson", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.177, "underPct": 0.823},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.162, "underPct": 0.838},
];const underdogReboundsHitRates = [
    {"name": "Duncan Robinson", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.712, "underPct": 0.288},
    {"name": "Yves Missi", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.646, "underPct": 0.354},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.602, "underPct": 0.398},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.597, "underPct": 0.403},
    {"name": "Darius Garland", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.555, "underPct": 0.445},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.553, "underPct": 0.447},
    {"name": "Ziaire Williams", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.516, "underPct": 0.484},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.493, "underPct": 0.507},
    {"name": "Klay Thompson", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.48, "underPct": 0.52},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.464, "underPct": 0.536},
    {"name": "John Collins", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.395, "underPct": 0.605},
    {"name": "Ryan Nembhard", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.395, "underPct": 0.605},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.394, "underPct": 0.606},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.322, "underPct": 0.678},
    {"name": "Kevin Porter Jr.", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.23, "underPct": 0.77},
    {"name": "Marvin Bagley III", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.183, "underPct": 0.817},
];const underdogBlocksHitRates = [
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.564, "underPct": 0.436},
];const underdogStealsHitRates = [
    {"name": "Jalen Johnson", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.382, "underPct": 0.618},
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.505, "underPct": 0.495},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Saddiq Bey", "line": 26.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 31.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 28.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Johnson", "line": 44.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Edwards", "line": 39.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Noah Clowney", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "John Collins", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 38.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keegan Murray", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Maxime Raynaud", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ausar Thompson", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Nembhard", "line": 17.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Hawkins", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 40.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Darius Garland", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 34.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Davis", "line": 33.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Danny Wolf", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tristan Vukcevic", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mouhamed Gueye", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Caris LeVert", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 34.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bryce McGowens", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kel'el Ware", "line": 23.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Holiday", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Okogie", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 29.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden Ivey", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyle Kuzma", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 30.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const underdogPRHitRates = [
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Saddiq Bey", "line": 23.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 19.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Johnson", "line": 35.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tobias Harris", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Porter Jr.", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Amen Thompson", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Darius Garland", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Davis", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 33.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 26.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandin Podziemski", "line": 21.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
];const underdogPAHitRates = [
    {"name": "Saddiq Bey", "line": 19.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 35.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Wiggins", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julius Randle", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Davis", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const underdogRAHitRates = [
    {"name": "Noah Clowney", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylon Tyson", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Rollins", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Davis", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
];const underdogTurnoversHitRates = [
    {"name": "Darius Garland", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
];const underdogBlocksStealsHitRates = [
    {"name": "Anthony Davis", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
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

