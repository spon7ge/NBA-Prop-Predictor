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
    {"name1": "Day'Ron Sharpe", "name2": "Jaden McDaniels", "line1": 6.5, "line2": 13.5, "prediction1": 14.18, "prediction2": 21.78, "side1": "over", "side2": "over", "recommendation": 1, "ev": 142.54, "kelly": 0.713, "sigma1": "Med", "sigma2": "Med", "prob1": 0.902, "prob2": 0.914, "hitRate1": 25.6, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 44.0, "l5_2": 0.2, "l15_2": 0.53},
    {"name1": "Payton Pritchard", "name2": "Donte DiVincenzo", "line1": 22.5, "line2": 13.5, "prediction1": 16.91, "prediction2": 21.12, "side1": "under", "side2": "over", "recommendation": 1, "ev": 126.54, "kelly": 0.633, "sigma1": "Med", "sigma2": "Med", "prob1": 0.87, "prob2": 0.885, "hitRate1": 80.9, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 57.1, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Tyrese Martin", "name2": "Derik Queen", "line1": 10.0, "line2": 14.5, "prediction1": 18.18, "prediction2": 23.16, "side1": "over", "side2": "over", "recommendation": 1, "ev": 122.72, "kelly": 0.614, "sigma1": "High", "sigma2": "High", "prob1": 0.87, "prob2": 0.871, "hitRate1": 44.2, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 45.3, "l5_2": 0.4, "l15_2": 0.4},
    {"name1": "Svi Mykhailiuk", "name2": "Mike Conley", "line1": 8.5, "line2": 4.5, "prediction1": 13.83, "prediction2": 8.37, "side1": "over", "side2": "over", "recommendation": 0, "ev": 109.98, "kelly": 0.55, "sigma1": "Med", "sigma2": "Low", "prob1": 0.825, "prob2": 0.865, "hitRate1": 63.3, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 49.9, "l5_2": 0.6, "l15_2": 0.8},
    {"name1": "Isaiah Collier", "name2": "Saddiq Bey", "line1": 7.0, "line2": 14.5, "prediction1": 11.61, "prediction2": 23.07, "side1": "over", "side2": "over", "recommendation": 1, "ev": 108.86, "kelly": 0.544, "sigma1": "Low", "sigma2": "High", "prob1": 0.822, "prob2": 0.864, "hitRate1": 24.6, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 77.9, "l5_2": 1.0, "l15_2": 0.67},
    {"name1": "Ziaire Williams", "name2": "Deandre Ayton", "line1": 10.5, "line2": 15.5, "prediction1": 15.73, "prediction2": 11.63, "side1": "over", "side2": "under", "recommendation": 0, "ev": 92.17, "kelly": 0.461, "sigma1": "Med", "sigma2": "Low", "prob1": 0.807, "prob2": 0.81, "hitRate1": 11.8, "l5_1": 0.0, "l15_1": 0.4, "hitRate2": 49.6, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "VJ Edgecombe", "name2": "Jordan Hawkins", "line1": 10.5, "line2": 7.0, "prediction1": 17.48, "prediction2": 11.66, "side1": "over", "side2": "over", "recommendation": 1, "ev": 88.29, "kelly": 0.441, "sigma1": "High", "sigma2": "Med", "prob1": 0.8, "prob2": 0.8, "hitRate1": 78.1, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 43.3, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Quinten Post", "name2": "Brice Sensabaugh", "line1": 7.5, "line2": 9.5, "prediction1": 11.88, "prediction2": 13.94, "side1": "over", "side2": "over", "recommendation": 1, "ev": 74.25, "kelly": 0.371, "sigma1": "Med", "sigma2": "Med", "prob1": 0.772, "prob2": 0.768, "hitRate1": 60.5, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 37.3, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Draymond Green", "name2": "Naz Reid", "line1": 9.5, "line2": 13.5, "prediction1": 13.57, "prediction2": 17.43, "side1": "over", "side2": "over", "recommendation": 0, "ev": 54.08, "kelly": 0.27, "sigma1": "High", "sigma2": "Med", "prob1": 0.724, "prob2": 0.724, "hitRate1": 46.0, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 59.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Quentin Grimes", "name2": "Josh Minott", "line1": 16.5, "line2": 7.5, "prediction1": 13.2, "prediction2": 10.25, "side1": "under", "side2": "over", "recommendation": 0, "ev": 49.6, "kelly": 0.248, "sigma1": "High", "sigma2": "Low", "prob1": 0.711, "prob2": 0.715, "hitRate1": 39.7, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 42.7, "l5_2": 0.6, "l15_2": 0.53},
];const prizepicksTriosData = [
    {"name1": "Payton Pritchard", "name2": "Day'Ron Sharpe", "name3": "Jaden McDaniels", "line1": 22.5, "line2": 6.5, "line3": 13.5, "prediction1": 16.91, "prediction2": 14.18, "prediction3": 21.78, "side1": "under", "side2": "over", "side3": "over", "recommendation": 1, "ev": 287.74, "kelly": 0.575, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "prob1": 0.87, "prob2": 0.902, "prob3": 0.914, "hitRate1": 80.9, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 25.6, "l5_2": 0.2, "l15_2": 0.27, "hitRate3": 44.0, "l5_3": 0.2, "l15_3": 0.53},
    {"name1": "Tyrese Martin", "name2": "Deandre Ayton", "name3": "Donte DiVincenzo", "line1": 10.0, "line2": 15.5, "line3": 13.5, "prediction1": 18.18, "prediction2": 11.63, "prediction3": 21.12, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 236.83, "kelly": 0.474, "sigma1": "High", "sigma2": "Low", "sigma3": "Med", "prob1": 0.87, "prob2": 0.81, "prob3": 0.885, "hitRate1": 44.2, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 49.6, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 57.1, "l5_3": 0.8, "l15_3": 0.6},
    {"name1": "VJ Edgecombe", "name2": "Svi Mykhailiuk", "name3": "Derik Queen", "line1": 10.5, "line2": 8.5, "line3": 14.5, "prediction1": 17.48, "prediction2": 13.83, "prediction3": 23.16, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 210.71, "kelly": 0.421, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.8, "prob2": 0.825, "prob3": 0.871, "hitRate1": 78.1, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 63.3, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 45.3, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Quinten Post", "name2": "Isaiah Collier", "name3": "Saddiq Bey", "line1": 7.5, "line2": 7.0, "line3": 14.5, "prediction1": 11.88, "prediction2": 11.61, "prediction3": 23.07, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 196.09, "kelly": 0.392, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.772, "prob2": 0.822, "prob3": 0.864, "hitRate1": 60.5, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 24.6, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 77.9, "l5_3": 1.0, "l15_3": 0.67},
    {"name1": "Draymond Green", "name2": "Ziaire Williams", "name3": "Jordan Hawkins", "line1": 9.5, "line2": 10.5, "line3": 7.0, "prediction1": 13.57, "prediction2": 15.73, "prediction3": 11.66, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 152.42, "kelly": 0.305, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.724, "prob2": 0.807, "prob3": 0.8, "hitRate1": 46.0, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 11.8, "l5_2": 0.0, "l15_2": 0.4, "hitRate3": 43.3, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Josh Minott", "name2": "Brice Sensabaugh", "name3": "Naz Reid", "line1": 7.5, "line2": 9.5, "line3": 13.5, "prediction1": 10.25, "prediction2": 13.94, "prediction3": 17.43, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 114.66, "kelly": 0.229, "sigma1": "Low", "sigma2": "Med", "sigma3": "Med", "prob1": 0.715, "prob2": 0.768, "prob3": 0.724, "hitRate1": 42.7, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 37.3, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 59.8, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Quentin Grimes", "name2": "Neemias Queta", "name3": "Ja'Kobe Walter", "line1": 16.5, "line2": 12.5, "line3": 7.5, "prediction1": 13.2, "prediction2": 16.38, "prediction3": 10.46, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 91.19, "kelly": 0.182, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.711, "prob2": 0.709, "prob3": 0.702, "hitRate1": 39.7, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 39.2, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 41.3, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Moses Moody", "name2": "Kyshawn George", "name3": "Danny Wolf", "line1": 12.5, "line2": 14.5, "line3": 11.5, "prediction1": 16.79, "prediction2": 18.97, "prediction3": 15.25, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 84.57, "kelly": 0.169, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.702, "prob2": 0.699, "prob3": 0.696, "hitRate1": 17.3, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 43.6, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 3.6, "l5_3": 0.2, "l15_3": 0.07},
    {"name1": "Brandin Podziemski", "name2": "Derrick White", "name3": "Trey Murphy III", "line1": 14.5, "line2": 21.5, "line3": 20.5, "prediction1": 18.29, "prediction2": 18.88, "prediction3": 17.31, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 72.01, "kelly": 0.144, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.687, "prob2": 0.683, "prob3": 0.679, "hitRate1": 35.2, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 86.1, "l5_2": 0.4, "l15_2": 0.2, "hitRate3": 46.3, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Jared McCain", "name2": "Cam Whitmore", "name3": "Jeremiah Fears", "line1": 12.5, "line2": 9.5, "line3": 16.0, "prediction1": 10.61, "prediction2": 12.57, "prediction3": 19.9, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 65.04, "kelly": 0.13, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.672, "prob2": 0.676, "prob3": 0.672, "hitRate1": 68.5, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 48.5, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 56.3, "l5_3": 0.4, "l15_3": 0.47},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Ben Saraf", "name2": "Jaden McDaniels", "line1": 6.5, "line2": 13.5, "prediction1": 15.61, "prediction2": 21.78, "side1": "over", "side2": "over", "recommendation": 1, "ev": 150.81, "kelly": 0.754, "sigma1": "Med", "sigma2": "Med", "prob1": 0.933, "prob2": 0.914, "hitRate1": 22.8, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 44.0, "l5_2": 0.2, "l15_2": 0.53},
    {"name1": "Andre Drummond", "name2": "Donte DiVincenzo", "line1": 5.5, "line2": 13.5, "prediction1": 11.68, "prediction2": 21.12, "side1": "over", "side2": "over", "recommendation": 1, "ev": 131.63, "kelly": 0.658, "sigma1": "Low", "sigma2": "Med", "prob1": 0.89, "prob2": 0.885, "hitRate1": 81.5, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 57.1, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Payton Pritchard", "name2": "Derik Queen", "line1": 22.5, "line2": 14.5, "prediction1": 16.91, "prediction2": 23.16, "side1": "under", "side2": "over", "recommendation": 1, "ev": 122.84, "kelly": 0.614, "sigma1": "Med", "sigma2": "High", "prob1": 0.87, "prob2": 0.871, "hitRate1": 80.9, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 45.3, "l5_2": 0.4, "l15_2": 0.4},
    {"name1": "Day'Ron Sharpe", "name2": "Mike Conley", "line1": 7.5, "line2": 4.5, "prediction1": 14.18, "prediction2": 8.37, "side1": "over", "side2": "over", "recommendation": 0, "ev": 120.55, "kelly": 0.603, "sigma1": "Med", "sigma2": "Low", "prob1": 0.867, "prob2": 0.865, "hitRate1": 14.7, "l5_1": 0.0, "l15_1": 0.13, "hitRate2": 49.9, "l5_2": 0.6, "l15_2": 0.8},
    {"name1": "Svi Mykhailiuk", "name2": "Saddiq Bey", "line1": 8.5, "line2": 14.5, "prediction1": 13.83, "prediction2": 23.07, "side1": "over", "side2": "over", "recommendation": 1, "ev": 109.79, "kelly": 0.549, "sigma1": "Med", "sigma2": "High", "prob1": 0.825, "prob2": 0.864, "hitRate1": 63.3, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 77.9, "l5_2": 1.0, "l15_2": 0.67},
    {"name1": "Tyrese Martin", "name2": "Deandre Ayton", "line1": 11.5, "line2": 15.5, "prediction1": 18.18, "prediction2": 11.63, "side1": "over", "side2": "under", "recommendation": 0, "ev": 94.63, "kelly": 0.473, "sigma1": "High", "sigma2": "Low", "prob1": 0.817, "prob2": 0.81, "hitRate1": 32.6, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 49.6, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "VJ Edgecombe", "name2": "Ziaire Williams", "line1": 10.5, "line2": 10.5, "prediction1": 17.48, "prediction2": 15.73, "side1": "over", "side2": "over", "recommendation": 1, "ev": 89.92, "kelly": 0.45, "sigma1": "High", "sigma2": "Med", "prob1": 0.8, "prob2": 0.807, "hitRate1": 78.1, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 11.8, "l5_2": 0.0, "l15_2": 0.4},
    {"name1": "Paul George", "name2": "Brice Sensabaugh", "line1": 15.5, "line2": 9.5, "prediction1": 22.27, "prediction2": 13.94, "side1": "over", "side2": "over", "recommendation": 1, "ev": 78.95, "kelly": 0.395, "sigma1": "High", "sigma2": "Med", "prob1": 0.793, "prob2": 0.768, "hitRate1": 32.5, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 37.3, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Quinten Post", "name2": "Josh Minott", "line1": 7.5, "line2": 7.5, "prediction1": 11.88, "prediction2": 10.25, "side1": "over", "side2": "over", "recommendation": 0, "ev": 62.29, "kelly": 0.311, "sigma1": "Med", "sigma2": "Low", "prob1": 0.772, "prob2": 0.715, "hitRate1": 60.5, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 42.7, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Draymond Green", "name2": "Neemias Queta", "line1": 9.5, "line2": 12.5, "prediction1": 13.57, "prediction2": 16.38, "side1": "over", "side2": "over", "recommendation": 0, "ev": 50.82, "kelly": 0.254, "sigma1": "High", "sigma2": "High", "prob1": 0.724, "prob2": 0.709, "hitRate1": 46.0, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 39.2, "l5_2": 0.4, "l15_2": 0.33},
];const underdogTriosData = [
    {"name1": "Andre Drummond", "name2": "Ben Saraf", "name3": "Jaden McDaniels", "line1": 5.5, "line2": 6.5, "line3": 13.5, "prediction1": 11.68, "prediction2": 15.61, "prediction3": 21.78, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 309.96, "kelly": 0.62, "sigma1": "Low", "sigma2": "Med", "sigma3": "Med", "prob1": 0.89, "prob2": 0.933, "prob3": 0.914, "hitRate1": 81.5, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 22.8, "l5_2": 0.4, "l15_2": 0.2, "hitRate3": 44.0, "l5_3": 0.2, "l15_3": 0.53},
    {"name1": "Payton Pritchard", "name2": "Day'Ron Sharpe", "name3": "Donte DiVincenzo", "line1": 22.5, "line2": 7.5, "line3": 13.5, "prediction1": 16.91, "prediction2": 14.18, "prediction3": 21.12, "side1": "under", "side2": "over", "side3": "over", "recommendation": 1, "ev": 260.73, "kelly": 0.521, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "prob1": 0.87, "prob2": 0.867, "prob3": 0.885, "hitRate1": 80.9, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 14.7, "l5_2": 0.0, "l15_2": 0.13, "hitRate3": 57.1, "l5_3": 0.8, "l15_3": 0.6},
    {"name1": "Svi Mykhailiuk", "name2": "Deandre Ayton", "name3": "Derik Queen", "line1": 8.5, "line2": 15.5, "line3": 14.5, "prediction1": 13.83, "prediction2": 11.63, "prediction3": 23.16, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 214.4, "kelly": 0.429, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.825, "prob2": 0.81, "prob3": 0.871, "hitRate1": 63.3, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 49.6, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 45.3, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "VJ Edgecombe", "name2": "Tyrese Martin", "name3": "Mike Conley", "line1": 10.5, "line2": 11.5, "line3": 4.5, "prediction1": 17.48, "prediction2": 18.18, "prediction3": 8.37, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 205.7, "kelly": 0.411, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.8, "prob2": 0.817, "prob3": 0.865, "hitRate1": 78.1, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 32.6, "l5_2": 0.2, "l15_2": 0.33, "hitRate3": 49.9, "l5_3": 0.6, "l15_3": 0.8},
    {"name1": "Paul George", "name2": "Ziaire Williams", "name3": "Saddiq Bey", "line1": 15.5, "line2": 10.5, "line3": 14.5, "prediction1": 22.27, "prediction2": 15.73, "prediction3": 23.07, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 198.63, "kelly": 0.397, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.793, "prob2": 0.807, "prob3": 0.864, "hitRate1": 32.5, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 11.8, "l5_2": 0.0, "l15_2": 0.4, "hitRate3": 77.9, "l5_3": 1.0, "l15_3": 0.67},
    {"name1": "Quinten Post", "name2": "Josh Minott", "name3": "Brice Sensabaugh", "line1": 7.5, "line2": 7.5, "line3": 9.5, "prediction1": 11.88, "prediction2": 10.25, "prediction3": 13.94, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 128.88, "kelly": 0.258, "sigma1": "Med", "sigma2": "Low", "sigma3": "Med", "prob1": 0.772, "prob2": 0.715, "prob3": 0.768, "hitRate1": 60.5, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 42.7, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 37.3, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Draymond Green", "name2": "Neemias Queta", "name3": "Danny Wolf", "line1": 9.5, "line2": 12.5, "line3": 11.5, "prediction1": 13.57, "prediction2": 16.38, "prediction3": 15.25, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 92.84, "kelly": 0.186, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.724, "prob2": 0.709, "prob3": 0.696, "hitRate1": 46.0, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 39.2, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 3.6, "l5_3": 0.2, "l15_3": 0.07},
    {"name1": "Quentin Grimes", "name2": "Kyshawn George", "name3": "Terance Mann", "line1": 16.5, "line2": 14.5, "line3": 9.5, "prediction1": 13.2, "prediction2": 18.97, "prediction3": 12.22, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 86.92, "kelly": 0.174, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.711, "prob2": 0.699, "prob3": 0.696, "hitRate1": 39.7, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 43.6, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 13.8, "l5_3": 0.2, "l15_3": 0.4},
    {"name1": "Brandin Podziemski", "name2": "Sandro Mamukelashvili", "name3": "Trey Murphy III", "line1": 14.5, "line2": 12.5, "line3": 20.5, "prediction1": 18.29, "prediction2": 10.62, "prediction3": 17.31, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 73.43, "kelly": 0.147, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "prob1": 0.687, "prob2": 0.688, "prob3": 0.679, "hitRate1": 35.2, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 71.6, "l5_2": 0.2, "l15_2": 0.27, "hitRate3": 46.3, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Joel Embiid", "name2": "Noah Clowney", "name3": "Jeremiah Fears", "line1": 18.5, "line2": 16.5, "line3": 16.5, "prediction1": 22.25, "prediction2": 20.19, "prediction3": 19.9, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 57.72, "kelly": 0.115, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.664, "prob2": 0.678, "prob3": 0.649, "hitRate1": 57.7, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 53.2, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 56.3, "l5_3": 0.4, "l15_3": 0.47},
];const prizepicksPointsHitRates = [
    {"name": "Anthony Edwards", "line": 30.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.803, "underPct": 0.197},
    {"name": "Tyrese Maxey", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.801, "underPct": 0.199},
    {"name": "VJ Edgecombe", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.781, "underPct": 0.219},
    {"name": "Saddiq Bey", "line": 14.5, "l5": 1.0, "l10": 0.6, "l15": 0.67, "overPct": 0.779, "underPct": 0.221},
    {"name": "Scottie Barnes", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.675, "underPct": 0.325},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.663, "underPct": 0.337},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.633, "underPct": 0.367},
    {"name": "Quinten Post", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.605, "underPct": 0.395},
    {"name": "Quentin Grimes", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.603, "underPct": 0.397},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.598, "underPct": 0.402},
    {"name": "Gary Payton II", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.595, "underPct": 0.405},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.593, "underPct": 0.407},
    {"name": "Joel Embiid", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.577, "underPct": 0.423},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.571, "underPct": 0.429},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.566, "underPct": 0.434},
    {"name": "Jeremiah Fears", "line": 16.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.563, "underPct": 0.437},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.539, "underPct": 0.461},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.537, "underPct": 0.463},
    {"name": "Deandre Ayton", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.504, "underPct": 0.496},
    {"name": "Mike Conley", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.499, "underPct": 0.501},
    {"name": "Cam Whitmore", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Micah Peavy", "line": 7.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.481, "underPct": 0.519},
    {"name": "Justin Edwards", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.466, "underPct": 0.534},
    {"name": "Draymond Green", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.46, "underPct": 0.54},
    {"name": "Derik Queen", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.453, "underPct": 0.547},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.447, "underPct": 0.553},
    {"name": "Tyrese Martin", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.442, "underPct": 0.558},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.44, "underPct": 0.56},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.436, "underPct": 0.564},
    {"name": "Kyshawn George", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.436, "underPct": 0.564},
    {"name": "Jordan Hawkins", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.433, "underPct": 0.567},
    {"name": "Josh Minott", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.424, "underPct": 0.576},
    {"name": "Ja'Kobe Walter", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.413, "underPct": 0.587},
    {"name": "Austin Reaves", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.406, "underPct": 0.594},
    {"name": "Neemias Queta", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.392, "underPct": 0.608},
    {"name": "Sandro Mamukelashvili", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.39, "underPct": 0.61},
    {"name": "Brice Sensabaugh", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.373, "underPct": 0.627},
    {"name": "Brandin Podziemski", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.352, "underPct": 0.648},
    {"name": "Khris Middleton", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.325, "underPct": 0.675},
    {"name": "Jared McCain", "line": 12.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.315, "underPct": 0.685},
    {"name": "Dalton Knecht", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.314, "underPct": 0.686},
    {"name": "Julius Randle", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.301, "underPct": 0.699},
    {"name": "Marvin Bagley III", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.294, "underPct": 0.706},
    {"name": "Kyle Filipowski", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.29, "underPct": 0.71},
    {"name": "Buddy Hield", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.267, "underPct": 0.733},
    {"name": "Al Horford", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.259, "underPct": 0.741},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.259, "underPct": 0.741},
    {"name": "Day'Ron Sharpe", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.256, "underPct": 0.744},
    {"name": "Isaiah Collier", "line": 7.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.246, "underPct": 0.754},
    {"name": "Rui Hachimura", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.246, "underPct": 0.754},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.242, "underPct": 0.758},
    {"name": "Tristan Vukcevic", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.236, "underPct": 0.764},
    {"name": "Jordan Walsh", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.232, "underPct": 0.768},
    {"name": "Payton Pritchard", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.191, "underPct": 0.809},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.173, "underPct": 0.827},
    {"name": "Jake LaRavia", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.16, "underPct": 0.84},
    {"name": "Anfernee Simons", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.141, "underPct": 0.859},
    {"name": "Derrick White", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.139, "underPct": 0.861},
    {"name": "Jamal Shead", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.136, "underPct": 0.864},
    {"name": "Collin Murray-Boyles", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.123, "underPct": 0.877},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.118, "underPct": 0.882},
    {"name": "Will Richard", "line": 9.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.103, "underPct": 0.897},
    {"name": "Gabe Vincent", "line": 6.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.088, "underPct": 0.912},
    {"name": "Terance Mann", "line": 10.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.077, "underPct": 0.923},
    {"name": "Sam Hauser", "line": 11.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.054, "underPct": 0.946},
    {"name": "LeBron James", "line": 22.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.049, "underPct": 0.951},
    {"name": "Danny Wolf", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.036, "underPct": 0.964},
];const prizepicksAssistsHitRates = [
    {"name": "Khris Middleton", "line": 3.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.678, "underPct": 0.322},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.609, "underPct": 0.391},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.602, "underPct": 0.398},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.56, "underPct": 0.44},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.543, "underPct": 0.457},
    {"name": "Mike Conley", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.541, "underPct": 0.459},
    {"name": "Derik Queen", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.54, "underPct": 0.46},
    {"name": "Tyrese Maxey", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.473, "underPct": 0.527},
    {"name": "Tristan Vukcevic", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.469, "underPct": 0.531},
    {"name": "Jamal Shead", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.459, "underPct": 0.541},
    {"name": "Jose Alvarado", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.446, "underPct": 0.554},
    {"name": "Bryce McGowens", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.428, "underPct": 0.572},
    {"name": "Immanuel Quickley", "line": 7.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.387, "underPct": 0.613},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.356, "underPct": 0.644},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.354, "underPct": 0.646},
    {"name": "Day'Ron Sharpe", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.348, "underPct": 0.652},
    {"name": "Terance Mann", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.348, "underPct": 0.652},
    {"name": "LeBron James", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.327, "underPct": 0.673},
    {"name": "Keyonte George", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.325, "underPct": 0.675},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.301, "underPct": 0.699},
    {"name": "Jordan Walsh", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.298, "underPct": 0.702},
    {"name": "Anfernee Simons", "line": 3.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.278, "underPct": 0.722},
    {"name": "Derrick White", "line": 6.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.245, "underPct": 0.755},
    {"name": "Brandin Podziemski", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.166, "underPct": 0.834},
    {"name": "Payton Pritchard", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.134, "underPct": 0.866},
    {"name": "Austin Reaves", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.114, "underPct": 0.886},
];const prizepicksReboundsHitRates = [
    {"name": "Draymond Green", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.666, "underPct": 0.334},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.651, "underPct": 0.349},
    {"name": "Gary Payton II", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.633, "underPct": 0.367},
    {"name": "Austin Reaves", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.633, "underPct": 0.367},
    {"name": "Terance Mann", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.558, "underPct": 0.442},
    {"name": "Jaden McDaniels", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jeremiah Fears", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Brandon Ingram", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.532, "underPct": 0.468},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.526, "underPct": 0.474},
    {"name": "Rudy Gobert", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.516, "underPct": 0.484},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.483, "underPct": 0.517},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.471, "underPct": 0.529},
    {"name": "Tyrese Martin", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.465, "underPct": 0.535},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.456, "underPct": 0.544},
    {"name": "Scottie Barnes", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.45, "underPct": 0.55},
    {"name": "Jonathan Kuminga", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.448, "underPct": 0.552},
    {"name": "Immanuel Quickley", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.43, "underPct": 0.57},
    {"name": "Khris Middleton", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.427, "underPct": 0.573},
    {"name": "Derrick White", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.421, "underPct": 0.579},
    {"name": "Trey Murphy III", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Jared McCain", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.41, "underPct": 0.59},
    {"name": "Sandro Mamukelashvili", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Deandre Ayton", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.397, "underPct": 0.603},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.384, "underPct": 0.616},
    {"name": "Anfernee Simons", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.372, "underPct": 0.628},
    {"name": "Payton Pritchard", "line": 5.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.366, "underPct": 0.634},
    {"name": "Keyonte George", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.361, "underPct": 0.639},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.335, "underPct": 0.665},
    {"name": "Sam Hauser", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.324, "underPct": 0.676},
    {"name": "Rui Hachimura", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.321, "underPct": 0.679},
    {"name": "Josh Minott", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.307, "underPct": 0.693},
    {"name": "Derik Queen", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.301, "underPct": 0.699},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.297, "underPct": 0.703},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.294, "underPct": 0.706},
    {"name": "Joel Embiid", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.289, "underPct": 0.711},
    {"name": "Jordan Walsh", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.287, "underPct": 0.713},
    {"name": "Jake LaRavia", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.47, "overPct": 0.232, "underPct": 0.768},
    {"name": "Kyle Filipowski", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.227, "underPct": 0.773},
    {"name": "Isaiah Collier", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.215, "underPct": 0.785},
    {"name": "Ace Bailey", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.204, "underPct": 0.796},
    {"name": "Collin Murray-Boyles", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.186, "underPct": 0.814},
    {"name": "Marvin Bagley III", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.151, "underPct": 0.849},
    {"name": "LeBron James", "line": 6.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.124, "underPct": 0.876},
    {"name": "Danny Wolf", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.09, "underPct": 0.91},
];const prizepicksBlocksHitRates = [
    {"name": "Kyshawn George", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.513, "underPct": 0.487},
    {"name": "Anthony Edwards", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.52, "underPct": 0.48},
];const prizepicksStealsHitRates = [
    {"name": "Al Horford", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.646, "underPct": 0.354},
    {"name": "Marvin Bagley III", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.423, "underPct": 0.577},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.281, "underPct": 0.719},
    {"name": "Kyle Filipowski", "line": 0.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.203, "underPct": 0.797},
    {"name": "Svi Mykhailiuk", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.253, "underPct": 0.747},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.556, "underPct": 0.444},
    {"name": "Bryce McGowens", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.375, "underPct": 0.625},
    {"name": "Micah Peavy", "line": 0.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.604, "underPct": 0.396},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Anthony Edwards", "line": 39.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 23.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jared McCain", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Draymond Green", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Micah Peavy", "line": 12.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brice Sensabaugh", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Minott", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 41.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 35.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tristan Vukcevic", "line": 17.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Justin Edwards", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Al Horford", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Joel Embiid", "line": 28.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dalton Knecht", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Murray-Boyles", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deandre Ayton", "line": 27.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaxson Hayes", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 19.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 22.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 40.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quinten Post", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Whitmore", "line": 15.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 22.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Walsh", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Buddy Hield", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 24.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 35.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 18.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gabe Vincent", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 35.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derrick White", "line": 32.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ja'Kobe Walter", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Danny Wolf", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jake LaRavia", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.47, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Shead", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anfernee Simons", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sandro Mamukelashvili", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sam Hauser", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 34.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 36.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Day'Ron Sharpe", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Collier", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Filipowski", "line": 23.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Terance Mann", "line": 18.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Rui Hachimura", "line": 19.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jonathan Kuminga", "line": 22.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const prizepicksPRHitRates = [
    {"name": "Saddiq Bey", "line": 21.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Micah Peavy", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Svi Mykhailiuk", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Khris Middleton", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dominick Barlow", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Joel Embiid", "line": 25.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 35.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 33.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 29.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Minott", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan Vukcevic", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brice Sensabaugh", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Justin Edwards", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jose Alvarado", "line": 13.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 18.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Gary Payton II", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 15.0, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deandre Ayton", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bryce McGowens", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rui Hachimura", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Murray-Boyles", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ja'Kobe Walter", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dalton Knecht", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 23.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaxson Hayes", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Marvin Bagley III", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Whitmore", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Al Horford", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quinten Post", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 20.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 22.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Martin", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 33.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 14.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandin Podziemski", "line": 20.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Danny Wolf", "line": 17.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Payton Pritchard", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Gradey Dick", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sam Hauser", "line": 15.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LeBron James", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jake LaRavia", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.47, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anfernee Simons", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sandro Mamukelashvili", "line": 17.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ziaire Williams", "line": 14.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 29.0, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gabe Vincent", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Kyle Filipowski", "line": 20.0, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Shead", "line": 9.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPAHitRates = [
    {"name": "Anthony Edwards", "line": 34.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 17.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Joel Embiid", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jose Alvarado", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Al Horford", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Draymond Green", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Micah Peavy", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ace Bailey", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Minott", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaxson Hayes", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brice Sensabaugh", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan Vukcevic", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Justin Edwards", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandin Podziemski", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sam Hauser", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Whitmore", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Walsh", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jordan Hawkins", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Filipowski", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 30.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyrese Martin", "line": 13.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jonathan Kuminga", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 27.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaden McDaniels", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sandro Mamukelashvili", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 14.0, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ja'Kobe Walter", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gabe Vincent", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rui Hachimura", "line": 14.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
];const prizepicksRAHitRates = [
    {"name": "VJ Edgecombe", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Draymond Green", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Micah Peavy", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 13.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 11.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bryce McGowens", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 8.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jonathan Kuminga", "line": 6.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Will Richard", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Minott", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deandre Ayton", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Neemias Queta", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sandro Mamukelashvili", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sam Hauser", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Austin Reaves", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jake LaRavia", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.47, "overPct": 0.2, "underPct": 0.8},
    {"name": "Rui Hachimura", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "LeBron James", "line": 14.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Derik Queen", "line": 12.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jared McCain", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kyle Filipowski", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Danny Wolf", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksTurnoversHitRates = [
    {"name": "Tristan Vukcevic", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Hawkins", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Joel Embiid", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gabe Vincent", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaxson Hayes", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Micah Peavy", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Will Richard", "line": 0.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sam Hauser", "line": 0.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 2.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dalton Knecht", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogPointsHitRates = [
    {"name": "Anthony Edwards", "line": 29.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.848, "underPct": 0.152},
    {"name": "Andre Drummond", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.815, "underPct": 0.185},
    {"name": "VJ Edgecombe", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.781, "underPct": 0.219},
    {"name": "Saddiq Bey", "line": 14.5, "l5": 1.0, "l10": 0.6, "l15": 0.67, "overPct": 0.779, "underPct": 0.221},
    {"name": "Scottie Barnes", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.675, "underPct": 0.325},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.633, "underPct": 0.367},
    {"name": "Quinten Post", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.605, "underPct": 0.395},
    {"name": "Quentin Grimes", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.603, "underPct": 0.397},
    {"name": "Gary Payton II", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.595, "underPct": 0.405},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.593, "underPct": 0.407},
    {"name": "Keyonte George", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.582, "underPct": 0.418},
    {"name": "Joel Embiid", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.577, "underPct": 0.423},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.571, "underPct": 0.429},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.563, "underPct": 0.437},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.539, "underPct": 0.461},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.537, "underPct": 0.463},
    {"name": "Noah Clowney", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.532, "underPct": 0.468},
    {"name": "Deandre Ayton", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.504, "underPct": 0.496},
    {"name": "Mike Conley", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.499, "underPct": 0.501},
    {"name": "Draymond Green", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.46, "underPct": 0.54},
    {"name": "Derik Queen", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.453, "underPct": 0.547},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.447, "underPct": 0.553},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.44, "underPct": 0.56},
    {"name": "Kyshawn George", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.436, "underPct": 0.564},
    {"name": "Josh Minott", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.424, "underPct": 0.576},
    {"name": "Austin Reaves", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.406, "underPct": 0.594},
    {"name": "Neemias Queta", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.392, "underPct": 0.608},
    {"name": "Brice Sensabaugh", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.373, "underPct": 0.627},
    {"name": "Cam Whitmore", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.361, "underPct": 0.639},
    {"name": "Brandin Podziemski", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.352, "underPct": 0.648},
    {"name": "Tyrese Martin", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.326, "underPct": 0.674},
    {"name": "Paul George", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.325, "underPct": 0.675},
    {"name": "Khris Middleton", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.325, "underPct": 0.675},
    {"name": "Dalton Knecht", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.314, "underPct": 0.686},
    {"name": "Julius Randle", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.301, "underPct": 0.699},
    {"name": "Marvin Bagley III", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.294, "underPct": 0.706},
    {"name": "Kyle Filipowski", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.29, "underPct": 0.71},
    {"name": "Sandro Mamukelashvili", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.284, "underPct": 0.716},
    {"name": "Ja'Kobe Walter", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.281, "underPct": 0.719},
    {"name": "Al Horford", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.259, "underPct": 0.741},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.259, "underPct": 0.741},
    {"name": "Rui Hachimura", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.246, "underPct": 0.754},
    {"name": "Tristan Vukcevic", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.236, "underPct": 0.764},
    {"name": "Jordan Walsh", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.232, "underPct": 0.768},
    {"name": "Ben Saraf", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.228, "underPct": 0.772},
    {"name": "Payton Pritchard", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.191, "underPct": 0.809},
    {"name": "Will Richard", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.181, "underPct": 0.819},
    {"name": "Jake LaRavia", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.16, "underPct": 0.84},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.147, "underPct": 0.853},
    {"name": "Anfernee Simons", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.141, "underPct": 0.859},
    {"name": "Terance Mann", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.138, "underPct": 0.862},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.118, "underPct": 0.882},
    {"name": "Sam Hauser", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.099, "underPct": 0.901},
    {"name": "Gabe Vincent", "line": 6.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.088, "underPct": 0.912},
    {"name": "LeBron James", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.076, "underPct": 0.924},
    {"name": "Jonathan Kuminga", "line": 15.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.046, "underPct": 0.954},
    {"name": "Danny Wolf", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.036, "underPct": 0.964},
];const underdogAssistsHitRates = [
    {"name": "Draymond Green", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.623, "underPct": 0.377},
    {"name": "Dominick Barlow", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.62, "underPct": 0.38},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.609, "underPct": 0.391},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.602, "underPct": 0.398},
    {"name": "Gary Payton II", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.542, "underPct": 0.458},
    {"name": "Mike Conley", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.541, "underPct": 0.459},
    {"name": "Tristan Vukcevic", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.469, "underPct": 0.531},
    {"name": "Jose Alvarado", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.446, "underPct": 0.554},
    {"name": "Bryce McGowens", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.428, "underPct": 0.572},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.374, "underPct": 0.626},
    {"name": "Day'Ron Sharpe", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.348, "underPct": 0.652},
    {"name": "Jordan Walsh", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.298, "underPct": 0.702},
    {"name": "Anfernee Simons", "line": 3.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.278, "underPct": 0.722},
];const underdogReboundsHitRates = [
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.651, "underPct": 0.349},
    {"name": "Gary Payton II", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.633, "underPct": 0.367},
    {"name": "Jaden McDaniels", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jeremiah Fears", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.526, "underPct": 0.474},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.483, "underPct": 0.517},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.459, "underPct": 0.541},
    {"name": "Mike Conley", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.449, "underPct": 0.551},
    {"name": "Jared McCain", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.41, "underPct": 0.59},
    {"name": "Sandro Mamukelashvili", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Ziaire Williams", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.401, "underPct": 0.599},
    {"name": "Anfernee Simons", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.372, "underPct": 0.628},
    {"name": "Derik Queen", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.301, "underPct": 0.699},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.297, "underPct": 0.703},
    {"name": "Isaiah Collier", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.215, "underPct": 0.785},
    {"name": "Marvin Bagley III", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.151, "underPct": 0.849},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Anthony Edwards", "line": 39.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 23.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Draymond Green", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Micah Peavy", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 34.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 41.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Svi Mykhailiuk", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Joel Embiid", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ja'Kobe Walter", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Minott", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gary Payton II", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaxson Hayes", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Neemias Queta", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandin Podziemski", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 39.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Justin Champagnie", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Terance Mann", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bryce McGowens", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Lauri Markkanen", "line": 35.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 34.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jake LaRavia", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.47, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sandro Mamukelashvili", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gabe Vincent", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rui Hachimura", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 34.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anfernee Simons", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 36.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Derrick White", "line": 32.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Payton Pritchard", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ben Saraf", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Danny Wolf", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyrese Martin", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sam Hauser", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyle Filipowski", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jonathan Kuminga", "line": 22.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
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
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Draymond Green", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 8.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Keyonte George", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Terance Mann", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paul George", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Neemias Queta", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deandre Ayton", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Danny Wolf", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogTurnoversHitRates = [
    {"name": "Joel Embiid", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 2.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
];const underdogBlocksStealsHitRates = [
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
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
            <th style="width: 2%">#</th>
            <th style="width: 14%">Player </th>
            <th style="width: 5%">Line </th>
            <th style="width: 5%">Proj. </th>
            <th style="width: 5%">Prob. </th>
            <th style="width: 14%">Player </th>
            <th style="width: 5%">Line </th>
            <th style="width: 5%">Proj. </th>
            <th style="width: 5%">Prob. </th>
            <th style="width: 8%">EV%</th>
            <th style="width: 7%">Kelly</th>
            <th style="width: 12%">Sigma</th>
            <th style="width: 3%">Rec</th>
        </tr>
    `;

    const tbody = data.map((row, index) => `
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
            <td style="font-weight: 600; color: ${(row.prob1 || 0) > 0.5 ? '#10b981' : '#f59e0b'}; font-size: 0.9rem;">
                ${((row.prob1 || 0) * 100).toFixed(1)}%
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
            <td style="font-weight: 600; color: ${(row.prob2 || 0) > 0.5 ? '#10b981' : '#f59e0b'}; font-size: 0.9rem;">
                ${((row.prob2 || 0) * 100).toFixed(1)}%
            </td>
            <td class="ev-cell ${getEVClass(row.ev)}">${row.ev.toFixed(2)}%</td>
            <td class="kelly-cell">${(row.kelly * 100).toFixed(1)}%</td>
            <td>
                <span class="sigma-badge ${getSigmaClass(row.sigma1)}">${row.sigma1}</span>
                <span class="sigma-badge ${getSigmaClass(row.sigma2)}">${row.sigma2}</span>
            </td>
            <td class="recommendation-cell">
                <span class="rec-badge rec-${row.recommendation}"></span>
            </td>
        </tr>
    `).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

function renderTriosTable(data) {
    const thead = `
        <tr>
            <th style="width: 2%">#</th>
            <th style="width: 11%">Player </th>
            <th style="width: 4%">Line </th>
            <th style="width: 4%">Proj. </th>
            <th style="width: 4%">Prob. </th>
            <th style="width: 11%">Player </th>
            <th style="width: 4%">Line </th>
            <th style="width: 4%">Proj. </th>
            <th style="width: 4%">Prob. </th>
            <th style="width: 11%">Player </th>
            <th style="width: 4%">Line </th>
            <th style="width: 4%">Proj. </th>
            <th style="width: 4%">Prob. </th>
            <th style="width: 7%">EV%</th>
            <th style="width: 7%">Kelly</th>
            <th style="width: 2%">Rec</th>
        </tr>
    `;

    const tbody = data.map((row, index) => `
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
            <td style="font-weight: 600; color: ${(row.prob1 || 0) > 0.5 ? '#10b981' : '#f59e0b'}; font-size: 0.85rem;">
                ${((row.prob1 || 0) * 100).toFixed(1)}%
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
            <td style="font-weight: 600; color: ${(row.prob2 || 0) > 0.5 ? '#10b981' : '#f59e0b'}; font-size: 0.85rem;">
                ${((row.prob2 || 0) * 100).toFixed(1)}%
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
            <td style="font-weight: 600; color: ${(row.prob3 || 0) > 0.5 ? '#10b981' : '#f59e0b'}; font-size: 0.85rem;">
                ${((row.prob3 || 0) * 100).toFixed(1)}%
            </td>
            <td class="ev-cell ${getEVClass(row.ev)}">${row.ev.toFixed(2)}%</td>
            <td class="kelly-cell">${(row.kelly * 100).toFixed(1)}%</td>
            <td class="recommendation-cell">
                <span class="rec-badge rec-${row.recommendation}"></span>
            </td>
        </tr>
    `).join('');

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

