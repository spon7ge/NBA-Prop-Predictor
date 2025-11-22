const prizepicksSinglesData = [
    {"name": "Jerami Grant", "bookmaker": "DraftKings", "line": 22.5, "prediction": 17.1, "side": "Under", "odds": -111, "recommendation": 1, "ev": 6.09, "kelly": 0.676, "sigma": "Med"},
    {"name": "Bennedict Mathurin", "bookmaker": "BetMGM", "line": 21.5, "prediction": 26.02, "side": "Over", "odds": 110, "recommendation": 1, "ev": 5.92, "kelly": 0.538, "sigma": "High"},
    {"name": "Alperen Sengun", "bookmaker": "BetRivers", "line": 24.5, "prediction": 28.14, "side": "Over", "odds": 112, "recommendation": 0, "ev": 5.09, "kelly": 0.455, "sigma": "High"},
    {"name": "Keyonte George", "bookmaker": "FanDuel", "line": 18.5, "prediction": 23.11, "side": "Over", "odds": 100, "recommendation": 1, "ev": 5.01, "kelly": 0.501, "sigma": "High"},
    {"name": "Tre Jones", "bookmaker": "BetMGM", "line": 9.5, "prediction": 13.85, "side": "Over", "odds": -110, "recommendation": 1, "ev": 4.76, "kelly": 0.524, "sigma": "Med"},
    {"name": "Ayo Dosunmu", "bookmaker": "BetMGM", "line": 13.5, "prediction": 17.6, "side": "Over", "odds": -110, "recommendation": 1, "ev": 4.4, "kelly": 0.484, "sigma": "Med"},
    {"name": "Aaron Gordon", "bookmaker": "BetRivers", "line": 18.5, "prediction": 21.33, "side": "Over", "odds": 112, "recommendation": 0, "ev": 4.38, "kelly": 0.391, "sigma": "High"},
    {"name": "Klay Thompson", "bookmaker": "BetRivers", "line": 10.5, "prediction": 7.83, "side": "Under", "odds": 112, "recommendation": 0, "ev": 4.37, "kelly": 0.39, "sigma": "Med"},
    {"name": "Jonas Valanciunas", "bookmaker": "BetMGM", "line": 6.5, "prediction": 8.65, "side": "Over", "odds": 105, "recommendation": 0, "ev": 4.35, "kelly": 0.415, "sigma": "Low"},
    {"name": "Lauri Markkanen", "bookmaker": "FanDuel", "line": 24.5, "prediction": 28.53, "side": "Over", "odds": -102, "recommendation": 1, "ev": 4.25, "kelly": 0.434, "sigma": "High"},
];const prizepicksPairsData = [
    {"name1": "Miles McBride", "name2": "Coby White", "line1": 8.5, "line2": 20.5, "prediction1": 13.06, "prediction2": 25.74, "side1": "over", "side2": "over", "recommendation": 1, "ev": 85.31, "kelly": 0.427, "sigma1": "High", "sigma2": "Med", "prob1": 0.764, "prob2": 0.825, "hitRate1": 72.8, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 30.1, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Jalen Duren", "name2": "Peyton Watson", "line1": 17.5, "line2": 12.5, "prediction1": 22.25, "prediction2": 9.44, "side1": "over", "side2": "under", "recommendation": 0, "ev": 67.07, "kelly": 0.335, "sigma1": "High", "sigma2": "Low", "prob1": 0.755, "prob2": 0.753, "hitRate1": 85.3, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 76.7, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "Landry Shamet", "name2": "Tobias Harris", "line1": 9.0, "line2": 10.5, "prediction1": 12.17, "prediction2": 14.89, "side1": "over", "side2": "over", "recommendation": 0, "ev": 61.07, "kelly": 0.305, "sigma1": "Med", "sigma2": "High", "prob1": 0.73, "prob2": 0.751, "hitRate1": 80.0, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 78.2, "l5_2": 0.6, "l15_2": 0.2},
    {"name1": "Karl-Anthony Towns", "name2": "Ausar Thompson", "line1": 21.5, "line2": 9.5, "prediction1": 25.82, "prediction2": 13.38, "side1": "over", "side2": "over", "recommendation": 0, "ev": 58.08, "kelly": 0.29, "sigma1": "High", "sigma2": "High", "prob1": 0.729, "prob2": 0.738, "hitRate1": 56.3, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 77.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Jonathan Isaac", "name2": "Cameron Johnson", "line1": 3.5, "line2": 13.5, "prediction1": 6.04, "prediction2": 10.05, "side1": "over", "side2": "under", "recommendation": 0, "ev": 52.13, "kelly": 0.261, "sigma1": "Low", "sigma2": "Med", "prob1": 0.711, "prob2": 0.728, "hitRate1": 46.9, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 94.4, "l5_2": 0.4, "l15_2": 0.2},
    {"name1": "Jalen Johnson", "name2": "Myles Turner", "line1": 22.5, "line2": 15.5, "prediction1": 25.33, "prediction2": 12.32, "side1": "over", "side2": "under", "recommendation": 0, "ev": 36.47, "kelly": 0.182, "sigma1": "High", "sigma2": "High", "prob1": 0.666, "prob2": 0.697, "hitRate1": 62.1, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 60.8, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Duncan Robinson", "name2": "Santi Aldama", "line1": 10.5, "line2": 17.5, "prediction1": 13.07, "prediction2": 14.94, "side1": "over", "side2": "under", "recommendation": 0, "ev": 28.89, "kelly": 0.144, "sigma1": "High", "sigma2": "High", "prob1": 0.664, "prob2": 0.66, "hitRate1": 83.1, "l5_1": 1.0, "l15_1": 0.67, "hitRate2": 64.1, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Mitchell Robinson", "name2": "Jamal Murray", "line1": 4.5, "line2": 22.5, "prediction1": 6.43, "prediction2": 25.47, "side1": "over", "side2": "over", "recommendation": 0, "ev": 27.75, "kelly": 0.139, "sigma1": "Low", "sigma2": "High", "prob1": 0.66, "prob2": 0.659, "hitRate1": 31.7, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 49.1, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Jordan Clarkson", "name2": "Jeremiah Fears", "line1": 10.5, "line2": 16.5, "prediction1": 12.76, "prediction2": 18.78, "side1": "over", "side2": "over", "recommendation": 0, "ev": 21.41, "kelly": 0.107, "sigma1": "High", "sigma2": "High", "prob1": 0.645, "prob2": 0.64, "hitRate1": 55.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 60.3, "l5_2": 0.8, "l15_2": 0.4},
    {"name1": "Tristan da Silva", "name2": "Zach LaVine", "line1": 12.5, "line2": 18.5, "prediction1": 14.88, "prediction2": 20.85, "side1": "over", "side2": "over", "recommendation": 0, "ev": 20.66, "kelly": 0.103, "sigma1": "High", "sigma2": "High", "prob1": 0.642, "prob2": 0.639, "hitRate1": 49.1, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 33.3, "l5_2": 0.4, "l15_2": 0.6},
];const prizepicksTriosData = [
    {"name1": "Miles McBride", "name2": "Coby White", "name3": "Jalen Duren", "line1": 8.5, "line2": 20.5, "line3": 17.5, "prediction1": 13.06, "prediction2": 25.74, "prediction3": 22.25, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 157.0, "kelly": 0.314, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.764, "prob2": 0.825, "prob3": 0.755, "hitRate1": 72.8, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 30.1, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 85.3, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Tobias Harris", "name2": "Ausar Thompson", "name3": "Peyton Watson", "line1": 10.5, "line2": 9.5, "line3": 12.5, "prediction1": 14.89, "prediction2": 13.38, "prediction3": 9.44, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 125.07, "kelly": 0.25, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.751, "prob2": 0.738, "prob3": 0.753, "hitRate1": 78.2, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 77.0, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 76.7, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Karl-Anthony Towns", "name2": "Landry Shamet", "name3": "Cameron Johnson", "line1": 21.5, "line2": 9.0, "line3": 13.5, "prediction1": 25.82, "prediction2": 12.17, "prediction3": 10.05, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 109.02, "kelly": 0.218, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.729, "prob2": 0.73, "prob3": 0.728, "hitRate1": 56.3, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 80.0, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 94.4, "l5_3": 0.4, "l15_3": 0.2},
    {"name1": "Jonathan Isaac", "name2": "Jalen Johnson", "name3": "Myles Turner", "line1": 3.5, "line2": 22.5, "line3": 15.5, "prediction1": 6.04, "prediction2": 25.33, "prediction3": 12.32, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 78.24, "kelly": 0.156, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.711, "prob2": 0.666, "prob3": 0.697, "hitRate1": 46.9, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 62.1, "l5_2": 1.0, "l15_2": 0.53, "hitRate3": 60.8, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Mitchell Robinson", "name2": "Duncan Robinson", "name3": "Santi Aldama", "line1": 4.5, "line2": 10.5, "line3": 17.5, "prediction1": 6.43, "prediction2": 13.07, "prediction3": 14.94, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 56.17, "kelly": 0.112, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.66, "prob2": 0.664, "prob3": 0.66, "hitRate1": 31.7, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 83.1, "l5_2": 1.0, "l15_2": 0.67, "hitRate3": 64.1, "l5_3": 0.4, "l15_3": 0.13},
    {"name1": "Jordan Clarkson", "name2": "Jeremiah Fears", "name3": "Jamal Murray", "line1": 10.5, "line2": 16.5, "line3": 22.5, "prediction1": 12.76, "prediction2": 18.78, "prediction3": 25.47, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 46.88, "kelly": 0.094, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.645, "prob2": 0.64, "prob3": 0.659, "hitRate1": 55.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 60.3, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 49.1, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Tristan da Silva", "name2": "Dyson Daniels", "name3": "Zach LaVine", "line1": 12.5, "line2": 11.5, "line3": 18.5, "prediction1": 14.88, "prediction2": 9.79, "prediction3": 20.85, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 41.0, "kelly": 0.082, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "prob1": 0.642, "prob2": 0.636, "prob3": 0.639, "hitRate1": 49.1, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 86.9, "l5_2": 0.2, "l15_2": 0.27, "hitRate3": 33.3, "l5_3": 0.4, "l15_3": 0.6},
    {"name1": "Patrick Williams", "name2": "Bobby Portis", "name3": "DeMar DeRozan", "line1": 10.5, "line2": 15.0, "line3": 17.5, "prediction1": 8.81, "prediction2": 13.15, "prediction3": 15.35, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 32.39, "kelly": 0.065, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.629, "prob2": 0.62, "prob3": 0.628, "hitRate1": 90.9, "l5_1": 0.0, "l15_1": 0.27, "hitRate2": 77.3, "l5_2": 0.2, "l15_2": 0.13, "hitRate3": 71.3, "l5_3": 0.2, "l15_3": 0.67},
    {"name1": "Jalen Brunson", "name2": "Mikal Bridges", "name3": "Nickeil Alexander-Walker", "line1": 28.5, "line2": 15.5, "line3": 18.5, "prediction1": 26.59, "prediction2": 17.39, "prediction3": 20.56, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 26.34, "kelly": 0.053, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.618, "prob2": 0.612, "prob3": 0.619, "hitRate1": 69.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 52.5, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 74.9, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Zaccharie Risacher", "name2": "Alex Sarr", "name3": "Dennis Schr\u00f6der", "line1": 12.0, "line2": 17.5, "line3": 11.5, "prediction1": 13.81, "prediction2": 19.36, "prediction3": 13.25, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 22.39, "kelly": 0.045, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.612, "prob2": 0.61, "prob3": 0.607, "hitRate1": 50.1, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 55.4, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Miles McBride", "name2": "Coby White", "line1": 8.5, "line2": 20.5, "prediction1": 13.06, "prediction2": 25.74, "side1": "over", "side2": "over", "recommendation": 1, "ev": 85.31, "kelly": 0.427, "sigma1": "High", "sigma2": "Med", "prob1": 0.764, "prob2": 0.825, "hitRate1": 72.8, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 30.1, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Jonathan Isaac", "name2": "Peyton Watson", "line1": 3.5, "line2": 12.5, "prediction1": 6.04, "prediction2": 9.44, "side1": "over", "side2": "under", "recommendation": 0, "ev": 57.33, "kelly": 0.287, "sigma1": "Low", "sigma2": "Low", "prob1": 0.711, "prob2": 0.753, "hitRate1": 46.9, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 76.7, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "Myles Turner", "name2": "Cameron Johnson", "line1": 15.5, "line2": 13.5, "prediction1": 12.32, "prediction2": 10.05, "side1": "under", "side2": "under", "recommendation": 0, "ev": 49.16, "kelly": 0.246, "sigma1": "High", "sigma2": "Med", "prob1": 0.697, "prob2": 0.728, "hitRate1": 60.8, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 94.4, "l5_2": 0.4, "l15_2": 0.2},
    {"name1": "Landry Shamet", "name2": "Duncan Robinson", "line1": 9.5, "line2": 10.5, "prediction1": 12.17, "prediction2": 13.07, "side1": "over", "side2": "over", "recommendation": 0, "ev": 36.02, "kelly": 0.18, "sigma1": "Med", "sigma2": "High", "prob1": 0.697, "prob2": 0.664, "hitRate1": 80.0, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 83.1, "l5_2": 1.0, "l15_2": 0.67},
    {"name1": "Karl-Anthony Towns", "name2": "Santi Aldama", "line1": 22.5, "line2": 17.5, "prediction1": 25.82, "prediction2": 14.94, "side1": "over", "side2": "under", "recommendation": 0, "ev": 32.07, "kelly": 0.16, "sigma1": "High", "sigma2": "High", "prob1": 0.68, "prob2": 0.66, "hitRate1": 47.8, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 64.1, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Mitchell Robinson", "name2": "Bobby Portis", "line1": 4.5, "line2": 15.5, "prediction1": 6.43, "prediction2": 13.15, "side1": "over", "side2": "under", "recommendation": 0, "ev": 26.26, "kelly": 0.131, "sigma1": "Low", "sigma2": "High", "prob1": 0.66, "prob2": 0.651, "hitRate1": 31.7, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 77.3, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "Jordan Clarkson", "name2": "Jeremiah Fears", "line1": 10.5, "line2": 16.5, "prediction1": 12.76, "prediction2": 18.78, "side1": "over", "side2": "over", "recommendation": 0, "ev": 21.41, "kelly": 0.107, "sigma1": "High", "sigma2": "High", "prob1": 0.645, "prob2": 0.64, "hitRate1": 55.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 60.3, "l5_2": 0.8, "l15_2": 0.4},
    {"name1": "Nickeil Alexander-Walker", "name2": "Zach LaVine", "line1": 18.5, "line2": 18.5, "prediction1": 20.56, "prediction2": 20.85, "side1": "over", "side2": "over", "recommendation": 0, "ev": 16.24, "kelly": 0.081, "sigma1": "High", "sigma2": "High", "prob1": 0.619, "prob2": 0.639, "hitRate1": 74.9, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 33.3, "l5_2": 0.4, "l15_2": 0.6},
    {"name1": "Jalen Brunson", "name2": "DeMar DeRozan", "line1": 28.5, "line2": 17.5, "prediction1": 26.59, "prediction2": 15.35, "side1": "under", "side2": "under", "recommendation": 0, "ev": 14.12, "kelly": 0.071, "sigma1": "High", "sigma2": "High", "prob1": 0.618, "prob2": 0.628, "hitRate1": 69.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 71.3, "l5_2": 0.2, "l15_2": 0.67},
    {"name1": "Mikal Bridges", "name2": "Alex Sarr", "line1": 15.5, "line2": 17.5, "prediction1": 17.39, "prediction2": 19.36, "side1": "over", "side2": "over", "recommendation": 0, "ev": 9.89, "kelly": 0.049, "sigma1": "High", "sigma2": "High", "prob1": 0.612, "prob2": 0.61, "hitRate1": 52.5, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 55.4, "l5_2": 0.4, "l15_2": 0.4},
];const underdogTriosData = [
    {"name1": "Miles McBride", "name2": "Coby White", "name3": "Peyton Watson", "line1": 8.5, "line2": 20.5, "line3": 12.5, "prediction1": 13.06, "prediction2": 25.74, "prediction3": 9.44, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 156.16, "kelly": 0.312, "sigma1": "High", "sigma2": "Med", "sigma3": "Low", "prob1": 0.764, "prob2": 0.825, "prob3": 0.753, "hitRate1": 72.8, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 30.1, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 76.7, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Jonathan Isaac", "name2": "Myles Turner", "name3": "Cameron Johnson", "line1": 3.5, "line2": 15.5, "line3": 13.5, "prediction1": 6.04, "prediction2": 12.32, "prediction3": 10.05, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 94.81, "kelly": 0.19, "sigma1": "Low", "sigma2": "High", "sigma3": "Med", "prob1": 0.711, "prob2": 0.697, "prob3": 0.728, "hitRate1": 46.9, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 60.8, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 94.4, "l5_3": 0.4, "l15_3": 0.2},
    {"name1": "Karl-Anthony Towns", "name2": "Landry Shamet", "name3": "Duncan Robinson", "line1": 22.5, "line2": 9.5, "line3": 10.5, "prediction1": 25.82, "prediction2": 12.17, "prediction3": 13.07, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 69.94, "kelly": 0.14, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.68, "prob2": 0.697, "prob3": 0.664, "hitRate1": 47.8, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 80.0, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 83.1, "l5_3": 1.0, "l15_3": 0.67},
    {"name1": "Mitchell Robinson", "name2": "Bobby Portis", "name3": "Santi Aldama", "line1": 4.5, "line2": 15.5, "line3": 17.5, "prediction1": 6.43, "prediction2": 13.15, "prediction3": 14.94, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 53.15, "kelly": 0.106, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.66, "prob2": 0.651, "prob3": 0.66, "hitRate1": 31.7, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 77.3, "l5_2": 0.2, "l15_2": 0.13, "hitRate3": 64.1, "l5_3": 0.4, "l15_3": 0.13},
    {"name1": "Jordan Clarkson", "name2": "Jeremiah Fears", "name3": "Zach LaVine", "line1": 10.5, "line2": 16.5, "line3": 18.5, "prediction1": 12.76, "prediction2": 18.78, "prediction3": 20.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 42.54, "kelly": 0.085, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.645, "prob2": 0.64, "prob3": 0.639, "hitRate1": 55.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 60.3, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 33.3, "l5_3": 0.4, "l15_3": 0.6},
    {"name1": "Jalen Brunson", "name2": "Nickeil Alexander-Walker", "name3": "DeMar DeRozan", "line1": 28.5, "line2": 18.5, "line3": 17.5, "prediction1": 26.59, "prediction2": 20.56, "prediction3": 15.35, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 29.65, "kelly": 0.059, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.618, "prob2": 0.619, "prob3": 0.628, "hitRate1": 69.5, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 74.9, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 71.3, "l5_3": 0.2, "l15_3": 0.67},
    {"name1": "Mikal Bridges", "name2": "Alex Sarr", "name3": "Jamal Murray", "line1": 15.5, "line2": 17.5, "line3": 23.5, "prediction1": 17.39, "prediction2": 19.36, "prediction3": 25.47, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 22.49, "kelly": 0.045, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.612, "prob2": 0.61, "prob3": 0.607, "hitRate1": 52.5, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 55.4, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 40.8, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Goga Bitadze", "name2": "Caris LeVert", "name3": "Dennis Schr\u00f6der", "line1": 4.5, "line2": 7.5, "line3": 11.5, "prediction1": 5.57, "prediction2": 8.82, "prediction3": 13.25, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 18.29, "kelly": 0.037, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "prob1": 0.605, "prob2": 0.597, "prob3": 0.607, "hitRate1": 76.5, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 86.5, "l5_2": 1.0, "l15_2": 0.47, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Bilal Coulibaly", "name2": "Brandon Williams", "name3": "Bruce Brown", "line1": 10.5, "line2": 14.5, "line3": 8.5, "prediction1": 12.02, "prediction2": 12.86, "prediction3": 7.16, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 14.11, "kelly": 0.028, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.594, "prob2": 0.595, "prob3": 0.598, "hitRate1": 39.5, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 75.6, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 71.1, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Saddiq Bey", "name2": "Cam Whitmore", "name3": "Cooper Flagg", "line1": 12.5, "line2": 10.5, "line3": 17.5, "prediction1": 14.04, "prediction2": 9.41, "prediction3": 19.04, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 11.03, "kelly": 0.022, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "prob1": 0.591, "prob2": 0.589, "prob3": 0.59, "hitRate1": 27.9, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 50.8, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 55.8, "l5_3": 0.4, "l15_3": 0.4},
];const prizepicksPointsHitRates = [
    {"name": "Caris LeVert", "line": 7.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.865, "underPct": 0.135},
    {"name": "Jalen Duren", "line": 17.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.853, "underPct": 0.147},
    {"name": "Duncan Robinson", "line": 10.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.831, "underPct": 0.169},
    {"name": "Landry Shamet", "line": 9.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.8, "underPct": 0.2},
    {"name": "Onyeka Okongwu", "line": 14.5, "l5": 1.0, "l10": 0.5, "l15": 0.47, "overPct": 0.795, "underPct": 0.205},
    {"name": "Tobias Harris", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.782, "underPct": 0.218},
    {"name": "Ausar Thompson", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.77, "underPct": 0.23},
    {"name": "Goga Bitadze", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.765, "underPct": 0.235},
    {"name": "Cade Cunningham", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.753, "underPct": 0.247},
    {"name": "Nickeil Alexander-Walker", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.749, "underPct": 0.251},
    {"name": "Miles McBride", "line": 8.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.728, "underPct": 0.272},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.666, "underPct": 0.334},
    {"name": "Jalen Johnson", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.621, "underPct": 0.379},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.603, "underPct": 0.397},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.574, "underPct": 0.426},
    {"name": "Karl-Anthony Towns", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.559, "underPct": 0.441},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.558, "underPct": 0.442},
    {"name": "Naji Marshall", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.554, "underPct": 0.446},
    {"name": "Alex Sarr", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.554, "underPct": 0.446},
    {"name": "Jordan Clarkson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.551, "underPct": 0.449},
    {"name": "Tre Johnson", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.526, "underPct": 0.474},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Anthony Black", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.51, "underPct": 0.49},
    {"name": "Derik Queen", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.502, "underPct": 0.498},
    {"name": "Zaccharie Risacher", "line": 12.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.499, "underPct": 0.501},
    {"name": "Cam Whitmore", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.492, "underPct": 0.508},
    {"name": "Tristan da Silva", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.491, "underPct": 0.509},
    {"name": "Jamal Murray", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.491, "underPct": 0.509},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.49, "underPct": 0.51},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.471, "underPct": 0.529},
    {"name": "Jonathan Isaac", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.469, "underPct": 0.531},
    {"name": "Ryan Rollins", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.452, "underPct": 0.548},
    {"name": "Luke Kennard", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.445, "underPct": 0.555},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.44, "underPct": 0.56},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Myles Turner", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.392, "underPct": 0.608},
    {"name": "Santi Aldama", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.359, "underPct": 0.641},
    {"name": "Zach LaVine", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.333, "underPct": 0.667},
    {"name": "Jaylen Wells", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.323, "underPct": 0.677},
    {"name": "D'Angelo Russell", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.32, "underPct": 0.68},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.318, "underPct": 0.682},
    {"name": "Mitchell Robinson", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.317, "underPct": 0.683},
    {"name": "Jalen Brunson", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.305, "underPct": 0.695},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.302, "underPct": 0.698},
    {"name": "Jose Alvarado", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.301, "underPct": 0.699},
    {"name": "Coby White", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.301, "underPct": 0.699},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.289, "underPct": 0.711},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.287, "underPct": 0.713},
    {"name": "Bilal Coulibaly", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.284, "underPct": 0.716},
    {"name": "Saddiq Bey", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.279, "underPct": 0.721},
    {"name": "Josh Giddey", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.278, "underPct": 0.722},
    {"name": "Cam Spencer", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.268, "underPct": 0.732},
    {"name": "Drew Eubanks", "line": 8.0, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.264, "underPct": 0.736},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.264, "underPct": 0.736},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.244, "underPct": 0.756},
    {"name": "Peyton Watson", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.233, "underPct": 0.767},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.229, "underPct": 0.771},
    {"name": "Bobby Portis", "line": 15.0, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.227, "underPct": 0.773},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.208, "underPct": 0.792},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.204, "underPct": 0.796},
    {"name": "Keegan Murray", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.198, "underPct": 0.802},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.131, "underPct": 0.869},
    {"name": "Patrick Williams", "line": 10.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.091, "underPct": 0.909},
    {"name": "Cole Anthony", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.068, "underPct": 0.932},
    {"name": "Cameron Johnson", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.056, "underPct": 0.944},
];const prizepicksAssistsHitRates = [
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.753, "underPct": 0.247},
    {"name": "Cade Cunningham", "line": 9.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.672, "underPct": 0.328},
    {"name": "Jamal Murray", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.666, "underPct": 0.334},
    {"name": "Miles McBride", "line": 1.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.65, "underPct": 0.35},
    {"name": "Josh Hart", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.623, "underPct": 0.377},
    {"name": "Mitchell Robinson", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.581, "underPct": 0.419},
    {"name": "Coby White", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.575, "underPct": 0.425},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.569, "underPct": 0.431},
    {"name": "Jalen Johnson", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.55, "underPct": 0.45},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.549, "underPct": 0.451},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.524, "underPct": 0.476},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.487, "underPct": 0.513},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.485, "underPct": 0.515},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.464, "underPct": 0.536},
    {"name": "Brandon Williams", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.458, "underPct": 0.542},
    {"name": "Ryan Rollins", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.453, "underPct": 0.547},
    {"name": "Cole Anthony", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Jalen Suggs", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.427, "underPct": 0.573},
    {"name": "Franz Wagner", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.384, "underPct": 0.616},
    {"name": "Tristan da Silva", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.338, "underPct": 0.662},
    {"name": "Derik Queen", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.299, "underPct": 0.701},
];const prizepicksReboundsHitRates = [
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.642, "underPct": 0.358},
    {"name": "Jalen Duren", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.619, "underPct": 0.381},
    {"name": "Franz Wagner", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.609, "underPct": 0.391},
    {"name": "Jock Landale", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.607, "underPct": 0.393},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.596, "underPct": 0.404},
    {"name": "Miles McBride", "line": 1.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.585, "underPct": 0.415},
    {"name": "Tobias Harris", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.571, "underPct": 0.429},
    {"name": "Duncan Robinson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.532, "underPct": 0.468},
    {"name": "Cade Cunningham", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.521, "underPct": 0.479},
    {"name": "Santi Aldama", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.506, "underPct": 0.494},
    {"name": "Alex Sarr", "line": 7.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.505, "underPct": 0.495},
    {"name": "Josh Giddey", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.493, "underPct": 0.507},
    {"name": "Mitchell Robinson", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.487, "underPct": 0.513},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Tristan da Silva", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.476, "underPct": 0.524},
    {"name": "Mikal Bridges", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.47, "underPct": 0.53},
    {"name": "Jalen Johnson", "line": 10.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.465, "underPct": 0.535},
    {"name": "Cooper Flagg", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.414, "underPct": 0.586},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.412, "underPct": 0.588},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.399, "underPct": 0.601},
    {"name": "Goga Bitadze", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.393, "underPct": 0.607},
    {"name": "P.J. Washington", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.391, "underPct": 0.609},
    {"name": "Bilal Coulibaly", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.391, "underPct": 0.609},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.371, "underPct": 0.629},
    {"name": "Zach Edey", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.367, "underPct": 0.633},
    {"name": "Ryan Rollins", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.359, "underPct": 0.641},
    {"name": "Cameron Johnson", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.344, "underPct": 0.656},
    {"name": "Onyeka Okongwu", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.337, "underPct": 0.663},
    {"name": "Klay Thompson", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.309, "underPct": 0.691},
    {"name": "Myles Turner", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.308, "underPct": 0.692},
    {"name": "Josh Hart", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.302, "underPct": 0.698},
    {"name": "Keegan Murray", "line": 5.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.256, "underPct": 0.744},
    {"name": "Drew Eubanks", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.212, "underPct": 0.788},
    {"name": "Bobby Portis", "line": 8.0, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.131, "underPct": 0.869},
];const prizepicksBlocksHitRates = [
    {"name": "Jalen Suggs", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.457, "underPct": 0.543},
    {"name": "Jonathan Isaac", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.259, "underPct": 0.741},
    {"name": "Bilal Coulibaly", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.676, "underPct": 0.324},
    {"name": "Kyshawn George", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.556, "underPct": 0.444},
    {"name": "Josh Giddey", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.33, "underPct": 0.67},
    {"name": "Kyle Kuzma", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.447, "underPct": 0.553},
    {"name": "Keegan Murray", "line": 0.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.5, "underPct": 0.5},
];const prizepicksStealsHitRates = [
    {"name": "Landry Shamet", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.349, "underPct": 0.651},
    {"name": "Miles McBride", "line": 0.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.507, "underPct": 0.493},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
    {"name": "Jalen Johnson", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.559, "underPct": 0.441},
    {"name": "Cam Whitmore", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.497, "underPct": 0.503},
    {"name": "Cole Anthony", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.491, "underPct": 0.509},
    {"name": "D'Angelo Russell", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.379, "underPct": 0.621},
    {"name": "Drew Eubanks", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.443, "underPct": 0.557},
    {"name": "Zach LaVine", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.528, "underPct": 0.472},
    {"name": "Malik Monk", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.548, "underPct": 0.452},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Duncan Robinson", "line": 14.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Onyeka Okongwu", "line": 23.5, "l5": 1.0, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 40.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Landry Shamet", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tobias Harris", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ayo Dosunmu", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyshawn George", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jock Landale", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Peyton Watson", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Spencer", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 27.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Precious Achiuwa", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 28.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jose Alvarado", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Rollins", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tre Johnson", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Huerter", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zaccharie Risacher", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 30.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 25.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "John Konchar", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Williams", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drew Eubanks", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cole Anthony", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keegan Murray", "line": 18.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Zach Edey", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
];const prizepicksPRHitRates = [
    {"name": "Jalen Duren", "line": 30.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Duncan Robinson", "line": 14.0, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 1.0, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jonathan Isaac", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Landry Shamet", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ausar Thompson", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cade Cunningham", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Johnson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 30.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Russell Westbrook", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cameron Johnson", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 23.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bobby Portis", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cole Anthony", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Smith", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drew Eubanks", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 8.5, "l5": 0.0, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 22.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Keegan Murray", "line": 16.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPAHitRates = [
    {"name": "Jalen Duren", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Black", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ayo Dosunmu", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Johnson", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Landry Shamet", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Wells", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cameron Johnson", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 34.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Coby White", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mitchell Robinson", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Saddiq Bey", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Williams", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cole Anthony", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kentavious Caldwell-Pope", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keegan Murray", "line": 12.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksRAHitRates = [
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 9.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 14.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kentavious Caldwell-Pope", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Santi Aldama", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tobias Harris", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cole Anthony", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cameron Johnson", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Peyton Watson", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 9.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Black", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 13.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 15.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trey Murphy III", "line": 9.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kevin Huerter", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const prizepicksTurnoversHitRates = [
    {"name": "Desmond Bane", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mitchell Robinson", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bobby Portis", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Ausar Thompson", "line": 1.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Anthony Black", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Sarr", "line": 2.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kyshawn George", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Konchar", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keegan Murray", "line": 1.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPointsHitRates = [
    {"name": "Caris LeVert", "line": 7.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.865, "underPct": 0.135},
    {"name": "Duncan Robinson", "line": 10.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.831, "underPct": 0.169},
    {"name": "Landry Shamet", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.8, "underPct": 0.2},
    {"name": "Onyeka Okongwu", "line": 14.5, "l5": 1.0, "l10": 0.5, "l15": 0.47, "overPct": 0.795, "underPct": 0.205},
    {"name": "Goga Bitadze", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.765, "underPct": 0.235},
    {"name": "Cade Cunningham", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.753, "underPct": 0.247},
    {"name": "Nickeil Alexander-Walker", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.749, "underPct": 0.251},
    {"name": "Miles McBride", "line": 8.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.728, "underPct": 0.272},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.666, "underPct": 0.334},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.603, "underPct": 0.397},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.574, "underPct": 0.426},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.559, "underPct": 0.441},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.558, "underPct": 0.442},
    {"name": "Alex Sarr", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.554, "underPct": 0.446},
    {"name": "Jordan Clarkson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.551, "underPct": 0.449},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.536, "underPct": 0.464},
    {"name": "Tre Johnson", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.526, "underPct": 0.474},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Derik Queen", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.502, "underPct": 0.498},
    {"name": "Zaccharie Risacher", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.499, "underPct": 0.501},
    {"name": "Cam Whitmore", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.492, "underPct": 0.508},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.49, "underPct": 0.51},
    {"name": "Karl-Anthony Towns", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.478, "underPct": 0.522},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.471, "underPct": 0.529},
    {"name": "Jonathan Isaac", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.469, "underPct": 0.531},
    {"name": "Luke Kennard", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.445, "underPct": 0.555},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.44, "underPct": 0.56},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.435, "underPct": 0.565},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.408, "underPct": 0.592},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.395, "underPct": 0.605},
    {"name": "Myles Turner", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.392, "underPct": 0.608},
    {"name": "Tristan da Silva", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.382, "underPct": 0.618},
    {"name": "Ryan Rollins", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.367, "underPct": 0.633},
    {"name": "Santi Aldama", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.359, "underPct": 0.641},
    {"name": "Zach LaVine", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.333, "underPct": 0.667},
    {"name": "Jaylen Wells", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.323, "underPct": 0.677},
    {"name": "D'Angelo Russell", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.32, "underPct": 0.68},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.318, "underPct": 0.682},
    {"name": "Mitchell Robinson", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.317, "underPct": 0.683},
    {"name": "Jalen Brunson", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.305, "underPct": 0.695},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.302, "underPct": 0.698},
    {"name": "Jose Alvarado", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.301, "underPct": 0.699},
    {"name": "Coby White", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.301, "underPct": 0.699},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.289, "underPct": 0.711},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.287, "underPct": 0.713},
    {"name": "Saddiq Bey", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.279, "underPct": 0.721},
    {"name": "Josh Giddey", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.278, "underPct": 0.722},
    {"name": "Cam Spencer", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.268, "underPct": 0.732},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.264, "underPct": 0.736},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.244, "underPct": 0.756},
    {"name": "Peyton Watson", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.233, "underPct": 0.767},
    {"name": "Bobby Portis", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.227, "underPct": 0.773},
    {"name": "Dyson Daniels", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.21, "underPct": 0.79},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.208, "underPct": 0.792},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.204, "underPct": 0.796},
    {"name": "Cole Anthony", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.068, "underPct": 0.932},
    {"name": "Cameron Johnson", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.056, "underPct": 0.944},
];const underdogAssistsHitRates = [
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.753, "underPct": 0.247},
    {"name": "Miles McBride", "line": 1.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.65, "underPct": 0.35},
    {"name": "Coby White", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.575, "underPct": 0.425},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.569, "underPct": 0.431},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.549, "underPct": 0.451},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.524, "underPct": 0.476},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.487, "underPct": 0.513},
    {"name": "Max Christie", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.368, "underPct": 0.632},
    {"name": "Tristan da Silva", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.338, "underPct": 0.662},
];const underdogReboundsHitRates = [
    {"name": "Bruce Brown", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.646, "underPct": 0.354},
    {"name": "Jalen Duren", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.619, "underPct": 0.381},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.596, "underPct": 0.404},
    {"name": "Tre Johnson", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.53, "underPct": 0.47},
    {"name": "Cade Cunningham", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.521, "underPct": 0.479},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.485, "underPct": 0.515},
    {"name": "Mikal Bridges", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.47, "underPct": 0.53},
    {"name": "Cooper Flagg", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.414, "underPct": 0.586},
    {"name": "P.J. Washington", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.391, "underPct": 0.609},
    {"name": "Cameron Johnson", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.344, "underPct": 0.656},
    {"name": "Onyeka Okongwu", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.337, "underPct": 0.663},
    {"name": "Klay Thompson", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.309, "underPct": 0.691},
    {"name": "Cam Spencer", "line": 2.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.263, "underPct": 0.737},
    {"name": "Drew Eubanks", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.212, "underPct": 0.788},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
    {"name": "Jalen Johnson", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.559, "underPct": 0.441},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Duncan Robinson", "line": 14.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jalen Johnson", "line": 40.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tristan da Silva", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles McBride", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tobias Harris", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Whitmore", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "P.J. Washington", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Spencer", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyle Kuzma", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Rollins", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Coby White", "line": 28.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mitchell Robinson", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zaccharie Risacher", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Konchar", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Williams", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 30.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Zach Edey", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keegan Murray", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Cole Anthony", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const underdogPRHitRates = [
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 1.0, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 30.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Coby White", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bobby Portis", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPAHitRates = [
    {"name": "Jalen Duren", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Sarr", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Murray", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Brunson", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const underdogRAHitRates = [
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Santi Aldama", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cole Anthony", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mitchell Robinson", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alex Sarr", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const underdogTurnoversHitRates = [
    {"name": "Kyshawn George", "line": 2.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Desmond Bane", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
];const underdogBlocksStealsHitRates = [
    {"name": "Mikal Bridges", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
];let currentPlatform = 'prizepicks';
let currentType = 'singles';
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
    
    if (currentType === 'singles') {
        statsHTML = `
            <div class="stat-card">
                <div class="stat-label">Projection</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Model's projected value given the context of the game and player performance</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Expected Value %</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value as a percentage of your stake (Ex. If EV% is 5%, you can expect to profit 5% of whatever stake you place on that bet on average.)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Kelly Criterion</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Optimal bet sizing percentage to maximize long-term bankroll growth while managing risk (Ex. If bankroll is $10 for the day, and kelly is 25%, bet $2.50)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Odds</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">American odds format. <span style="color: #34d399;">+</span> = underdog, <span style="color: #f87171;">-</span> = favorite</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sigma</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (volatile, less reliable projections), Med, Low (consistent, more reliable projections)</div>
            </div>
        `;
    } else if (currentType === 'hitrates') {
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
        if (currentType === 'singles') return prizepicksSinglesData;
        if (currentType === 'pairs') return prizepicksPairsData;
        return prizepicksTriosData;
    } else {
        if (currentType === 'singles') return underdogSinglesData;
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
    
    if (currentType === 'singles') {
        platformToggle.style.display = 'none';
        propTypeGroup.style.display = 'none';
        searchGroup.style.display = 'none';
    } else if (currentType === 'hitrates') {
        platformToggle.style.display = 'flex';
        propTypeGroup.style.display = 'flex';
        searchGroup.style.display = 'flex';
    } else {
        platformToggle.style.display = 'flex';
        propTypeGroup.style.display = 'none';
        searchGroup.style.display = 'none';
    }
    
    if (currentType === 'singles') {
        renderSinglesTable(data);
    } else if (currentType === 'pairs') {
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
    
    if (currentType === 'singles') {
        document.getElementById('picksCount').textContent = `Showing top ${data.length} ${betTypeLabel}`;
    } else if (currentType === 'hitrates') {
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

