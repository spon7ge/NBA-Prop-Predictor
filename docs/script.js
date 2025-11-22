const prizepicksSinglesData = [
    {"name": "Coby White", "bookmaker": "BetRivers", "line": 22.5, "prediction": 25.74, "side": "Over", "odds": 120, "recommendation": 0, "ev": 57.98, "kelly": 0.483, "sigma": "Med"},
    {"name": "Ayo Dosunmu", "bookmaker": "BetRivers", "line": 15.5, "prediction": 19.48, "side": "Over", "odds": 114, "recommendation": 0, "ev": 57.15, "kelly": 0.501, "sigma": "High"},
    {"name": "Jalen Duren", "bookmaker": "BetRivers", "line": 18.5, "prediction": 22.25, "side": "Over", "odds": 107, "recommendation": 0, "ev": 46.38, "kelly": 0.433, "sigma": "High"},
    {"name": "Miles McBride", "bookmaker": "FanDuel", "line": 8.5, "prediction": 11.82, "side": "Over", "odds": 102, "recommendation": 0, "ev": 41.49, "kelly": 0.407, "sigma": "High"},
    {"name": "Ausar Thompson", "bookmaker": "BetRivers", "line": 10.5, "prediction": 13.38, "side": "Over", "odds": 104, "recommendation": 0, "ev": 39.06, "kelly": 0.376, "sigma": "High"},
    {"name": "Tobias Harris", "bookmaker": "DraftKings", "line": 11.5, "prediction": 14.89, "side": "Over", "odds": -102, "recommendation": 0, "ev": 38.5, "kelly": 0.393, "sigma": "High"},
    {"name": "Jalen Johnson", "bookmaker": "FanDuel", "line": 21.5, "prediction": 25.33, "side": "Over", "odds": -108, "recommendation": 0, "ev": 38.46, "kelly": 0.415, "sigma": "High"},
    {"name": "Jalen Smith", "bookmaker": "BetMGM", "line": 8.5, "prediction": 13.11, "side": "Over", "odds": -130, "recommendation": 1, "ev": 38.24, "kelly": 0.497, "sigma": "Med"},
    {"name": "Cameron Johnson", "bookmaker": "FanDuel", "line": 13.5, "prediction": 10.05, "side": "Under", "odds": -114, "recommendation": 0, "ev": 36.6, "kelly": 0.417, "sigma": "Med"},
    {"name": "Peyton Watson", "bookmaker": "DraftKings", "line": 12.5, "prediction": 9.44, "side": "Under", "odds": -126, "recommendation": 0, "ev": 34.99, "kelly": 0.441, "sigma": "Low"},
];const prizepicksPairsData = [
    {"name1": "Coby White", "name2": "Jalen Duren", "line1": 20.5, "line2": 17.5, "prediction1": 25.74, "prediction2": 22.25, "side1": "over", "side2": "over", "recommendation": 1, "ev": 83.07, "kelly": 0.415, "sigma1": "Med", "sigma2": "High", "prob1": 0.825, "prob2": 0.755, "hitRate1": 30.1, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 85.3, "l5_2": 1.0, "l15_2": 0.6},
    {"name1": "Ayo Dosunmu", "name2": "Peyton Watson", "line1": 14.5, "line2": 12.5, "prediction1": 19.48, "prediction2": 9.44, "side1": "over", "side2": "under", "recommendation": 0, "ev": 73.31, "kelly": 0.367, "sigma1": "High", "sigma2": "Low", "prob1": 0.783, "prob2": 0.753, "hitRate1": 66.4, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 76.7, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "Tobias Harris", "name2": "Cameron Johnson", "line1": 10.5, "line2": 13.5, "prediction1": 14.89, "prediction2": 10.05, "side1": "over", "side2": "under", "recommendation": 0, "ev": 60.59, "kelly": 0.303, "sigma1": "High", "sigma2": "Med", "prob1": 0.751, "prob2": 0.728, "hitRate1": 78.2, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 94.4, "l5_2": 0.4, "l15_2": 0.2},
    {"name1": "Jonathan Isaac", "name2": "Ausar Thompson", "line1": 3.5, "line2": 9.5, "prediction1": 6.04, "prediction2": 13.38, "side1": "over", "side2": "over", "recommendation": 0, "ev": 54.24, "kelly": 0.271, "sigma1": "Low", "sigma2": "High", "prob1": 0.711, "prob2": 0.738, "hitRate1": 46.9, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 77.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Miles McBride", "name2": "Myles Turner", "line1": 8.5, "line2": 15.5, "prediction1": 11.82, "prediction2": 12.32, "side1": "over", "side2": "under", "recommendation": 0, "ev": 43.57, "kelly": 0.218, "sigma1": "High", "sigma2": "High", "prob1": 0.7, "prob2": 0.697, "hitRate1": 72.8, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 60.8, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Karl-Anthony Towns", "name2": "Jock Landale", "line1": 21.5, "line2": 8.5, "prediction1": 24.58, "prediction2": 10.98, "side1": "over", "side2": "over", "recommendation": 0, "ev": 33.09, "kelly": 0.165, "sigma1": "High", "sigma2": "Med", "prob1": 0.669, "prob2": 0.677, "hitRate1": 56.3, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 64.9, "l5_2": 0.4, "l15_2": 0.6},
    {"name1": "Jalen Johnson", "name2": "Duncan Robinson", "line1": 22.5, "line2": 10.5, "prediction1": 25.33, "prediction2": 13.07, "side1": "over", "side2": "over", "recommendation": 0, "ev": 29.94, "kelly": 0.15, "sigma1": "High", "sigma2": "High", "prob1": 0.666, "prob2": 0.664, "hitRate1": 62.1, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 83.1, "l5_2": 1.0, "l15_2": 0.67},
    {"name1": "Jordan Clarkson", "name2": "Santi Aldama", "line1": 9.5, "line2": 17.5, "prediction1": 11.95, "prediction2": 14.94, "side1": "over", "side2": "under", "recommendation": 0, "ev": 27.81, "kelly": 0.139, "sigma1": "High", "sigma2": "High", "prob1": 0.658, "prob2": 0.66, "hitRate1": 66.9, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 64.1, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Jeremiah Fears", "name2": "Bobby Portis", "line1": 16.5, "line2": 15.5, "prediction1": 18.78, "prediction2": 13.15, "side1": "over", "side2": "under", "recommendation": 0, "ev": 22.45, "kelly": 0.112, "sigma1": "High", "sigma2": "High", "prob1": 0.64, "prob2": 0.651, "hitRate1": 60.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 77.3, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "D'Angelo Russell", "name2": "Zach LaVine", "line1": 12.5, "line2": 18.5, "prediction1": 14.79, "prediction2": 20.85, "side1": "over", "side2": "over", "recommendation": 0, "ev": 18.37, "kelly": 0.092, "sigma1": "High", "sigma2": "High", "prob1": 0.63, "prob2": 0.639, "hitRate1": 42.5, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 33.3, "l5_2": 0.4, "l15_2": 0.6},
];const prizepicksTriosData = [
    {"name1": "Coby White", "name2": "Ayo Dosunmu", "name3": "Jalen Duren", "line1": 20.5, "line2": 14.5, "line3": 17.5, "prediction1": 25.74, "prediction2": 19.48, "prediction3": 22.25, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 163.38, "kelly": 0.327, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.825, "prob2": 0.783, "prob3": 0.755, "hitRate1": 30.1, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 66.4, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 85.3, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Tobias Harris", "name2": "Ausar Thompson", "name3": "Peyton Watson", "line1": 10.5, "line2": 9.5, "line3": 12.5, "prediction1": 14.89, "prediction2": 13.38, "prediction3": 9.44, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 125.07, "kelly": 0.25, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.751, "prob2": 0.738, "prob3": 0.753, "hitRate1": 78.2, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 77.0, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 76.7, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Jonathan Isaac", "name2": "Myles Turner", "name3": "Cameron Johnson", "line1": 3.5, "line2": 15.5, "line3": 13.5, "prediction1": 6.04, "prediction2": 12.32, "prediction3": 10.05, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 94.81, "kelly": 0.19, "sigma1": "Low", "sigma2": "High", "sigma3": "Med", "prob1": 0.711, "prob2": 0.697, "prob3": 0.728, "hitRate1": 46.9, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 60.8, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 94.4, "l5_3": 0.4, "l15_3": 0.2},
    {"name1": "Karl-Anthony Towns", "name2": "Miles McBride", "name3": "Jock Landale", "line1": 21.5, "line2": 8.5, "line3": 8.5, "prediction1": 24.58, "prediction2": 11.82, "prediction3": 10.98, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 71.22, "kelly": 0.142, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.669, "prob2": 0.7, "prob3": 0.677, "hitRate1": 56.3, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 72.8, "l5_2": 1.0, "l15_2": 0.53, "hitRate3": 64.9, "l5_3": 0.4, "l15_3": 0.6},
    {"name1": "Jalen Johnson", "name2": "Duncan Robinson", "name3": "Santi Aldama", "line1": 22.5, "line2": 10.5, "line3": 17.5, "prediction1": 25.33, "prediction2": 13.07, "prediction3": 14.94, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 57.61, "kelly": 0.115, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.666, "prob2": 0.664, "prob3": 0.66, "hitRate1": 62.1, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 83.1, "l5_2": 1.0, "l15_2": 0.67, "hitRate3": 64.1, "l5_3": 0.4, "l15_3": 0.13},
    {"name1": "Jordan Clarkson", "name2": "Jeremiah Fears", "name3": "Bobby Portis", "line1": 9.5, "line2": 16.5, "line3": 15.5, "prediction1": 11.95, "prediction2": 18.78, "prediction3": 13.15, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 48.05, "kelly": 0.096, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.658, "prob2": 0.64, "prob3": 0.651, "hitRate1": 66.9, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 60.3, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 77.3, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "D'Angelo Russell", "name2": "Zach LaVine", "name3": "DeMar DeRozan", "line1": 12.5, "line2": 18.5, "line3": 17.5, "prediction1": 14.79, "prediction2": 20.85, "prediction3": 15.35, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 36.64, "kelly": 0.073, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.63, "prob2": 0.639, "prob3": 0.628, "hitRate1": 42.5, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 33.3, "l5_2": 0.4, "l15_2": 0.6, "hitRate3": 71.3, "l5_3": 0.2, "l15_3": 0.67},
    {"name1": "Nickeil Alexander-Walker", "name2": "Zaccharie Risacher", "name3": "Alex Sarr", "line1": 18.5, "line2": 12.0, "line3": 17.5, "prediction1": 20.56, "prediction2": 13.81, "prediction3": 19.36, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 24.78, "kelly": 0.05, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.619, "prob2": 0.612, "prob3": 0.61, "hitRate1": 74.9, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 50.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 55.4, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Anthony Black", "name2": "Goga Bitadze", "name3": "Jamal Murray", "line1": 11.5, "line2": 4.5, "line3": 23.5, "prediction1": 13.15, "prediction2": 5.57, "prediction3": 25.47, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 19.32, "kelly": 0.039, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "prob1": 0.602, "prob2": 0.605, "prob3": 0.607, "hitRate1": 51.0, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 76.5, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 40.8, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Landry Shamet", "name2": "Caris LeVert", "name3": "Dennis Schr\u00f6der", "line1": 9.0, "line2": 7.5, "line3": 11.5, "prediction1": 10.33, "prediction2": 8.82, "prediction3": 13.25, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 17.83, "kelly": 0.036, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.602, "prob2": 0.597, "prob3": 0.607, "hitRate1": 80.0, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 86.5, "l5_2": 1.0, "l15_2": 0.47, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Coby White", "name2": "Cameron Johnson", "line1": 20.5, "line2": 13.5, "prediction1": 25.74, "prediction2": 10.05, "side1": "over", "side2": "under", "recommendation": 0, "ev": 76.44, "kelly": 0.382, "sigma1": "Med", "sigma2": "Med", "prob1": 0.825, "prob2": 0.728, "hitRate1": 30.1, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 94.4, "l5_2": 0.4, "l15_2": 0.2},
    {"name1": "Jonathan Isaac", "name2": "Ayo Dosunmu", "line1": 3.5, "line2": 14.5, "prediction1": 6.04, "prediction2": 19.48, "side1": "over", "side2": "over", "recommendation": 0, "ev": 63.75, "kelly": 0.319, "sigma1": "Low", "sigma2": "High", "prob1": 0.711, "prob2": 0.783, "hitRate1": 46.9, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 66.4, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Karl-Anthony Towns", "name2": "Myles Turner", "line1": 21.5, "line2": 15.5, "prediction1": 24.58, "prediction2": 12.32, "side1": "over", "side2": "under", "recommendation": 0, "ev": 37.16, "kelly": 0.186, "sigma1": "High", "sigma2": "High", "prob1": 0.669, "prob2": 0.697, "hitRate1": 56.3, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 60.8, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Duncan Robinson", "name2": "Santi Aldama", "line1": 10.5, "line2": 17.5, "prediction1": 13.07, "prediction2": 14.94, "side1": "over", "side2": "under", "recommendation": 0, "ev": 28.89, "kelly": 0.144, "sigma1": "High", "sigma2": "High", "prob1": 0.664, "prob2": 0.66, "hitRate1": 83.1, "l5_1": 1.0, "l15_1": 0.67, "hitRate2": 64.1, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Jordan Clarkson", "name2": "Bobby Portis", "line1": 9.5, "line2": 15.5, "prediction1": 11.95, "prediction2": 13.15, "side1": "over", "side2": "under", "recommendation": 0, "ev": 25.98, "kelly": 0.13, "sigma1": "High", "sigma2": "High", "prob1": 0.658, "prob2": 0.651, "hitRate1": 66.9, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 77.3, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "Landry Shamet", "name2": "Jeremiah Fears", "line1": 8.5, "line2": 16.5, "prediction1": 10.33, "prediction2": 18.78, "side1": "over", "side2": "over", "recommendation": 0, "ev": 20.33, "kelly": 0.102, "sigma1": "Med", "sigma2": "High", "prob1": 0.64, "prob2": 0.64, "hitRate1": 87.6, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 60.3, "l5_2": 0.8, "l15_2": 0.4},
    {"name1": "Nickeil Alexander-Walker", "name2": "Zach LaVine", "line1": 18.5, "line2": 18.5, "prediction1": 20.56, "prediction2": 20.85, "side1": "over", "side2": "over", "recommendation": 0, "ev": 16.24, "kelly": 0.081, "sigma1": "High", "sigma2": "High", "prob1": 0.619, "prob2": 0.639, "hitRate1": 74.9, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 33.3, "l5_2": 0.4, "l15_2": 0.6},
    {"name1": "Alex Sarr", "name2": "DeMar DeRozan", "line1": 17.5, "line2": 17.5, "prediction1": 19.36, "prediction2": 15.35, "side1": "over", "side2": "under", "recommendation": 0, "ev": 12.76, "kelly": 0.064, "sigma1": "High", "sigma2": "High", "prob1": 0.61, "prob2": 0.628, "hitRate1": 55.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 71.3, "l5_2": 0.2, "l15_2": 0.67},
    {"name1": "Goga Bitadze", "name2": "Jamal Murray", "line1": 4.5, "line2": 23.5, "prediction1": 5.57, "prediction2": 25.47, "side1": "over", "side2": "over", "recommendation": 0, "ev": 7.91, "kelly": 0.04, "sigma1": "Low", "sigma2": "High", "prob1": 0.605, "prob2": 0.607, "hitRate1": 76.5, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 40.8, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Caris LeVert", "name2": "Dennis Schr\u00f6der", "line1": 7.5, "line2": 11.5, "prediction1": 8.82, "prediction2": 13.25, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.48, "kelly": 0.032, "sigma1": "Med", "sigma2": "High", "prob1": 0.597, "prob2": 0.607, "hitRate1": 86.5, "l5_1": 1.0, "l15_1": 0.47, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
];const underdogTriosData = [
    {"name1": "Coby White", "name2": "Ayo Dosunmu", "name3": "Cameron Johnson", "line1": 20.5, "line2": 14.5, "line3": 13.5, "prediction1": 25.74, "prediction2": 19.48, "prediction3": 10.05, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 153.83, "kelly": 0.308, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.825, "prob2": 0.783, "prob3": 0.728, "hitRate1": 30.1, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 66.4, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 94.4, "l5_3": 0.4, "l15_3": 0.2},
    {"name1": "Jonathan Isaac", "name2": "Myles Turner", "name3": "Santi Aldama", "line1": 3.5, "line2": 15.5, "line3": 17.5, "prediction1": 6.04, "prediction2": 12.32, "prediction3": 14.94, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 76.8, "kelly": 0.154, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.711, "prob2": 0.697, "prob3": 0.66, "hitRate1": 46.9, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 60.8, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 64.1, "l5_3": 0.4, "l15_3": 0.13},
    {"name1": "Karl-Anthony Towns", "name2": "Jordan Clarkson", "name3": "Duncan Robinson", "line1": 21.5, "line2": 9.5, "line3": 10.5, "prediction1": 24.58, "prediction2": 11.95, "prediction3": 13.07, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 57.89, "kelly": 0.116, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.669, "prob2": 0.658, "prob3": 0.664, "hitRate1": 56.3, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 66.9, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 83.1, "l5_3": 1.0, "l15_3": 0.67},
    {"name1": "Landry Shamet", "name2": "Jeremiah Fears", "name3": "Bobby Portis", "line1": 8.5, "line2": 16.5, "line3": 15.5, "prediction1": 10.33, "prediction2": 18.78, "prediction3": 13.15, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 43.88, "kelly": 0.088, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.64, "prob2": 0.64, "prob3": 0.651, "hitRate1": 87.6, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 60.3, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 77.3, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Nickeil Alexander-Walker", "name2": "Zach LaVine", "name3": "DeMar DeRozan", "line1": 18.5, "line2": 18.5, "line3": 17.5, "prediction1": 20.56, "prediction2": 20.85, "prediction3": 15.35, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 34.18, "kelly": 0.068, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.619, "prob2": 0.639, "prob3": 0.628, "hitRate1": 74.9, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 33.3, "l5_2": 0.4, "l15_2": 0.6, "hitRate3": 71.3, "l5_3": 0.2, "l15_3": 0.67},
    {"name1": "Goga Bitadze", "name2": "Alex Sarr", "name3": "Jamal Murray", "line1": 4.5, "line2": 17.5, "line3": 23.5, "prediction1": 5.57, "prediction2": 19.36, "prediction3": 25.47, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 20.96, "kelly": 0.042, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.605, "prob2": 0.61, "prob3": 0.607, "hitRate1": 76.5, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 55.4, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 40.8, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Caris LeVert", "name2": "Brandon Williams", "name3": "Dennis Schr\u00f6der", "line1": 7.5, "line2": 14.5, "line3": 11.5, "prediction1": 8.82, "prediction2": 12.86, "prediction3": 13.25, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 16.33, "kelly": 0.033, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.597, "prob2": 0.595, "prob3": 0.607, "hitRate1": 86.5, "l5_1": 1.0, "l15_1": 0.47, "hitRate2": 75.6, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Mikal Bridges", "name2": "Cooper Flagg", "name3": "Bruce Brown", "line1": 15.5, "line2": 17.5, "line3": 8.5, "prediction1": 17.01, "prediction2": 19.04, "prediction3": 7.16, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 12.56, "kelly": 0.025, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.591, "prob2": 0.59, "prob3": 0.598, "hitRate1": 52.5, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 55.8, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 71.1, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Cam Whitmore", "name2": "Kentavious Caldwell-Pope", "name3": "Malik Monk", "line1": 10.5, "line2": 7.5, "line3": 11.5, "prediction1": 9.41, "prediction2": 6.31, "prediction3": 10.16, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 9.71, "kelly": 0.019, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "prob1": 0.589, "prob2": 0.59, "prob3": 0.585, "hitRate1": 50.8, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 79.6, "l5_2": 0.0, "l15_2": 0.4, "hitRate3": 79.2, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Derik Queen", "name2": "Josh Giddey", "name3": "Klay Thompson", "line1": 15.5, "line2": 21.5, "line3": 10.5, "prediction1": 16.89, "prediction2": 22.93, "prediction3": 9.23, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 6.5, "kelly": 0.013, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.584, "prob2": 0.58, "prob3": 0.582, "hitRate1": 50.2, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 27.8, "l5_2": 0.2, "l15_2": 0.27, "hitRate3": 50.1, "l5_3": 0.6, "l15_3": 0.4},
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
    {"name": "Jordan Clarkson", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.669, "underPct": 0.331},
    {"name": "Josh Hart", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.666, "underPct": 0.334},
    {"name": "Ayo Dosunmu", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.664, "underPct": 0.336},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.649, "underPct": 0.351},
    {"name": "Jalen Johnson", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.621, "underPct": 0.379},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.603, "underPct": 0.397},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.574, "underPct": 0.426},
    {"name": "Kyle Kuzma", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.566, "underPct": 0.434},
    {"name": "Karl-Anthony Towns", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.559, "underPct": 0.441},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.558, "underPct": 0.442},
    {"name": "Naji Marshall", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.554, "underPct": 0.446},
    {"name": "Alex Sarr", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.554, "underPct": 0.446},
    {"name": "Tre Johnson", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.526, "underPct": 0.474},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Anthony Black", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.51, "underPct": 0.49},
    {"name": "Derik Queen", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.502, "underPct": 0.498},
    {"name": "Zaccharie Risacher", "line": 12.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.499, "underPct": 0.501},
    {"name": "Cam Whitmore", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.492, "underPct": 0.508},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.49, "underPct": 0.51},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.471, "underPct": 0.529},
    {"name": "Jonathan Isaac", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.469, "underPct": 0.531},
    {"name": "Franz Wagner", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.455, "underPct": 0.545},
    {"name": "Luke Kennard", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.445, "underPct": 0.555},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.44, "underPct": 0.56},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.435, "underPct": 0.565},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.425, "underPct": 0.575},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Jalen Suggs", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.414, "underPct": 0.586},
    {"name": "Cedric Coward", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.413, "underPct": 0.587},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.408, "underPct": 0.592},
    {"name": "Myles Turner", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.392, "underPct": 0.608},
    {"name": "Tristan da Silva", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.382, "underPct": 0.618},
    {"name": "Matas Buzelis", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.372, "underPct": 0.628},
    {"name": "Ryan Rollins", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.367, "underPct": 0.633},
    {"name": "Santi Aldama", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.359, "underPct": 0.641},
    {"name": "Corey Kispert", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.344, "underPct": 0.656},
    {"name": "Zach LaVine", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.333, "underPct": 0.667},
    {"name": "Jaylen Wells", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.323, "underPct": 0.677},
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
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.264, "underPct": 0.736},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.264, "underPct": 0.736},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.244, "underPct": 0.756},
    {"name": "Peyton Watson", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.233, "underPct": 0.767},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.229, "underPct": 0.771},
    {"name": "Bobby Portis", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.227, "underPct": 0.773},
    {"name": "Dyson Daniels", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.21, "underPct": 0.79},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.208, "underPct": 0.792},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.204, "underPct": 0.796},
    {"name": "Keegan Murray", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.198, "underPct": 0.802},
    {"name": "Patrick Williams", "line": 10.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.091, "underPct": 0.909},
    {"name": "Cole Anthony", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.068, "underPct": 0.932},
    {"name": "Cameron Johnson", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.056, "underPct": 0.944},
];const prizepicksAssistsHitRates = [
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.753, "underPct": 0.247},
    {"name": "Cade Cunningham", "line": 9.0, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.672, "underPct": 0.328},
    {"name": "Jamal Murray", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.666, "underPct": 0.334},
    {"name": "Miles McBride", "line": 1.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.65, "underPct": 0.35},
    {"name": "Josh Hart", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.623, "underPct": 0.377},
    {"name": "Coby White", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.575, "underPct": 0.425},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.569, "underPct": 0.431},
    {"name": "Jalen Johnson", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.55, "underPct": 0.45},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.549, "underPct": 0.451},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Kentavious Caldwell-Pope", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.526, "underPct": 0.474},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.524, "underPct": 0.476},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.487, "underPct": 0.513},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.485, "underPct": 0.515},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.464, "underPct": 0.536},
    {"name": "Brandon Williams", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.458, "underPct": 0.542},
    {"name": "Ryan Rollins", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.453, "underPct": 0.547},
    {"name": "Cole Anthony", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Jalen Suggs", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.427, "underPct": 0.573},
    {"name": "Corey Kispert", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.415, "underPct": 0.585},
    {"name": "Franz Wagner", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.384, "underPct": 0.616},
    {"name": "Tristan da Silva", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.338, "underPct": 0.662},
    {"name": "Derik Queen", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.299, "underPct": 0.701},
];const prizepicksReboundsHitRates = [
    {"name": "Jock Landale", "line": 4.5, "l5": 1.0, "l10": 0.9, "l15": 0.87, "overPct": 0.758, "underPct": 0.242},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.642, "underPct": 0.358},
    {"name": "Jalen Duren", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.619, "underPct": 0.381},
    {"name": "Franz Wagner", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.609, "underPct": 0.391},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.596, "underPct": 0.404},
    {"name": "Miles McBride", "line": 1.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.585, "underPct": 0.415},
    {"name": "Ausar Thompson", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.577, "underPct": 0.423},
    {"name": "Tobias Harris", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.571, "underPct": 0.429},
    {"name": "Duncan Robinson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.552, "underPct": 0.448},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.532, "underPct": 0.468},
    {"name": "Cade Cunningham", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.521, "underPct": 0.479},
    {"name": "Santi Aldama", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.506, "underPct": 0.494},
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
    {"name": "Bilal Coulibaly", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.391, "underPct": 0.609},
    {"name": "P.J. Washington", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.391, "underPct": 0.609},
    {"name": "Daniel Gafford", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.382, "underPct": 0.618},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.371, "underPct": 0.629},
    {"name": "Ryan Rollins", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.359, "underPct": 0.641},
    {"name": "Cameron Johnson", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.344, "underPct": 0.656},
    {"name": "Onyeka Okongwu", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.337, "underPct": 0.663},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.31, "underPct": 0.69},
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
    {"name": "Drew Eubanks", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.443, "underPct": 0.557},
    {"name": "Zach LaVine", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.528, "underPct": 0.472},
    {"name": "Malik Monk", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.548, "underPct": 0.452},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Jalen Duren", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 23.5, "l5": 1.0, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 40.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Duncan Robinson", "line": 14.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Tobias Harris", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tristan da Silva", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Landry Shamet", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles McBride", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Smith", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Whitmore", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaac Okoro", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyshawn George", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Naji Marshall", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jock Landale", "line": 14.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jonathan Isaac", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kennard", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mitchell Robinson", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Coby White", "line": 28.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cedric Coward", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tre Johnson", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Corey Kispert", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 30.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Konchar", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bobby Portis", "line": 25.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Williams", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 18.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Cole Anthony", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Duncan Robinson", "line": 12.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Duren", "line": 30.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 1.0, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 16.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Landry Shamet", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jonathan Isaac", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Whitmore", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Brunson", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Johnson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 30.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 24.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Peyton Watson", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Spencer", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Corey Kispert", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Matas Buzelis", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 23.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Precious Achiuwa", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaac Okoro", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Wells", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jalen Smith", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cameron Johnson", "line": 17.0, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Drew Eubanks", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cole Anthony", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kentavious Caldwell-Pope", "line": 8.5, "l5": 0.0, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Patrick Williams", "line": 12.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Keegan Murray", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Zach Edey", "line": 22.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPAHitRates = [
    {"name": "Duncan Robinson", "line": 11.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jalen Duren", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ayo Dosunmu", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Johnson", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Rollins", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Landry Shamet", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Goga Bitadze", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "P.J. Washington", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Wells", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jock Landale", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 34.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Corey Kispert", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bilal Coulibaly", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Huerter", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaac Okoro", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cole Anthony", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 31.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Peyton Watson", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Williams", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Patrick Williams", "line": 10.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keegan Murray", "line": 12.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksRAHitRates = [
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Daniel Gafford", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 9.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 19.0, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kentavious Caldwell-Pope", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Santi Aldama", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tobias Harris", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cole Anthony", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alex Sarr", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 9.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 13.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 15.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaac Okoro", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cameron Johnson", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Wells", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kyle Kuzma", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trey Murphy III", "line": 9.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Huerter", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Whitmore", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const prizepicksTurnoversHitRates = [
    {"name": "Desmond Bane", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mitchell Robinson", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Smith", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Karl-Anthony Towns", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Landry Shamet", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jose Alvarado", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Corey Kispert", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bobby Portis", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Santi Aldama", "line": 1.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Ausar Thompson", "line": 1.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Anthony Black", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Konchar", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mitchell Robinson", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const underdogPointsHitRates = [
    {"name": "Landry Shamet", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.876, "underPct": 0.124},
    {"name": "Caris LeVert", "line": 7.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.865, "underPct": 0.135},
    {"name": "Duncan Robinson", "line": 10.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.831, "underPct": 0.169},
    {"name": "Onyeka Okongwu", "line": 14.5, "l5": 1.0, "l10": 0.5, "l15": 0.47, "overPct": 0.795, "underPct": 0.205},
    {"name": "Goga Bitadze", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.765, "underPct": 0.235},
    {"name": "Cade Cunningham", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.753, "underPct": 0.247},
    {"name": "Nickeil Alexander-Walker", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.749, "underPct": 0.251},
    {"name": "Jordan Clarkson", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.669, "underPct": 0.331},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.666, "underPct": 0.334},
    {"name": "Ayo Dosunmu", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.664, "underPct": 0.336},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.603, "underPct": 0.397},
    {"name": "Karl-Anthony Towns", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.559, "underPct": 0.441},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.558, "underPct": 0.442},
    {"name": "Alex Sarr", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.554, "underPct": 0.446},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.536, "underPct": 0.464},
    {"name": "Tre Johnson", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.526, "underPct": 0.474},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Derik Queen", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.502, "underPct": 0.498},
    {"name": "Zaccharie Risacher", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.499, "underPct": 0.501},
    {"name": "Cam Whitmore", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.492, "underPct": 0.508},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.49, "underPct": 0.51},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.471, "underPct": 0.529},
    {"name": "Jonathan Isaac", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.469, "underPct": 0.531},
    {"name": "Luke Kennard", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.445, "underPct": 0.555},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.44, "underPct": 0.56},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.435, "underPct": 0.565},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.408, "underPct": 0.592},
    {"name": "Myles Turner", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.392, "underPct": 0.608},
    {"name": "Matas Buzelis", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.372, "underPct": 0.628},
    {"name": "Ryan Rollins", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.367, "underPct": 0.633},
    {"name": "Santi Aldama", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.359, "underPct": 0.641},
    {"name": "Corey Kispert", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.344, "underPct": 0.656},
    {"name": "Zach LaVine", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.333, "underPct": 0.667},
    {"name": "Jaylen Wells", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.323, "underPct": 0.677},
    {"name": "D'Angelo Russell", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.32, "underPct": 0.68},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.318, "underPct": 0.682},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.302, "underPct": 0.698},
    {"name": "Coby White", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.301, "underPct": 0.699},
    {"name": "Jose Alvarado", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.301, "underPct": 0.699},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.289, "underPct": 0.711},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.287, "underPct": 0.713},
    {"name": "Josh Giddey", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.278, "underPct": 0.722},
    {"name": "Cam Spencer", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.268, "underPct": 0.732},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.264, "underPct": 0.736},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.244, "underPct": 0.756},
    {"name": "Bobby Portis", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.227, "underPct": 0.773},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.208, "underPct": 0.792},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.204, "underPct": 0.796},
    {"name": "Patrick Williams", "line": 9.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.159, "underPct": 0.841},
    {"name": "Cole Anthony", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.068, "underPct": 0.932},
    {"name": "Cameron Johnson", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.056, "underPct": 0.944},
];const underdogAssistsHitRates = [
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.752, "underPct": 0.248},
    {"name": "Miles McBride", "line": 1.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.65, "underPct": 0.35},
    {"name": "Coby White", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.575, "underPct": 0.425},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.569, "underPct": 0.431},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Kentavious Caldwell-Pope", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.526, "underPct": 0.474},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.524, "underPct": 0.476},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.491, "underPct": 0.509},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.487, "underPct": 0.513},
    {"name": "Corey Kispert", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.415, "underPct": 0.585},
    {"name": "Ayo Dosunmu", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.384, "underPct": 0.616},
    {"name": "P.J. Washington", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.357, "underPct": 0.643},
    {"name": "Tristan da Silva", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.338, "underPct": 0.662},
];const underdogReboundsHitRates = [
    {"name": "Cedric Coward", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.737, "underPct": 0.263},
    {"name": "Jalen Duren", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.619, "underPct": 0.381},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.596, "underPct": 0.404},
    {"name": "Cade Cunningham", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.521, "underPct": 0.479},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.485, "underPct": 0.515},
    {"name": "Mikal Bridges", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.471, "underPct": 0.529},
    {"name": "Cooper Flagg", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.414, "underPct": 0.586},
    {"name": "P.J. Washington", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.391, "underPct": 0.609},
    {"name": "Onyeka Okongwu", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.337, "underPct": 0.663},
    {"name": "Cam Spencer", "line": 2.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.264, "underPct": 0.736},
    {"name": "Drew Eubanks", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.212, "underPct": 0.788},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
    {"name": "Jalen Johnson", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.559, "underPct": 0.441},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Jalen Johnson", "line": 40.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Duncan Robinson", "line": 14.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Miles McBride", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tobias Harris", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Landry Shamet", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaac Okoro", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Smith", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kennard", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Spencer", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mitchell Robinson", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jose Alvarado", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Corey Kispert", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 28.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Saddiq Bey", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Drew Eubanks", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Konchar", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Trey Murphy III", "line": 30.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 23.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Cole Anthony", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Patrick Williams", "line": 14.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
];const underdogPRHitRates = [
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 1.0, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 30.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Alex Sarr", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPAHitRates = [
    {"name": "Jalen Duren", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 36.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alex Sarr", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Coby White", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Brunson", "line": 34.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 31.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const underdogRAHitRates = [
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Santi Aldama", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cole Anthony", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Alex Sarr", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const underdogTurnoversHitRates = [
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

