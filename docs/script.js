const prizepicksSinglesData = [
    {"name": "Lauri Markkanen", "bookmaker": "BetRivers", "line": 27.5, "prediction": 34.77, "side": "Over", "odds": 108, "recommendation": 1, "ev": 6.73, "roi": 67.3, "kelly": 0.623, "sigma": "High"},
    {"name": "Lauri Markkanen", "bookmaker": "FanDuel", "line": 26.5, "prediction": 34.77, "side": "Over", "odds": 102, "recommendation": 1, "ev": 6.73, "roi": 67.3, "kelly": 0.66, "sigma": "High"},
    {"name": "Lauri Markkanen", "bookmaker": "BetRivers", "line": 26.5, "prediction": 34.77, "side": "Over", "odds": -107, "recommendation": 1, "ev": 6.28, "roi": 62.8, "kelly": 0.672, "sigma": "High"},
    {"name": "Keyonte George", "bookmaker": "BetRivers", "line": 20.5, "prediction": 25.05, "side": "Over", "odds": 123, "recommendation": 1, "ev": 6.15, "roi": 61.5, "kelly": 0.5, "sigma": "High"},
    {"name": "Lauri Markkanen", "bookmaker": "BetMGM", "line": 26.5, "prediction": 34.77, "side": "Over", "odds": -110, "recommendation": 1, "ev": 5.96, "roi": 59.6, "kelly": 0.656, "sigma": "High"},
    {"name": "Nick Richards", "bookmaker": "DraftKings", "line": 4.5, "prediction": 7.39, "side": "Over", "odds": 107, "recommendation": 0, "ev": 5.89, "roi": 58.9, "kelly": 0.55, "sigma": "Med"},
    {"name": "Lauri Markkanen", "bookmaker": "DraftKings", "line": 26.5, "prediction": 34.77, "side": "Over", "odds": -113, "recommendation": 1, "ev": 5.79, "roi": 57.9, "kelly": 0.654, "sigma": "High"},
    {"name": "Marcus Smart", "bookmaker": "DraftKings", "line": 7.5, "prediction": 13.65, "side": "Over", "odds": -112, "recommendation": 1, "ev": 5.71, "roi": 57.1, "kelly": 0.639, "sigma": "High"},
    {"name": "Marcus Smart", "bookmaker": "FanDuel", "line": 7.5, "prediction": 13.65, "side": "Over", "odds": -113, "recommendation": 1, "ev": 5.7, "roi": 57.0, "kelly": 0.644, "sigma": "High"},
    {"name": "Lauri Markkanen", "bookmaker": "BetRivers", "line": 25.5, "prediction": 34.77, "side": "Over", "odds": -125, "recommendation": 1, "ev": 5.49, "roi": 54.9, "kelly": 0.686, "sigma": "High"},
    {"name": "Keyonte George", "bookmaker": "BetRivers", "line": 19.5, "prediction": 25.05, "side": "Over", "odds": 104, "recommendation": 1, "ev": 5.45, "roi": 54.5, "kelly": 0.524, "sigma": "High"},
    {"name": "Nick Richards", "bookmaker": "FanDuel", "line": 4.5, "prediction": 7.39, "side": "Over", "odds": -106, "recommendation": 0, "ev": 5.16, "roi": 51.6, "kelly": 0.547, "sigma": "Med"},
    {"name": "Keyonte George", "bookmaker": "FanDuel", "line": 18.5, "prediction": 25.05, "side": "Over", "odds": -108, "recommendation": 1, "ev": 5.13, "roi": 51.3, "kelly": 0.554, "sigma": "High"},
    {"name": "Keyonte George", "bookmaker": "BetMGM", "line": 19.5, "prediction": 25.05, "side": "Over", "odds": 100, "recommendation": 1, "ev": 5.11, "roi": 51.1, "kelly": 0.511, "sigma": "High"},
    {"name": "Isaiah Collier", "bookmaker": "BetRivers", "line": 9.5, "prediction": 12.69, "side": "Over", "odds": 110, "recommendation": 0, "ev": 5.09, "roi": 50.9, "kelly": 0.463, "sigma": "Med"},
];const prizepicksPairsData = [
    {"name1": "Lauri Markkanen", "name2": "Marcus Smart", "line1": 26.0, "line2": 7.0, "prediction1": 34.77, "prediction2": 13.65, "side1": "over", "side2": "over", "recommendation": 1, "ev": 9.47, "kelly": 0.474, "sigma1": "High", "sigma2": "High", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 83.7, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Lauri Markkanen", "name2": "Nick Richards", "line1": 26.0, "line2": 4.0, "prediction1": 34.77, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 8.64, "kelly": 0.432, "sigma1": "High", "sigma2": "Med", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Marcus Smart", "name2": "Nick Richards", "line1": 7.0, "line2": 4.0, "prediction1": 13.65, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 8.49, "kelly": 0.425, "sigma1": "High", "sigma2": "Med", "hitRate1": 83.7, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Keyonte George", "name2": "Marcus Smart", "line1": 18.5, "line2": 7.0, "prediction1": 25.05, "prediction2": 13.65, "side1": "over", "side2": "over", "recommendation": 1, "ev": 8.07, "kelly": 0.403, "sigma1": "High", "sigma2": "High", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 83.7, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Lauri Markkanen", "name2": "Dillon Brooks", "line1": 26.0, "line2": 18.5, "prediction1": 34.77, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.62, "kelly": 0.381, "sigma1": "High", "sigma2": "High", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Keyonte George", "name2": "Nick Richards", "line1": 18.5, "line2": 4.0, "prediction1": 25.05, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 7.32, "kelly": 0.366, "sigma1": "High", "sigma2": "Med", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Anthony Black", "name2": "Keyonte George", "line1": 9.5, "line2": 18.5, "prediction1": 14.28, "prediction2": 25.05, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.46, "kelly": 0.323, "sigma1": "High", "sigma2": "High", "hitRate1": 52.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 75.3, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Isaiah Collier", "name2": "Dillon Brooks", "line1": 8.5, "line2": 18.5, "prediction1": 12.69, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.24, "kelly": 0.312, "sigma1": "Med", "sigma2": "High", "hitRate1": 75.2, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Anthony Black", "name2": "Isaiah Collier", "line1": 9.5, "line2": 8.5, "prediction1": 14.28, "prediction2": 12.69, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.01, "kelly": 0.301, "sigma1": "High", "sigma2": "Med", "hitRate1": 52.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 75.2, "l5_2": 0.8, "l15_2": 0.27},
    {"name1": "Isaiah Collier", "name2": "Jake LaRavia", "line1": 8.5, "line2": 7.0, "prediction1": 12.69, "prediction2": 10.95, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.91, "kelly": 0.296, "sigma1": "Med", "sigma2": "High", "hitRate1": 75.2, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 51.8, "l5_2": 0.2, "l15_2": 0.47},
];const prizepicksTriosData = [
    {"name1": "Lauri Markkanen", "name2": "Marcus Smart", "name3": "Nick Richards", "line1": 26.0, "line2": 7.0, "line3": 4.0, "prediction1": 34.77, "prediction2": 13.65, "prediction3": 7.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 18.35, "kelly": 0.367, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 83.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 26.3, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Lauri Markkanen", "name2": "Marcus Smart", "name3": "Dillon Brooks", "line1": 26.0, "line2": 7.0, "line3": 18.5, "prediction1": 34.77, "prediction2": 13.65, "prediction3": 23.84, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 17.07, "kelly": 0.341, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 83.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 75.9, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Keyonte George", "name2": "Isaiah Collier", "name3": "Nick Richards", "line1": 18.5, "line2": 8.5, "line3": 4.0, "prediction1": 25.05, "prediction2": 12.69, "prediction3": 7.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 14.01, "kelly": 0.28, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.2, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 26.3, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Keyonte George", "name2": "Isaiah Collier", "name3": "Dillon Brooks", "line1": 18.5, "line2": 8.5, "line3": 18.5, "prediction1": 25.05, "prediction2": 12.69, "prediction3": 23.84, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 12.92, "kelly": 0.258, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.2, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 75.9, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Anthony Black", "name2": "Day'Ron Sharpe", "name3": "Jake LaRavia", "line1": 9.5, "line2": 5.5, "line3": 7.0, "prediction1": 14.28, "prediction2": 8.4, "prediction3": 10.95, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.21, "kelly": 0.224, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 52.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 46.3, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 51.8, "l5_3": 0.2, "l15_3": 0.47},
    {"name1": "Anthony Black", "name2": "Draymond Green", "name3": "Jake LaRavia", "line1": 9.5, "line2": 7.5, "line3": 7.0, "prediction1": 14.28, "prediction2": 11.05, "prediction3": 10.95, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.19, "kelly": 0.224, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 52.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 35.0, "l5_2": 0.2, "l15_2": 0.4, "hitRate3": 51.8, "l5_3": 0.2, "l15_3": 0.47},
    {"name1": "Tristan da Silva", "name2": "Draymond Green", "name3": "Day'Ron Sharpe", "line1": 11.5, "line2": 7.5, "line3": 5.5, "prediction1": 16.21, "prediction2": 11.05, "prediction3": 8.4, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 10.35, "kelly": 0.207, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "hitRate1": 56.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 35.0, "l5_2": 0.2, "l15_2": 0.4, "hitRate3": 46.3, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Tristan da Silva", "name2": "Will Richard", "name3": "Ryan Kalkbrenner", "line1": 11.5, "line2": 7.5, "line3": 8.5, "prediction1": 16.21, "prediction2": 11.13, "prediction3": 11.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.72, "kelly": 0.194, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 56.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 91.2, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 61.9, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Will Richard", "name2": "Svi Mykhailiuk", "name3": "Ryan Kalkbrenner", "line1": 7.5, "line2": 7.5, "line3": 8.5, "prediction1": 11.13, "prediction2": 10.66, "prediction3": 11.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.26, "kelly": 0.185, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 91.2, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 88.9, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 61.9, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Cade Cunningham", "name2": "Rui Hachimura", "name3": "Svi Mykhailiuk", "line1": 24.5, "line2": 11.5, "line3": 7.5, "prediction1": 29.03, "prediction2": 15.02, "prediction3": 10.66, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.76, "kelly": 0.155, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 89.1, "l5_1": 1.0, "l15_1": 0.47, "hitRate2": 89.9, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 88.9, "l5_3": 0.6, "l15_3": 0.53},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Lauri Markkanen", "name2": "Nick Richards", "line1": 26.5, "line2": 4.5, "prediction1": 34.77, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 7.51, "kelly": 0.376, "sigma1": "High", "sigma2": "Med", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Lauri Markkanen", "name2": "Dillon Brooks", "line1": 26.5, "line2": 18.5, "prediction1": 34.77, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.3, "kelly": 0.365, "sigma1": "High", "sigma2": "High", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Anthony Black", "name2": "Lauri Markkanen", "line1": 9.5, "line2": 26.5, "prediction1": 14.28, "prediction2": 34.77, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.27, "kelly": 0.364, "sigma1": "High", "sigma2": "High", "hitRate1": 52.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 60.4, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Anthony Black", "name2": "Keyonte George", "line1": 9.5, "line2": 18.5, "prediction1": 14.28, "prediction2": 25.05, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.55, "kelly": 0.327, "sigma1": "High", "sigma2": "High", "hitRate1": 52.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 75.3, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Keyonte George", "name2": "Nick Richards", "line1": 18.5, "line2": 4.5, "prediction1": 25.05, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.52, "kelly": 0.326, "sigma1": "High", "sigma2": "Med", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Keyonte George", "name2": "Dillon Brooks", "line1": 18.5, "line2": 18.5, "prediction1": 25.05, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.45, "kelly": 0.322, "sigma1": "High", "sigma2": "High", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Draymond Green", "name2": "Anthony Black", "line1": 7.5, "line2": 9.5, "prediction1": 11.05, "prediction2": 14.28, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.8, "kelly": 0.29, "sigma1": "Med", "sigma2": "High", "hitRate1": 35.0, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 52.4, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Draymond Green", "name2": "Nick Richards", "line1": 7.5, "line2": 4.5, "prediction1": 11.05, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.7, "kelly": 0.285, "sigma1": "Med", "sigma2": "Med", "hitRate1": 35.0, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Draymond Green", "name2": "Dillon Brooks", "line1": 7.5, "line2": 18.5, "prediction1": 11.05, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.68, "kelly": 0.284, "sigma1": "Med", "sigma2": "High", "hitRate1": 35.0, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Tristan da Silva", "name2": "Day'Ron Sharpe", "line1": 11.5, "line2": 5.5, "prediction1": 16.21, "prediction2": 8.4, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.81, "kelly": 0.24, "sigma1": "High", "sigma2": "Med", "hitRate1": 56.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 46.3, "l5_2": 0.6, "l15_2": 0.4},
];const underdogTriosData = [
    {"name1": "Lauri Markkanen", "name2": "Keyonte George", "name3": "Nick Richards", "line1": 26.5, "line2": 18.5, "line3": 4.5, "prediction1": 34.77, "prediction2": 25.05, "prediction3": 7.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 14.85, "kelly": 0.297, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.3, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 26.3, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Lauri Markkanen", "name2": "Keyonte George", "name3": "Dillon Brooks", "line1": 26.5, "line2": 18.5, "line3": 18.5, "prediction1": 34.77, "prediction2": 25.05, "prediction3": 23.84, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 14.8, "kelly": 0.296, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.3, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 75.9, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Anthony Black", "name2": "Dillon Brooks", "name3": "Nick Richards", "line1": 9.5, "line2": 18.5, "line3": 4.5, "prediction1": 14.28, "prediction2": 23.84, "prediction3": 7.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 12.06, "kelly": 0.241, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 52.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 26.3, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Draymond Green", "name2": "Anthony Black", "name3": "Day'Ron Sharpe", "line1": 7.5, "line2": 9.5, "line3": 5.5, "prediction1": 11.05, "prediction2": 14.28, "prediction3": 8.4, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.01, "kelly": 0.22, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "hitRate1": 35.0, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 52.4, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 46.3, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Draymond Green", "name2": "Tristan da Silva", "name3": "Day'Ron Sharpe", "line1": 7.5, "line2": 11.5, "line3": 5.5, "prediction1": 11.05, "prediction2": 16.21, "prediction3": 8.4, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 10.35, "kelly": 0.207, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "hitRate1": 35.0, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 56.1, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 46.3, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Tristan da Silva", "name2": "Dyson Daniels", "name3": "Svi Mykhailiuk", "line1": 11.5, "line2": 13.5, "line3": 7.5, "prediction1": 16.21, "prediction2": 10.12, "prediction3": 10.66, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 8.14, "kelly": 0.163, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 56.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 93.4, "l5_2": 0.0, "l15_2": 0.2, "hitRate3": 88.9, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Dyson Daniels", "name2": "Julian Champagnie", "name3": "Svi Mykhailiuk", "line1": 13.5, "line2": 12.5, "line3": 7.5, "prediction1": 10.12, "prediction2": 8.89, "prediction3": 10.66, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 6.94, "kelly": 0.139, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 93.4, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 66.3, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 88.9, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Buddy Hield", "name2": "Julian Champagnie", "name3": "Luke Kornet", "line1": 6.5, "line2": 12.5, "line3": 10.5, "prediction1": 8.18, "prediction2": 8.89, "prediction3": 12.83, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 4.92, "kelly": 0.098, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "hitRate1": 42.4, "l5_1": 0.4, "l15_1": 0.53, "hitRate2": 66.3, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 42.4, "l5_3": 0.8, "l15_3": 0.27},
    {"name1": "Buddy Hield", "name2": "Cade Cunningham", "name3": "Luke Kornet", "line1": 6.5, "line2": 25.5, "line3": 10.5, "prediction1": 8.18, "prediction2": 29.03, "prediction3": 12.83, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 4.34, "kelly": 0.087, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "hitRate1": 42.4, "l5_1": 0.4, "l15_1": 0.53, "hitRate2": 85.1, "l5_2": 1.0, "l15_2": 0.4, "hitRate3": 42.4, "l5_3": 0.8, "l15_3": 0.27},
    {"name1": "Jaylen Brown", "name2": "Cade Cunningham", "name3": "De'Aaron Fox", "line1": 26.5, "line2": 25.5, "line3": 26.5, "prediction1": 29.97, "prediction2": 29.03, "prediction3": 23.58, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 3.89, "kelly": 0.078, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 59.8, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 85.1, "l5_2": 1.0, "l15_2": 0.4, "hitRate3": 83.7, "l5_3": 0.2, "l15_3": 0.07},
];// This is a large data file - I'll create a simplified version that includes all the hit rates data
// For brevity, I'll include a condensed version with the key structures
const prizepicksPointsHitRates = [
    {"name": "Will Richard", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.912, "underPct": 0.088},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.899, "underPct": 0.101},
    {"name": "Cade Cunningham", "line": 24.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.891, "underPct": 0.109},
    {"name": "Svi Mykhailiuk", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.889, "underPct": 0.111},
    {"name": "Marcus Smart", "line": 7.0, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.837, "underPct": 0.163},
    {"name": "Deandre Ayton", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.794, "underPct": 0.206},
    {"name": "Jalen Duren", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.766, "underPct": 0.234},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.759, "underPct": 0.241},
    {"name": "Keyonte George", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.753, "underPct": 0.247},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.752, "underPct": 0.248},
    {"name": "Kon Knueppel", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.74, "underPct": 0.26},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.72, "underPct": 0.28},
    {"name": "Payton Pritchard", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.704, "underPct": 0.296},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.675, "underPct": 0.325},
    {"name": "Noah Clowney", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.639, "underPct": 0.361},
    {"name": "Cam Spencer", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.637, "underPct": 0.363},
    {"name": "Anfernee Simons", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.626, "underPct": 0.374},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.619, "underPct": 0.381},
    {"name": "Stephen Curry", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.61, "underPct": 0.39},
    {"name": "Lauri Markkanen", "line": 26.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.604, "underPct": 0.396},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.598, "underPct": 0.402},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.595, "underPct": 0.405},
    {"name": "Miles Bridges", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.595, "underPct": 0.405},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "Nickeil Alexander-Walker", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "LaMelo Ball", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.568, "underPct": 0.432},
    {"name": "Tyrese Martin", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.566, "underPct": 0.434},
    {"name": "Santi Aldama", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.562, "underPct": 0.438},
    {"name": "Tristan da Silva", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.561, "underPct": 0.439},
    {"name": "Bennedict Mathurin", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.552, "underPct": 0.448},
    {"name": "Cedric Coward", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.547, "underPct": 0.453},
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.538, "underPct": 0.462},
    {"name": "Jalen Johnson", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.536, "underPct": 0.464},
    {"name": "Austin Reaves", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.535, "underPct": 0.465},
    {"name": "Anthony Black", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.524, "underPct": 0.476},
    {"name": "Jake LaRavia", "line": 7.0, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.518, "underPct": 0.482},
    {"name": "Kelly Olynyk", "line": 6.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.489, "underPct": 0.511},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.463, "underPct": 0.537},
    {"name": "Shaedon Sharpe", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.462, "underPct": 0.538},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.46, "underPct": 0.54},
    {"name": "Franz Wagner", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.442, "underPct": 0.558},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.439, "underPct": 0.561},
    {"name": "Brandin Podziemski", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.439, "underPct": 0.561},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.425, "underPct": 0.575},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.424, "underPct": 0.576},
    {"name": "Buddy Hield", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.424, "underPct": 0.576},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.423, "underPct": 0.577},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.415, "underPct": 0.585},
    {"name": "Ausar Thompson", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.403, "underPct": 0.597},
    {"name": "Desmond Bane", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.399, "underPct": 0.601},
    {"name": "Goga Bitadze", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.395, "underPct": 0.605},
    {"name": "Jalen Suggs", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.391, "underPct": 0.609},
    {"name": "Luka Garza", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.387, "underPct": 0.613},
    {"name": "Al Horford", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.385, "underPct": 0.615},
    {"name": "Neemias Queta", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.384, "underPct": 0.616},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.378, "underPct": 0.622},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.358, "underPct": 0.642},
    {"name": "Pascal Siakam", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.356, "underPct": 0.644},
    {"name": "Draymond Green", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.35, "underPct": 0.65},
    {"name": "Devin Booker", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.349, "underPct": 0.651},
    {"name": "Mouhamed Gueye", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.349, "underPct": 0.651},
    {"name": "Julian Champagnie", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.337, "underPct": 0.663},
    {"name": "Terance Mann", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.33, "underPct": 0.67},
    {"name": "Jaylen Wells", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.325, "underPct": 0.675},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.318, "underPct": 0.682},
    {"name": "Jordan Goodwin", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.314, "underPct": 0.686},
    {"name": "Ryan Dunn", "line": 8.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.308, "underPct": 0.692},
    {"name": "Mark Williams", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.306, "underPct": 0.694},
    {"name": "Luka Doncic", "line": 31.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.288, "underPct": 0.712},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.268, "underPct": 0.732},
    {"name": "Nick Richards", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.263, "underPct": 0.737},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.239, "underPct": 0.761},
    {"name": "Jarace Walker", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.227, "underPct": 0.773},
    {"name": "Kris Murray", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.183, "underPct": 0.817},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.182, "underPct": 0.818},
    {"name": "De'Aaron Fox", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.163, "underPct": 0.837},
    {"name": "Devin Vassell", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.151, "underPct": 0.849},
    {"name": "Keldon Johnson", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.126, "underPct": 0.874},
    {"name": "Collin Gillespie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.125, "underPct": 0.875},
    {"name": "Dyson Daniels", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.066, "underPct": 0.934},
];const prizepicksAssistsHitRates = [
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.692, "underPct": 0.308},
    {"name": "Austin Reaves", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.661, "underPct": 0.339},
    {"name": "Jake LaRavia", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.62, "underPct": 0.38},
    {"name": "Derrick White", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.606, "underPct": 0.394},
    {"name": "Isaiah Collier", "line": 6.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.587, "underPct": 0.413},
    {"name": "Jalen Johnson", "line": 6.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.577, "underPct": 0.423},
    {"name": "Desmond Bane", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.566, "underPct": 0.434},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.556, "underPct": 0.444},
    {"name": "Sion James", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.501, "underPct": 0.499},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.467, "underPct": 0.533},
    {"name": "Andrew Nembhard", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.443, "underPct": 0.557},
    {"name": "Cedric Coward", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.428, "underPct": 0.572},
    {"name": "Draymond Green", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.419, "underPct": 0.581},
    {"name": "Franz Wagner", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.415, "underPct": 0.585},
    {"name": "Jordan Goodwin", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.397, "underPct": 0.603},
    {"name": "Keyonte George", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.328, "underPct": 0.672},
    {"name": "Devin Booker", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.315, "underPct": 0.685},
    {"name": "Jaylen Brown", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.309, "underPct": 0.691},
    {"name": "Payton Pritchard", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.256, "underPct": 0.744},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.245, "underPct": 0.755},
    {"name": "Deni Avdija", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.243, "underPct": 0.757},
    {"name": "De'Aaron Fox", "line": 8.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.239, "underPct": 0.761},
    {"name": "Stephen Curry", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.167, "underPct": 0.833},
    {"name": "Luka Doncic", "line": 8.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.111, "underPct": 0.889},
];const prizepicksReboundsHitRates = [
    {"name": "Cedric Coward", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.693, "underPct": 0.307},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.682, "underPct": 0.318},
    {"name": "Kon Knueppel", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.68, "underPct": 0.32},
    {"name": "Jock Landale", "line": 4.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.613, "underPct": 0.387},
    {"name": "Royce O'Neale", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.58, "underPct": 0.42},
    {"name": "Jalen Johnson", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Franz Wagner", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.525, "underPct": 0.475},
    {"name": "Rui Hachimura", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "Shaedon Sharpe", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.507, "underPct": 0.493},
    {"name": "Collin Gillespie", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.506, "underPct": 0.494},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.505, "underPct": 0.495},
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.494, "underPct": 0.506},
    {"name": "Jaylen Brown", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.485, "underPct": 0.515},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Cade Cunningham", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.48, "underPct": 0.52},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.468, "underPct": 0.532},
    {"name": "Santi Aldama", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.455, "underPct": 0.545},
    {"name": "Ausar Thompson", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.425, "underPct": 0.575},
    {"name": "Devin Booker", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.404, "underPct": 0.596},
    {"name": "Austin Reaves", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.403, "underPct": 0.597},
    {"name": "Derrick White", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.398, "underPct": 0.602},
    {"name": "Jordan Walsh", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.39, "underPct": 0.61},
    {"name": "Ace Bailey", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.388, "underPct": 0.612},
    {"name": "Miles Bridges", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.378, "underPct": 0.622},
    {"name": "Deandre Ayton", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.363, "underPct": 0.637},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.347, "underPct": 0.653},
    {"name": "Al Horford", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.347, "underPct": 0.653},
    {"name": "Goga Bitadze", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.345, "underPct": 0.655},
    {"name": "Draymond Green", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.337, "underPct": 0.663},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.336, "underPct": 0.664},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.335, "underPct": 0.665},
    {"name": "Tristan da Silva", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.325, "underPct": 0.675},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.312, "underPct": 0.688},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.286, "underPct": 0.714},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.282, "underPct": 0.718},
    {"name": "Pascal Siakam", "line": 6.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.277, "underPct": 0.723},
    {"name": "Andrew Nembhard", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.277, "underPct": 0.723},
    {"name": "Harrison Barnes", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.27, "underPct": 0.73},
    {"name": "Zach Edey", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.268, "underPct": 0.732},
    {"name": "Stephen Curry", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.266, "underPct": 0.734},
    {"name": "Luka Garza", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.263, "underPct": 0.737},
    {"name": "Mark Williams", "line": 9.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.214, "underPct": 0.786},
    {"name": "Devin Vassell", "line": 4.0, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.209, "underPct": 0.791},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.208, "underPct": 0.792},
    {"name": "Ryan Dunn", "line": 5.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.178, "underPct": 0.822},
    {"name": "Julian Champagnie", "line": 5.0, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.172, "underPct": 0.828},
    {"name": "Jeremy Sochan", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.152, "underPct": 0.848},
    {"name": "Luka Doncic", "line": 8.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.103, "underPct": 0.897},
];const prizepicksBlocksHitRates = [
    {"name": "Draymond Green", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.494, "underPct": 0.506},
    {"name": "Moses Moody", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.58, "underPct": 0.42},
    {"name": "Ausar Thompson", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Cade Cunningham", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.597, "underPct": 0.403},
    {"name": "Nickeil Alexander-Walker", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.637, "underPct": 0.363},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Miles Bridges", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.391, "underPct": 0.609},
];const prizepicksStealsHitRates = [
    {"name": "Buddy Hield", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.616, "underPct": 0.384},
    {"name": "Gary Payton II", "line": 0.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.511, "underPct": 0.489},
    {"name": "Day'Ron Sharpe", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.532, "underPct": 0.468},
    {"name": "Ausar Thompson", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.548, "underPct": 0.452},
    {"name": "Cam Spencer", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.578, "underPct": 0.422},
    {"name": "Marcus Smart", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.659, "underPct": 0.341},
    {"name": "Svi Mykhailiuk", "line": 0.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.226, "underPct": 0.774},
    {"name": "Bennedict Mathurin", "line": 0.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.426, "underPct": 0.574},
    {"name": "Sion James", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.546, "underPct": 0.454},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Derrick White", "line": 27.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 39.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Santi Aldama", "line": 23.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Andrew Nembhard", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 32.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Duncan Robinson", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Garza", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 18.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 33.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 34.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gary Payton II", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 37.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Walsh", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keyonte George", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Marcus Smart", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 18.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tristan da Silva", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sion James", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 39.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremy Sochan", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Onyeka Okongwu", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dyson Daniels", "line": 24.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 25.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 47.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "De'Aaron Fox", "line": 38.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Vassell", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jake LaRavia", "line": 11.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 41.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 16.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 23.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Julian Champagnie", "line": 19.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 19.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPRHitRates = [
    {"name": "Cade Cunningham", "line": 29.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jock Landale", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Marcus Smart", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ausar Thompson", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Day'Ron Sharpe", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rui Hachimura", "line": 15.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Stephen Curry", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Goodwin", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 12.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jonathan Isaac", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Moses Moody", "line": 15.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 20.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 32.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Suggs", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 31.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Al Horford", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Buddy Hield", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sion James", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mark Williams", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Onyeka Okongwu", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bennedict Mathurin", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 19.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Draymond Green", "line": 14.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Terance Mann", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mouhamed Gueye", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 39.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nick Richards", "line": 9.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ryan Dunn", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremy Sochan", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Collin Sexton", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 17.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kelly Olynyk", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "De'Aaron Fox", "line": 30.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Zach Edey", "line": 20.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jarace Walker", "line": 16.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const prizepicksPAHitRates = [
    {"name": "Duncan Robinson", "line": 13.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 35.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 23.5, "l5": 1.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 29.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Rui Hachimura", "line": 13.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Ausar Thompson", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Duren", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Will Richard", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sion James", "line": 7.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Svi Mykhailiuk", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Goodwin", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Moses Moody", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 29.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marcus Smart", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Isaiah Collier", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ace Bailey", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Santi Aldama", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tristan da Silva", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Black", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deandre Ayton", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jake LaRavia", "line": 8.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kentavious Caldwell-Pope", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 29.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jonathan Isaac", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "De'Aaron Fox", "line": 34.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyrese Martin", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kelly Olynyk", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Julian Champagnie", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 37.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Terance Mann", "line": 13.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 8.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const prizepicksRAHitRates = [
    {"name": "Jaylen Brown", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 7.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luka Garza", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Duncan Robinson", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derrick White", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Suggs", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Terance Mann", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Murray", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 11.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Royce O'Neale", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 12.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Onyeka Okongwu", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Martin", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandin Podziemski", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Stephen Curry", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rui Hachimura", "line": 4.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 14.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deandre Ayton", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sion James", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 6.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julian Champagnie", "line": 6.5, "l5": 0.0, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Doncic", "line": 16.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const prizepicksTurnoversHitRates = [
    {"name": "Buddy Hield", "line": 0.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Toumani Camara", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Nickeil Alexander-Walker", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarace Walker", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ben Sheppard", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jordan Walsh", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Franz Wagner", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Jordan Walsh", "line": 1.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Marcus Smart", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Nickeil Alexander-Walker", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const underdogPointsHitRates = [
    {"name": "Svi Mykhailiuk", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.889, "underPct": 0.111},
    {"name": "Cade Cunningham", "line": 25.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.851, "underPct": 0.149},
    {"name": "Jalen Duren", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.766, "underPct": 0.234},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.759, "underPct": 0.241},
    {"name": "Keyonte George", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.753, "underPct": 0.247},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.72, "underPct": 0.28},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.675, "underPct": 0.325},
    {"name": "Anfernee Simons", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.626, "underPct": 0.374},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.604, "underPct": 0.396},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.598, "underPct": 0.402},
    {"name": "Nickeil Alexander-Walker", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "Santi Aldama", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.562, "underPct": 0.438},
    {"name": "Tristan da Silva", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.561, "underPct": 0.439},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.541, "underPct": 0.459},
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.538, "underPct": 0.462},
    {"name": "Jalen Johnson", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.536, "underPct": 0.464},
    {"name": "Austin Reaves", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.535, "underPct": 0.465},
    {"name": "Stephen Curry", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.535, "underPct": 0.465},
    {"name": "Noah Clowney", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.529, "underPct": 0.471},
    {"name": "Anthony Black", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.524, "underPct": 0.476},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.523, "underPct": 0.477},
    {"name": "Kelly Olynyk", "line": 6.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.489, "underPct": 0.511},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.463, "underPct": 0.537},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.439, "underPct": 0.561},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.424, "underPct": 0.576},
    {"name": "Buddy Hield", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.424, "underPct": 0.576},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.423, "underPct": 0.577},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.415, "underPct": 0.585},
    {"name": "Ausar Thompson", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.403, "underPct": 0.597},
    {"name": "Desmond Bane", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.399, "underPct": 0.601},
    {"name": "Jalen Suggs", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.391, "underPct": 0.609},
    {"name": "Luka Garza", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.387, "underPct": 0.613},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.358, "underPct": 0.642},
    {"name": "Luka Doncic", "line": 30.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.353, "underPct": 0.647},
    {"name": "Draymond Green", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.35, "underPct": 0.65},
    {"name": "Devin Booker", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.349, "underPct": 0.651},
    {"name": "Julian Champagnie", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.337, "underPct": 0.663},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.318, "underPct": 0.682},
    {"name": "Jordan Goodwin", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.314, "underPct": 0.686},
    {"name": "Ryan Dunn", "line": 8.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.308, "underPct": 0.692},
    {"name": "Mark Williams", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.306, "underPct": 0.694},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.268, "underPct": 0.732},
    {"name": "Nick Richards", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.263, "underPct": 0.737},
    {"name": "De'Aaron Fox", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.163, "underPct": 0.837},
    {"name": "Devin Vassell", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.151, "underPct": 0.849},
    {"name": "Keldon Johnson", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.126, "underPct": 0.874},
    {"name": "Collin Gillespie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.125, "underPct": 0.875},
    {"name": "Dyson Daniels", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.066, "underPct": 0.934},
];const underdogAssistsHitRates = [
    {"name": "Duncan Robinson", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.659, "underPct": 0.341},
    {"name": "Desmond Bane", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.566, "underPct": 0.434},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.556, "underPct": 0.444},
    {"name": "Ausar Thompson", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.544, "underPct": 0.456},
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.534, "underPct": 0.466},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.525, "underPct": 0.475},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.448, "underPct": 0.552},
    {"name": "Anthony Black", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.443, "underPct": 0.557},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.427, "underPct": 0.573},
    {"name": "Franz Wagner", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.415, "underPct": 0.585},
    {"name": "Jordan Goodwin", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.397, "underPct": 0.603},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.245, "underPct": 0.755},
    {"name": "Julian Champagnie", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.196, "underPct": 0.804},
];const underdogReboundsHitRates = [
    {"name": "Cedric Coward", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.693, "underPct": 0.307},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.682, "underPct": 0.318},
    {"name": "Jock Landale", "line": 4.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.613, "underPct": 0.387},
    {"name": "Ausar Thompson", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.585, "underPct": 0.415},
    {"name": "Royce O'Neale", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.58, "underPct": 0.42},
    {"name": "Jalen Johnson", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Jalen Suggs", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.528, "underPct": 0.472},
    {"name": "Franz Wagner", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.525, "underPct": 0.475},
    {"name": "Rui Hachimura", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "Tristan da Silva", "line": 4.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.498, "underPct": 0.502},
    {"name": "Brandin Podziemski", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Jordan Goodwin", "line": 4.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.465, "underPct": 0.535},
    {"name": "Drake Powell", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.392, "underPct": 0.608},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.336, "underPct": 0.664},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.296, "underPct": 0.704},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.286, "underPct": 0.714},
    {"name": "Zach Edey", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.268, "underPct": 0.732},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.208, "underPct": 0.792},
];const underdogBlocksHitRates = [
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
];const underdogStealsHitRates = [
    {"name": "Ausar Thompson", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.548, "underPct": 0.452},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Cade Cunningham", "line": 39.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derrick White", "line": 27.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Duncan Robinson", "line": 15.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luka Garza", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ace Bailey", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Santi Aldama", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 32.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ausar Thompson", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keyonte George", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Al Horford", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 36.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marcus Smart", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Onyeka Okongwu", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Brown", "line": 36.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Buddy Hield", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deandre Ayton", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 39.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremy Sochan", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nick Richards", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Vassell", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 38.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jonathan Isaac", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Wells", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luka Doncic", "line": 47.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ryan Dunn", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 41.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Julian Champagnie", "line": 19.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
];const underdogPRHitRates = [
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 30.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Santi Aldama", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luke Kornet", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 33.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Clingan", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 34.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 39.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "De'Aaron Fox", "line": 30.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const underdogPAHitRates = [
    {"name": "Derrick White", "line": 23.5, "l5": 1.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 35.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 29.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shaedon Sharpe", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Luka Doncic", "line": 39.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 37.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
];const underdogRAHitRates = [
    {"name": "Jalen Johnson", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 16.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const underdogTurnoversHitRates = [
    {"name": "Desmond Bane", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogBlocksStealsHitRates = [
    {"name": "Donovan Clingan", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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
            <th style="width: 10%">Prediction</th>
            <th style="width: 10%">Side</th>
            <th style="width: 8%">Odds</th>
            <th style="width: 9%">EV %</th>
            <th style="width: 9%">ROI %</th>
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
            <td class="kelly-cell">${row.roi.toFixed(1)}%</td>
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
            <th style="width: 16%">Player 1</th>
            <th style="width: 6%">Line 1</th>
            <th style="width: 6%">Pred 1</th>
            <th style="width: 16%">Player 2</th>
            <th style="width: 6%">Line 2</th>
            <th style="width: 6%">Pred 2</th>
            <th style="width: 9%">EV %</th>
            <th style="width: 9%">Kelly</th>
            <th style="width: 14%">Sigma</th>
            <th style="width: 4%">Rec</th>
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
            <th style="width: 13%">Player 1</th>
            <th style="width: 5%">Line 1</th>
            <th style="width: 5%">Pred 1</th>
            <th style="width: 13%">Player 2</th>
            <th style="width: 5%">Line 2</th>
            <th style="width: 5%">Pred 2</th>
            <th style="width: 13%">Player 3</th>
            <th style="width: 5%">Line 3</th>
            <th style="width: 5%">Pred 3</th>
            <th style="width: 7%">EV %</th>
            <th style="width: 7%">Kelly</th>
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
                <div class="stat-label">Prediction & Side</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Model's projected value and betting direction</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">EV % & ROI %</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value & Return on Investment percentage</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Odds</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">American odds format. <span style="color: #34d399;">+</span> = underdog, <span style="color: #f87171;">-</span> = favorite</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sigma</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (volatile), Med, Low (consistent)</div>
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
                <div class="stat-label">Pred (Prediction)</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Model's predicted value. <span style="color: #10b981;">Green</span> = over line, <span style="color: #f59e0b;">Orange</span> = under</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">EV % & Kelly</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value(How much you can expect to win per 10$ bet) & Kelly Criterion bet sizing %</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sigma</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (volatile), Med, Low (consistent)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Rec (Recommendation)</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">1 = Strong play, 0 = Consider</div>
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

// Initial render
render();

