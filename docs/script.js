const prizepicksSinglesData = [
    {"name": "Lauri Markkanen", "bookmaker": "BetRivers", "line": 27.5, "prediction": 34.77, "side": "Over", "odds": 108, "recommendation": 1, "ev": 6.73, "kelly": 0.623, "sigma": "High"},
    {"name": "Lauri Markkanen", "bookmaker": "FanDuel", "line": 26.5, "prediction": 34.77, "side": "Over", "odds": 102, "recommendation": 1, "ev": 6.73, "kelly": 0.66, "sigma": "High"},
    {"name": "Lauri Markkanen", "bookmaker": "BetRivers", "line": 26.5, "prediction": 34.77, "side": "Over", "odds": -107, "recommendation": 1, "ev": 6.28, "kelly": 0.672, "sigma": "High"},
    {"name": "Keyonte George", "bookmaker": "BetRivers", "line": 20.5, "prediction": 25.05, "side": "Over", "odds": 123, "recommendation": 1, "ev": 6.27, "kelly": 0.51, "sigma": "High"},
    {"name": "Lauri Markkanen", "bookmaker": "BetMGM", "line": 26.5, "prediction": 34.77, "side": "Over", "odds": -110, "recommendation": 1, "ev": 5.96, "kelly": 0.656, "sigma": "High"},
    {"name": "Nick Richards", "bookmaker": "DraftKings", "line": 4.5, "prediction": 7.39, "side": "Over", "odds": 107, "recommendation": 0, "ev": 5.89, "kelly": 0.55, "sigma": "Med"},
    {"name": "Lauri Markkanen", "bookmaker": "DraftKings", "line": 26.5, "prediction": 34.77, "side": "Over", "odds": -113, "recommendation": 1, "ev": 5.79, "kelly": 0.654, "sigma": "High"},
    {"name": "Keyonte George", "bookmaker": "BetRivers", "line": 19.5, "prediction": 25.05, "side": "Over", "odds": 104, "recommendation": 1, "ev": 5.56, "kelly": 0.534, "sigma": "High"},
    {"name": "Lauri Markkanen", "bookmaker": "BetRivers", "line": 25.5, "prediction": 34.77, "side": "Over", "odds": -125, "recommendation": 1, "ev": 5.49, "kelly": 0.686, "sigma": "High"},
    {"name": "Keyonte George", "bookmaker": "FanDuel", "line": 18.5, "prediction": 25.05, "side": "Over", "odds": -108, "recommendation": 1, "ev": 5.27, "kelly": 0.569, "sigma": "High"},
    {"name": "Keyonte George", "bookmaker": "BetMGM", "line": 19.5, "prediction": 25.05, "side": "Over", "odds": 100, "recommendation": 1, "ev": 5.23, "kelly": 0.523, "sigma": "High"},
    {"name": "Cade Cunningham", "bookmaker": "BetRivers", "line": 26.5, "prediction": 30.82, "side": "Over", "odds": 120, "recommendation": 0, "ev": 5.21, "kelly": 0.434, "sigma": "High"},
    {"name": "Nick Richards", "bookmaker": "FanDuel", "line": 4.5, "prediction": 7.39, "side": "Over", "odds": -106, "recommendation": 0, "ev": 5.16, "kelly": 0.547, "sigma": "Med"},
    {"name": "Keyonte George", "bookmaker": "BetRivers", "line": 18.5, "prediction": 25.05, "side": "Over", "odds": -115, "recommendation": 1, "ev": 5.0, "kelly": 0.575, "sigma": "High"},
    {"name": "Cade Cunningham", "bookmaker": "BetRivers", "line": 25.5, "prediction": 30.82, "side": "Over", "odds": 102, "recommendation": 1, "ev": 4.96, "kelly": 0.486, "sigma": "High"},
];const prizepicksPairsData = [
    {"name1": "Luka Don\u010di\u0107", "name2": "Lauri Markkanen", "line1": 0.5, "line2": 26.0, "prediction1": 29.3, "prediction2": 34.77, "side1": "over", "side2": "over", "recommendation": 1, "ev": 13.03, "kelly": 0.651, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 60.4, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Luka Don\u010di\u0107", "name2": "Nick Richards", "line1": 0.5, "line2": 4.0, "prediction1": 29.3, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 11.73, "kelly": 0.586, "sigma1": "High", "sigma2": "Med", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Luka Don\u010di\u0107", "name2": "Keyonte George", "line1": 0.5, "line2": 18.5, "prediction1": 29.3, "prediction2": 25.05, "side1": "over", "side2": "over", "recommendation": 1, "ev": 11.27, "kelly": 0.564, "sigma1": "High", "sigma2": "High", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 75.3, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Lauri Markkanen", "name2": "Marcus Smart", "line1": 26.0, "line2": 7.0, "prediction1": 34.77, "prediction2": 13.65, "side1": "over", "side2": "over", "recommendation": 1, "ev": 9.45, "kelly": 0.472, "sigma1": "High", "sigma2": "High", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 83.7, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Marcus Smart", "name2": "Nick Richards", "line1": 7.0, "line2": 4.0, "prediction1": 13.65, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 8.62, "kelly": 0.431, "sigma1": "High", "sigma2": "Med", "hitRate1": 83.7, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Lauri Markkanen", "name2": "Nick Richards", "line1": 26.0, "line2": 4.0, "prediction1": 34.77, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 8.56, "kelly": 0.428, "sigma1": "High", "sigma2": "Med", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Keyonte George", "name2": "Marcus Smart", "line1": 18.5, "line2": 7.0, "prediction1": 25.05, "prediction2": 13.65, "side1": "over", "side2": "over", "recommendation": 1, "ev": 8.27, "kelly": 0.413, "sigma1": "High", "sigma2": "High", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 83.7, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Keyonte George", "name2": "Dillon Brooks", "line1": 18.5, "line2": 18.5, "prediction1": 25.05, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.45, "kelly": 0.322, "sigma1": "High", "sigma2": "High", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Isaiah Collier", "name2": "Dillon Brooks", "line1": 8.5, "line2": 18.5, "prediction1": 12.69, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.0, "kelly": 0.3, "sigma1": "Med", "sigma2": "High", "hitRate1": 75.2, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Ausar Thompson", "name2": "Isaiah Collier", "line1": 10.5, "line2": 8.5, "prediction1": 15.04, "prediction2": 12.69, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.88, "kelly": 0.294, "sigma1": "High", "sigma2": "Med", "hitRate1": 72.9, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 75.2, "l5_2": 0.8, "l15_2": 0.27},
];const prizepicksTriosData = [
    {"name1": "Luka Don\u010di\u0107", "name2": "Marcus Smart", "name3": "Nick Richards", "line1": 0.5, "line2": 7.0, "line3": 4.0, "prediction1": 29.3, "prediction2": 13.65, "prediction3": 7.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 23.33, "kelly": 0.467, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 83.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 26.3, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Luka Don\u010di\u0107", "name2": "Lauri Markkanen", "name3": "Nick Richards", "line1": 0.5, "line2": 26.0, "line3": 4.0, "prediction1": 29.3, "prediction2": 34.77, "prediction3": 7.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 23.3, "kelly": 0.466, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 60.4, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 26.3, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Lauri Markkanen", "name2": "Marcus Smart", "name3": "Dillon Brooks", "line1": 26.0, "line2": 7.0, "line3": 18.5, "prediction1": 34.77, "prediction2": 13.65, "prediction3": 23.84, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 17.07, "kelly": 0.341, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 83.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 75.9, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Keyonte George", "name2": "Isaiah Collier", "name3": "Dillon Brooks", "line1": 18.5, "line2": 8.5, "line3": 18.5, "prediction1": 25.05, "prediction2": 12.69, "prediction3": 23.84, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 12.92, "kelly": 0.258, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.2, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 75.9, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Ausar Thompson", "name2": "Keyonte George", "name3": "Isaiah Collier", "line1": 10.5, "line2": 18.5, "line3": 8.5, "prediction1": 15.04, "prediction2": 25.05, "prediction3": 12.69, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 12.72, "kelly": 0.254, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 72.9, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 75.3, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 75.2, "l5_3": 0.8, "l15_3": 0.27},
    {"name1": "Day'Ron Sharpe", "name2": "Ausar Thompson", "name3": "Jake LaRavia", "line1": 5.5, "line2": 10.5, "line3": 7.0, "prediction1": 8.4, "prediction2": 15.04, "prediction3": 10.95, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.13, "kelly": 0.223, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 46.3, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 72.9, "l5_2": 0.8, "l15_2": 0.53, "hitRate3": 51.8, "l5_3": 0.2, "l15_3": 0.47},
    {"name1": "Day'Ron Sharpe", "name2": "Jake LaRavia", "name3": "Ryan Kalkbrenner", "line1": 5.5, "line2": 7.0, "line3": 8.5, "prediction1": 8.4, "prediction2": 10.95, "prediction3": 11.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 10.53, "kelly": 0.211, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "hitRate1": 46.3, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 51.8, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 61.9, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Jock Landale", "name2": "Svi Mykhailiuk", "name3": "Ryan Kalkbrenner", "line1": 6.5, "line2": 7.5, "line3": 8.5, "prediction1": 9.71, "prediction2": 10.66, "prediction3": 11.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.28, "kelly": 0.186, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 85.5, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 88.9, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 61.9, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Jock Landale", "name2": "Rui Hachimura", "name3": "Svi Mykhailiuk", "line1": 6.5, "line2": 11.5, "line3": 7.5, "prediction1": 9.71, "prediction2": 15.02, "prediction3": 10.66, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.55, "kelly": 0.171, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 85.5, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 89.9, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 88.9, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Cade Cunningham", "name2": "Rui Hachimura", "name3": "Sion James", "line1": 24.5, "line2": 11.5, "line3": 6.5, "prediction1": 29.03, "prediction2": 15.02, "prediction3": 8.81, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.18, "kelly": 0.144, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 89.1, "l5_1": 1.0, "l15_1": 0.47, "hitRate2": 89.9, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 69.7, "l5_3": 0.6, "l15_3": 0.6},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Daniss Jenkins", "name2": "Lauri Markkanen", "line1": 10.5, "line2": 26.5, "prediction1": 16.35, "prediction2": 34.77, "side1": "over", "side2": "over", "recommendation": 1, "ev": 8.97, "kelly": 0.448, "sigma1": "High", "sigma2": "High", "hitRate1": 46.6, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 60.4, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Daniss Jenkins", "name2": "Keyonte George", "line1": 10.5, "line2": 18.5, "prediction1": 16.35, "prediction2": 25.05, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.64, "kelly": 0.382, "sigma1": "High", "sigma2": "High", "hitRate1": 46.6, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 75.3, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Ausar Thompson", "name2": "Lauri Markkanen", "line1": 10.5, "line2": 26.5, "prediction1": 15.04, "prediction2": 34.77, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.44, "kelly": 0.372, "sigma1": "High", "sigma2": "High", "hitRate1": 72.9, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 60.4, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Lauri Markkanen", "name2": "Dillon Brooks", "line1": 26.5, "line2": 18.5, "prediction1": 34.77, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 1, "ev": 7.43, "kelly": 0.372, "sigma1": "High", "sigma2": "High", "hitRate1": 60.4, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Daniss Jenkins", "name2": "Nick Richards", "line1": 10.5, "line2": 4.5, "prediction1": 16.35, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 7.41, "kelly": 0.37, "sigma1": "High", "sigma2": "Med", "hitRate1": 46.6, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Keyonte George", "name2": "Dillon Brooks", "line1": 18.5, "line2": 18.5, "prediction1": 25.05, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 1, "ev": 6.59, "kelly": 0.33, "sigma1": "High", "sigma2": "High", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Keyonte George", "name2": "Nick Richards", "line1": 18.5, "line2": 4.5, "prediction1": 25.05, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.38, "kelly": 0.319, "sigma1": "High", "sigma2": "Med", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Day'Ron Sharpe", "name2": "Nick Richards", "line1": 5.5, "line2": 4.5, "prediction1": 8.4, "prediction2": 7.39, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.9, "kelly": 0.295, "sigma1": "Med", "sigma2": "Med", "hitRate1": 46.3, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 26.3, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Ausar Thompson", "name2": "Dillon Brooks", "line1": 10.5, "line2": 18.5, "prediction1": 15.04, "prediction2": 23.84, "side1": "over", "side2": "over", "recommendation": 1, "ev": 5.88, "kelly": 0.294, "sigma1": "High", "sigma2": "High", "hitRate1": 72.9, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Ausar Thompson", "name2": "Ryan Kalkbrenner", "line1": 10.5, "line2": 8.5, "prediction1": 15.04, "prediction2": 11.85, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.53, "kelly": 0.276, "sigma1": "High", "sigma2": "Med", "hitRate1": 72.9, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 61.9, "l5_2": 1.0, "l15_2": 0.6},
];const underdogTriosData = [
    {"name1": "Daniss Jenkins", "name2": "Lauri Markkanen", "name3": "Keyonte George", "line1": 10.5, "line2": 26.5, "line3": 18.5, "prediction1": 16.35, "prediction2": 34.77, "prediction3": 25.05, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 16.73, "kelly": 0.335, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 46.6, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 60.4, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 75.3, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Daniss Jenkins", "name2": "Lauri Markkanen", "name3": "Nick Richards", "line1": 10.5, "line2": 26.5, "line3": 4.5, "prediction1": 16.35, "prediction2": 34.77, "prediction3": 7.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 15.98, "kelly": 0.32, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 46.6, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 60.4, "l5_2": 0.6, "l15_2": 0.53, "hitRate3": 26.3, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Keyonte George", "name2": "Dillon Brooks", "name3": "Nick Richards", "line1": 18.5, "line2": 18.5, "line3": 4.5, "prediction1": 25.05, "prediction2": 23.84, "prediction3": 7.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 12.86, "kelly": 0.257, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 75.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 75.9, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 26.3, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Day'Ron Sharpe", "name2": "Ausar Thompson", "name3": "Dillon Brooks", "line1": 5.5, "line2": 10.5, "line3": 18.5, "prediction1": 8.4, "prediction2": 15.04, "prediction3": 23.84, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.43, "kelly": 0.229, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 46.3, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 72.9, "l5_2": 0.8, "l15_2": 0.53, "hitRate3": 75.9, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Day'Ron Sharpe", "name2": "Ausar Thompson", "name3": "Ryan Kalkbrenner", "line1": 5.5, "line2": 10.5, "line3": 8.5, "prediction1": 8.4, "prediction2": 15.04, "prediction3": 11.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 10.64, "kelly": 0.213, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "hitRate1": 46.3, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 72.9, "l5_2": 0.8, "l15_2": 0.53, "hitRate3": 61.9, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Svi Mykhailiuk", "name2": "Ryan Kalkbrenner", "name3": "Sion James", "line1": 7.5, "line2": 8.5, "line3": 6.5, "prediction1": 10.66, "prediction2": 11.85, "prediction3": 8.81, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.25, "kelly": 0.165, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 88.9, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 61.9, "l5_2": 1.0, "l15_2": 0.6, "hitRate3": 69.7, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Svi Mykhailiuk", "name2": "Sion James", "name3": "Tyrese Maxey", "line1": 7.5, "line2": 6.5, "line3": 31.5, "prediction1": 10.66, "prediction2": 8.81, "prediction3": 27.11, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 7.2, "kelly": 0.144, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 88.9, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 69.7, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 59.5, "l5_3": 0.2, "l15_3": 0.33},
    {"name1": "Dyson Daniels", "name2": "Julian Champagnie", "name3": "Tyrese Maxey", "line1": 13.5, "line2": 12.5, "line3": 31.5, "prediction1": 10.12, "prediction2": 8.89, "prediction3": 27.11, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 6.34, "kelly": 0.127, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 93.4, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 66.3, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 59.5, "l5_3": 0.2, "l15_3": 0.33},
    {"name1": "Dyson Daniels", "name2": "Javonte Green", "name3": "Julian Champagnie", "line1": 13.5, "line2": 8.5, "line3": 12.5, "prediction1": 10.12, "prediction2": 10.71, "prediction3": 8.89, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 5.59, "kelly": 0.112, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 93.4, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 46.4, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 66.3, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Cade Cunningham", "name2": "Javonte Green", "name3": "Luke Kornet", "line1": 25.5, "line2": 8.5, "line3": 10.5, "prediction1": 29.03, "prediction2": 10.71, "prediction3": 12.83, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 4.32, "kelly": 0.086, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 85.1, "l5_1": 1.0, "l15_1": 0.4, "hitRate2": 46.4, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 42.4, "l5_3": 0.8, "l15_3": 0.27},
];const prizepicksPointsHitRates = [
    {"name": "Luka Doncic", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 1.0, "underPct": 0.0},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.899, "underPct": 0.101},
    {"name": "Cade Cunningham", "line": 24.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.891, "underPct": 0.109},
    {"name": "Svi Mykhailiuk", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.889, "underPct": 0.111},
    {"name": "Jock Landale", "line": 6.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.855, "underPct": 0.145},
    {"name": "Marcus Smart", "line": 7.0, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.837, "underPct": 0.163},
    {"name": "Deandre Ayton", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.794, "underPct": 0.206},
    {"name": "Jalen Duren", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.766, "underPct": 0.234},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.759, "underPct": 0.241},
    {"name": "Keyonte George", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.753, "underPct": 0.247},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.752, "underPct": 0.248},
    {"name": "Kon Knueppel", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.74, "underPct": 0.26},
    {"name": "Ausar Thompson", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.729, "underPct": 0.271},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.72, "underPct": 0.28},
    {"name": "Sandro Mamukelashvili", "line": 8.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.708, "underPct": 0.292},
    {"name": "Immanuel Quickley", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.707, "underPct": 0.293},
    {"name": "Payton Pritchard", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.704, "underPct": 0.296},
    {"name": "Sion James", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.697, "underPct": 0.303},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.675, "underPct": 0.325},
    {"name": "Kyle Filipowski", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.656, "underPct": 0.344},
    {"name": "Noah Clowney", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.639, "underPct": 0.361},
    {"name": "Cam Spencer", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.637, "underPct": 0.363},
    {"name": "Anfernee Simons", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.626, "underPct": 0.374},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.619, "underPct": 0.381},
    {"name": "Lauri Markkanen", "line": 26.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.604, "underPct": 0.396},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.598, "underPct": 0.402},
    {"name": "Miles Bridges", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.595, "underPct": 0.405},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.595, "underPct": 0.405},
    {"name": "Nickeil Alexander-Walker", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "LaMelo Ball", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.568, "underPct": 0.432},
    {"name": "Tyrese Martin", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.566, "underPct": 0.434},
    {"name": "Santi Aldama", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.562, "underPct": 0.438},
    {"name": "Bennedict Mathurin", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.552, "underPct": 0.448},
    {"name": "Cedric Coward", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.547, "underPct": 0.453},
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.538, "underPct": 0.462},
    {"name": "Jalen Johnson", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.536, "underPct": 0.464},
    {"name": "Austin Reaves", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.535, "underPct": 0.465},
    {"name": "Jake LaRavia", "line": 7.0, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.518, "underPct": 0.482},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.513, "underPct": 0.487},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.491, "underPct": 0.509},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.471, "underPct": 0.529},
    {"name": "Javonte Green", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.464, "underPct": 0.536},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.463, "underPct": 0.537},
    {"name": "Shaedon Sharpe", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.462, "underPct": 0.538},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.46, "underPct": 0.54},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.439, "underPct": 0.561},
    {"name": "Quentin Grimes", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.426, "underPct": 0.574},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.425, "underPct": 0.575},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.424, "underPct": 0.576},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.423, "underPct": 0.577},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.415, "underPct": 0.585},
    {"name": "Tyrese Maxey", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.405, "underPct": 0.595},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.403, "underPct": 0.597},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.399, "underPct": 0.601},
    {"name": "Luka Garza", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.387, "underPct": 0.613},
    {"name": "Neemias Queta", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.384, "underPct": 0.616},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.378, "underPct": 0.622},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.358, "underPct": 0.642},
    {"name": "Pascal Siakam", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.356, "underPct": 0.644},
    {"name": "Mouhamed Gueye", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.349, "underPct": 0.651},
    {"name": "Julian Champagnie", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.337, "underPct": 0.663},
    {"name": "Terance Mann", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.33, "underPct": 0.67},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.318, "underPct": 0.682},
    {"name": "Jordan Goodwin", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.314, "underPct": 0.686},
    {"name": "Ryan Dunn", "line": 8.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.308, "underPct": 0.692},
    {"name": "Mark Williams", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.306, "underPct": 0.694},
    {"name": "Devin Booker", "line": 30.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.283, "underPct": 0.717},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.268, "underPct": 0.732},
    {"name": "Nick Richards", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.263, "underPct": 0.737},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.239, "underPct": 0.761},
    {"name": "Jarace Walker", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.227, "underPct": 0.773},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.182, "underPct": 0.818},
    {"name": "De'Aaron Fox", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.163, "underPct": 0.837},
    {"name": "VJ Edgecombe", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.156, "underPct": 0.844},
    {"name": "Devin Vassell", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.151, "underPct": 0.849},
    {"name": "Keldon Johnson", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.126, "underPct": 0.874},
    {"name": "Collin Gillespie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.125, "underPct": 0.875},
    {"name": "Dyson Daniels", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.066, "underPct": 0.934},
];const prizepicksAssistsHitRates = [
    {"name": "Pascal Siakam", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.701, "underPct": 0.299},
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.692, "underPct": 0.308},
    {"name": "Austin Reaves", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.661, "underPct": 0.339},
    {"name": "Jake LaRavia", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.62, "underPct": 0.38},
    {"name": "Derrick White", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.606, "underPct": 0.394},
    {"name": "Isaiah Collier", "line": 6.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.587, "underPct": 0.413},
    {"name": "Jalen Johnson", "line": 6.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.577, "underPct": 0.423},
    {"name": "Jock Landale", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.566, "underPct": 0.434},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.556, "underPct": 0.444},
    {"name": "Sandro Mamukelashvili", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.54, "underPct": 0.46},
    {"name": "Sion James", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.501, "underPct": 0.499},
    {"name": "Andrew Nembhard", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.443, "underPct": 0.557},
    {"name": "Cedric Coward", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.428, "underPct": 0.572},
    {"name": "Jordan Goodwin", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.397, "underPct": 0.603},
    {"name": "Daniss Jenkins", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.391, "underPct": 0.609},
    {"name": "Keyonte George", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.328, "underPct": 0.672},
    {"name": "Devin Booker", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.315, "underPct": 0.685},
    {"name": "Jaylen Brown", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.309, "underPct": 0.691},
    {"name": "Payton Pritchard", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.256, "underPct": 0.744},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.245, "underPct": 0.755},
    {"name": "Deni Avdija", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.243, "underPct": 0.757},
    {"name": "De'Aaron Fox", "line": 8.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.239, "underPct": 0.761},
    {"name": "Luka Doncic", "line": 8.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.111, "underPct": 0.889},
];const prizepicksReboundsHitRates = [
    {"name": "Javonte Green", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.749, "underPct": 0.251},
    {"name": "Cedric Coward", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.693, "underPct": 0.307},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.682, "underPct": 0.318},
    {"name": "Kon Knueppel", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.68, "underPct": 0.32},
    {"name": "Royce O'Neale", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.58, "underPct": 0.42},
    {"name": "Jalen Johnson", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Duncan Robinson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.557, "underPct": 0.443},
    {"name": "Rui Hachimura", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "Shaedon Sharpe", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.507, "underPct": 0.493},
    {"name": "Collin Gillespie", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.506, "underPct": 0.494},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.505, "underPct": 0.495},
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.494, "underPct": 0.506},
    {"name": "Jaylen Brown", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.485, "underPct": 0.515},
    {"name": "Cade Cunningham", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.48, "underPct": 0.52},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.468, "underPct": 0.532},
    {"name": "Santi Aldama", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.455, "underPct": 0.545},
    {"name": "Luka Garza", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.429, "underPct": 0.571},
    {"name": "Devin Booker", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.404, "underPct": 0.596},
    {"name": "Austin Reaves", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.403, "underPct": 0.597},
    {"name": "Derrick White", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.398, "underPct": 0.602},
    {"name": "Jamal Shead", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.396, "underPct": 0.604},
    {"name": "Ace Bailey", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.388, "underPct": 0.612},
    {"name": "Miles Bridges", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.378, "underPct": 0.622},
    {"name": "Jakob Poeltl", "line": 8.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.365, "underPct": 0.635},
    {"name": "Deandre Ayton", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.363, "underPct": 0.637},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.347, "underPct": 0.653},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.336, "underPct": 0.664},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.335, "underPct": 0.665},
    {"name": "Jarace Walker", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.333, "underPct": 0.667},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.312, "underPct": 0.688},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.282, "underPct": 0.718},
    {"name": "Pascal Siakam", "line": 6.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.277, "underPct": 0.723},
    {"name": "Andrew Nembhard", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.277, "underPct": 0.723},
    {"name": "Harrison Barnes", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.27, "underPct": 0.73},
    {"name": "Mark Williams", "line": 9.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.214, "underPct": 0.786},
    {"name": "Devin Vassell", "line": 4.0, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.209, "underPct": 0.791},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.208, "underPct": 0.792},
    {"name": "Ryan Dunn", "line": 5.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.178, "underPct": 0.822},
    {"name": "Julian Champagnie", "line": 5.0, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.172, "underPct": 0.828},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.166, "underPct": 0.834},
    {"name": "Jeremy Sochan", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.152, "underPct": 0.848},
    {"name": "Luka Doncic", "line": 8.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.103, "underPct": 0.897},
];const prizepicksBlocksHitRates = [
    {"name": "Cade Cunningham", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.597, "underPct": 0.403},
    {"name": "Nickeil Alexander-Walker", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.637, "underPct": 0.363},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Miles Bridges", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.391, "underPct": 0.609},
];const prizepicksStealsHitRates = [
    {"name": "Day'Ron Sharpe", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.532, "underPct": 0.468},
    {"name": "Cam Spencer", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.578, "underPct": 0.422},
    {"name": "Svi Mykhailiuk", "line": 0.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.226, "underPct": 0.774},
    {"name": "Bennedict Mathurin", "line": 0.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.426, "underPct": 0.574},
    {"name": "Sion James", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.546, "underPct": 0.454},
    {"name": "Jakob Poeltl", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.492, "underPct": 0.508},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Derrick White", "line": 27.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 38.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Luka Garza", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ace Bailey", "line": 18.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Duncan Robinson", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Santi Aldama", "line": 23.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "LaMelo Ball", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anfernee Simons", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 18.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Daniss Jenkins", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Johnson", "line": 37.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Marcus Smart", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shaedon Sharpe", "line": 33.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bennedict Mathurin", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyrese Martin", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Neemias Queta", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremy Sochan", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 39.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Dunn", "line": 16.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luka Doncic", "line": 47.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jake LaRavia", "line": 11.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mouhamed Gueye", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 38.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jerami Grant", "line": 25.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 41.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 24.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 43.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 23.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Julian Champagnie", "line": 19.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 19.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPRHitRates = [
    {"name": "Cade Cunningham", "line": 29.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sandro Mamukelashvili", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Day'Ron Sharpe", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Immanuel Quickley", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Marcus Smart", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jock Landale", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ace Bailey", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rui Hachimura", "line": 15.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Duncan Robinson", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jakob Poeltl", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Javonte Green", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Kalkbrenner", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drake Powell", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Johnson", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ziaire Williams", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniss Jenkins", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 32.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Collier", "line": 12.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Goodwin", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Santi Aldama", "line": 20.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kon Knueppel", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anfernee Simons", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jake LaRavia", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cedric Coward", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Shead", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sion James", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "LaMelo Ball", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Clingan", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mouhamed Gueye", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jeremy Sochan", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 32.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Dunn", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nick Richards", "line": 9.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Doncic", "line": 38.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kelly Olynyk", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Julian Champagnie", "line": 17.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 22.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "De'Aaron Fox", "line": 30.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jarace Walker", "line": 17.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const prizepicksPAHitRates = [
    {"name": "Derrick White", "line": 23.5, "l5": 1.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 33.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Duncan Robinson", "line": 13.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 29.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Nickeil Alexander-Walker", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rui Hachimura", "line": 13.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ausar Thompson", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Andrew Nembhard", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Goodwin", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Collier", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Svi Mykhailiuk", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Marcus Smart", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sion James", "line": 7.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jakob Poeltl", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Johnson", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniss Jenkins", "line": 15.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Onyeka Okongwu", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Sexton", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anfernee Simons", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 29.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jake LaRavia", "line": 8.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deandre Ayton", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mouhamed Gueye", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Spencer", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kentavious Caldwell-Pope", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Dunn", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Martin", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Julian Champagnie", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "De'Aaron Fox", "line": 34.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kelly Olynyk", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Keldon Johnson", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Scottie Barnes", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 37.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 39.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 13.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 14.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const prizepicksRAHitRates = [
    {"name": "Jaylen Brown", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 7.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Walsh", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 11.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyrese Martin", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Terance Mann", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kon Knueppel", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 12.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "VJ Edgecombe", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deandre Ayton", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Clingan", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 14.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Rui Hachimura", "line": 4.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Dunn", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sion James", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julian Champagnie", "line": 6.5, "l5": 0.0, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Harrison Barnes", "line": 6.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luka Doncic", "line": 16.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksTurnoversHitRates = [
    {"name": "Javonte Green", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Toumani Camara", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarace Walker", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Daniss Jenkins", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ben Sheppard", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Walsh", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Jordan Walsh", "line": 1.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Marcus Smart", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "VJ Edgecombe", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mouhamed Gueye", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Clingan", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quentin Grimes", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogPointsHitRates = [
    {"name": "Svi Mykhailiuk", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.889, "underPct": 0.111},
    {"name": "Cade Cunningham", "line": 25.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.851, "underPct": 0.149},
    {"name": "Jalen Duren", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.766, "underPct": 0.234},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.759, "underPct": 0.241},
    {"name": "Keyonte George", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.753, "underPct": 0.247},
    {"name": "Kon Knueppel", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.74, "underPct": 0.26},
    {"name": "Ausar Thompson", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.729, "underPct": 0.271},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.72, "underPct": 0.28},
    {"name": "Sandro Mamukelashvili", "line": 8.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.708, "underPct": 0.292},
    {"name": "Immanuel Quickley", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.707, "underPct": 0.293},
    {"name": "Sion James", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.697, "underPct": 0.303},
    {"name": "Deni Avdija", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.675, "underPct": 0.325},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.637, "underPct": 0.363},
    {"name": "Anfernee Simons", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.626, "underPct": 0.374},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.619, "underPct": 0.381},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.604, "underPct": 0.396},
    {"name": "Jaylen Brown", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.598, "underPct": 0.402},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "Nickeil Alexander-Walker", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "Santi Aldama", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.562, "underPct": 0.438},
    {"name": "Bennedict Mathurin", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.552, "underPct": 0.448},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.541, "underPct": 0.459},
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.538, "underPct": 0.462},
    {"name": "Jalen Johnson", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.536, "underPct": 0.464},
    {"name": "Austin Reaves", "line": 23.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.535, "underPct": 0.465},
    {"name": "Noah Clowney", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.529, "underPct": 0.471},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.513, "underPct": 0.487},
    {"name": "Miles Bridges", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.511, "underPct": 0.489},
    {"name": "Andrew Nembhard", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.495, "underPct": 0.505},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.491, "underPct": 0.509},
    {"name": "Kelly Olynyk", "line": 6.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.489, "underPct": 0.511},
    {"name": "LaMelo Ball", "line": 22.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.484, "underPct": 0.516},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.471, "underPct": 0.529},
    {"name": "Daniss Jenkins", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.466, "underPct": 0.534},
    {"name": "Javonte Green", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.464, "underPct": 0.536},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.463, "underPct": 0.537},
    {"name": "Luke Kennard", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.452, "underPct": 0.548},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.439, "underPct": 0.561},
    {"name": "Quentin Grimes", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.426, "underPct": 0.574},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.424, "underPct": 0.576},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.423, "underPct": 0.577},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.415, "underPct": 0.585},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.413, "underPct": 0.587},
    {"name": "Tyrese Maxey", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.405, "underPct": 0.595},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.403, "underPct": 0.597},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.399, "underPct": 0.601},
    {"name": "Andre Drummond", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.389, "underPct": 0.611},
    {"name": "Luka Garza", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.387, "underPct": 0.613},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.384, "underPct": 0.616},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.378, "underPct": 0.622},
    {"name": "Derrick White", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.358, "underPct": 0.642},
    {"name": "Pascal Siakam", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.356, "underPct": 0.644},
    {"name": "Luka Doncic", "line": 30.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.353, "underPct": 0.647},
    {"name": "Mouhamed Gueye", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.349, "underPct": 0.651},
    {"name": "Devin Booker", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.349, "underPct": 0.651},
    {"name": "Julian Champagnie", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.337, "underPct": 0.663},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.318, "underPct": 0.682},
    {"name": "Jordan Goodwin", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.314, "underPct": 0.686},
    {"name": "Ryan Dunn", "line": 8.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.308, "underPct": 0.692},
    {"name": "Mark Williams", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.306, "underPct": 0.694},
    {"name": "Nick Richards", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.263, "underPct": 0.737},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.228, "underPct": 0.772},
    {"name": "Jarace Walker", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.227, "underPct": 0.773},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.182, "underPct": 0.818},
    {"name": "De'Aaron Fox", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.163, "underPct": 0.837},
    {"name": "Devin Vassell", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.151, "underPct": 0.849},
    {"name": "Keldon Johnson", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.126, "underPct": 0.874},
    {"name": "Collin Gillespie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.125, "underPct": 0.875},
    {"name": "Dyson Daniels", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.066, "underPct": 0.934},
];const underdogAssistsHitRates = [
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.692, "underPct": 0.308},
    {"name": "Jake LaRavia", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.62, "underPct": 0.38},
    {"name": "Collin Sexton", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.585, "underPct": 0.415},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.556, "underPct": 0.444},
    {"name": "Sandro Mamukelashvili", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.54, "underPct": 0.46},
    {"name": "Ryan Dunn", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.534, "underPct": 0.466},
    {"name": "Sion James", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.501, "underPct": 0.499},
    {"name": "VJ Edgecombe", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.453, "underPct": 0.547},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.427, "underPct": 0.573},
    {"name": "Jarace Walker", "line": 2.5, "l5": 0.0, "l10": 0.5, "l15": 0.53, "overPct": 0.425, "underPct": 0.575},
    {"name": "Jordan Goodwin", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.397, "underPct": 0.603},
    {"name": "Ben Sheppard", "line": 1.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.291, "underPct": 0.709},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.245, "underPct": 0.755},
    {"name": "Deni Avdija", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.243, "underPct": 0.757},
    {"name": "Julian Champagnie", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.196, "underPct": 0.804},
];const underdogReboundsHitRates = [
    {"name": "Javonte Green", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.749, "underPct": 0.251},
    {"name": "Cedric Coward", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.693, "underPct": 0.307},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.682, "underPct": 0.318},
    {"name": "Jock Landale", "line": 4.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.613, "underPct": 0.387},
    {"name": "Royce O'Neale", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.58, "underPct": 0.42},
    {"name": "Jalen Johnson", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Duncan Robinson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.557, "underPct": 0.443},
    {"name": "Rui Hachimura", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.494, "underPct": 0.506},
    {"name": "Jordan Goodwin", "line": 4.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.465, "underPct": 0.535},
    {"name": "Jamal Shead", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.396, "underPct": 0.604},
    {"name": "Drake Powell", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.392, "underPct": 0.608},
    {"name": "Collin Sexton", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.391, "underPct": 0.609},
    {"name": "Miles Bridges", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.378, "underPct": 0.622},
    {"name": "Jakob Poeltl", "line": 8.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.365, "underPct": 0.635},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.347, "underPct": 0.653},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.286, "underPct": 0.714},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.282, "underPct": 0.718},
    {"name": "Pascal Siakam", "line": 6.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.277, "underPct": 0.723},
    {"name": "Andrew Nembhard", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.277, "underPct": 0.723},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.208, "underPct": 0.792},
    {"name": "Luka Doncic", "line": 7.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.187, "underPct": 0.813},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.166, "underPct": 0.834},
];const underdogBlocksHitRates = [
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
];const underdogStealsHitRates = [
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Cade Cunningham", "line": 39.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Duncan Robinson", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Santi Aldama", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 26.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brice Sensabaugh", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Javonte Green", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Day'Ron Sharpe", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Walsh", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 37.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Marcus Smart", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luke Kornet", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Daniss Jenkins", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andre Drummond", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Kalkbrenner", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 36.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Neemias Queta", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Jackson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremy Sochan", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kris Murray", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 42.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles Bridges", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 39.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mouhamed Gueye", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 38.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Luka Doncic", "line": 47.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Collin Sexton", "line": 24.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 41.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Wells", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jake LaRavia", "line": 11.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nick Richards", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 23.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julian Champagnie", "line": 19.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jarace Walker", "line": 19.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const underdogPRHitRates = [
    {"name": "Derrick White", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 30.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Santi Aldama", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Neemias Queta", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Quentin Grimes", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 39.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Pascal Siakam", "line": 32.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 34.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Sexton", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 30.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const underdogPAHitRates = [
    {"name": "Derrick White", "line": 23.5, "l5": 1.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 34.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 29.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Andrew Nembhard", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Luka Doncic", "line": 39.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 37.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 30.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Scottie Barnes", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 38.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const underdogRAHitRates = [
    {"name": "Jaylen Brown", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "VJ Edgecombe", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jakob Poeltl", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 16.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogTurnoversHitRates = [
    {"name": "Quentin Grimes", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogBlocksStealsHitRates = [
    {"name": "Donovan Clingan", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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
            <th style="width: 9%">EV $</th>
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
            <td class="ev-cell ${getEVClass(row.ev)}">$${row.ev.toFixed(2)}</td>
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
            <th style="width: 16%">Player 1</th>
            <th style="width: 6%">Line 1</th>
            <th style="width: 6%">Proj. 1</th>
            <th style="width: 16%">Player 2</th>
            <th style="width: 6%">Line 2</th>
            <th style="width: 6%">Proj. 2</th>
            <th style="width: 9%">EV $</th>
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
            <td class="ev-cell ${getEVClass(row.ev)}">$${row.ev.toFixed(2)}</td>
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
            <th style="width: 5%">Proj. 1</th>
            <th style="width: 13%">Player 2</th>
            <th style="width: 5%">Line 2</th>
            <th style="width: 5%">Proj. 2</th>
            <th style="width: 13%">Player 3</th>
            <th style="width: 5%">Line 3</th>
            <th style="width: 5%">Proj. 3</th>
            <th style="width: 7%">EV $</th>
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
            <td class="ev-cell ${getEVClass(row.ev)}">$${row.ev.toFixed(2)}</td>
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
                <div class="stat-label">Expected Value $</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value on a $10 stake (Ex. If EV is $2.00, you can expect to win $2.00 per $10 bet on average)</div>
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
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (volatile, less reliable predictions), Med, Low (consistent, more reliable predictions)</div>
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
                <div class="stat-label">Projection</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Model's projected value given the context of the game and player performance</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Expected Value $</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value on a $10 stake (Ex. If EV is $2.00, you can expect to win $2.00 per $10 bet on average)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sigma</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Confidence level: High (volatile, less reliable predictions), Med, Low (consistent, more reliable predictions)</div>
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

