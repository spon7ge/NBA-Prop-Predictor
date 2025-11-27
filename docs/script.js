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
    {"name1": "Julius Randle", "name2": "Clint Capela", "line1": 20.5, "line2": 5.5, "prediction1": 27.53, "prediction2": 0.71, "side1": "over", "side2": "under", "recommendation": 1, "ev": 112.89, "kelly": 0.564, "sigma1": "High", "sigma2": "Low", "prob1": 0.838, "prob2": 0.864, "hitRate1": 71.5, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 97.4, "l5_2": 0.0, "l15_2": 0.13},
    {"name1": "LaMelo Ball", "name2": "Alex Caruso", "line1": 20.5, "line2": 7.0, "prediction1": 27.56, "prediction2": 2.18, "side1": "over", "side2": "under", "recommendation": 1, "ev": 100.72, "kelly": 0.504, "sigma1": "High", "sigma2": "Med", "prob1": 0.829, "prob2": 0.824, "hitRate1": 31.8, "l5_1": 0.0, "l15_1": 0.13, "hitRate2": 75.2, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Ryan Rollins", "name2": "Harrison Barnes", "line1": 18.5, "line2": 12.5, "prediction1": 24.54, "prediction2": 17.48, "side1": "over", "side2": "over", "recommendation": 1, "ev": 84.79, "kelly": 0.424, "sigma1": "High", "sigma2": "High", "prob1": 0.793, "prob2": 0.792, "hitRate1": 62.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 59.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Ja'Kobe Walter", "name2": "Norman Powell", "line1": 8.5, "line2": 19.5, "prediction1": 4.36, "prediction2": 24.98, "side1": "under", "side2": "over", "recommendation": 1, "ev": 78.44, "kelly": 0.392, "sigma1": "Med", "sigma2": "High", "prob1": 0.78, "prob2": 0.778, "hitRate1": 89.2, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 94.0, "l5_2": 0.6, "l15_2": 0.73},
    {"name1": "Ben Sheppard", "name2": "Will Richard", "line1": 6.5, "line2": 7.0, "prediction1": 2.54, "prediction2": 3.04, "side1": "under", "side2": "under", "recommendation": 0, "ev": 73.66, "kelly": 0.368, "sigma1": "Med", "sigma2": "Med", "prob1": 0.765, "prob2": 0.772, "hitRate1": 76.4, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 69.6, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Bennedict Mathurin", "name2": "Tyler Herro", "line1": 20.5, "line2": 19.5, "prediction1": 25.0, "prediction2": 22.23, "side1": "over", "side2": "over", "recommendation": 0, "ev": 66.37, "kelly": 0.332, "sigma1": "High", "sigma2": "Low", "prob1": 0.751, "prob2": 0.754, "hitRate1": 76.2, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 97.3, "l5_2": 0.2, "l15_2": 0.07},
    {"name1": "Andrew Wiggins", "name2": "Saddiq Bey", "line1": 13.5, "line2": 12.5, "prediction1": 17.66, "prediction2": 16.95, "side1": "over", "side2": "over", "recommendation": 1, "ev": 53.84, "kelly": 0.269, "sigma1": "High", "sigma2": "High", "prob1": 0.72, "prob2": 0.727, "hitRate1": 76.3, "l5_1": 0.8, "l15_1": 0.8, "hitRate2": 56.3, "l5_2": 0.4, "l15_2": 0.4},
    {"name1": "Pascal Siakam", "name2": "Jeremiah Fears", "line1": 23.5, "line2": 14.5, "prediction1": 27.68, "prediction2": 18.4, "side1": "over", "side2": "over", "recommendation": 0, "ev": 49.68, "kelly": 0.248, "sigma1": "High", "sigma2": "High", "prob1": 0.716, "prob2": 0.711, "hitRate1": 68.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 77.7, "l5_2": 0.8, "l15_2": 0.73},
    {"name1": "Zion Williamson", "name2": "Precious Achiuwa", "line1": 22.5, "line2": 7.5, "prediction1": 26.36, "prediction2": 4.67, "side1": "over", "side2": "under", "recommendation": 0, "ev": 46.7, "kelly": 0.233, "sigma1": "High", "sigma2": "Med", "prob1": 0.707, "prob2": 0.705, "hitRate1": 47.5, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 62.3, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "T.J. McConnell", "name2": "Luke Kornet", "line1": 9.5, "line2": 8.5, "prediction1": 6.39, "prediction2": 5.93, "side1": "under", "side2": "under", "recommendation": 0, "ev": 45.3, "kelly": 0.227, "sigma1": "Med", "sigma2": "Low", "prob1": 0.704, "prob2": 0.702, "hitRate1": 55.9, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 55.1, "l5_2": 0.2, "l15_2": 0.27},
];const prizepicksTriosData = [
    {"name1": "LaMelo Ball", "name2": "Julius Randle", "name3": "Clint Capela", "line1": 20.5, "line2": 20.5, "line3": 5.5, "prediction1": 27.56, "prediction2": 27.53, "prediction3": 0.71, "side1": "over", "side2": "over", "side3": "under", "recommendation": 1, "ev": 224.01, "kelly": 0.448, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.829, "prob2": 0.838, "prob3": 0.864, "hitRate1": 31.8, "l5_1": 0.0, "l15_1": 0.13, "hitRate2": 71.5, "l5_2": 0.4, "l15_2": 0.6, "hitRate3": 97.4, "l5_3": 0.0, "l15_3": 0.13},
    {"name1": "Ryan Rollins", "name2": "Alex Caruso", "name3": "Harrison Barnes", "line1": 18.5, "line2": 7.0, "line3": 12.5, "prediction1": 24.54, "prediction2": 2.18, "prediction3": 17.48, "side1": "over", "side2": "under", "side3": "over", "recommendation": 1, "ev": 179.66, "kelly": 0.359, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.793, "prob2": 0.824, "prob3": 0.792, "hitRate1": 62.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 75.2, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 59.0, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Ja'Kobe Walter", "name2": "Norman Powell", "name3": "Will Richard", "line1": 8.5, "line2": 19.5, "line3": 7.0, "prediction1": 4.36, "prediction2": 24.98, "prediction3": 3.04, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 152.92, "kelly": 0.306, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.78, "prob2": 0.778, "prob3": 0.772, "hitRate1": 89.2, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 94.0, "l5_2": 0.6, "l15_2": 0.73, "hitRate3": 69.6, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Ben Sheppard", "name2": "Tyler Herro", "name3": "Saddiq Bey", "line1": 6.5, "line2": 19.5, "line3": 12.5, "prediction1": 2.54, "prediction2": 22.23, "prediction3": 16.95, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 126.46, "kelly": 0.253, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.765, "prob2": 0.754, "prob3": 0.727, "hitRate1": 76.4, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 97.3, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 56.3, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Bennedict Mathurin", "name2": "Andrew Wiggins", "name3": "Jeremiah Fears", "line1": 20.5, "line2": 13.5, "line3": 14.5, "prediction1": 25.0, "prediction2": 17.66, "prediction3": 18.4, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 107.61, "kelly": 0.215, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.751, "prob2": 0.72, "prob3": 0.711, "hitRate1": 76.2, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 76.3, "l5_2": 0.8, "l15_2": 0.8, "hitRate3": 77.7, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Pascal Siakam", "name2": "Zion Williamson", "name3": "Precious Achiuwa", "line1": 23.5, "line2": 22.5, "line3": 7.5, "prediction1": 27.68, "prediction2": 26.36, "prediction3": 4.67, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 92.84, "kelly": 0.186, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.716, "prob2": 0.707, "prob3": 0.705, "hitRate1": 68.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 47.5, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 62.3, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "T.J. McConnell", "name2": "Trey Murphy III", "name3": "Luke Kornet", "line1": 9.5, "line2": 19.5, "line3": 8.5, "prediction1": 6.39, "prediction2": 23.29, "prediction3": 5.93, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 85.93, "kelly": 0.172, "sigma1": "Med", "sigma2": "High", "sigma3": "Low", "prob1": 0.704, "prob2": 0.697, "prob3": 0.702, "hitRate1": 55.9, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 62.3, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 55.1, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Jamal Shead", "name2": "Josh Okogie", "name3": "Julian Champagnie", "line1": 7.5, "line2": 9.5, "line3": 11.5, "prediction1": 5.08, "prediction2": 6.64, "prediction3": 14.64, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 73.51, "kelly": 0.147, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.679, "prob2": 0.694, "prob3": 0.682, "hitRate1": 70.8, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 93.2, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 38.5, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Andrew Nembhard", "name2": "Luguentz Dort", "name3": "Drew Eubanks", "line1": 16.5, "line2": 8.5, "line3": 6.5, "prediction1": 19.63, "prediction2": 6.19, "prediction3": 3.98, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 59.62, "kelly": 0.119, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.667, "prob2": 0.656, "prob3": 0.675, "hitRate1": 81.6, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 43.8, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 78.1, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Miles Bridges", "name2": "Kentavious Caldwell-Pope", "name3": "Keegan Murray", "line1": 18.5, "line2": 7.5, "line3": 15.5, "prediction1": 21.42, "prediction2": 5.36, "prediction3": 18.1, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 52.93, "kelly": 0.106, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.656, "prob2": 0.655, "prob3": 0.659, "hitRate1": 76.7, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 84.4, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 55.0, "l5_3": 0.4, "l15_3": 0.13},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Julius Randle", "name2": "Clint Capela", "line1": 20.5, "line2": 5.5, "prediction1": 27.53, "prediction2": 0.71, "side1": "over", "side2": "under", "recommendation": 1, "ev": 112.89, "kelly": 0.564, "sigma1": "High", "sigma2": "Low", "prob1": 0.838, "prob2": 0.864, "hitRate1": 71.5, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 97.4, "l5_2": 0.0, "l15_2": 0.13},
    {"name1": "LaMelo Ball", "name2": "Yves Missi", "line1": 20.5, "line2": 6.5, "prediction1": 27.56, "prediction2": 1.98, "side1": "over", "side2": "under", "recommendation": 1, "ev": 103.41, "kelly": 0.517, "sigma1": "High", "sigma2": "Low", "prob1": 0.829, "prob2": 0.835, "hitRate1": 31.8, "l5_1": 0.0, "l15_1": 0.13, "hitRate2": 66.7, "l5_2": 0.4, "l15_2": 0.4},
    {"name1": "Jalen Brunson", "name2": "Shai Gilgeous-Alexander", "line1": 28.5, "line2": 31.5, "prediction1": 35.48, "prediction2": 36.74, "side1": "over", "side2": "over", "recommendation": 1, "ev": 93.1, "kelly": 0.465, "sigma1": "High", "sigma2": "High", "prob1": 0.828, "prob2": 0.793, "hitRate1": 44.6, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 61.8, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Giannis Antetokounmpo", "name2": "Anthony Edwards", "line1": 28.5, "line2": 25.5, "prediction1": 34.55, "prediction2": 31.4, "side1": "over", "side2": "over", "recommendation": 1, "ev": 82.67, "kelly": 0.413, "sigma1": "High", "sigma2": "High", "prob1": 0.79, "prob2": 0.786, "hitRate1": 44.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 75.8, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Norman Powell", "name2": "Stephen Curry", "line1": 19.5, "line2": 26.5, "prediction1": 24.98, "prediction2": 32.37, "side1": "over", "side2": "over", "recommendation": 1, "ev": 76.77, "kelly": 0.384, "sigma1": "High", "sigma2": "High", "prob1": 0.778, "prob2": 0.772, "hitRate1": 94.0, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 77.0, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Sion James", "name2": "Devin Booker", "line1": 4.5, "line2": 28.5, "prediction1": 1.07, "prediction2": 33.67, "side1": "under", "side2": "over", "recommendation": 0, "ev": 71.61, "kelly": 0.358, "sigma1": "Low", "sigma2": "High", "prob1": 0.766, "prob2": 0.762, "hitRate1": 43.6, "l5_1": 0.2, "l15_1": 0.67, "hitRate2": 23.9, "l5_2": 0.0, "l15_2": 0.33},
    {"name1": "Tyler Herro", "name2": "Kris Murray", "line1": 20.5, "line2": 4.5, "prediction1": 23.33, "prediction2": 1.17, "side1": "over", "side2": "under", "recommendation": 0, "ev": 68.38, "kelly": 0.342, "sigma1": "Low", "sigma2": "Low", "prob1": 0.762, "prob2": 0.752, "hitRate1": 95.6, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 35.1, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Pascal Siakam", "name2": "Zion Williamson", "line1": 23.5, "line2": 22.5, "prediction1": 27.68, "prediction2": 26.36, "side1": "over", "side2": "over", "recommendation": 0, "ev": 48.83, "kelly": 0.244, "sigma1": "High", "sigma2": "High", "prob1": 0.716, "prob2": 0.707, "hitRate1": 68.6, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 47.5, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "T.J. McConnell", "name2": "Precious Achiuwa", "line1": 9.5, "line2": 7.5, "prediction1": 6.39, "prediction2": 4.67, "side1": "under", "side2": "under", "recommendation": 0, "ev": 45.94, "kelly": 0.23, "sigma1": "Med", "sigma2": "Med", "prob1": 0.704, "prob2": 0.705, "hitRate1": 55.9, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 62.3, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Trey Murphy III", "name2": "Luke Kornet", "line1": 19.5, "line2": 8.5, "prediction1": 23.29, "prediction2": 5.93, "side1": "over", "side2": "under", "recommendation": 0, "ev": 43.86, "kelly": 0.219, "sigma1": "High", "sigma2": "Low", "prob1": 0.697, "prob2": 0.702, "hitRate1": 62.3, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 55.1, "l5_2": 0.2, "l15_2": 0.27},
];const underdogTriosData = [
    {"name1": "Julius Randle", "name2": "Yves Missi", "name3": "Clint Capela", "line1": 20.5, "line2": 6.5, "line3": 5.5, "prediction1": 27.53, "prediction2": 1.98, "prediction3": 0.71, "side1": "over", "side2": "under", "side3": "under", "recommendation": 1, "ev": 226.5, "kelly": 0.453, "sigma1": "High", "sigma2": "Low", "sigma3": "Low", "prob1": 0.838, "prob2": 0.835, "prob3": 0.864, "hitRate1": 71.5, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 66.7, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 97.4, "l5_3": 0.0, "l15_3": 0.13},
    {"name1": "LaMelo Ball", "name2": "Pascal Siakam", "name3": "Norman Powell", "line1": 20.5, "line2": 23.5, "line3": 19.5, "prediction1": 27.56, "prediction2": 27.68, "prediction3": 24.98, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 149.29, "kelly": 0.299, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.829, "prob2": 0.716, "prob3": 0.778, "hitRate1": 31.8, "l5_1": 0.0, "l15_1": 0.13, "hitRate2": 68.6, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 94.0, "l5_3": 0.6, "l15_3": 0.73},
    {"name1": "Tyler Herro", "name2": "Zion Williamson", "name3": "Precious Achiuwa", "line1": 20.5, "line2": 22.5, "line3": 7.5, "prediction1": 23.33, "prediction2": 26.36, "prediction3": 4.67, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 105.26, "kelly": 0.211, "sigma1": "Low", "sigma2": "High", "sigma3": "Med", "prob1": 0.762, "prob2": 0.707, "prob3": 0.705, "hitRate1": 95.6, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 47.5, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 62.3, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "T.J. McConnell", "name2": "Trey Murphy III", "name3": "Luke Kornet", "line1": 9.5, "line2": 19.5, "line3": 8.5, "prediction1": 6.39, "prediction2": 23.29, "prediction3": 5.93, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 85.93, "kelly": 0.172, "sigma1": "Med", "sigma2": "High", "sigma3": "Low", "prob1": 0.704, "prob2": 0.697, "prob3": 0.702, "hitRate1": 55.9, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 62.3, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 55.1, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Josh Hart", "name2": "Josh Okogie", "name3": "Julian Champagnie", "line1": 13.5, "line2": 9.5, "line3": 11.5, "prediction1": 10.42, "prediction2": 6.64, "prediction3": 14.64, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 75.71, "kelly": 0.151, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.688, "prob2": 0.694, "prob3": 0.682, "hitRate1": 74.1, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 93.2, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 38.5, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Jamal Shead", "name2": "Andrew Wiggins", "name3": "Derik Queen", "line1": 7.5, "line2": 14.5, "line3": 13.5, "prediction1": 5.08, "prediction2": 17.66, "prediction3": 16.49, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 63.12, "kelly": 0.126, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.679, "prob2": 0.671, "prob3": 0.663, "hitRate1": 70.8, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 67.6, "l5_2": 0.8, "l15_2": 0.8, "hitRate3": 56.6, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Luguentz Dort", "name2": "Kentavious Caldwell-Pope", "name3": "Keegan Murray", "line1": 8.5, "line2": 7.5, "line3": 15.5, "prediction1": 6.19, "prediction2": 5.36, "prediction3": 18.1, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 53.02, "kelly": 0.106, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.656, "prob2": 0.655, "prob3": 0.659, "hitRate1": 43.8, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 84.4, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 55.0, "l5_3": 0.4, "l15_3": 0.13},
    {"name1": "Davion Mitchell", "name2": "Dillon Brooks", "name3": "Shaedon Sharpe", "line1": 7.5, "line2": 20.5, "line3": 22.5, "prediction1": 9.87, "prediction2": 23.4, "prediction3": 25.23, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 49.12, "kelly": 0.098, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.649, "prob2": 0.654, "prob3": 0.651, "hitRate1": 88.5, "l5_1": 0.8, "l15_1": 0.93, "hitRate2": 77.9, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 70.5, "l5_3": 0.8, "l15_3": 0.33},
    {"name1": "Bam Adebayo", "name2": "Alperen Sengun", "name3": "Russell Westbrook", "line1": 16.5, "line2": 23.5, "line3": 12.5, "prediction1": 19.26, "prediction2": 25.42, "prediction3": 15.11, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 35.61, "kelly": 0.071, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.649, "prob2": 0.602, "prob3": 0.643, "hitRate1": 60.5, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 36.7, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 43.6, "l5_3": 0.4, "l15_3": 0.53},
    {"name1": "Kyle Kuzma", "name2": "Quinten Post", "name3": "Donovan Clingan", "line1": 12.5, "line2": 7.5, "line3": 10.5, "prediction1": 14.79, "prediction2": 5.97, "prediction3": 11.93, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 21.68, "kelly": 0.043, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.635, "prob2": 0.602, "prob3": 0.59, "hitRate1": 59.0, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 70.5, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 74.2, "l5_3": 0.6, "l15_3": 0.4},
];const prizepicksPointsHitRates = [
    {"name": "Tyler Herro", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.973, "underPct": 0.027},
    {"name": "Norman Powell", "line": 19.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.94, "underPct": 0.06},
    {"name": "Davion Mitchell", "line": 7.5, "l5": 0.8, "l10": 0.9, "l15": 0.93, "overPct": 0.885, "underPct": 0.115},
    {"name": "Kon Knueppel", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.88, "underPct": 0.12},
    {"name": "Collin Sexton", "line": 12.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.859, "underPct": 0.141},
    {"name": "Naz Reid", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.854, "underPct": 0.146},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.816, "underPct": 0.184},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.779, "underPct": 0.221},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.777, "underPct": 0.223},
    {"name": "Stephen Curry", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.77, "underPct": 0.23},
    {"name": "Brandin Podziemski", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.767, "underPct": 0.233},
    {"name": "Miles Bridges", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.767, "underPct": 0.233},
    {"name": "Andrew Wiggins", "line": 13.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.763, "underPct": 0.237},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.762, "underPct": 0.238},
    {"name": "Anthony Edwards", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.758, "underPct": 0.242},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.729, "underPct": 0.271},
    {"name": "Dylan Harper", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.72, "underPct": 0.28},
    {"name": "Julius Randle", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.715, "underPct": 0.285},
    {"name": "Kel'el Ware", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.715, "underPct": 0.285},
    {"name": "Shaedon Sharpe", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.705, "underPct": 0.295},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.695, "underPct": 0.305},
    {"name": "Pascal Siakam", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.686, "underPct": 0.314},
    {"name": "Moses Moody", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.678, "underPct": 0.322},
    {"name": "Kris Murray", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.649, "underPct": 0.351},
    {"name": "Jock Landale", "line": 9.0, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.646, "underPct": 0.354},
    {"name": "Sandro Mamukelashvili", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.645, "underPct": 0.355},
    {"name": "Donte DiVincenzo", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.64, "underPct": 0.36},
    {"name": "Donovan Clingan", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.639, "underPct": 0.361},
    {"name": "Collin Murray-Boyles", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.632, "underPct": 0.368},
    {"name": "Bobby Portis", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.627, "underPct": 0.373},
    {"name": "Ryan Rollins", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.627, "underPct": 0.373},
    {"name": "Jakob Poeltl", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.626, "underPct": 0.374},
    {"name": "Trey Murphy III", "line": 19.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.623, "underPct": 0.377},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.618, "underPct": 0.382},
    {"name": "Bam Adebayo", "line": 16.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.605, "underPct": 0.395},
    {"name": "Oso Ighodaro", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.603, "underPct": 0.397},
    {"name": "Kyle Kuzma", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.59, "underPct": 0.41},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.59, "underPct": 0.41},
    {"name": "Brandon Miller", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.576, "underPct": 0.424},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.57, "underPct": 0.43},
    {"name": "Sion James", "line": 4.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.564, "underPct": 0.436},
    {"name": "Saddiq Bey", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.562, "underPct": 0.438},
    {"name": "Jordan Goodwin", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.553, "underPct": 0.447},
    {"name": "Keegan Murray", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.55, "underPct": 0.45},
    {"name": "Myles Turner", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.55, "underPct": 0.45},
    {"name": "Karl-Anthony Towns", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.54, "underPct": 0.46},
    {"name": "Jarace Walker", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.522, "underPct": 0.478},
    {"name": "Buddy Hield", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.522, "underPct": 0.478},
    {"name": "Zion Williamson", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.475, "underPct": 0.525},
    {"name": "Jay Huff", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.474, "underPct": 0.526},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.473, "underPct": 0.527},
    {"name": "Derik Queen", "line": 14.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.46, "underPct": 0.54},
    {"name": "Mitchell Robinson", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.455, "underPct": 0.545},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.451, "underPct": 0.549},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.449, "underPct": 0.551},
    {"name": "Giannis Antetokounmpo", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.447, "underPct": 0.553},
    {"name": "Jalen Brunson", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.446, "underPct": 0.554},
    {"name": "T.J. McConnell", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.441, "underPct": 0.559},
    {"name": "Russell Westbrook", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.436, "underPct": 0.564},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.427, "underPct": 0.573},
    {"name": "Reed Sheppard", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.425, "underPct": 0.575},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.422, "underPct": 0.578},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.419, "underPct": 0.581},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Collin Gillespie", "line": 13.5, "l5": 1.0, "l10": 0.5, "l15": 0.53, "overPct": 0.412, "underPct": 0.588},
    {"name": "Cam Spencer", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.411, "underPct": 0.589},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.396, "underPct": 0.604},
    {"name": "Malik Monk", "line": 12.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.386, "underPct": 0.614},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.377, "underPct": 0.623},
    {"name": "Cedric Coward", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.375, "underPct": 0.625},
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.37, "underPct": 0.63},
    {"name": "Alperen Sengun", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.367, "underPct": 0.633},
    {"name": "Miles McBride", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.366, "underPct": 0.634},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.358, "underPct": 0.642},
    {"name": "Jerami Grant", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.347, "underPct": 0.653},
    {"name": "Keldon Johnson", "line": 13.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.347, "underPct": 0.653},
    {"name": "Ajay Mitchell", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.344, "underPct": 0.656},
    {"name": "Amen Thompson", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.336, "underPct": 0.664},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.335, "underPct": 0.665},
    {"name": "Royce O'Neale", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.325, "underPct": 0.675},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.323, "underPct": 0.677},
    {"name": "LaMelo Ball", "line": 20.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.318, "underPct": 0.682},
    {"name": "Will Richard", "line": 7.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.304, "underPct": 0.696},
    {"name": "Draymond Green", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.303, "underPct": 0.697},
    {"name": "Quinten Post", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.295, "underPct": 0.705},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.292, "underPct": 0.708},
    {"name": "Brandon Ingram", "line": 24.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.288, "underPct": 0.712},
    {"name": "Deni Avdija", "line": 26.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.258, "underPct": 0.742},
    {"name": "Alex Caruso", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.248, "underPct": 0.752},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.239, "underPct": 0.761},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.237, "underPct": 0.763},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.236, "underPct": 0.764},
    {"name": "Aaron Holiday", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.236, "underPct": 0.764},
    {"name": "Drew Eubanks", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.219, "underPct": 0.781},
    {"name": "Mark Williams", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.205, "underPct": 0.795},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.176, "underPct": 0.824},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.156, "underPct": 0.844},
    {"name": "Ja'Kobe Walter", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.108, "underPct": 0.892},
    {"name": "Josh Okogie", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.068, "underPct": 0.932},
    {"name": "Clint Capela", "line": 5.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.026, "underPct": 0.974},
];const prizepicksAssistsHitRates = [
    {"name": "Tyler Herro", "line": 4.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.763, "underPct": 0.237},
    {"name": "Davion Mitchell", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.707, "underPct": 0.293},
    {"name": "LaMelo Ball", "line": 7.5, "l5": 0.8, "l10": 0.9, "l15": 0.6, "overPct": 0.676, "underPct": 0.324},
    {"name": "Naz Reid", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.641, "underPct": 0.359},
    {"name": "Mike Conley", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.628, "underPct": 0.372},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.565, "underPct": 0.435},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.564, "underPct": 0.436},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.558, "underPct": 0.442},
    {"name": "Josh Hart", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.552, "underPct": 0.448},
    {"name": "Giannis Antetokounmpo", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.536, "underPct": 0.464},
    {"name": "Julius Randle", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.529, "underPct": 0.471},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.516, "underPct": 0.484},
    {"name": "Luguentz Dort", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.504, "underPct": 0.496},
    {"name": "Alperen Sengun", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.498, "underPct": 0.502},
    {"name": "Shai Gilgeous-Alexander", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.496, "underPct": 0.504},
    {"name": "Ryan Rollins", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.487, "underPct": 0.513},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.478, "underPct": 0.522},
    {"name": "De'Aaron Fox", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.464, "underPct": 0.536},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.458, "underPct": 0.542},
    {"name": "Jamal Shead", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.454, "underPct": 0.546},
    {"name": "Andrew Nembhard", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.448, "underPct": 0.552},
    {"name": "Jock Landale", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.438, "underPct": 0.562},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Draymond Green", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.428, "underPct": 0.572},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.404, "underPct": 0.596},
    {"name": "Toumani Camara", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.373, "underPct": 0.627},
    {"name": "Zion Williamson", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.363, "underPct": 0.637},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.362, "underPct": 0.638},
    {"name": "Amen Thompson", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.359, "underPct": 0.641},
    {"name": "Stephen Curry", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.338, "underPct": 0.662},
    {"name": "Jarace Walker", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.325, "underPct": 0.675},
    {"name": "T.J. McConnell", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.309, "underPct": 0.691},
    {"name": "Pascal Siakam", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.29, "underPct": 0.71},
    {"name": "Deni Avdija", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.278, "underPct": 0.722},
    {"name": "Kris Murray", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.258, "underPct": 0.742},
];const prizepicksReboundsHitRates = [
    {"name": "Tyler Herro", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.827, "underPct": 0.173},
    {"name": "Kel'el Ware", "line": 11.0, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.706, "underPct": 0.294},
    {"name": "Naz Reid", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.687, "underPct": 0.313},
    {"name": "Saddiq Bey", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.683, "underPct": 0.317},
    {"name": "Julius Randle", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.679, "underPct": 0.321},
    {"name": "LaMelo Ball", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.678, "underPct": 0.322},
    {"name": "Jock Landale", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.674, "underPct": 0.326},
    {"name": "Bennedict Mathurin", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.652, "underPct": 0.348},
    {"name": "Jaden McDaniels", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.65, "underPct": 0.35},
    {"name": "Kris Murray", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.629, "underPct": 0.371},
    {"name": "Jordan Goodwin", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.628, "underPct": 0.372},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.598, "underPct": 0.402},
    {"name": "Andrew Wiggins", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.597, "underPct": 0.403},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.588, "underPct": 0.412},
    {"name": "Keegan Murray", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.587, "underPct": 0.413},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.559, "underPct": 0.441},
    {"name": "Keldon Johnson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.556, "underPct": 0.444},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.553, "underPct": 0.447},
    {"name": "Mitchell Robinson", "line": 7.0, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.553, "underPct": 0.447},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.545, "underPct": 0.455},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.54, "underPct": 0.46},
    {"name": "Isaiah Jackson", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.538, "underPct": 0.462},
    {"name": "Jerami Grant", "line": 3.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.535, "underPct": 0.465},
    {"name": "Shai Gilgeous-Alexander", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.529, "underPct": 0.471},
    {"name": "Immanuel Quickley", "line": 4.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.52, "underPct": 0.48},
    {"name": "Bobby Portis", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.512, "underPct": 0.488},
    {"name": "Scottie Barnes", "line": 8.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.507, "underPct": 0.493},
    {"name": "Rudy Gobert", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.502, "underPct": 0.498},
    {"name": "Isaiah Hartenstein", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.498, "underPct": 0.502},
    {"name": "Collin Gillespie", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.498, "underPct": 0.502},
    {"name": "Ryan Rollins", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.484, "underPct": 0.516},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Chet Holmgren", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Mike Conley", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.468, "underPct": 0.532},
    {"name": "Ryan Kalkbrenner", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.468, "underPct": 0.532},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.468, "underPct": 0.532},
    {"name": "Jarace Walker", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.468, "underPct": 0.532},
    {"name": "Jakob Poeltl", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.463, "underPct": 0.537},
    {"name": "Kyle Kuzma", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.455, "underPct": 0.545},
    {"name": "Devin Booker", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.452, "underPct": 0.548},
    {"name": "Harrison Barnes", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Bam Adebayo", "line": 8.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.444, "underPct": 0.556},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.441, "underPct": 0.559},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.439, "underPct": 0.561},
    {"name": "Josh Hart", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Kon Knueppel", "line": 5.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.43, "underPct": 0.57},
    {"name": "Pascal Siakam", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.419, "underPct": 0.581},
    {"name": "Royce O'Neale", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.415, "underPct": 0.585},
    {"name": "Brandin Podziemski", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.411, "underPct": 0.589},
    {"name": "Jaylin Williams", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.395, "underPct": 0.605},
    {"name": "Zach LaVine", "line": 3.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.39, "underPct": 0.61},
    {"name": "Miles Bridges", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.388, "underPct": 0.612},
    {"name": "Amen Thompson", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.38, "underPct": 0.62},
    {"name": "Jaylen Wells", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.361, "underPct": 0.639},
    {"name": "Deni Avdija", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.355, "underPct": 0.645},
    {"name": "Myles Turner", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.343, "underPct": 0.657},
    {"name": "Sandro Mamukelashvili", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.334, "underPct": 0.666},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.323, "underPct": 0.677},
    {"name": "Giannis Antetokounmpo", "line": 10.0, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.316, "underPct": 0.684},
    {"name": "T.J. McConnell", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.312, "underPct": 0.688},
    {"name": "Jay Huff", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.29, "underPct": 0.71},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.289, "underPct": 0.711},
    {"name": "Brandon Ingram", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.288, "underPct": 0.712},
    {"name": "Yves Missi", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.28, "underPct": 0.72},
    {"name": "Quinten Post", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.274, "underPct": 0.726},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.268, "underPct": 0.732},
    {"name": "Zion Williamson", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.264, "underPct": 0.736},
    {"name": "Luke Kornet", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.263, "underPct": 0.737},
    {"name": "Luguentz Dort", "line": 4.0, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.261, "underPct": 0.739},
    {"name": "Draymond Green", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.246, "underPct": 0.754},
    {"name": "Stephen Curry", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.237, "underPct": 0.763},
    {"name": "Precious Achiuwa", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.213, "underPct": 0.787},
    {"name": "Mark Williams", "line": 9.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.207, "underPct": 0.793},
    {"name": "Julian Champagnie", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.184, "underPct": 0.816},
    {"name": "Josh Okogie", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.182, "underPct": 0.818},
    {"name": "Clint Capela", "line": 6.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.071, "underPct": 0.929},
];const prizepicksBlocksHitRates = [
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.465, "underPct": 0.535},
    {"name": "Jakob Poeltl", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.448, "underPct": 0.552},
    {"name": "Noah Clowney", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.654, "underPct": 0.346},
    {"name": "Matas Buzelis", "line": 1.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Josh Giddey", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.401, "underPct": 0.599},
    {"name": "Zion Williamson", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.259, "underPct": 0.741},
    {"name": "Kevin Huerter", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.533, "underPct": 0.467},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.515, "underPct": 0.485},
    {"name": "Kyle Kuzma", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.456, "underPct": 0.544},
    {"name": "Amen Thompson", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Moses Moody", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.584, "underPct": 0.416},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.258, "underPct": 0.742},
];const prizepicksStealsHitRates = [
    {"name": "Ryan Kalkbrenner", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.495, "underPct": 0.505},
    {"name": "Isaiah Jackson", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.594, "underPct": 0.406},
    {"name": "Kyle Kuzma", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.574, "underPct": 0.426},
    {"name": "Cason Wallace", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.623, "underPct": 0.377},
    {"name": "Chet Holmgren", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.436, "underPct": 0.564},
    {"name": "Cedric Coward", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.447, "underPct": 0.553},
    {"name": "Kentavious Caldwell-Pope", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Aaron Holiday", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.331, "underPct": 0.669},
    {"name": "Quinten Post", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.488, "underPct": 0.512},
    {"name": "Jordan Goodwin", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Drew Eubanks", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.481, "underPct": 0.519},
    {"name": "Donovan Clingan", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.481, "underPct": 0.519},
    {"name": "Jeremy Sochan", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.333, "underPct": 0.667},
    {"name": "Luke Kornet", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.105, "underPct": 0.895},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Shaedon Sharpe", "line": 29.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Wiggins", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Kalkbrenner", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Gillespie", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dillon Brooks", "line": 26.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Draymond Green", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 19.0, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 25.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bobby Portis", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 15.5, "l5": 0.6, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shai Gilgeous-Alexander", "line": 43.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Goodwin", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 41.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gary Payton II", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Caruso", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mike Conley", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 17.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "T.J. McConnell", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "LaMelo Ball", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles Bridges", "line": 28.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Josh Hart", "line": 26.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremy Sochan", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 34.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dylan Harper", "line": 19.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Oso Ighodaro", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Nembhard", "line": 24.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Aaron Fox", "line": 35.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bam Adebayo", "line": 28.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Giannis Antetokounmpo", "line": 44.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Shead", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quinten Post", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Okogie", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Holiday", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 40.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julian Champagnie", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 39.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 22.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Miller", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles McBride", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jay Huff", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Murray-Boyles", "line": 10.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 33.0, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Murray", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Yves Missi", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 28.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Jackson", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ja'Kobe Walter", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 18.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ben Sheppard", "line": 12.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyler Herro", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 33.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Clint Capela", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 40.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kentavious Caldwell-Pope", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
];const prizepicksPRHitRates = [
    {"name": "Shaedon Sharpe", "line": 26.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Gillespie", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.8, "l10": 0.9, "l15": 0.87, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jerami Grant", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 26.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephen Curry", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 22.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 20.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Buddy Hield", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Norman Powell", "line": 22.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 25.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dylan Harper", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarace Walker", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 12.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Will Richard", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julian Champagnie", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremy Sochan", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 15.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Okogie", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quinten Post", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Drew Eubanks", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Holiday", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Miller", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Hart", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ja'Kobe Walter", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Giannis Antetokounmpo", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylin Williams", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Caruso", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mitchell Robinson", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sion James", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Scottie Barnes", "line": 28.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 20.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 34.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Oso Ighodaro", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Jackson", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ben Sheppard", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kentavious Caldwell-Pope", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyler Herro", "line": 24.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 12.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Clint Capela", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 25.0, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const prizepicksPAHitRates = [
    {"name": "Stephen Curry", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Gillespie", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 16.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 21.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Giannis Antetokounmpo", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shaedon Sharpe", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 14.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dylan Harper", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kel'el Ware", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 22.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 20.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Chet Holmgren", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Davion Mitchell", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Buddy Hield", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cason Wallace", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles Bridges", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Aaron Fox", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quinten Post", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Draymond Green", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Holiday", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julian Champagnie", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keegan Murray", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Drew Eubanks", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Precious Achiuwa", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Caruso", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 19.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Karl-Anthony Towns", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Miller", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ja'Kobe Walter", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jay Huff", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Murray-Boyles", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Murray", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kentavious Caldwell-Pope", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Saddiq Bey", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 27.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jose Alvarado", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ben Sheppard", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Oso Ighodaro", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 36.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyler Herro", "line": 24.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksRAHitRates = [
    {"name": "Kel'el Ware", "line": 12.0, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaden McDaniels", "line": 6.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Draymond Green", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jay Huff", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 13.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quinten Post", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mike Conley", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Caruso", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 11.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 7.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sion James", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Wells", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Yves Missi", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Saddiq Bey", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 14.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Will Richard", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 15.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 13.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Chet Holmgren", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 10.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luke Kornet", "line": 9.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keldon Johnson", "line": 9.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles McBride", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Miller", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Trey Murphy III", "line": 9.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Jackson", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keegan Murray", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Giannis Antetokounmpo", "line": 16.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Stephen Curry", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ja'Kobe Walter", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 6.0, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 8.0, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyler Herro", "line": 8.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksTurnoversHitRates = [
    {"name": "Dillon Brooks", "line": 1.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Gillespie", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Will Richard", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 1.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dylan Harper", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gary Payton II", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Naz Reid", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ben Sheppard", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Oso Ighodaro", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Stephen Curry", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dru Smith", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 1.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brice Sensabaugh", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
];const underdogPointsHitRates = [
    {"name": "Pelle Larsson", "line": 5.5, "l5": 1.0, "l10": 1.0, "l15": 0.93, "overPct": 0.958, "underPct": 0.042},
    {"name": "Tyler Herro", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.956, "underPct": 0.044},
    {"name": "Norman Powell", "line": 19.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.94, "underPct": 0.06},
    {"name": "Davion Mitchell", "line": 7.5, "l5": 0.8, "l10": 0.9, "l15": 0.93, "overPct": 0.885, "underPct": 0.115},
    {"name": "Naz Reid", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.854, "underPct": 0.146},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.779, "underPct": 0.221},
    {"name": "Stephen Curry", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.77, "underPct": 0.23},
    {"name": "Brandin Podziemski", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.767, "underPct": 0.233},
    {"name": "Anthony Edwards", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.758, "underPct": 0.242},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.742, "underPct": 0.258},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.729, "underPct": 0.271},
    {"name": "Julius Randle", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.715, "underPct": 0.285},
    {"name": "Shaedon Sharpe", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.705, "underPct": 0.295},
    {"name": "Jaden McDaniels", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.695, "underPct": 0.305},
    {"name": "Pascal Siakam", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.686, "underPct": 0.314},
    {"name": "Moses Moody", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.678, "underPct": 0.322},
    {"name": "Andrew Wiggins", "line": 14.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.676, "underPct": 0.324},
    {"name": "Myles Turner", "line": 12.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.655, "underPct": 0.345},
    {"name": "Kris Murray", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.649, "underPct": 0.351},
    {"name": "Donte DiVincenzo", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.64, "underPct": 0.36},
    {"name": "Collin Murray-Boyles", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.632, "underPct": 0.368},
    {"name": "Jakob Poeltl", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.626, "underPct": 0.374},
    {"name": "Trey Murphy III", "line": 19.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.623, "underPct": 0.377},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.618, "underPct": 0.382},
    {"name": "Bam Adebayo", "line": 16.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.605, "underPct": 0.395},
    {"name": "Kyle Kuzma", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.59, "underPct": 0.41},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.566, "underPct": 0.434},
    {"name": "Sion James", "line": 4.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.564, "underPct": 0.436},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.562, "underPct": 0.438},
    {"name": "Jordan Goodwin", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.553, "underPct": 0.447},
    {"name": "Keegan Murray", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.55, "underPct": 0.45},
    {"name": "Karl-Anthony Towns", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.54, "underPct": 0.46},
    {"name": "Buddy Hield", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.522, "underPct": 0.478},
    {"name": "Zion Williamson", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.475, "underPct": 0.525},
    {"name": "Jay Huff", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.474, "underPct": 0.526},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.473, "underPct": 0.527},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.449, "underPct": 0.551},
    {"name": "Giannis Antetokounmpo", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.447, "underPct": 0.553},
    {"name": "Jalen Brunson", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.446, "underPct": 0.554},
    {"name": "T.J. McConnell", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.441, "underPct": 0.559},
    {"name": "Russell Westbrook", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.436, "underPct": 0.564},
    {"name": "Jerami Grant", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.433, "underPct": 0.567},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Cam Spencer", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.411, "underPct": 0.589},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.396, "underPct": 0.604},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.377, "underPct": 0.623},
    {"name": "Cedric Coward", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.375, "underPct": 0.625},
    {"name": "Alperen Sengun", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.367, "underPct": 0.633},
    {"name": "Ajay Mitchell", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.344, "underPct": 0.656},
    {"name": "Amen Thompson", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.336, "underPct": 0.664},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.335, "underPct": 0.665},
    {"name": "Yves Missi", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.333, "underPct": 0.667},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.323, "underPct": 0.677},
    {"name": "LaMelo Ball", "line": 20.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.318, "underPct": 0.682},
    {"name": "Quinten Post", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.295, "underPct": 0.705},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.292, "underPct": 0.708},
    {"name": "Josh Hart", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.259, "underPct": 0.741},
    {"name": "Deni Avdija", "line": 26.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.258, "underPct": 0.742},
    {"name": "Devin Booker", "line": 28.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.239, "underPct": 0.761},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.237, "underPct": 0.763},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.156, "underPct": 0.844},
    {"name": "Josh Okogie", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.068, "underPct": 0.932},
    {"name": "Clint Capela", "line": 5.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.026, "underPct": 0.974},
];const underdogAssistsHitRates = [
    {"name": "Davion Mitchell", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.707, "underPct": 0.293},
    {"name": "LaMelo Ball", "line": 7.5, "l5": 0.8, "l10": 0.9, "l15": 0.6, "overPct": 0.676, "underPct": 0.324},
    {"name": "Andrew Wiggins", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.657, "underPct": 0.343},
    {"name": "Naz Reid", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.641, "underPct": 0.359},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.558, "underPct": 0.442},
    {"name": "Kon Knueppel", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.551, "underPct": 0.449},
    {"name": "Ryan Rollins", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.487, "underPct": 0.513},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.458, "underPct": 0.542},
    {"name": "Jock Landale", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.438, "underPct": 0.562},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Toumani Camara", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.373, "underPct": 0.627},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.362, "underPct": 0.638},
    {"name": "Jarace Walker", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.325, "underPct": 0.675},
    {"name": "Kris Murray", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.258, "underPct": 0.742},
];const underdogReboundsHitRates = [
    {"name": "Saddiq Bey", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.683, "underPct": 0.317},
    {"name": "Jock Landale", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.674, "underPct": 0.326},
    {"name": "Bennedict Mathurin", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.652, "underPct": 0.348},
    {"name": "Jaden McDaniels", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.65, "underPct": 0.35},
    {"name": "Kris Murray", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.629, "underPct": 0.371},
    {"name": "Jordan Goodwin", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.628, "underPct": 0.372},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.598, "underPct": 0.402},
    {"name": "Jose Alvarado", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.588, "underPct": 0.412},
    {"name": "Keegan Murray", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.587, "underPct": 0.413},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.559, "underPct": 0.441},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.553, "underPct": 0.447},
    {"name": "Isaiah Jackson", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.538, "underPct": 0.462},
    {"name": "Shai Gilgeous-Alexander", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.529, "underPct": 0.471},
    {"name": "Bobby Portis", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.512, "underPct": 0.488},
    {"name": "Isaiah Hartenstein", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.498, "underPct": 0.502},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Ryan Kalkbrenner", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.468, "underPct": 0.532},
    {"name": "Mike Conley", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.468, "underPct": 0.532},
    {"name": "Jakob Poeltl", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.463, "underPct": 0.537},
    {"name": "Dillon Brooks", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.452, "underPct": 0.548},
    {"name": "Harrison Barnes", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.439, "underPct": 0.561},
    {"name": "Stephen Curry", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.419, "underPct": 0.581},
    {"name": "Pascal Siakam", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.419, "underPct": 0.581},
    {"name": "Jaylin Williams", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.395, "underPct": 0.605},
    {"name": "Jaylen Wells", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.361, "underPct": 0.639},
    {"name": "T.J. McConnell", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.312, "underPct": 0.688},
    {"name": "Jay Huff", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.29, "underPct": 0.71},
    {"name": "Luke Kornet", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.263, "underPct": 0.737},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Shaedon Sharpe", "line": 29.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Draymond Green", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 25.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Pascal Siakam", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kel'el Ware", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Gillespie", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Wiggins", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dylan Harper", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Spencer", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Hartenstein", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mike Conley", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 40.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremy Sochan", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Buddy Hield", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gary Payton II", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 35.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Oso Ighodaro", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Caruso", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Goodwin", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 44.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ben Sheppard", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sandro Mamukelashvili", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 28.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Myles Turner", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "T.J. McConnell", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 34.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Norman Powell", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Bobby Portis", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 15.5, "l5": 0.6, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Miles Bridges", "line": 28.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ryan Rollins", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 43.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keegan Murray", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Aaron Holiday", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Precious Achiuwa", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jay Huff", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 32.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylin Williams", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Miller", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Murray", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Yves Missi", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 40.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Saddiq Bey", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jock Landale", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 33.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Okogie", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 33.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Wells", "line": 18.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trey Murphy III", "line": 28.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyler Herro", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Isaiah Jackson", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 40.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "Clint Capela", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kentavious Caldwell-Pope", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
];const underdogPRHitRates = [
    {"name": "Shaedon Sharpe", "line": 26.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 27.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Brunson", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 22.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Bam Adebayo", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyler Herro", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Scottie Barnes", "line": 28.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 32.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 25.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const underdogPAHitRates = [
    {"name": "Trey Murphy III", "line": 22.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Giannis Antetokounmpo", "line": 34.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 21.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Shaedon Sharpe", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles Bridges", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Rollins", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 20.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 36.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyler Herro", "line": 24.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "LaMelo Ball", "line": 28.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogRAHitRates = [
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Draymond Green", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bam Adebayo", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
];const underdogTurnoversHitRates = [
    {"name": "Alperen Sengun", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
];const underdogBlocksStealsHitRates = [
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
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

