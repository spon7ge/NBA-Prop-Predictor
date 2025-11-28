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
    {"name1": "Drake Powell", "name2": "Cameron Johnson", "line1": 9.5, "line2": 14.5, "prediction1": 5.41, "prediction2": 10.39, "side1": "under", "side2": "under", "recommendation": 1, "ev": 94.82, "kelly": 0.474, "sigma1": "Low", "sigma2": "Low", "prob1": 0.823, "prob2": 0.805, "hitRate1": 83.4, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 69.0, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "Jalen Wilson", "name2": "Precious Achiuwa", "line1": 7.5, "line2": 7.5, "prediction1": 3.96, "prediction2": 3.92, "side1": "under", "side2": "under", "recommendation": 0, "ev": 87.25, "kelly": 0.436, "sigma1": "Low", "sigma2": "Low", "prob1": 0.797, "prob2": 0.799, "hitRate1": 94.3, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 61.7, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Cam Whitmore", "name2": "Austin Reaves", "line1": 11.5, "line2": 21.5, "prediction1": 7.42, "prediction2": 26.28, "side1": "under", "side2": "over", "recommendation": 1, "ev": 85.46, "kelly": 0.427, "sigma1": "Low", "sigma2": "Med", "prob1": 0.795, "prob2": 0.794, "hitRate1": 63.3, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 77.9, "l5_2": 1.0, "l15_2": 0.8},
    {"name1": "LaMelo Ball", "name2": "Bennedict Mathurin", "line1": 18.5, "line2": 23.5, "prediction1": 22.77, "prediction2": 27.81, "side1": "over", "side2": "over", "recommendation": 1, "ev": 73.84, "kelly": 0.369, "sigma1": "Med", "sigma2": "Med", "prob1": 0.764, "prob2": 0.774, "hitRate1": 48.0, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 53.2, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "T.J. McConnell", "name2": "LeBron James", "line1": 12.5, "line2": 20.5, "prediction1": 8.8, "prediction2": 16.88, "side1": "under", "side2": "under", "recommendation": 0, "ev": 73.09, "kelly": 0.365, "sigma1": "Low", "sigma2": "Med", "prob1": 0.772, "prob2": 0.763, "hitRate1": 86.2, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 19.4, "l5_2": 0.2, "l15_2": 0.07},
    {"name1": "Bruce Brown", "name2": "Isaiah Collier", "line1": 8.5, "line2": 8.5, "prediction1": 5.36, "prediction2": 5.58, "side1": "under", "side2": "under", "recommendation": 0, "ev": 70.64, "kelly": 0.353, "sigma1": "Low", "sigma2": "Low", "prob1": 0.762, "prob2": 0.762, "hitRate1": 81.6, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 44.3, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Jalen Johnson", "name2": "Franz Wagner", "line1": 20.5, "line2": 23.5, "prediction1": 24.37, "prediction2": 27.43, "side1": "over", "side2": "over", "recommendation": 0, "ev": 68.88, "kelly": 0.344, "sigma1": "Med", "sigma2": "Med", "prob1": 0.761, "prob2": 0.754, "hitRate1": 75.0, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 53.3, "l5_2": 0.2, "l15_2": 0.4},
    {"name1": "Jared McCain", "name2": "Cedric Coward", "line1": 12.5, "line2": 12.5, "prediction1": 9.16, "prediction2": 16.13, "side1": "under", "side2": "over", "recommendation": 0, "ev": 66.36, "kelly": 0.332, "sigma1": "Low", "sigma2": "Med", "prob1": 0.751, "prob2": 0.753, "hitRate1": 99.1, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 48.0, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Dillon Brooks", "name2": "Kyle Filipowski", "line1": 17.0, "line2": 9.5, "prediction1": 20.5, "prediction2": 6.37, "side1": "over", "side2": "under", "recommendation": 0, "ev": 61.57, "kelly": 0.308, "sigma1": "Med", "sigma2": "Low", "prob1": 0.736, "prob2": 0.747, "hitRate1": 92.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 47.1, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Coby White", "name2": "Walter Clayton Jr.", "line1": 23.5, "line2": 5.5, "prediction1": 27.03, "prediction2": 2.91, "side1": "over", "side2": "under", "recommendation": 0, "ev": 57.15, "kelly": 0.286, "sigma1": "Med", "sigma2": "Low", "prob1": 0.729, "prob2": 0.733, "hitRate1": 57.3, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 65.1, "l5_2": 0.4, "l15_2": 0.47},
];const prizepicksTriosData = [
    {"name1": "Drake Powell", "name2": "Cameron Johnson", "name3": "Precious Achiuwa", "line1": 9.5, "line2": 14.5, "line3": 7.5, "prediction1": 5.41, "prediction2": 10.39, "prediction3": 3.92, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 185.86, "kelly": 0.372, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.823, "prob2": 0.805, "prob3": 0.799, "hitRate1": 83.4, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 69.0, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 61.7, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Jalen Wilson", "name2": "Cam Whitmore", "name3": "Austin Reaves", "line1": 7.5, "line2": 11.5, "line3": 21.5, "prediction1": 3.96, "prediction2": 7.42, "prediction3": 26.28, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 171.59, "kelly": 0.343, "sigma1": "Low", "sigma2": "Low", "sigma3": "Med", "prob1": 0.797, "prob2": 0.795, "prob3": 0.794, "hitRate1": 94.3, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 63.3, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 77.9, "l5_3": 1.0, "l15_3": 0.8},
    {"name1": "LaMelo Ball", "name2": "Bennedict Mathurin", "name3": "LeBron James", "line1": 18.5, "line2": 23.5, "line3": 20.5, "prediction1": 22.77, "prediction2": 27.81, "prediction3": 16.88, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 143.58, "kelly": 0.287, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "prob1": 0.764, "prob2": 0.774, "prob3": 0.763, "hitRate1": 48.0, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 53.2, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 19.4, "l5_3": 0.2, "l15_3": 0.07},
    {"name1": "T.J. McConnell", "name2": "Bruce Brown", "name3": "Isaiah Collier", "line1": 12.5, "line2": 8.5, "line3": 8.5, "prediction1": 8.8, "prediction2": 5.36, "prediction3": 5.58, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 141.88, "kelly": 0.284, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.772, "prob2": 0.762, "prob3": 0.762, "hitRate1": 86.2, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 81.6, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 44.3, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Jalen Johnson", "name2": "Franz Wagner", "name3": "Cedric Coward", "line1": 20.5, "line2": 23.5, "line3": 12.5, "prediction1": 24.37, "prediction2": 27.43, "prediction3": 16.13, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 133.6, "kelly": 0.267, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "prob1": 0.761, "prob2": 0.754, "prob3": 0.753, "hitRate1": 75.0, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 53.3, "l5_2": 0.2, "l15_2": 0.4, "hitRate3": 48.0, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Jared McCain", "name2": "Dillon Brooks", "name3": "Kyle Filipowski", "line1": 12.5, "line2": 17.0, "line3": 9.5, "prediction1": 9.16, "prediction2": 20.5, "prediction3": 6.37, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 122.97, "kelly": 0.246, "sigma1": "Low", "sigma2": "Med", "sigma3": "Low", "prob1": 0.751, "prob2": 0.736, "prob3": 0.747, "hitRate1": 99.1, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 92.3, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 47.1, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Coby White", "name2": "Karl-Anthony Towns", "name3": "Walter Clayton Jr.", "line1": 23.5, "line2": 23.5, "line3": 5.5, "prediction1": 27.03, "prediction2": 26.97, "prediction3": 2.91, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 108.41, "kelly": 0.217, "sigma1": "Med", "sigma2": "Med", "sigma3": "Low", "prob1": 0.729, "prob2": 0.722, "prob3": 0.733, "hitRate1": 57.3, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 59.1, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 65.1, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Nikola Vu\u010devi\u0107", "name2": "Tristan da Silva", "name3": "Jonas Valan\u010di\u016bnas", "line1": 18.5, "line2": 11.0, "line3": 7.5, "prediction1": 15.48, "prediction2": 13.83, "prediction3": 5.1, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 95.25, "kelly": 0.191, "sigma1": "Med", "sigma2": "Med", "sigma3": "Low", "prob1": 0.717, "prob2": 0.707, "prob3": 0.713, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 68.9, "l5_2": 0.4, "l15_2": 0.53, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Luke Kennard", "name2": "Ryan Kalkbrenner", "name3": "Jay Huff", "line1": 6.5, "line2": 7.5, "line3": 10.5, "prediction1": 4.15, "prediction2": 10.04, "prediction3": 8.05, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 92.17, "kelly": 0.184, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.706, "prob2": 0.715, "prob3": 0.704, "hitRate1": 31.2, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 72.1, "l5_2": 0.6, "l15_2": 0.67, "hitRate3": 86.4, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Lonzo Ball", "name2": "Brandon Miller", "name3": "Khris Middleton", "line1": 5.5, "line2": 18.5, "line3": 9.5, "prediction1": 3.11, "prediction2": 15.36, "prediction3": 7.07, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 88.15, "kelly": 0.176, "sigma1": "Low", "sigma2": "Med", "sigma3": "Low", "prob1": 0.698, "prob2": 0.715, "prob3": 0.699, "hitRate1": 21.5, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 63.7, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 64.8, "l5_3": 0.6, "l15_3": 0.4},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Pascal Siakam", "name2": "Devin Booker", "line1": 25.5, "line2": 24.5, "prediction1": 31.71, "prediction2": 30.52, "side1": "over", "side2": "over", "recommendation": 1, "ev": 115.75, "kelly": 0.579, "sigma1": "High", "sigma2": "Med", "prob1": 0.845, "prob2": 0.868, "hitRate1": 41.7, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 46.2, "l5_2": 0.2, "l15_2": 0.53},
    {"name1": "Cade Cunningham", "name2": "Lauri Markkanen", "line1": 28.5, "line2": 26.5, "prediction1": 34.2, "prediction2": 31.69, "side1": "over", "side2": "over", "recommendation": 1, "ev": 100.93, "kelly": 0.505, "sigma1": "High", "sigma2": "Med", "prob1": 0.823, "prob2": 0.831, "hitRate1": 70.9, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 79.9, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "LeBron James", "name2": "James Harden", "line1": 21.5, "line2": 26.5, "prediction1": 16.88, "prediction2": 32.04, "side1": "under", "side2": "over", "recommendation": 1, "ev": 95.2, "kelly": 0.476, "sigma1": "Med", "sigma2": "High", "prob1": 0.819, "prob2": 0.81, "hitRate1": 25.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 87.0, "l5_2": 0.8, "l15_2": 0.53},
    {"name1": "Cameron Johnson", "name2": "Precious Achiuwa", "line1": 14.5, "line2": 7.5, "prediction1": 10.39, "prediction2": 3.92, "side1": "under", "side2": "under", "recommendation": 0, "ev": 89.02, "kelly": 0.445, "sigma1": "Low", "sigma2": "Low", "prob1": 0.805, "prob2": 0.799, "hitRate1": 69.0, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 61.7, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Bennedict Mathurin", "name2": "De'Aaron Fox", "line1": 23.5, "line2": 24.5, "prediction1": 27.81, "prediction2": 28.91, "side1": "over", "side2": "over", "recommendation": 1, "ev": 75.73, "kelly": 0.379, "sigma1": "Med", "sigma2": "Med", "prob1": 0.774, "prob2": 0.772, "hitRate1": 53.2, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 42.3, "l5_2": 0.8, "l15_2": 0.27},
    {"name1": "LaMelo Ball", "name2": "T.J. McConnell", "line1": 18.5, "line2": 12.5, "prediction1": 22.77, "prediction2": 8.8, "side1": "over", "side2": "under", "recommendation": 0, "ev": 73.25, "kelly": 0.366, "sigma1": "Med", "sigma2": "Low", "prob1": 0.764, "prob2": 0.772, "hitRate1": 48.0, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 86.2, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Day'Ron Sharpe", "name2": "Isaiah Collier", "line1": 7.5, "line2": 8.5, "prediction1": 4.23, "prediction2": 5.58, "side1": "under", "side2": "under", "recommendation": 0, "ev": 70.89, "kelly": 0.354, "sigma1": "Low", "sigma2": "Low", "prob1": 0.763, "prob2": 0.762, "hitRate1": 64.3, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 44.3, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Jalen Johnson", "name2": "Bruce Brown", "line1": 20.5, "line2": 8.5, "prediction1": 24.37, "prediction2": 5.36, "side1": "over", "side2": "under", "recommendation": 0, "ev": 70.47, "kelly": 0.352, "sigma1": "Med", "sigma2": "Low", "prob1": 0.761, "prob2": 0.762, "hitRate1": 75.0, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 81.6, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Franz Wagner", "name2": "Cedric Coward", "line1": 23.5, "line2": 12.5, "prediction1": 27.43, "prediction2": 16.13, "side1": "over", "side2": "over", "recommendation": 0, "ev": 67.04, "kelly": 0.335, "sigma1": "Med", "sigma2": "Med", "prob1": 0.754, "prob2": 0.753, "hitRate1": 53.3, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 48.0, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Donovan Mitchell", "name2": "Jared McCain", "line1": 29.5, "line2": 12.5, "prediction1": 33.48, "prediction2": 9.16, "side1": "over", "side2": "under", "recommendation": 0, "ev": 65.82, "kelly": 0.329, "sigma1": "Med", "sigma2": "Low", "prob1": 0.751, "prob2": 0.751, "hitRate1": 61.0, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 99.1, "l5_2": 0.4, "l15_2": 0.13},
];const underdogTriosData = [
    {"name1": "Cameron Johnson", "name2": "Precious Achiuwa", "name3": "LeBron James", "line1": 14.5, "line2": 7.5, "line3": 21.5, "prediction1": 10.39, "prediction2": 3.92, "prediction3": 16.88, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 184.49, "kelly": 0.369, "sigma1": "Low", "sigma2": "Low", "sigma3": "Med", "prob1": 0.805, "prob2": 0.799, "prob3": 0.819, "hitRate1": 69.0, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 61.7, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 25.8, "l5_3": 0.2, "l15_3": 0.07},
    {"name1": "Day'Ron Sharpe", "name2": "LaMelo Ball", "name3": "Bennedict Mathurin", "line1": 7.5, "line2": 18.5, "line3": 23.5, "prediction1": 4.23, "prediction2": 22.77, "prediction3": 27.81, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 143.51, "kelly": 0.287, "sigma1": "Low", "sigma2": "Med", "sigma3": "Med", "prob1": 0.763, "prob2": 0.764, "prob3": 0.774, "hitRate1": 64.3, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 48.0, "l5_2": 0.2, "l15_2": 0.33, "hitRate3": 53.2, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "T.J. McConnell", "name2": "Bruce Brown", "name3": "Isaiah Collier", "line1": 12.5, "line2": 8.5, "line3": 8.5, "prediction1": 8.8, "prediction2": 5.36, "prediction3": 5.58, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 141.88, "kelly": 0.284, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.772, "prob2": 0.762, "prob3": 0.762, "hitRate1": 86.2, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 81.6, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 44.3, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Jalen Johnson", "name2": "Franz Wagner", "name3": "Cedric Coward", "line1": 20.5, "line2": 23.5, "line3": 12.5, "prediction1": 24.37, "prediction2": 27.43, "prediction3": 16.13, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 133.6, "kelly": 0.267, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "prob1": 0.761, "prob2": 0.754, "prob3": 0.753, "hitRate1": 75.0, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 53.3, "l5_2": 0.2, "l15_2": 0.4, "hitRate3": 48.0, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Jared McCain", "name2": "Kyle Filipowski", "name3": "Austin Reaves", "line1": 12.5, "line2": 9.5, "line3": 22.5, "prediction1": 9.16, "prediction2": 6.37, "prediction3": 26.28, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 124.71, "kelly": 0.249, "sigma1": "Low", "sigma2": "Low", "sigma3": "Med", "prob1": 0.751, "prob2": 0.747, "prob3": 0.742, "hitRate1": 99.1, "l5_1": 0.4, "l15_1": 0.13, "hitRate2": 47.1, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 71.3, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Luke Kennard", "name2": "Coby White", "name3": "Jonas Valan\u010di\u016bnas", "line1": 6.5, "line2": 23.5, "line3": 7.5, "prediction1": 4.15, "prediction2": 27.03, "prediction3": 5.1, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 98.38, "kelly": 0.197, "sigma1": "Low", "sigma2": "Med", "sigma3": "Low", "prob1": 0.706, "prob2": 0.729, "prob3": 0.713, "hitRate1": 31.2, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 57.3, "l5_2": 0.6, "l15_2": 0.2, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Nikola Vu\u010devi\u0107", "name2": "Jay Huff", "name3": "Dillon Brooks", "line1": 18.5, "line2": 9.5, "line3": 17.5, "prediction1": 15.48, "prediction2": 7.05, "prediction3": 20.5, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 92.52, "kelly": 0.185, "sigma1": "Med", "sigma2": "Low", "sigma3": "Med", "prob1": 0.717, "prob2": 0.704, "prob3": 0.706, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 77.8, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 92.3, "l5_3": 0.8, "l15_3": 0.6},
    {"name1": "Ryan Kalkbrenner", "name2": "Khris Middleton", "name3": "Kawhi Leonard", "line1": 7.5, "line2": 9.5, "line3": 20.5, "prediction1": 10.04, "prediction2": 7.07, "prediction3": 23.5, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 88.37, "kelly": 0.177, "sigma1": "Low", "sigma2": "Low", "sigma3": "Med", "prob1": 0.715, "prob2": 0.699, "prob3": 0.698, "hitRate1": 72.1, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 70.3, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 82.8, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Alex Caruso", "name2": "Keegan Murray", "name3": "Kentavious Caldwell-Pope", "line1": 5.5, "line2": 17.5, "line3": 5.5, "prediction1": 3.27, "prediction2": 15.02, "prediction3": 3.29, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 76.34, "kelly": 0.153, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.685, "prob2": 0.69, "prob3": 0.691, "hitRate1": 45.7, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 69.8, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 63.8, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Josh Hart", "name2": "Julian Champagnie", "name3": "Ace Bailey", "line1": 12.5, "line2": 11.5, "line3": 12.5, "prediction1": 10.1, "prediction2": 13.99, "prediction3": 10.24, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 71.26, "kelly": 0.143, "sigma1": "Med", "sigma2": "Med", "sigma3": "Low", "prob1": 0.682, "prob2": 0.684, "prob3": 0.68, "hitRate1": 68.3, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 39.6, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 41.3, "l5_3": 0.6, "l15_3": 0.33},
];const prizepicksPointsHitRates = [
    {"name": "Jaden Ivey", "line": 9.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.949, "underPct": 0.051},
    {"name": "Onyeka Okongwu", "line": 13.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.944, "underPct": 0.056},
    {"name": "Dillon Brooks", "line": 17.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.923, "underPct": 0.077},
    {"name": "Jett Howard", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.903, "underPct": 0.097},
    {"name": "Duncan Robinson", "line": 10.0, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.89, "underPct": 0.11},
    {"name": "Kon Knueppel", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.878, "underPct": 0.122},
    {"name": "Jaylon Tyson", "line": 8.0, "l5": 0.8, "l10": 0.9, "l15": 0.6, "overPct": 0.859, "underPct": 0.141},
    {"name": "Jalen Duren", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.849, "underPct": 0.151},
    {"name": "Naji Marshall", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.831, "underPct": 0.169},
    {"name": "Caris LeVert", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.828, "underPct": 0.172},
    {"name": "Marcus Smart", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.823, "underPct": 0.177},
    {"name": "James Harden", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.817, "underPct": 0.183},
    {"name": "Dylan Harper", "line": 10.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.81, "underPct": 0.19},
    {"name": "LeBron James", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.806, "underPct": 0.194},
    {"name": "Luguentz Dort", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.806, "underPct": 0.194},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.799, "underPct": 0.201},
    {"name": "Isaiah Joe", "line": 9.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.795, "underPct": 0.205},
    {"name": "Collin Sexton", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.788, "underPct": 0.212},
    {"name": "Lonzo Ball", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.785, "underPct": 0.215},
    {"name": "Austin Reaves", "line": 21.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.779, "underPct": 0.221},
    {"name": "Oso Ighodaro", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.77, "underPct": 0.23},
    {"name": "Kawhi Leonard", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.768, "underPct": 0.232},
    {"name": "Rui Hachimura", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.765, "underPct": 0.235},
    {"name": "Miles Bridges", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.764, "underPct": 0.236},
    {"name": "Santi Aldama", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.757, "underPct": 0.243},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.75, "underPct": 0.25},
    {"name": "Ajay Mitchell", "line": 11.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.75, "underPct": 0.25},
    {"name": "Jalen Johnson", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.75, "underPct": 0.25},
    {"name": "Nickeil Alexander-Walker", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.739, "underPct": 0.261},
    {"name": "Ayo Dosunmu", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.738, "underPct": 0.262},
    {"name": "Tobias Harris", "line": 12.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.728, "underPct": 0.272},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.721, "underPct": 0.279},
    {"name": "Ryan Rollins", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.712, "underPct": 0.288},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.711, "underPct": 0.289},
    {"name": "Cade Cunningham", "line": 28.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.709, "underPct": 0.291},
    {"name": "Chet Holmgren", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.698, "underPct": 0.302},
    {"name": "Andrew Nembhard", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.694, "underPct": 0.306},
    {"name": "Tristan da Silva", "line": 11.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.689, "underPct": 0.311},
    {"name": "Luke Kennard", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.688, "underPct": 0.312},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.688, "underPct": 0.312},
    {"name": "Keyonte George", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.687, "underPct": 0.313},
    {"name": "Anthony Black", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.673, "underPct": 0.327},
    {"name": "Max Christie", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.67, "underPct": 0.33},
    {"name": "Dyson Daniels", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.65, "underPct": 0.35},
    {"name": "Cooper Flagg", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.637, "underPct": 0.363},
    {"name": "Isaac Okoro", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.628, "underPct": 0.372},
    {"name": "De'Andre Hunter", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.619, "underPct": 0.381},
    {"name": "Donovan Mitchell", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.61, "underPct": 0.39},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.607, "underPct": 0.393},
    {"name": "Kevin Huerter", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.603, "underPct": 0.397},
    {"name": "Svi Mykhailiuk", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.602, "underPct": 0.398},
    {"name": "Mikal Bridges", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.598, "underPct": 0.402},
    {"name": "Karl-Anthony Towns", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.591, "underPct": 0.409},
    {"name": "Goga Bitadze", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.587, "underPct": 0.413},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.587, "underPct": 0.413},
    {"name": "Collin Gillespie", "line": 12.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.586, "underPct": 0.414},
    {"name": "Luke Kornet", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.582, "underPct": 0.418},
    {"name": "Coby White", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.573, "underPct": 0.427},
    {"name": "Mark Williams", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.564, "underPct": 0.436},
    {"name": "Cason Wallace", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.562, "underPct": 0.438},
    {"name": "Harrison Barnes", "line": 13.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.558, "underPct": 0.442},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.557, "underPct": 0.443},
    {"name": "Alex Caruso", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.543, "underPct": 0.457},
    {"name": "Anthony Davis", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.54, "underPct": 0.46},
    {"name": "Brice Sensabaugh", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.538, "underPct": 0.462},
    {"name": "Cam Spencer", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.536, "underPct": 0.464},
    {"name": "Royce O'Neale", "line": 10.0, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.535, "underPct": 0.465},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Bennedict Mathurin", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.532, "underPct": 0.468},
    {"name": "Kris Dunn", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.531, "underPct": 0.469},
    {"name": "Ausar Thompson", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.531, "underPct": 0.469},
    {"name": "Kyle Filipowski", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.529, "underPct": 0.471},
    {"name": "Jake LaRavia", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.529, "underPct": 0.471},
    {"name": "Noah Clowney", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.518, "underPct": 0.482},
    {"name": "Jalen Suggs", "line": 15.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.518, "underPct": 0.482},
    {"name": "Peyton Watson", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.515, "underPct": 0.485},
    {"name": "Daniel Gafford", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.512, "underPct": 0.488},
    {"name": "Desmond Bane", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.509, "underPct": 0.491},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.498, "underPct": 0.502},
    {"name": "Kelly Olynyk", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.498, "underPct": 0.502},
    {"name": "Pascal Siakam", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.497, "underPct": 0.503},
    {"name": "Jalen Brunson", "line": 29.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.491, "underPct": 0.509},
    {"name": "Miles McBride", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.488, "underPct": 0.512},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.486, "underPct": 0.514},
    {"name": "Isaiah Hartenstein", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.486, "underPct": 0.514},
    {"name": "Deandre Ayton", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.483, "underPct": 0.517},
    {"name": "Cedric Coward", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.48, "underPct": 0.52},
    {"name": "LaMelo Ball", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.48, "underPct": 0.52},
    {"name": "Alex Sarr", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.466, "underPct": 0.534},
    {"name": "Devin Booker", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.462, "underPct": 0.538},
    {"name": "Jarrett Allen", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.458, "underPct": 0.542},
    {"name": "Jordan Clarkson", "line": 12.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.458, "underPct": 0.542},
    {"name": "Jamal Murray", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.447, "underPct": 0.553},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Matas Buzelis", "line": 13.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Giannis Antetokounmpo", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.431, "underPct": 0.569},
    {"name": "Brook Lopez", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.429, "underPct": 0.571},
    {"name": "John Collins", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.423, "underPct": 0.577},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.396, "underPct": 0.604},
    {"name": "Keldon Johnson", "line": 13.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.392, "underPct": 0.608},
    {"name": "Zaccharie Risacher", "line": 11.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.391, "underPct": 0.609},
    {"name": "Brandon Williams", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.388, "underPct": 0.612},
    {"name": "Bilal Coulibaly", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.384, "underPct": 0.616},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.383, "underPct": 0.617},
    {"name": "Cam Whitmore", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.367, "underPct": 0.633},
    {"name": "Kyshawn George", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.365, "underPct": 0.635},
    {"name": "Brandon Miller", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.363, "underPct": 0.637},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.363, "underPct": 0.637},
    {"name": "Khris Middleton", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.352, "underPct": 0.648},
    {"name": "Walter Clayton Jr.", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.349, "underPct": 0.651},
    {"name": "Myles Turner", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.342, "underPct": 0.658},
    {"name": "Tyrese Maxey", "line": 32.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.335, "underPct": 0.665},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.335, "underPct": 0.665},
    {"name": "Justin Edwards", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.327, "underPct": 0.673},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.324, "underPct": 0.676},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.311, "underPct": 0.689},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.31, "underPct": 0.69},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.308, "underPct": 0.692},
    {"name": "Keegan Murray", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.302, "underPct": 0.698},
    {"name": "Devin Vassell", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.294, "underPct": 0.706},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.278, "underPct": 0.722},
    {"name": "Jarace Walker", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.273, "underPct": 0.727},
    {"name": "Darius Garland", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.272, "underPct": 0.728},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.264, "underPct": 0.736},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.245, "underPct": 0.755},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.235, "underPct": 0.765},
    {"name": "Russell Westbrook", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.216, "underPct": 0.784},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.215, "underPct": 0.785},
    {"name": "Paul George", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.212, "underPct": 0.788},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.204, "underPct": 0.796},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.184, "underPct": 0.816},
    {"name": "Drake Powell", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.166, "underPct": 0.834},
    {"name": "T.J. McConnell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.138, "underPct": 0.862},
    {"name": "Jay Huff", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.136, "underPct": 0.864},
    {"name": "Drew Eubanks", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.11, "underPct": 0.89},
    {"name": "Jalen Wilson", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.057, "underPct": 0.943},
    {"name": "Jared McCain", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.009, "underPct": 0.991},
];const prizepicksAssistsHitRates = [
    {"name": "Caris LeVert", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.767, "underPct": 0.233},
    {"name": "LeBron James", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.7, "underPct": 0.3},
    {"name": "Jalen Johnson", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.677, "underPct": 0.323},
    {"name": "LaMelo Ball", "line": 7.5, "l5": 0.8, "l10": 0.9, "l15": 0.6, "overPct": 0.672, "underPct": 0.328},
    {"name": "Cade Cunningham", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.672, "underPct": 0.328},
    {"name": "Jamal Murray", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.671, "underPct": 0.329},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.638, "underPct": 0.362},
    {"name": "Kyshawn George", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.635, "underPct": 0.365},
    {"name": "Jaylon Tyson", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.607, "underPct": 0.393},
    {"name": "Dyson Daniels", "line": 5.0, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.603, "underPct": 0.397},
    {"name": "Jalen Brunson", "line": 6.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.593, "underPct": 0.407},
    {"name": "Shai Gilgeous-Alexander", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.585, "underPct": 0.415},
    {"name": "Coby White", "line": 5.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.57, "underPct": 0.43},
    {"name": "Terance Mann", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.555, "underPct": 0.445},
    {"name": "Peyton Watson", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.542, "underPct": 0.458},
    {"name": "Josh Hart", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.539, "underPct": 0.461},
    {"name": "Kon Knueppel", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.534, "underPct": 0.466},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.534, "underPct": 0.466},
    {"name": "Brandon Williams", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Giannis Antetokounmpo", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.529, "underPct": 0.471},
    {"name": "Isaiah Collier", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.511, "underPct": 0.489},
    {"name": "De'Aaron Fox", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.505, "underPct": 0.495},
    {"name": "Devin Booker", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Evan Mobley", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.498, "underPct": 0.502},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.497, "underPct": 0.503},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.489, "underPct": 0.511},
    {"name": "Desmond Bane", "line": 4.0, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.474, "underPct": 0.526},
    {"name": "Deandre Ayton", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.465, "underPct": 0.535},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.459, "underPct": 0.541},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.435, "underPct": 0.565},
    {"name": "Andre Drummond", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.433, "underPct": 0.567},
    {"name": "Paul George", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.418, "underPct": 0.582},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.417, "underPct": 0.583},
    {"name": "Franz Wagner", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.405, "underPct": 0.595},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.392, "underPct": 0.608},
    {"name": "Darius Garland", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.391, "underPct": 0.609},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.377, "underPct": 0.623},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.373, "underPct": 0.627},
    {"name": "Anthony Davis", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.367, "underPct": 0.633},
    {"name": "DeMar DeRozan", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.347, "underPct": 0.653},
    {"name": "Tyrese Maxey", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.338, "underPct": 0.662},
    {"name": "Pascal Siakam", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.289, "underPct": 0.711},
    {"name": "Andrew Nembhard", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.288, "underPct": 0.712},
    {"name": "Jared McCain", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.08, "underPct": 0.92},
];const prizepicksReboundsHitRates = [
    {"name": "LeBron James", "line": 6.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.794, "underPct": 0.206},
    {"name": "Austin Reaves", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.756, "underPct": 0.244},
    {"name": "Lonzo Ball", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.753, "underPct": 0.247},
    {"name": "Mitchell Robinson", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.751, "underPct": 0.249},
    {"name": "Jock Landale", "line": 4.0, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.691, "underPct": 0.309},
    {"name": "Ivica Zubac", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.677, "underPct": 0.323},
    {"name": "Bennedict Mathurin", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.667, "underPct": 0.333},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.666, "underPct": 0.334},
    {"name": "Santi Aldama", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.66, "underPct": 0.34},
    {"name": "Naji Marshall", "line": 4.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.655, "underPct": 0.345},
    {"name": "Jalen Johnson", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.647, "underPct": 0.353},
    {"name": "Duncan Robinson", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.646, "underPct": 0.354},
    {"name": "Ryan Rollins", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.64, "underPct": 0.36},
    {"name": "Franz Wagner", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.63, "underPct": 0.37},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.608, "underPct": 0.392},
    {"name": "Donovan Mitchell", "line": 4.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.606, "underPct": 0.394},
    {"name": "P.J. Washington", "line": 6.0, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.589, "underPct": 0.411},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.576, "underPct": 0.424},
    {"name": "Keldon Johnson", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.568, "underPct": 0.432},
    {"name": "Anthony Davis", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.555, "underPct": 0.445},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.546, "underPct": 0.454},
    {"name": "James Harden", "line": 5.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.539, "underPct": 0.461},
    {"name": "Matas Buzelis", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.538, "underPct": 0.462},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.537, "underPct": 0.463},
    {"name": "LaMelo Ball", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.532, "underPct": 0.468},
    {"name": "Daniel Gafford", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.531, "underPct": 0.469},
    {"name": "Kentavious Caldwell-Pope", "line": 1.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.521, "underPct": 0.479},
    {"name": "Josh Giddey", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.52, "underPct": 0.48},
    {"name": "Kyle Filipowski", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.518, "underPct": 0.482},
    {"name": "Tobias Harris", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.508, "underPct": 0.492},
    {"name": "Myles Turner", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.497, "underPct": 0.503},
    {"name": "De'Andre Hunter", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.496, "underPct": 0.504},
    {"name": "Desmond Bane", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.484, "underPct": 0.516},
    {"name": "Keyonte George", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.48, "underPct": 0.52},
    {"name": "Kawhi Leonard", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.475, "underPct": 0.525},
    {"name": "Svi Mykhailiuk", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.473, "underPct": 0.527},
    {"name": "Chet Holmgren", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.469, "underPct": 0.531},
    {"name": "Goga Bitadze", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.463, "underPct": 0.537},
    {"name": "Kyle Kuzma", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.462, "underPct": 0.538},
    {"name": "Cade Cunningham", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.46, "underPct": 0.54},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
    {"name": "Miles Bridges", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.444, "underPct": 0.556},
    {"name": "Dillon Brooks", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.437, "underPct": 0.563},
    {"name": "Lauri Markkanen", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.429, "underPct": 0.571},
    {"name": "Keegan Murray", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.418, "underPct": 0.582},
    {"name": "Evan Mobley", "line": 9.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.417, "underPct": 0.583},
    {"name": "Isaiah Jackson", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
    {"name": "Cooper Flagg", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.407, "underPct": 0.593},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.405, "underPct": 0.595},
    {"name": "Bruce Brown", "line": 3.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.403, "underPct": 0.597},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.395, "underPct": 0.605},
    {"name": "Devin Booker", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.393, "underPct": 0.607},
    {"name": "Peyton Watson", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.381, "underPct": 0.619},
    {"name": "Cameron Johnson", "line": 4.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.348, "underPct": 0.652},
    {"name": "Kyshawn George", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.53, "overPct": 0.344, "underPct": 0.656},
    {"name": "Khris Middleton", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.34, "underPct": 0.66},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.34, "underPct": 0.66},
    {"name": "John Collins", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.332, "underPct": 0.668},
    {"name": "Josh Hart", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.33, "underPct": 0.67},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.33, "underPct": 0.67},
    {"name": "Pascal Siakam", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.312, "underPct": 0.688},
    {"name": "Ausar Thompson", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.308, "underPct": 0.692},
    {"name": "Andrew Nembhard", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.306, "underPct": 0.694},
    {"name": "Justin Edwards", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.289, "underPct": 0.711},
    {"name": "Bilal Coulibaly", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.287, "underPct": 0.713},
    {"name": "Jarrett Allen", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.275, "underPct": 0.725},
    {"name": "Jarace Walker", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.275, "underPct": 0.725},
    {"name": "Harrison Barnes", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.273, "underPct": 0.727},
    {"name": "Spencer Jones", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.269, "underPct": 0.731},
    {"name": "Onyeka Okongwu", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.267, "underPct": 0.733},
    {"name": "Luke Kornet", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.261, "underPct": 0.739},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.249, "underPct": 0.751},
    {"name": "Giannis Antetokounmpo", "line": 10.0, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.223, "underPct": 0.777},
    {"name": "Ace Bailey", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.199, "underPct": 0.801},
    {"name": "Mark Williams", "line": 9.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.185, "underPct": 0.815},
    {"name": "Brandon Miller", "line": 4.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.179, "underPct": 0.821},
    {"name": "Zach Edey", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.172, "underPct": 0.828},
    {"name": "Drew Eubanks", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.126, "underPct": 0.874},
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
    {"name": "Onyeka Okongwu", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.597, "underPct": 0.403},
    {"name": "Zaccharie Risacher", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.683, "underPct": 0.317},
    {"name": "Luke Kennard", "line": 0.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.678, "underPct": 0.322},
    {"name": "Jared McCain", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.222, "underPct": 0.778},
    {"name": "Collin Sexton", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.473, "underPct": 0.527},
    {"name": "Ausar Thompson", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.525, "underPct": 0.475},
    {"name": "Jalen Suggs", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.553, "underPct": 0.447},
    {"name": "Caris LeVert", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.486, "underPct": 0.514},
    {"name": "Mikal Bridges", "line": 1.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.706, "underPct": 0.294},
    {"name": "Kyle Kuzma", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.574, "underPct": 0.426},
    {"name": "Mitchell Robinson", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.356, "underPct": 0.644},
    {"name": "Bruce Brown", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.591, "underPct": 0.409},
    {"name": "Dylan Harper", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.476, "underPct": 0.524},
    {"name": "Keldon Johnson", "line": 0.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.545, "underPct": 0.455},
    {"name": "Brice Sensabaugh", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.446, "underPct": 0.554},
    {"name": "Svi Mykhailiuk", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.211, "underPct": 0.789},
    {"name": "Drew Eubanks", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.454, "underPct": 0.546},
    {"name": "Daniel Gafford", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.559, "underPct": 0.441},
    {"name": "Zach Edey", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.632, "underPct": 0.368},
    {"name": "Kentavious Caldwell-Pope", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.549, "underPct": 0.451},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Austin Reaves", "line": 31.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Davis", "line": 32.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Kalkbrenner", "line": 16.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Black", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 25.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Duncan Robinson", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alex Caruso", "line": 9.5, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Keyonte George", "line": 33.0, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Caris LeVert", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Filipowski", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dylan Harper", "line": 17.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 27.0, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 18.0, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylon Tyson", "line": 13.5, "l5": 0.8, "l10": 0.9, "l15": 0.6, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Santi Aldama", "line": 21.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 37.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Harrison Barnes", "line": 19.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 22.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Peyton Watson", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 35.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Mitchell", "line": 39.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 41.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deandre Ayton", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 40.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 16.0, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 16.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "P.J. Washington", "line": 24.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Joe", "line": 12.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cason Wallace", "line": 12.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 46.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 21.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 22.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 31.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alex Sarr", "line": 31.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ausar Thompson", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 29.0, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Goga Bitadze", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kennard", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lonzo Ball", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 39.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Giddey", "line": 39.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brice Sensabaugh", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kawhi Leonard", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 44.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Brunson", "line": 40.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Dunn", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles McBride", "line": 16.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julian Champagnie", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 28.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 21.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jarrett Allen", "line": 24.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jarace Walker", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.33, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bilal Coulibaly", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Evan Mobley", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 37.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden Ivey", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandon Miller", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paul George", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LeBron James", "line": 34.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Isaiah Collier", "line": 17.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drake Powell", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keegan Murray", "line": 25.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Franz Wagner", "line": 33.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 30.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Quentin Grimes", "line": 25.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Devin Booker", "line": 36.0, "l5": 0.0, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
];const prizepicksPRHitRates = [
    {"name": "Collin Gillespie", "line": 16.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 26.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Black", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Sexton", "line": 16.0, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dylan Harper", "line": 14.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 22.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ajay Mitchell", "line": 15.0, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jordan Goodwin", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Filipowski", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kelly Olynyk", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Davis", "line": 29.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Duncan Robinson", "line": 13.0, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Caris LeVert", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyshawn George", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 31.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dyson Daniels", "line": 17.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylon Tyson", "line": 11.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Rui Hachimura", "line": 14.0, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ryan Kalkbrenner", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 20.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 17.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 33.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 35.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Spencer", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Santi Aldama", "line": 18.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jake LaRavia", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 14.0, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Deandre Ayton", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Joe", "line": 11.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Khris Middleton", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mark Williams", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ayo Dosunmu", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 34.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tobias Harris", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 25.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Nickeil Alexander-Walker", "line": 21.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 28.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luguentz Dort", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Caruso", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Oso Ighodaro", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 33.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jock Landale", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brice Sensabaugh", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 26.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kennard", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jared McCain", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mitchell Robinson", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 19.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Royce O'Neale", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tristan da Silva", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 39.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Huerter", "line": 13.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "T.J. McConnell", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 23.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "John Collins", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden Ivey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ben Sheppard", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jarrett Allen", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Jackson", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 30.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Paul George", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jay Huff", "line": 15.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Miller", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Matas Buzelis", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drake Powell", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 23.0, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bruce Brown", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 13.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drew Eubanks", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Justin Edwards", "line": 14.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 29.0, "l5": 0.0, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
];const prizepicksPAHitRates = [
    {"name": "Collin Gillespie", "line": 17.5, "l5": 1.0, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 15.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ajay Mitchell", "line": 14.5, "l5": 0.8, "l10": 0.9, "l15": 0.87, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Austin Reaves", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 19.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 21.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Keyonte George", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Davis", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dyson Daniels", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Caris LeVert", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dylan Harper", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 9.5, "l5": 0.8, "l10": 0.9, "l15": 0.6, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Brandon Williams", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "De'Aaron Fox", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Johnson", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 36.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Giannis Antetokounmpo", "line": 36.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 34.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 37.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Hartenstein", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Joe", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alex Caruso", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Santi Aldama", "line": 15.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cam Spencer", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Khris Middleton", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nicolas Batum", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 36.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ayo Dosunmu", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kennard", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lonzo Ball", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles Bridges", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jared McCain", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Collier", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brice Sensabaugh", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ace Bailey", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keegan Murray", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyrese Maxey", "line": 39.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Martin", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Wells", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Darius Garland", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "DeMar DeRozan", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alex Sarr", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tobias Harris", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden Ivey", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Karl-Anthony Towns", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Miller", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "LaMelo Ball", "line": 26.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Royce O'Neale", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 24.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "T.J. McConnell", "line": 18.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "John Collins", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Justin Edwards", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Daniel Gafford", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paul George", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Franz Wagner", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Huerter", "line": 13.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "LeBron James", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 31.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drake Powell", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
];const prizepicksRAHitRates = [
    {"name": "P.J. Washington", "line": 8.5, "l5": 1.0, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cade Cunningham", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "LaMelo Ball", "line": 12.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandon Williams", "line": 6.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Austin Reaves", "line": 9.0, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Davis", "line": 12.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 8.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Gillespie", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Peyton Watson", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Josh Hart", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jock Landale", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Terance Mann", "line": 7.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lonzo Ball", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylon Tyson", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shai Gilgeous-Alexander", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Coby White", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Goodwin", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deandre Ayton", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rui Hachimura", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 14.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Justin Edwards", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Franz Wagner", "line": 10.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Desmond Bane", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Drake Powell", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 14.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naji Marshall", "line": 6.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Khris Middleton", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 15.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 6.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cason Wallace", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Dunn", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "James Harden", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kawhi Leonard", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jarrett Allen", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Giannis Antetokounmpo", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brice Sensabaugh", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 12.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "T.J. McConnell", "line": 9.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jarace Walker", "line": 8.0, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paul George", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ace Bailey", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jared McCain", "line": 4.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Drew Eubanks", "line": 5.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Zach LaVine", "line": 6.0, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 7.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
];const prizepicksTurnoversHitRates = [
    {"name": "Ryan Rollins", "line": 2.5, "l5": 1.0, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Suggs", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Onyeka Okongwu", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Collier", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jake LaRavia", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rui Hachimura", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ivica Zubac", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 1.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
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
    {"name": "Onyeka Okongwu", "line": 13.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.961, "underPct": 0.039},
    {"name": "Duncan Robinson", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.936, "underPct": 0.064},
    {"name": "Dillon Brooks", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.923, "underPct": 0.077},
    {"name": "Kon Knueppel", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.878, "underPct": 0.122},
    {"name": "Isaiah Joe", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.873, "underPct": 0.127},
    {"name": "James Harden", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.87, "underPct": 0.13},
    {"name": "Jordan Goodwin", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.864, "underPct": 0.136},
    {"name": "Naji Marshall", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.831, "underPct": 0.169},
    {"name": "Kawhi Leonard", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.828, "underPct": 0.172},
    {"name": "Marcus Smart", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.823, "underPct": 0.177},
    {"name": "Tobias Harris", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.812, "underPct": 0.188},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.799, "underPct": 0.201},
    {"name": "Collin Sexton", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.788, "underPct": 0.212},
    {"name": "Rui Hachimura", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.765, "underPct": 0.235},
    {"name": "Miles Bridges", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.764, "underPct": 0.236},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.75, "underPct": 0.25},
    {"name": "Jalen Johnson", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.75, "underPct": 0.25},
    {"name": "LeBron James", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.742, "underPct": 0.258},
    {"name": "Nickeil Alexander-Walker", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.739, "underPct": 0.261},
    {"name": "Ayo Dosunmu", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.738, "underPct": 0.262},
    {"name": "Anthony Black", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.732, "underPct": 0.268},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.721, "underPct": 0.279},
    {"name": "Austin Reaves", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.713, "underPct": 0.287},
    {"name": "Ryan Rollins", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.712, "underPct": 0.288},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.711, "underPct": 0.289},
    {"name": "Cade Cunningham", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.709, "underPct": 0.291},
    {"name": "Chet Holmgren", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.698, "underPct": 0.302},
    {"name": "Andrew Nembhard", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.694, "underPct": 0.306},
    {"name": "Luguentz Dort", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.693, "underPct": 0.307},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.689, "underPct": 0.311},
    {"name": "Luke Kennard", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.688, "underPct": 0.312},
    {"name": "Keyonte George", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.687, "underPct": 0.313},
    {"name": "Jake LaRavia", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.68, "underPct": 0.32},
    {"name": "Max Christie", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.67, "underPct": 0.33},
    {"name": "Brice Sensabaugh", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.664, "underPct": 0.336},
    {"name": "Royce O'Neale", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.655, "underPct": 0.345},
    {"name": "Ajay Mitchell", "line": 12.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.653, "underPct": 0.347},
    {"name": "Dyson Daniels", "line": 10.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.65, "underPct": 0.35},
    {"name": "Cooper Flagg", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.637, "underPct": 0.363},
    {"name": "Daniel Gafford", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.634, "underPct": 0.366},
    {"name": "De'Andre Hunter", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.619, "underPct": 0.381},
    {"name": "Donovan Mitchell", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.61, "underPct": 0.39},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.607, "underPct": 0.393},
    {"name": "Kevin Huerter", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.603, "underPct": 0.397},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.587, "underPct": 0.413},
    {"name": "Collin Gillespie", "line": 12.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.586, "underPct": 0.414},
    {"name": "Luke Kornet", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.582, "underPct": 0.418},
    {"name": "Coby White", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.573, "underPct": 0.427},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.571, "underPct": 0.429},
    {"name": "Deandre Ayton", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.57, "underPct": 0.43},
    {"name": "Mark Williams", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.564, "underPct": 0.436},
    {"name": "Cason Wallace", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.562, "underPct": 0.438},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.557, "underPct": 0.443},
    {"name": "Alex Caruso", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.543, "underPct": 0.457},
    {"name": "Anthony Davis", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.54, "underPct": 0.46},
    {"name": "Cam Spencer", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.536, "underPct": 0.464},
    {"name": "Franz Wagner", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Bennedict Mathurin", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.532, "underPct": 0.468},
    {"name": "Ausar Thompson", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.531, "underPct": 0.469},
    {"name": "Kyle Filipowski", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.529, "underPct": 0.471},
    {"name": "Noah Clowney", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.518, "underPct": 0.482},
    {"name": "Peyton Watson", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.515, "underPct": 0.485},
    {"name": "Desmond Bane", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.509, "underPct": 0.491},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.498, "underPct": 0.502},
    {"name": "Miles McBride", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.488, "underPct": 0.512},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.486, "underPct": 0.514},
    {"name": "LaMelo Ball", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.48, "underPct": 0.52},
    {"name": "Cedric Coward", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.48, "underPct": 0.52},
    {"name": "Alex Sarr", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.466, "underPct": 0.534},
    {"name": "Kris Dunn", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.463, "underPct": 0.537},
    {"name": "Devin Booker", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.462, "underPct": 0.538},
    {"name": "Jamal Murray", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.447, "underPct": 0.553},
    {"name": "Myles Turner", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Matas Buzelis", "line": 13.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Giannis Antetokounmpo", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.431, "underPct": 0.569},
    {"name": "Kyshawn George", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.424, "underPct": 0.576},
    {"name": "De'Aaron Fox", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.423, "underPct": 0.577},
    {"name": "Jalen Brunson", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.419, "underPct": 0.581},
    {"name": "Pascal Siakam", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.417, "underPct": 0.583},
    {"name": "Paul George", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.404, "underPct": 0.596},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.396, "underPct": 0.604},
    {"name": "Keldon Johnson", "line": 13.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.392, "underPct": 0.608},
    {"name": "Brandon Williams", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.388, "underPct": 0.612},
    {"name": "Bilal Coulibaly", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.384, "underPct": 0.616},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.383, "underPct": 0.617},
    {"name": "Terance Mann", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.378, "underPct": 0.622},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.363, "underPct": 0.637},
    {"name": "Kentavious Caldwell-Pope", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.362, "underPct": 0.638},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.357, "underPct": 0.643},
    {"name": "Tyrese Maxey", "line": 32.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.335, "underPct": 0.665},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.311, "underPct": 0.689},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.31, "underPct": 0.69},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.308, "underPct": 0.692},
    {"name": "Keegan Murray", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.302, "underPct": 0.698},
    {"name": "Justin Edwards", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.297, "underPct": 0.703},
    {"name": "Khris Middleton", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.297, "underPct": 0.703},
    {"name": "Devin Vassell", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.294, "underPct": 0.706},
    {"name": "Darius Garland", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.272, "underPct": 0.728},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.264, "underPct": 0.736},
    {"name": "DeMar DeRozan", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.252, "underPct": 0.748},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.235, "underPct": 0.765},
    {"name": "Jay Huff", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.222, "underPct": 0.778},
    {"name": "Russell Westbrook", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.216, "underPct": 0.784},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.204, "underPct": 0.796},
    {"name": "Drew Eubanks", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.204, "underPct": 0.796},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.184, "underPct": 0.816},
    {"name": "Tyrese Martin", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.16, "underPct": 0.84},
    {"name": "T.J. McConnell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.138, "underPct": 0.862},
    {"name": "Jared McCain", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.009, "underPct": 0.991},
];const underdogAssistsHitRates = [
    {"name": "Caris LeVert", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.767, "underPct": 0.233},
    {"name": "Isaiah Hartenstein", "line": 2.5, "l5": 1.0, "l10": 0.7, "l15": 0.73, "overPct": 0.732, "underPct": 0.268},
    {"name": "Cade Cunningham", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.672, "underPct": 0.328},
    {"name": "LaMelo Ball", "line": 7.5, "l5": 0.8, "l10": 0.9, "l15": 0.6, "overPct": 0.672, "underPct": 0.328},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.638, "underPct": 0.362},
    {"name": "Anthony Black", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.631, "underPct": 0.369},
    {"name": "Jaylon Tyson", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.607, "underPct": 0.393},
    {"name": "Peyton Watson", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.542, "underPct": 0.458},
    {"name": "Kon Knueppel", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.534, "underPct": 0.466},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.534, "underPct": 0.466},
    {"name": "Isaiah Collier", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.511, "underPct": 0.489},
    {"name": "Paul George", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.511, "underPct": 0.489},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.497, "underPct": 0.503},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.435, "underPct": 0.565},
    {"name": "Keegan Murray", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.404, "underPct": 0.596},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.377, "underPct": 0.623},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.373, "underPct": 0.627},
    {"name": "Drake Powell", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.37, "underPct": 0.63},
    {"name": "Tyrese Martin", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.338, "underPct": 0.662},
    {"name": "Tyrese Maxey", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.338, "underPct": 0.662},
    {"name": "Pascal Siakam", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.289, "underPct": 0.711},
];const underdogReboundsHitRates = [
    {"name": "Jaylon Tyson", "line": 3.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.875, "underPct": 0.125},
    {"name": "LeBron James", "line": 6.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.794, "underPct": 0.206},
    {"name": "Austin Reaves", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.756, "underPct": 0.244},
    {"name": "Lonzo Ball", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.753, "underPct": 0.247},
    {"name": "Ivica Zubac", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.677, "underPct": 0.323},
    {"name": "Daniel Gafford", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.672, "underPct": 0.328},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.666, "underPct": 0.334},
    {"name": "Santi Aldama", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.66, "underPct": 0.34},
    {"name": "Duncan Robinson", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.646, "underPct": 0.354},
    {"name": "Ryan Rollins", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.64, "underPct": 0.36},
    {"name": "Ajay Mitchell", "line": 2.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.612, "underPct": 0.388},
    {"name": "Donovan Mitchell", "line": 4.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.606, "underPct": 0.394},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.576, "underPct": 0.424},
    {"name": "Rui Hachimura", "line": 3.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.559, "underPct": 0.441},
    {"name": "Anthony Davis", "line": 9.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.555, "underPct": 0.445},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.546, "underPct": 0.454},
    {"name": "Kentavious Caldwell-Pope", "line": 1.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.521, "underPct": 0.479},
    {"name": "Jake LaRavia", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.501, "underPct": 0.499},
    {"name": "De'Andre Hunter", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.496, "underPct": 0.504},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Svi Mykhailiuk", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.473, "underPct": 0.527},
    {"name": "Chet Holmgren", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.471, "underPct": 0.529},
    {"name": "Dillon Brooks", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.437, "underPct": 0.563},
    {"name": "Keegan Murray", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.418, "underPct": 0.582},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.405, "underPct": 0.595},
    {"name": "Bruce Brown", "line": 3.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.403, "underPct": 0.597},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.395, "underPct": 0.605},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.347, "underPct": 0.653},
    {"name": "Justin Edwards", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.273, "underPct": 0.727},
    {"name": "Spencer Jones", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.269, "underPct": 0.731},
    {"name": "Luke Kornet", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.261, "underPct": 0.739},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.243, "underPct": 0.757},
    {"name": "Zach Edey", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.172, "underPct": 0.828},
    {"name": "Drew Eubanks", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.126, "underPct": 0.874},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Austin Reaves", "line": 31.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Black", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 27.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Hartenstein", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kon Knueppel", "line": 25.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Gillespie", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 16.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Filipowski", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ajay Mitchell", "line": 17.5, "l5": 0.8, "l10": 0.9, "l15": 0.87, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Santi Aldama", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 13.5, "l5": 0.8, "l10": 0.9, "l15": 0.6, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Dyson Daniels", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dylan Harper", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 37.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Caris LeVert", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Davis", "line": 31.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Khris Middleton", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 46.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Joe", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mark Williams", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 39.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mitchell Robinson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 35.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Murray", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Vassell", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 41.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 16.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "James Harden", "line": 40.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kennard", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 16.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Deandre Ayton", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 31.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles Bridges", "line": 28.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 31.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jake LaRavia", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brice Sensabaugh", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 35.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Malik Monk", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Darius Garland", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luke Kornet", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 44.5, "l5": 0.4, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 39.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tobias Harris", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles McBride", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Kuzma", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 40.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Karl-Anthony Towns", "line": 39.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 37.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Paul George", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "John Collins", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jared McCain", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Noah Clowney", "line": 24.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Day'Ron Sharpe", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Drake Powell", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Miller", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Isaiah Jackson", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Huerter", "line": 16.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "LeBron James", "line": 33.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jay Huff", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 33.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drew Eubanks", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden Ivey", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bilal Coulibaly", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 30.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 25.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 18.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Justin Edwards", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quentin Grimes", "line": 25.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
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
    {"name": "Isaiah Hartenstein", "line": 11.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Johnson", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Lonzo Ball", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Davis", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 12.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Gillespie", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 7.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ivica Zubac", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Goodwin", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Booker", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Santi Aldama", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Coby White", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "DeMar DeRozan", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ausar Thompson", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyrese Maxey", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Giannis Antetokounmpo", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
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

