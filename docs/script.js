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
    {"name1": "Isaiah Jackson", "name2": "Jordan Goodwin", "line1": 6.5, "line2": 5.5, "prediction1": 9.03, "prediction2": 9.37, "side1": "over", "side2": "over", "recommendation": 0, "ev": 193.47, "kelly": 0.967, "sigma1": "Low", "sigma2": "Low", "prob1": 0.998, "prob2": 1.0, "hitRate1": 71.2, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 94.1, "l5_2": 1.0, "l15_2": 0.8},
    {"name1": "Pelle Larsson", "name2": "Grayson Allen", "line1": 7.5, "line2": 13.5, "prediction1": 10.21, "prediction2": 18.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 192.51, "kelly": 0.963, "sigma1": "Low", "sigma2": "Low", "prob1": 0.997, "prob2": 0.998, "hitRate1": 75.9, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 86.5, "l5_2": 0.6, "l15_2": 0.73},
    {"name1": "Keyonte George", "name2": "Rui Hachimura", "line1": 19.5, "line2": 11.5, "prediction1": 25.37, "prediction2": 14.86, "side1": "over", "side2": "over", "recommendation": 0, "ev": 189.76, "kelly": 0.949, "sigma1": "Low", "sigma2": "Low", "prob1": 0.994, "prob2": 0.991, "hitRate1": 92.6, "l5_1": 1.0, "l15_1": 0.67, "hitRate2": 74.6, "l5_2": 0.8, "l15_2": 0.8},
    {"name1": "Bruce Brown", "name2": "Svi Mykhailiuk", "line1": 8.0, "line2": 7.5, "prediction1": 6.29, "prediction2": 9.72, "side1": "under", "side2": "over", "recommendation": 0, "ev": 186.0, "kelly": 0.93, "sigma1": "Low", "sigma2": "Low", "prob1": 0.983, "prob2": 0.99, "hitRate1": 87.2, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 55.5, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Duncan Robinson", "name2": "Brandon Miller", "line1": 11.5, "line2": 19.5, "prediction1": 14.28, "prediction2": 15.92, "side1": "over", "side2": "under", "recommendation": 0, "ev": 179.54, "kelly": 0.898, "sigma1": "Low", "sigma2": "Low", "prob1": 0.978, "prob2": 0.973, "hitRate1": 70.8, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 75.3, "l5_2": 0.4, "l15_2": 0.2},
    {"name1": "Tobias Harris", "name2": "Austin Reaves", "line1": 13.5, "line2": 23.5, "prediction1": 16.42, "prediction2": 28.63, "side1": "over", "side2": "over", "recommendation": 0, "ev": 176.14, "kelly": 0.881, "sigma1": "Low", "sigma2": "Low", "prob1": 0.967, "prob2": 0.971, "hitRate1": 82.9, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 75.1, "l5_2": 0.8, "l15_2": 0.8},
    {"name1": "Davion Mitchell", "name2": "Collin Gillespie", "line1": 8.0, "line2": 11.5, "prediction1": 9.66, "prediction2": 13.92, "side1": "over", "side2": "over", "recommendation": 0, "ev": 169.91, "kelly": 0.85, "sigma1": "Low", "sigma2": "Low", "prob1": 0.955, "prob2": 0.962, "hitRate1": 56.7, "l5_1": 0.2, "l15_1": 0.53, "hitRate2": 87.3, "l5_2": 1.0, "l15_2": 0.67},
    {"name1": "Kyle Kuzma", "name2": "Kawhi Leonard", "line1": 11.0, "line2": 22.5, "prediction1": 13.18, "prediction2": 26.75, "side1": "over", "side2": "over", "recommendation": 0, "ev": 166.44, "kelly": 0.832, "sigma1": "Low", "sigma2": "Low", "prob1": 0.952, "prob2": 0.952, "hitRate1": 66.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 81.3, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Norman Powell", "name2": "Miles Bridges", "line1": 20.5, "line2": 19.5, "prediction1": 24.32, "prediction2": 23.03, "side1": "over", "side2": "over", "recommendation": 0, "ev": 163.37, "kelly": 0.817, "sigma1": "Low", "sigma2": "Low", "prob1": 0.949, "prob2": 0.944, "hitRate1": 71.4, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 70.1, "l5_2": 0.6, "l15_2": 0.73},
    {"name1": "Kel'el Ware", "name2": "LeBron James", "line1": 11.5, "line2": 19.5, "prediction1": 13.61, "prediction2": 16.08, "side1": "over", "side2": "under", "recommendation": 0, "ev": 158.68, "kelly": 0.793, "sigma1": "Low", "sigma2": "Low", "prob1": 0.941, "prob2": 0.935, "hitRate1": 77.8, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 38.8, "l5_2": 0.2, "l15_2": 0.07},
];const prizepicksTriosData = [
    {"name1": "Isaiah Jackson", "name2": "Pelle Larsson", "name3": "Jordan Goodwin", "line1": 6.5, "line2": 7.5, "line3": 5.5, "prediction1": 9.03, "prediction2": 10.21, "prediction3": 9.37, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 437.49, "kelly": 0.875, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.998, "prob2": 0.997, "prob3": 1.0, "hitRate1": 71.2, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 75.9, "l5_2": 0.6, "l15_2": 0.73, "hitRate3": 94.1, "l5_3": 1.0, "l15_3": 0.8},
    {"name1": "Bruce Brown", "name2": "Keyonte George", "name3": "Grayson Allen", "line1": 8.0, "line2": 19.5, "line3": 13.5, "prediction1": 6.29, "prediction2": 25.37, "prediction3": 18.27, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 426.36, "kelly": 0.853, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.983, "prob2": 0.994, "prob3": 0.998, "hitRate1": 87.2, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 92.6, "l5_2": 1.0, "l15_2": 0.67, "hitRate3": 86.5, "l5_3": 0.6, "l15_3": 0.73},
    {"name1": "Duncan Robinson", "name2": "Svi Mykhailiuk", "name3": "Rui Hachimura", "line1": 11.5, "line2": 7.5, "line3": 11.5, "prediction1": 14.28, "prediction2": 9.72, "prediction3": 14.86, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 418.06, "kelly": 0.836, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.978, "prob2": 0.99, "prob3": 0.991, "hitRate1": 70.8, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 55.5, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 74.6, "l5_3": 0.8, "l15_3": 0.8},
    {"name1": "Tobias Harris", "name2": "Brandon Miller", "name3": "Austin Reaves", "line1": 13.5, "line2": 19.5, "line3": 23.5, "prediction1": 16.42, "prediction2": 15.92, "prediction3": 28.63, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 393.33, "kelly": 0.787, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.967, "prob2": 0.973, "prob3": 0.971, "hitRate1": 82.9, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 75.3, "l5_2": 0.4, "l15_2": 0.2, "hitRate3": 75.1, "l5_3": 0.8, "l15_3": 0.8},
    {"name1": "Kyle Kuzma", "name2": "Davion Mitchell", "name3": "Collin Gillespie", "line1": 11.0, "line2": 8.0, "line3": 11.5, "prediction1": 13.18, "prediction2": 9.66, "prediction3": 13.92, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 372.01, "kelly": 0.744, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.952, "prob2": 0.955, "prob3": 0.962, "hitRate1": 66.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 56.7, "l5_2": 0.2, "l15_2": 0.53, "hitRate3": 87.3, "l5_3": 1.0, "l15_3": 0.67},
    {"name1": "Kawhi Leonard", "name2": "Miles Bridges", "name3": "LeBron James", "line1": 22.5, "line2": 19.5, "line3": 19.5, "prediction1": 26.75, "prediction2": 23.03, "prediction3": 16.08, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 353.84, "kelly": 0.708, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.952, "prob2": 0.944, "prob3": 0.935, "hitRate1": 81.3, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 70.1, "l5_2": 0.6, "l15_2": 0.73, "hitRate3": 38.8, "l5_3": 0.2, "l15_3": 0.07},
    {"name1": "Norman Powell", "name2": "Goga Bitadze", "name3": "Anthony Davis", "line1": 20.5, "line2": 5.5, "line3": 19.5, "prediction1": 24.32, "prediction2": 6.54, "prediction3": 16.67, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 344.69, "kelly": 0.689, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.949, "prob2": 0.931, "prob3": 0.932, "hitRate1": 71.4, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 58.7, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 58.4, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Khris Middleton", "name2": "Kel'el Ware", "name3": "Day'Ron Sharpe", "line1": 11.5, "line2": 11.5, "line3": 7.5, "prediction1": 9.79, "prediction2": 13.61, "prediction3": 6.33, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 339.06, "kelly": 0.678, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.929, "prob2": 0.941, "prob3": 0.931, "hitRate1": 73.4, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 77.8, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 67.9, "l5_3": 0.0, "l15_3": 0.2},
    {"name1": "Andrew Wiggins", "name2": "Klay Thompson", "name3": "Deandre Ayton", "line1": 15.5, "line2": 10.5, "line3": 13.5, "prediction1": 17.94, "prediction2": 12.23, "prediction3": 15.79, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 322.28, "kelly": 0.645, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.916, "prob2": 0.919, "prob3": 0.929, "hitRate1": 59.5, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 67.0, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 59.9, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "John Collins", "name2": "Kon Knueppel", "name3": "Jalen Suggs", "line1": 14.5, "line2": 16.0, "line3": 16.5, "prediction1": 12.51, "prediction2": 18.52, "prediction3": 14.31, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 312.15, "kelly": 0.624, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.916, "prob2": 0.916, "prob3": 0.91, "hitRate1": 74.2, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 74.1, "l5_2": 0.6, "l15_2": 0.67, "hitRate3": 56.2, "l5_3": 0.4, "l15_3": 0.2},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Dru Smith", "name2": "Jordan Goodwin", "line1": 4.5, "line2": 5.5, "prediction1": 6.39, "prediction2": 9.37, "side1": "over", "side2": "over", "recommendation": 0, "ev": 193.53, "kelly": 0.968, "sigma1": "Low", "sigma2": "Low", "prob1": 0.998, "prob2": 1.0, "hitRate1": 78.5, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 94.1, "l5_2": 1.0, "l15_2": 0.8},
    {"name1": "Isaiah Jackson", "name2": "Pelle Larsson", "line1": 6.5, "line2": 7.5, "prediction1": 9.03, "prediction2": 10.21, "side1": "over", "side2": "over", "recommendation": 0, "ev": 192.63, "kelly": 0.963, "sigma1": "Low", "sigma2": "Low", "prob1": 0.998, "prob2": 0.997, "hitRate1": 71.2, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 75.9, "l5_2": 0.6, "l15_2": 0.73},
    {"name1": "Keyonte George", "name2": "Rui Hachimura", "line1": 19.5, "line2": 11.5, "prediction1": 25.37, "prediction2": 14.86, "side1": "over", "side2": "over", "recommendation": 0, "ev": 189.76, "kelly": 0.949, "sigma1": "Low", "sigma2": "Low", "prob1": 0.994, "prob2": 0.991, "hitRate1": 92.6, "l5_1": 1.0, "l15_1": 0.67, "hitRate2": 74.6, "l5_2": 0.8, "l15_2": 0.8},
    {"name1": "Spencer Jones", "name2": "Svi Mykhailiuk", "line1": 5.5, "line2": 7.5, "prediction1": 4.04, "prediction2": 9.72, "side1": "under", "side2": "over", "recommendation": 0, "ev": 185.07, "kelly": 0.925, "sigma1": "Low", "sigma2": "Low", "prob1": 0.979, "prob2": 0.99, "hitRate1": 77.2, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 55.5, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Duncan Robinson", "name2": "Brandon Miller", "line1": 11.5, "line2": 19.5, "prediction1": 14.28, "prediction2": 15.92, "side1": "over", "side2": "under", "recommendation": 0, "ev": 179.54, "kelly": 0.898, "sigma1": "Low", "sigma2": "Low", "prob1": 0.978, "prob2": 0.973, "hitRate1": 70.8, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 75.3, "l5_2": 0.4, "l15_2": 0.2},
    {"name1": "Kawhi Leonard", "name2": "Austin Reaves", "line1": 22.5, "line2": 23.5, "prediction1": 26.75, "prediction2": 28.63, "side1": "over", "side2": "over", "recommendation": 1, "ev": 171.84, "kelly": 0.859, "sigma1": "Low", "sigma2": "Low", "prob1": 0.952, "prob2": 0.971, "hitRate1": 81.3, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 75.1, "l5_2": 0.8, "l15_2": 0.8},
    {"name1": "Norman Powell", "name2": "Miles Bridges", "line1": 20.5, "line2": 19.5, "prediction1": 24.32, "prediction2": 23.03, "side1": "over", "side2": "over", "recommendation": 0, "ev": 163.37, "kelly": 0.817, "sigma1": "Low", "sigma2": "Low", "prob1": 0.949, "prob2": 0.944, "hitRate1": 71.4, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 70.1, "l5_2": 0.6, "l15_2": 0.73},
    {"name1": "Kel'el Ware", "name2": "LeBron James", "line1": 11.5, "line2": 19.5, "prediction1": 13.61, "prediction2": 16.08, "side1": "over", "side2": "under", "recommendation": 0, "ev": 158.68, "kelly": 0.793, "sigma1": "Low", "sigma2": "Low", "prob1": 0.941, "prob2": 0.935, "hitRate1": 77.8, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 38.8, "l5_2": 0.2, "l15_2": 0.07},
    {"name1": "Goga Bitadze", "name2": "Anthony Davis", "line1": 5.5, "line2": 19.5, "prediction1": 6.54, "prediction2": 16.67, "side1": "over", "side2": "under", "recommendation": 0, "ev": 155.1, "kelly": 0.776, "sigma1": "Low", "sigma2": "Low", "prob1": 0.931, "prob2": 0.932, "hitRate1": 58.7, "l5_1": 0.8, "l15_1": 0.67, "hitRate2": 58.4, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "Khris Middleton", "name2": "Reed Sheppard", "line1": 11.5, "line2": 13.5, "prediction1": 9.79, "prediction2": 15.76, "side1": "under", "side2": "over", "recommendation": 0, "ev": 152.89, "kelly": 0.764, "sigma1": "Low", "sigma2": "Low", "prob1": 0.929, "prob2": 0.926, "hitRate1": 78.6, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 77.8, "l5_2": 0.6, "l15_2": 0.47},
];const underdogTriosData = [
    {"name1": "Isaiah Jackson", "name2": "Pelle Larsson", "name3": "Jordan Goodwin", "line1": 6.5, "line2": 7.5, "line3": 5.5, "prediction1": 9.03, "prediction2": 10.21, "prediction3": 9.37, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 437.49, "kelly": 0.875, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.998, "prob2": 0.997, "prob3": 1.0, "hitRate1": 71.2, "l5_1": 0.8, "l15_1": 0.73, "hitRate2": 75.9, "l5_2": 0.6, "l15_2": 0.73, "hitRate3": 94.1, "l5_3": 1.0, "l15_3": 0.8},
    {"name1": "Spencer Jones", "name2": "Keyonte George", "name3": "Rui Hachimura", "line1": 5.5, "line2": 19.5, "line3": 11.5, "prediction1": 4.04, "prediction2": 25.37, "prediction3": 14.86, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 421.28, "kelly": 0.843, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.979, "prob2": 0.994, "prob3": 0.991, "hitRate1": 77.2, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 92.6, "l5_2": 1.0, "l15_2": 0.67, "hitRate3": 74.6, "l5_3": 0.8, "l15_3": 0.8},
    {"name1": "Duncan Robinson", "name2": "Brandon Miller", "name3": "Svi Mykhailiuk", "line1": 11.5, "line2": 19.5, "line3": 7.5, "prediction1": 14.28, "prediction2": 15.92, "prediction3": 9.72, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 408.27, "kelly": 0.817, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.978, "prob2": 0.973, "prob3": 0.99, "hitRate1": 70.8, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 75.3, "l5_2": 0.4, "l15_2": 0.2, "hitRate3": 55.5, "l5_3": 0.8, "l15_3": 0.6},
    {"name1": "Kawhi Leonard", "name2": "Miles Bridges", "name3": "Austin Reaves", "line1": 22.5, "line2": 19.5, "line3": 23.5, "prediction1": 26.75, "prediction2": 23.03, "prediction3": 28.63, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 371.29, "kelly": 0.743, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.952, "prob2": 0.944, "prob3": 0.971, "hitRate1": 81.3, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 70.1, "l5_2": 0.6, "l15_2": 0.73, "hitRate3": 75.1, "l5_3": 0.8, "l15_3": 0.8},
    {"name1": "Norman Powell", "name2": "Anthony Davis", "name3": "LeBron James", "line1": 20.5, "line2": 19.5, "line3": 19.5, "prediction1": 24.32, "prediction2": 16.67, "prediction3": 16.08, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 346.58, "kelly": 0.693, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.949, "prob2": 0.932, "prob3": 0.935, "hitRate1": 71.4, "l5_1": 0.6, "l15_1": 0.73, "hitRate2": 58.4, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 38.8, "l5_3": 0.2, "l15_3": 0.07},
    {"name1": "Khris Middleton", "name2": "Kel'el Ware", "name3": "Goga Bitadze", "line1": 11.5, "line2": 11.5, "line3": 5.5, "prediction1": 9.79, "prediction2": 13.61, "prediction3": 6.54, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 339.52, "kelly": 0.679, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.929, "prob2": 0.941, "prob3": 0.931, "hitRate1": 78.6, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 77.8, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 58.7, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Caris LeVert", "name2": "Klay Thompson", "name3": "Reed Sheppard", "line1": 8.5, "line2": 10.5, "line3": 13.5, "prediction1": 9.93, "prediction2": 12.23, "prediction3": 15.76, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 322.67, "kelly": 0.645, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.92, "prob2": 0.919, "prob3": 0.926, "hitRate1": 75.0, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 67.0, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 77.8, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Jay Huff", "name2": "Andrew Wiggins", "name3": "Jalen Suggs", "line1": 10.5, "line2": 15.5, "line3": 16.5, "prediction1": 9.11, "prediction2": 17.94, "prediction3": 14.31, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 305.49, "kelly": 0.611, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.901, "prob2": 0.916, "prob3": 0.91, "hitRate1": 61.7, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 59.5, "l5_2": 0.4, "l15_2": 0.6, "hitRate3": 64.1, "l5_3": 0.4, "l15_3": 0.2},
    {"name1": "John Collins", "name2": "Nikola Vu\u010devi\u0107", "name3": "Brandon Williams", "line1": 14.5, "line2": 17.5, "line3": 13.5, "prediction1": 12.51, "prediction2": 15.23, "prediction3": 11.8, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": 300.06, "kelly": 0.6, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.916, "prob2": 0.905, "prob3": 0.894, "hitRate1": 74.2, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 78.3, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Nicolas Batum", "name2": "Collin Sexton", "name3": "Cameron Johnson", "line1": 5.5, "line2": 13.5, "line3": 15.5, "prediction1": 4.68, "prediction2": 15.26, "prediction3": 13.65, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 267.89, "kelly": 0.536, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.883, "prob2": 0.874, "prob3": 0.883, "hitRate1": 65.2, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 79.2, "l5_2": 1.0, "l15_2": 0.67, "hitRate3": 33.7, "l5_3": 0.6, "l15_3": 0.27},
];const prizepicksPointsHitRates = [
    {"name": "Jordan Goodwin", "line": 5.5, "l5": 1.0, "l10": 1.0, "l15": 0.8, "overPct": 0.941, "underPct": 0.059},
    {"name": "Keyonte George", "line": 19.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.926, "underPct": 0.074},
    {"name": "Tyler Herro", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.879, "underPct": 0.121},
    {"name": "Peyton Watson", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.874, "underPct": 0.126},
    {"name": "Collin Gillespie", "line": 11.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.873, "underPct": 0.127},
    {"name": "Grayson Allen", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.865, "underPct": 0.135},
    {"name": "Tobias Harris", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.829, "underPct": 0.171},
    {"name": "Kawhi Leonard", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.813, "underPct": 0.187},
    {"name": "Lauri Markkanen", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.789, "underPct": 0.211},
    {"name": "Jalen Duren", "line": 17.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.789, "underPct": 0.211},
    {"name": "Dru Smith", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.785, "underPct": 0.215},
    {"name": "Nickeil Alexander-Walker", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.782, "underPct": 0.218},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.778, "underPct": 0.222},
    {"name": "Ryan Rollins", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.776, "underPct": 0.224},
    {"name": "Cade Cunningham", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.769, "underPct": 0.231},
    {"name": "Pelle Larsson", "line": 7.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.759, "underPct": 0.241},
    {"name": "Onyeka Okongwu", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.757, "underPct": 0.243},
    {"name": "Austin Reaves", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.751, "underPct": 0.249},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.749, "underPct": 0.251},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.746, "underPct": 0.254},
    {"name": "Kon Knueppel", "line": 16.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.741, "underPct": 0.259},
    {"name": "Jake LaRavia", "line": 4.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.722, "underPct": 0.278},
    {"name": "Jamal Murray", "line": 24.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.721, "underPct": 0.279},
    {"name": "Bobby Portis", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.715, "underPct": 0.285},
    {"name": "Norman Powell", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.714, "underPct": 0.286},
    {"name": "Isaiah Jackson", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.712, "underPct": 0.288},
    {"name": "Collin Sexton", "line": 14.0, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.71, "underPct": 0.29},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.708, "underPct": 0.292},
    {"name": "Miles Bridges", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.701, "underPct": 0.299},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.694, "underPct": 0.306},
    {"name": "Reed Sheppard", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.693, "underPct": 0.307},
    {"name": "James Harden", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.69, "underPct": 0.31},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.677, "underPct": 0.323},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.67, "underPct": 0.33},
    {"name": "Kyle Kuzma", "line": 11.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.663, "underPct": 0.337},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.662, "underPct": 0.338},
    {"name": "Alperen Sengun", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.638, "underPct": 0.362},
    {"name": "Caris LeVert", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.637, "underPct": 0.363},
    {"name": "Tyrese Martin", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.625, "underPct": 0.375},
    {"name": "Cooper Flagg", "line": 17.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.62, "underPct": 0.38},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.618, "underPct": 0.382},
    {"name": "LeBron James", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.612, "underPct": 0.388},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.609, "underPct": 0.391},
    {"name": "Jarace Walker", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.607, "underPct": 0.393},
    {"name": "Dillon Brooks", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.599, "underPct": 0.401},
    {"name": "Deandre Ayton", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.599, "underPct": 0.401},
    {"name": "Andrew Wiggins", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.595, "underPct": 0.405},
    {"name": "Ben Sheppard", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.591, "underPct": 0.409},
    {"name": "Goga Bitadze", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.587, "underPct": 0.413},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.582, "underPct": 0.418},
    {"name": "Ayo Dosunmu", "line": 15.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.57, "underPct": 0.43},
    {"name": "Davion Mitchell", "line": 8.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.567, "underPct": 0.433},
    {"name": "Kris Dunn", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.561, "underPct": 0.439},
    {"name": "Svi Mykhailiuk", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.555, "underPct": 0.445},
    {"name": "Aaron Holiday", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.548, "underPct": 0.452},
    {"name": "Tristan da Silva", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.541, "underPct": 0.459},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.539, "underPct": 0.461},
    {"name": "Pascal Siakam", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.535, "underPct": 0.465},
    {"name": "Desmond Bane", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Steven Adams", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.525, "underPct": 0.475},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.518, "underPct": 0.482},
    {"name": "Jalen Johnson", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "T.J. McConnell", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.516, "underPct": 0.484},
    {"name": "Cam Whitmore", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brice Sensabaugh", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.498, "underPct": 0.502},
    {"name": "Kyshawn George", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.482, "underPct": 0.518},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.478, "underPct": 0.522},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.463, "underPct": 0.537},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.462, "underPct": 0.538},
    {"name": "Mark Williams", "line": 13.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.459, "underPct": 0.541},
    {"name": "Myles Turner", "line": 13.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.454, "underPct": 0.546},
    {"name": "Jaden Ivey", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.441, "underPct": 0.559},
    {"name": "Jalen Suggs", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.438, "underPct": 0.562},
    {"name": "Kevin Durant", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.423, "underPct": 0.577},
    {"name": "Bam Adebayo", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.418, "underPct": 0.582},
    {"name": "Anthony Davis", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.416, "underPct": 0.584},
    {"name": "Ziaire Williams", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.409, "underPct": 0.591},
    {"name": "Noah Clowney", "line": 16.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.404, "underPct": 0.596},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.398, "underPct": 0.602},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "Jay Huff", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.383, "underPct": 0.617},
    {"name": "Franz Wagner", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.37, "underPct": 0.63},
    {"name": "Kobe Brown", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.369, "underPct": 0.631},
    {"name": "Brook Lopez", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.366, "underPct": 0.634},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.365, "underPct": 0.635},
    {"name": "Nicolas Batum", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.348, "underPct": 0.652},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.321, "underPct": 0.679},
    {"name": "Ausar Thompson", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.283, "underPct": 0.717},
    {"name": "Khris Middleton", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.266, "underPct": 0.734},
    {"name": "John Collins", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.258, "underPct": 0.742},
    {"name": "LaMelo Ball", "line": 18.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.256, "underPct": 0.744},
    {"name": "Giannis Antetokounmpo", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.251, "underPct": 0.749},
    {"name": "Brandon Miller", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.247, "underPct": 0.753},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.217, "underPct": 0.783},
    {"name": "Ryan Dunn", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.196, "underPct": 0.804},
    {"name": "Matas Buzelis", "line": 13.0, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.169, "underPct": 0.831},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.156, "underPct": 0.844},
    {"name": "Bilal Coulibaly", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.144, "underPct": 0.856},
    {"name": "Bruce Brown", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.128, "underPct": 0.872},
    {"name": "Ryan Nembhard", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.124, "underPct": 0.876},
    {"name": "Devin Booker", "line": 27.0, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.115, "underPct": 0.885},
    {"name": "Kevin Porter Jr.", "line": 13.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.074, "underPct": 0.926},
];const prizepicksAssistsHitRates = [
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.713, "underPct": 0.287},
    {"name": "Jalen Johnson", "line": 7.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.697, "underPct": 0.303},
    {"name": "Caris LeVert", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.677, "underPct": 0.323},
    {"name": "Ivica Zubac", "line": 2.5, "l5": 1.0, "l10": 0.5, "l15": 0.47, "overPct": 0.668, "underPct": 0.332},
    {"name": "Davion Mitchell", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.666, "underPct": 0.334},
    {"name": "Andrew Wiggins", "line": 2.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.66, "underPct": 0.34},
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.653, "underPct": 0.347},
    {"name": "LeBron James", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.642, "underPct": 0.358},
    {"name": "Isaiah Collier", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.632, "underPct": 0.368},
    {"name": "Kris Dunn", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.62, "underPct": 0.38},
    {"name": "Tyler Herro", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.585, "underPct": 0.415},
    {"name": "Keyonte George", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.576, "underPct": 0.424},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.572, "underPct": 0.428},
    {"name": "Josh Giddey", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.53, "underPct": 0.47},
    {"name": "Brandon Williams", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.519, "underPct": 0.481},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.495, "underPct": 0.505},
    {"name": "Alperen Sengun", "line": 6.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.469, "underPct": 0.531},
    {"name": "Jalen Suggs", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.467, "underPct": 0.533},
    {"name": "Nickeil Alexander-Walker", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.462, "underPct": 0.538},
    {"name": "Collin Gillespie", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.444, "underPct": 0.556},
    {"name": "Cade Cunningham", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.442, "underPct": 0.558},
    {"name": "T.J. McConnell", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.437, "underPct": 0.563},
    {"name": "Giannis Antetokounmpo", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.425, "underPct": 0.575},
    {"name": "Kevin Love", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Ausar Thompson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.424, "underPct": 0.576},
    {"name": "LaMelo Ball", "line": 7.0, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.407, "underPct": 0.593},
    {"name": "Andrew Nembhard", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.392, "underPct": 0.608},
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.356, "underPct": 0.644},
    {"name": "Kevin Huerter", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.341, "underPct": 0.659},
    {"name": "Devin Booker", "line": 7.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.335, "underPct": 0.665},
    {"name": "Franz Wagner", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.316, "underPct": 0.684},
    {"name": "Amen Thompson", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.307, "underPct": 0.693},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.303, "underPct": 0.697},
    {"name": "James Harden", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.282, "underPct": 0.718},
    {"name": "Collin Sexton", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.28, "underPct": 0.72},
    {"name": "Ryan Nembhard", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.254, "underPct": 0.746},
    {"name": "Pascal Siakam", "line": 4.0, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.254, "underPct": 0.746},
    {"name": "Kevin Porter Jr.", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.07, "underPct": 0.93},
];const prizepicksReboundsHitRates = [
    {"name": "Kel'el Ware", "line": 9.0, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.789, "underPct": 0.211},
    {"name": "Cade Cunningham", "line": 6.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.762, "underPct": 0.238},
    {"name": "Tyler Herro", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.759, "underPct": 0.241},
    {"name": "P.J. Washington", "line": 6.5, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.725, "underPct": 0.275},
    {"name": "Peyton Watson", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.689, "underPct": 0.311},
    {"name": "Steven Adams", "line": 7.0, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.669, "underPct": 0.331},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.663, "underPct": 0.337},
    {"name": "Austin Reaves", "line": 5.0, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.622, "underPct": 0.378},
    {"name": "Lauri Markkanen", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.611, "underPct": 0.389},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.599, "underPct": 0.401},
    {"name": "Bobby Portis", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.574, "underPct": 0.426},
    {"name": "Anthony Davis", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.57, "underPct": 0.43},
    {"name": "Franz Wagner", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.56, "underPct": 0.44},
    {"name": "Devin Booker", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.546, "underPct": 0.454},
    {"name": "Andrew Wiggins", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.542, "underPct": 0.458},
    {"name": "Kyle Kuzma", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.54, "underPct": 0.46},
    {"name": "Khris Middleton", "line": 4.0, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.522, "underPct": 0.478},
    {"name": "Pascal Siakam", "line": 7.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.521, "underPct": 0.479},
    {"name": "Ryan Rollins", "line": 4.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.52, "underPct": 0.48},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.517, "underPct": 0.483},
    {"name": "Kyle Filipowski", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.514, "underPct": 0.486},
    {"name": "Goga Bitadze", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.512, "underPct": 0.488},
    {"name": "Ziaire Williams", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.483, "underPct": 0.517},
    {"name": "Alperen Sengun", "line": 9.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.475, "underPct": 0.525},
    {"name": "Bennedict Mathurin", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.473, "underPct": 0.527},
    {"name": "James Harden", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.469, "underPct": 0.531},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.464, "underPct": 0.536},
    {"name": "Amen Thompson", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.462, "underPct": 0.538},
    {"name": "Kon Knueppel", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.458, "underPct": 0.542},
    {"name": "Royce O'Neale", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.458, "underPct": 0.542},
    {"name": "Cooper Flagg", "line": 6.0, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.457, "underPct": 0.543},
    {"name": "Jalen Johnson", "line": 9.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.457, "underPct": 0.543},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.431, "underPct": 0.569},
    {"name": "Collin Gillespie", "line": 3.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.424, "underPct": 0.576},
    {"name": "Rui Hachimura", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.417, "underPct": 0.583},
    {"name": "Tristan da Silva", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.41, "underPct": 0.59},
    {"name": "Isaiah Jackson", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.399, "underPct": 0.601},
    {"name": "LeBron James", "line": 6.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.381, "underPct": 0.619},
    {"name": "Mark Williams", "line": 9.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.38, "underPct": 0.62},
    {"name": "Miles Bridges", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.378, "underPct": 0.622},
    {"name": "Kawhi Leonard", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.375, "underPct": 0.625},
    {"name": "LaMelo Ball", "line": 5.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.363, "underPct": 0.637},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.353, "underPct": 0.647},
    {"name": "Bam Adebayo", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.346, "underPct": 0.654},
    {"name": "Ryan Kalkbrenner", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.341, "underPct": 0.659},
    {"name": "Jay Huff", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.297, "underPct": 0.703},
    {"name": "Ryan Dunn", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.288, "underPct": 0.712},
    {"name": "Ausar Thompson", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.287, "underPct": 0.713},
    {"name": "John Collins", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.275, "underPct": 0.725},
    {"name": "Tyus Jones", "line": 1.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.25, "underPct": 0.75},
    {"name": "Bilal Coulibaly", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.248, "underPct": 0.752},
    {"name": "Kevin Durant", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.24, "underPct": 0.76},
    {"name": "Giannis Antetokounmpo", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.237, "underPct": 0.763},
    {"name": "Tobias Harris", "line": 5.0, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.23, "underPct": 0.77},
    {"name": "Onyeka Okongwu", "line": 7.0, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.189, "underPct": 0.811},
    {"name": "Bruce Brown", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.177, "underPct": 0.823},
    {"name": "Ace Bailey", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.164, "underPct": 0.836},
    {"name": "Aaron Holiday", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.163, "underPct": 0.837},
    {"name": "Kevin Porter Jr.", "line": 4.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.147, "underPct": 0.853},
    {"name": "Kobe Brown", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.145, "underPct": 0.855},
    {"name": "Brandon Miller", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.049, "underPct": 0.951},
];const prizepicksBlocksHitRates = [
    {"name": "Myles Turner", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.476, "underPct": 0.524},
    {"name": "Zaccharie Risacher", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.566, "underPct": 0.434},
    {"name": "Bam Adebayo", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.648, "underPct": 0.352},
    {"name": "Nicolas Batum", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.319, "underPct": 0.681},
    {"name": "Ryan Kalkbrenner", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.419, "underPct": 0.581},
    {"name": "Brandon Miller", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.675, "underPct": 0.325},
    {"name": "Noah Clowney", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.689, "underPct": 0.311},
    {"name": "Alperen Sengun", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.615, "underPct": 0.385},
    {"name": "Steven Adams", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.488, "underPct": 0.512},
    {"name": "Reed Sheppard", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.371, "underPct": 0.629},
];const prizepicksStealsHitRates = [
    {"name": "Bobby Portis", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.383, "underPct": 0.617},
    {"name": "Cam Whitmore", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.39, "underPct": 0.61},
    {"name": "Ben Sheppard", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.549, "underPct": 0.451},
    {"name": "De'Andre Hunter", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.549, "underPct": 0.451},
    {"name": "Isaiah Jackson", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.578, "underPct": 0.422},
    {"name": "Ivica Zubac", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.409, "underPct": 0.591},
    {"name": "Sion James", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.61, "underPct": 0.39},
    {"name": "Matas Buzelis", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.441, "underPct": 0.559},
    {"name": "Tyus Jones", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.571, "underPct": 0.429},
    {"name": "Max Christie", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.382, "underPct": 0.618},
    {"name": "Ryan Nembhard", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.115, "underPct": 0.885},
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.519, "underPct": 0.481},
    {"name": "Aaron Holiday", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.33, "underPct": 0.67},
    {"name": "Steven Adams", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.399, "underPct": 0.601},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Jordan Goodwin", "line": 10.5, "l5": 1.0, "l10": 1.0, "l15": 0.8, "overPct": 1.0, "underPct": 0.0},
    {"name": "Keyonte George", "line": 29.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Gillespie", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cameron Johnson", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Sexton", "line": 20.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Austin Reaves", "line": 32.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deandre Ayton", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Steven Adams", "line": 14.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyle Filipowski", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Jones", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Goga Bitadze", "line": 13.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Davis", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Dunn", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Holiday", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Johnson", "line": 39.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dru Smith", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Svi Mykhailiuk", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "T.J. McConnell", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Caris LeVert", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jay Huff", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tobias Harris", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 41.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Love", "line": 13.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Wiggins", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 44.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 20.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Davion Mitchell", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Dunn", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 12.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Grayson Allen", "line": 20.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 36.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 33.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brice Sensabaugh", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 48.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 21.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Duncan Robinson", "line": 16.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "John Collins", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Nicolas Batum", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 17.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 15.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "LeBron James", "line": 33.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bilal Coulibaly", "line": 18.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Duren", "line": 32.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Porter Jr.", "line": 22.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ausar Thompson", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bennedict Mathurin", "line": 29.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kobe Brown", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 32.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Miller", "line": 26.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Huerter", "line": 18.0, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Nembhard", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyler Herro", "line": 31.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 38.0, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPRHitRates = [
    {"name": "Steven Adams", "line": 12.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 28.5, "l5": 1.0, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Keyonte George", "line": 22.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deandre Ayton", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Svi Mykhailiuk", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 31.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaden Ivey", "line": 10.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Noah Clowney", "line": 21.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 22.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 15.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Gillespie", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Martin", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Dunn", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kon Knueppel", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Wiggins", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sion James", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 20.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Davis", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 24.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "T.J. McConnell", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dillon Brooks", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 7.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Aaron Holiday", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyler Herro", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jay Huff", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyle Filipowski", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brice Sensabaugh", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 14.0, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Grayson Allen", "line": 16.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jake LaRavia", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ben Sheppard", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 29.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Duncan Robinson", "line": 13.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "John Collins", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Nicolas Batum", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 18.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Jackson", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 15.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Duren", "line": 29.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ryan Dunn", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyus Jones", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Nembhard", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Miller", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ryan Kalkbrenner", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Nembhard", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bruce Brown", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 31.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Huerter", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 42.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 30.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kevin Porter Jr.", "line": 17.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPAHitRates = [
    {"name": "Keyonte George", "line": 25.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kyle Kuzma", "line": 13.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Onyeka Okongwu", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Rui Hachimura", "line": 12.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ace Bailey", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cameron Johnson", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tobias Harris", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Gillespie", "line": 16.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Khris Middleton", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mark Williams", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Martin", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Jones", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Davis", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Holiday", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 10.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Royce O'Neale", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Caris LeVert", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "T.J. McConnell", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Duncan Robinson", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jay Huff", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 36.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden Ivey", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Whitmore", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brice Sensabaugh", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 29.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zaccharie Risacher", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Grayson Allen", "line": 17.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Reed Sheppard", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Giannis Antetokounmpo", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylon Tyson", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nicolas Batum", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Miller", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kobe Brown", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bam Adebayo", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Porter Jr.", "line": 17.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bilal Coulibaly", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyler Herro", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 26.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Huerter", "line": 15.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Nembhard", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bennedict Mathurin", "line": 23.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 33.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.0, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const prizepicksRAHitRates = [
    {"name": "Peyton Watson", "line": 7.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 8.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ivica Zubac", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 3.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Lauri Markkanen", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Austin Reaves", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyle Filipowski", "line": 8.0, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 16.0, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Deandre Ayton", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Rollins", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Suggs", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Davis", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dru Smith", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Wiggins", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 13.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Martin", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 11.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jarace Walker", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Royce O'Neale", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 8.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ryan Nembhard", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Alperen Sengun", "line": 15.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cameron Johnson", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Giannis Antetokounmpo", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Dunn", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 12.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bennedict Mathurin", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Porter Jr.", "line": 9.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bilal Coulibaly", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Duren", "line": 14.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Jackson", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kawhi Leonard", "line": 8.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Klay Thompson", "line": 3.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 5.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Kalkbrenner", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tre Jones", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyus Jones", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyler Herro", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kevin Huerter", "line": 6.0, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Nembhard", "line": 9.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Miller", "line": 7.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksTurnoversHitRates = [
    {"name": "Ryan Dunn", "line": 0.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alperen Sengun", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Goga Bitadze", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kobe Brown", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deandre Ayton", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nicolas Batum", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Lauri Markkanen", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Wiggins", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 1.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kevin Porter Jr.", "line": 1.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Jay Huff", "line": 2.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden Ivey", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kawhi Leonard", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kobe Brown", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Williams", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const underdogPointsHitRates = [
    {"name": "Jordan Goodwin", "line": 5.5, "l5": 1.0, "l10": 1.0, "l15": 0.8, "overPct": 0.941, "underPct": 0.059},
    {"name": "Keyonte George", "line": 19.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.926, "underPct": 0.074},
    {"name": "Tyler Herro", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.879, "underPct": 0.121},
    {"name": "Onyeka Okongwu", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.825, "underPct": 0.175},
    {"name": "Kawhi Leonard", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.813, "underPct": 0.187},
    {"name": "Collin Sexton", "line": 13.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.792, "underPct": 0.208},
    {"name": "Lauri Markkanen", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.789, "underPct": 0.211},
    {"name": "James Harden", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.787, "underPct": 0.213},
    {"name": "Dru Smith", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.785, "underPct": 0.215},
    {"name": "Nickeil Alexander-Walker", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.782, "underPct": 0.218},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.778, "underPct": 0.222},
    {"name": "Reed Sheppard", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.778, "underPct": 0.222},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.769, "underPct": 0.231},
    {"name": "Cade Cunningham", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.769, "underPct": 0.231},
    {"name": "Pelle Larsson", "line": 7.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.759, "underPct": 0.241},
    {"name": "Austin Reaves", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.751, "underPct": 0.249},
    {"name": "Caris LeVert", "line": 8.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.75, "underPct": 0.25},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.746, "underPct": 0.254},
    {"name": "Jamal Murray", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.721, "underPct": 0.279},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.72, "underPct": 0.28},
    {"name": "Jalen Duren", "line": 18.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.717, "underPct": 0.283},
    {"name": "Norman Powell", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.714, "underPct": 0.286},
    {"name": "Isaiah Jackson", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.712, "underPct": 0.288},
    {"name": "Duncan Robinson", "line": 11.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.708, "underPct": 0.292},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.706, "underPct": 0.294},
    {"name": "Miles Bridges", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.701, "underPct": 0.299},
    {"name": "Ryan Rollins", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.697, "underPct": 0.303},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.677, "underPct": 0.323},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.67, "underPct": 0.33},
    {"name": "Ayo Dosunmu", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.666, "underPct": 0.334},
    {"name": "Cameron Johnson", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.663, "underPct": 0.337},
    {"name": "Alperen Sengun", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.638, "underPct": 0.362},
    {"name": "Tyrese Martin", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.625, "underPct": 0.375},
    {"name": "LeBron James", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.612, "underPct": 0.388},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.609, "underPct": 0.391},
    {"name": "Dillon Brooks", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.599, "underPct": 0.401},
    {"name": "Andrew Wiggins", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.595, "underPct": 0.405},
    {"name": "Goga Bitadze", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.587, "underPct": 0.413},
    {"name": "Royce O'Neale", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.582, "underPct": 0.418},
    {"name": "Myles Turner", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.564, "underPct": 0.436},
    {"name": "Svi Mykhailiuk", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.555, "underPct": 0.445},
    {"name": "Jaylon Tyson", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.546, "underPct": 0.454},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.539, "underPct": 0.461},
    {"name": "Pascal Siakam", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.535, "underPct": 0.465},
    {"name": "Desmond Bane", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Steven Adams", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.525, "underPct": 0.475},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.518, "underPct": 0.482},
    {"name": "Jalen Johnson", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "T.J. McConnell", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.516, "underPct": 0.484},
    {"name": "Noah Clowney", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.503, "underPct": 0.497},
    {"name": "Kevin Durant", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.502, "underPct": 0.498},
    {"name": "Cam Whitmore", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brice Sensabaugh", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.498, "underPct": 0.502},
    {"name": "Kris Dunn", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.489, "underPct": 0.511},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.489, "underPct": 0.511},
    {"name": "Kyshawn George", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.472, "underPct": 0.528},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.463, "underPct": 0.537},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.462, "underPct": 0.538},
    {"name": "Jaden Ivey", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.441, "underPct": 0.559},
    {"name": "Anthony Davis", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.416, "underPct": 0.584},
    {"name": "Ziaire Williams", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.409, "underPct": 0.591},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.398, "underPct": 0.602},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.385, "underPct": 0.615},
    {"name": "Jay Huff", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.383, "underPct": 0.617},
    {"name": "Franz Wagner", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.37, "underPct": 0.63},
    {"name": "Kobe Brown", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.369, "underPct": 0.631},
    {"name": "Jalen Suggs", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.359, "underPct": 0.641},
    {"name": "Nicolas Batum", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.348, "underPct": 0.652},
    {"name": "De'Andre Hunter", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.294, "underPct": 0.706},
    {"name": "Ausar Thompson", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.283, "underPct": 0.717},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.268, "underPct": 0.732},
    {"name": "John Collins", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.258, "underPct": 0.742},
    {"name": "Brandon Miller", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.247, "underPct": 0.753},
    {"name": "Spencer Jones", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.228, "underPct": 0.772},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.217, "underPct": 0.783},
    {"name": "Khris Middleton", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.214, "underPct": 0.786},
    {"name": "Matas Buzelis", "line": 13.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.169, "underPct": 0.831},
    {"name": "Devin Booker", "line": 26.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.158, "underPct": 0.842},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.156, "underPct": 0.844},
    {"name": "Ryan Nembhard", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.124, "underPct": 0.876},
    {"name": "Kevin Porter Jr.", "line": 13.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.074, "underPct": 0.926},
];const underdogAssistsHitRates = [
    {"name": "Kyshawn George", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.679, "underPct": 0.321},
    {"name": "Caris LeVert", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.677, "underPct": 0.323},
    {"name": "Ivica Zubac", "line": 2.5, "l5": 1.0, "l10": 0.5, "l15": 0.47, "overPct": 0.668, "underPct": 0.332},
    {"name": "Andrew Wiggins", "line": 2.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.66, "underPct": 0.34},
    {"name": "Isaiah Collier", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.632, "underPct": 0.368},
    {"name": "Ryan Rollins", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.617, "underPct": 0.383},
    {"name": "Cooper Flagg", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.572, "underPct": 0.428},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.526, "underPct": 0.474},
    {"name": "Brandon Williams", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.519, "underPct": 0.481},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.495, "underPct": 0.505},
    {"name": "De'Andre Hunter", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.455, "underPct": 0.545},
    {"name": "T.J. McConnell", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.437, "underPct": 0.563},
    {"name": "Dru Smith", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.436, "underPct": 0.564},
    {"name": "Jalen Suggs", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.428, "underPct": 0.572},
    {"name": "Ausar Thompson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.424, "underPct": 0.576},
    {"name": "Kevin Love", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Evan Mobley", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.379, "underPct": 0.621},
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.356, "underPct": 0.644},
    {"name": "Kevin Huerter", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.341, "underPct": 0.659},
    {"name": "Amen Thompson", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.307, "underPct": 0.693},
    {"name": "Ryan Nembhard", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.254, "underPct": 0.746},
];const underdogReboundsHitRates = [
    {"name": "P.J. Washington", "line": 6.5, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.725, "underPct": 0.275},
    {"name": "Peyton Watson", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.689, "underPct": 0.311},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.651, "underPct": 0.349},
    {"name": "Lauri Markkanen", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.611, "underPct": 0.389},
    {"name": "Jaden Ivey", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.605, "underPct": 0.395},
    {"name": "Anthony Davis", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.57, "underPct": 0.43},
    {"name": "Franz Wagner", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.563, "underPct": 0.437},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.517, "underPct": 0.483},
    {"name": "Brandon Williams", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.483, "underPct": 0.517},
    {"name": "Bennedict Mathurin", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.473, "underPct": 0.527},
    {"name": "De'Andre Hunter", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "Collin Gillespie", "line": 3.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.424, "underPct": 0.576},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.422, "underPct": 0.578},
    {"name": "Tyrese Martin", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.411, "underPct": 0.589},
    {"name": "Evan Mobley", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.403, "underPct": 0.597},
    {"name": "Isaiah Jackson", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.399, "underPct": 0.601},
    {"name": "Miles Bridges", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.378, "underPct": 0.622},
    {"name": "Aaron Holiday", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.163, "underPct": 0.837},
    {"name": "Kobe Brown", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.145, "underPct": 0.855},
];const underdogBlocksHitRates = [
    {"name": "Myles Turner", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.476, "underPct": 0.524},
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.519, "underPct": 0.481},
    {"name": "Ryan Kalkbrenner", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.419, "underPct": 0.581},
    {"name": "Anthony Davis", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.441, "underPct": 0.559},
];const underdogStealsHitRates = [
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.519, "underPct": 0.481},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Jordan Goodwin", "line": 10.5, "l5": 1.0, "l10": 1.0, "l15": 0.8, "overPct": 1.0, "underPct": 0.0},
    {"name": "Keyonte George", "line": 29.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Mark Williams", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Steven Adams", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 32.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deandre Ayton", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Peyton Watson", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Khris Middleton", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaden Ivey", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 44.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dru Smith", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Reed Sheppard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bobby Portis", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Williams", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Jones", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Davis", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyshawn George", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kel'el Ware", "line": 20.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jay Huff", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tobias Harris", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Caris LeVert", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 40.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Dunn", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Holiday", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Wiggins", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Collier", "line": 12.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jarace Walker", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kawhi Leonard", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 41.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brice Sensabaugh", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 33.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alperen Sengun", "line": 36.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Spencer Jones", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Royce O'Neale", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Grayson Allen", "line": 20.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 48.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ziaire Williams", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Day'Ron Sharpe", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nicolas Batum", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 25.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Kalkbrenner", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 29.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 32.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Porter Jr.", "line": 22.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 33.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ryan Nembhard", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyler Herro", "line": 31.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 31.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Miller", "line": 26.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Huerter", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Jackson", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 38.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const underdogPRHitRates = [
    {"name": "Mark Williams", "line": 20.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 28.5, "l5": 1.0, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Keyonte George", "line": 22.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Deandre Ayton", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 31.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "P.J. Washington", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Noah Clowney", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kawhi Leonard", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dillon Brooks", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyler Herro", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Davis", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alperen Sengun", "line": 30.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Franz Wagner", "line": 31.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Miller", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "LaMelo Ball", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jalen Duren", "line": 29.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Giannis Antetokounmpo", "line": 42.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 30.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
];const underdogPAHitRates = [
    {"name": "Keyonte George", "line": 25.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 19.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jamal Murray", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Davis", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 34.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Johnson", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Duren", "line": 20.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Giannis Antetokounmpo", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Miller", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 28.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 26.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyler Herro", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Andrew Nembhard", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bam Adebayo", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 23.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 33.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const underdogRAHitRates = [
    {"name": "P.J. Washington", "line": 8.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Keyonte George", "line": 9.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Peyton Watson", "line": 7.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ivica Zubac", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deandre Ayton", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Khris Middleton", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Davis", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kel'el Ware", "line": 9.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Pascal Siakam", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ausar Thompson", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Reed Sheppard", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Giannis Antetokounmpo", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Day'Ron Sharpe", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyler Herro", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const underdogTurnoversHitRates = [
    {"name": "Alperen Sengun", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const underdogBlocksStealsHitRates = [
    {"name": "Jay Huff", "line": 2.5, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Davis", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
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

