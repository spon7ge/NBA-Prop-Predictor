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
    {"name1": "Sandro Mamukelashvili", "name2": "Jaden Ivey", "line1": 8.5, "line2": 8.5, "prediction1": 11.86, "prediction2": 11.83, "side1": "over", "side2": "over", "recommendation": 0, "ev": 60.19, "kelly": 0.301, "sigma1": "Med", "sigma2": "Low", "prob1": 0.721, "prob2": 0.756, "hitRate1": 82.3, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 95.8, "l5_2": 0.2, "l15_2": 0.07},
    {"name1": "Ben Sheppard", "name2": "Pelle Larsson", "line1": 6.5, "line2": 12.5, "prediction1": 9.53, "prediction2": 15.66, "side1": "over", "side2": "over", "recommendation": 0, "ev": 38.51, "kelly": 0.193, "sigma1": "Med", "sigma2": "High", "prob1": 0.693, "prob2": 0.68, "hitRate1": 31.4, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 30.6, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Tobias Harris", "name2": "Miles McBride", "line1": 11.5, "line2": 10.5, "prediction1": 14.55, "prediction2": 13.76, "side1": "over", "side2": "over", "recommendation": 0, "ev": 31.93, "kelly": 0.16, "sigma1": "High", "sigma2": "High", "prob1": 0.662, "prob2": 0.677, "hitRate1": 76.3, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 50.6, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Noah Clowney", "name2": "Cooper Flagg", "line1": 13.5, "line2": 15.5, "prediction1": 16.39, "prediction2": 18.1, "side1": "over", "side2": "over", "recommendation": 0, "ev": 22.9, "kelly": 0.115, "sigma1": "High", "sigma2": "High", "prob1": 0.653, "prob2": 0.64, "hitRate1": 47.5, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 77.6, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Collin Murray-Boyles", "name2": "Ausar Thompson", "line1": 6.5, "line2": 11.0, "prediction1": 8.39, "prediction2": 13.49, "side1": "over", "side2": "over", "recommendation": 0, "ev": 20.16, "kelly": 0.101, "sigma1": "Med", "sigma2": "High", "prob1": 0.64, "prob2": 0.639, "hitRate1": 39.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 50.2, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Bennedict Mathurin", "name2": "Daniel Gafford", "line1": 21.5, "line2": 10.5, "prediction1": 24.01, "prediction2": 8.78, "side1": "over", "side2": "under", "recommendation": 0, "ev": 16.45, "kelly": 0.082, "sigma1": "High", "sigma2": "Med", "prob1": 0.636, "prob2": 0.623, "hitRate1": 83.9, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 54.1, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Jaylon Tyson", "name2": "Davion Mitchell", "line1": 13.5, "line2": 10.5, "prediction1": 11.63, "prediction2": 12.29, "side1": "under", "side2": "over", "recommendation": 0, "ev": 9.83, "kelly": 0.049, "sigma1": "High", "sigma2": "High", "prob1": 0.61, "prob2": 0.613, "hitRate1": 64.7, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 63.1, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Andrew Nembhard", "name2": "Karl-Anthony Towns", "line1": 16.5, "line2": 23.5, "prediction1": 18.46, "prediction2": 21.57, "side1": "over", "side2": "under", "recommendation": 0, "ev": 7.93, "kelly": 0.04, "sigma1": "High", "sigma2": "High", "prob1": 0.607, "prob2": 0.605, "hitRate1": 80.7, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 45.5, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Jordan Clarkson", "name2": "Simone Fontecchio", "line1": 13.0, "line2": 11.5, "prediction1": 11.45, "prediction2": 13.16, "side1": "under", "side2": "over", "recommendation": 0, "ev": 5.33, "kelly": 0.027, "sigma1": "High", "sigma2": "High", "prob1": 0.598, "prob2": 0.599, "hitRate1": 66.9, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 31.7, "l5_2": 0.2, "l15_2": 0.33},
    {"name1": "Gradey Dick", "name2": "Josh Hart", "line1": 8.5, "line2": 12.5, "prediction1": 9.78, "prediction2": 11.05, "side1": "over", "side2": "under", "recommendation": 0, "ev": 3.63, "kelly": 0.018, "sigma1": "Med", "sigma2": "Med", "prob1": 0.591, "prob2": 0.596, "hitRate1": 33.0, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 60.0, "l5_2": 0.4, "l15_2": 0.27},
];const prizepicksTriosData = [
    {"name1": "Sandro Mamukelashvili", "name2": "Jaden Ivey", "name3": "Pelle Larsson", "line1": 8.5, "line2": 8.5, "line3": 12.5, "prediction1": 11.86, "prediction2": 11.83, "prediction3": 15.66, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 100.0, "kelly": 0.2, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.721, "prob2": 0.756, "prob3": 0.68, "hitRate1": 82.3, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 95.8, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 30.6, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Ben Sheppard", "name2": "Miles McBride", "name3": "Cooper Flagg", "line1": 6.5, "line2": 10.5, "line3": 15.5, "prediction1": 9.53, "prediction2": 13.76, "prediction3": 18.1, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 62.27, "kelly": 0.125, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.693, "prob2": 0.677, "prob3": 0.64, "hitRate1": 31.4, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 50.6, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 77.6, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Collin Murray-Boyles", "name2": "Tobias Harris", "name3": "Noah Clowney", "line1": 6.5, "line2": 11.5, "line3": 13.5, "prediction1": 8.39, "prediction2": 14.55, "prediction3": 16.39, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 49.43, "kelly": 0.099, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.64, "prob2": 0.662, "prob3": 0.653, "hitRate1": 39.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 76.3, "l5_2": 0.6, "l15_2": 0.2, "hitRate3": 47.5, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Jaylon Tyson", "name2": "Ausar Thompson", "name3": "Daniel Gafford", "line1": 13.5, "line2": 11.0, "line3": 10.5, "prediction1": 11.63, "prediction2": 13.49, "prediction3": 8.78, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": 31.0, "kelly": 0.062, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.61, "prob2": 0.639, "prob3": 0.623, "hitRate1": 64.7, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 50.2, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 54.1, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Bennedict Mathurin", "name2": "Andrew Nembhard", "name3": "Davion Mitchell", "line1": 21.5, "line2": 16.5, "line3": 10.5, "prediction1": 24.01, "prediction2": 18.46, "prediction3": 12.29, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 27.69, "kelly": 0.055, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.636, "prob2": 0.607, "prob3": 0.613, "hitRate1": 83.9, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 80.7, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 63.1, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Karl-Anthony Towns", "name2": "Jordan Clarkson", "name3": "Simone Fontecchio", "line1": 23.5, "line2": 13.0, "line3": 11.5, "prediction1": 21.57, "prediction2": 11.45, "prediction3": 13.16, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 17.09, "kelly": 0.034, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.605, "prob2": 0.598, "prob3": 0.599, "hitRate1": 45.5, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 66.9, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 31.7, "l5_3": 0.2, "l15_3": 0.33},
    {"name1": "Gradey Dick", "name2": "Josh Hart", "name3": "Klay Thompson", "line1": 8.5, "line2": 12.5, "line3": 10.5, "prediction1": 9.78, "prediction2": 11.05, "prediction3": 12.01, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 11.81, "kelly": 0.024, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.591, "prob2": 0.596, "prob3": 0.587, "hitRate1": 33.0, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 60.0, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 75.7, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Scottie Barnes", "name2": "Tyrese Martin", "name3": "D'Angelo Russell", "line1": 20.5, "line2": 8.5, "line3": 12.5, "prediction1": 19.1, "prediction2": 9.83, "prediction3": 13.42, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 0.94, "kelly": 0.002, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.58, "prob2": 0.586, "prob3": 0.55, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 61.6, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 48.9, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Dean Wade", "name2": "Day'Ron Sharpe", "name3": "Max Christie", "line1": 6.5, "line2": 6.5, "line3": 11.5, "prediction1": 7.46, "prediction2": 7.11, "prediction3": 12.14, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": -8.33, "kelly": 0.0, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.57, "prob2": 0.548, "prob3": 0.543, "hitRate1": 18.2, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 58.0, "l5_2": 0.2, "l15_2": 0.33, "hitRate3": 61.7, "l5_3": 0.6, "l15_3": 0.73},
    {"name1": "Jakob Poeltl", "name2": "Ja'Kobe Walter", "name3": "Pascal Siakam", "line1": 12.0, "line2": 7.5, "line3": 23.5, "prediction1": 12.72, "prediction2": 6.92, "prediction3": 24.28, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": -12.56, "kelly": 0.0, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.548, "prob2": 0.547, "prob3": 0.541, "hitRate1": 70.7, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 78.8, "l5_2": 0.4, "l15_2": 0.2, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Jaden Ivey", "name2": "Pelle Larsson", "line1": 8.5, "line2": 12.5, "prediction1": 11.83, "prediction2": 15.66, "side1": "over", "side2": "over", "recommendation": 0, "ev": 51.02, "kelly": 0.255, "sigma1": "Low", "sigma2": "High", "prob1": 0.756, "prob2": 0.68, "hitRate1": 95.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 30.6, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Evan Mobley", "name2": "Miles McBride", "line1": 21.5, "line2": 10.5, "prediction1": 18.18, "prediction2": 13.76, "side1": "under", "side2": "over", "recommendation": 0, "ev": 35.23, "kelly": 0.176, "sigma1": "High", "sigma2": "High", "prob1": 0.679, "prob2": 0.677, "hitRate1": 81.8, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 50.6, "l5_2": 0.4, "l15_2": 0.27},
    {"name1": "Bennedict Mathurin", "name2": "Daniel Gafford", "line1": 21.5, "line2": 10.5, "prediction1": 24.01, "prediction2": 8.78, "side1": "over", "side2": "under", "recommendation": 0, "ev": 16.45, "kelly": 0.082, "sigma1": "High", "sigma2": "Med", "prob1": 0.636, "prob2": 0.623, "hitRate1": 83.9, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 54.1, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Duncan Robinson", "name2": "Davion Mitchell", "line1": 10.5, "line2": 10.5, "prediction1": 12.75, "prediction2": 12.29, "side1": "over", "side2": "over", "recommendation": 0, "ev": 14.34, "kelly": 0.072, "sigma1": "High", "sigma2": "High", "prob1": 0.635, "prob2": 0.613, "hitRate1": 90.0, "l5_1": 1.0, "l15_1": 0.73, "hitRate2": 63.1, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Andrew Nembhard", "name2": "Karl-Anthony Towns", "line1": 16.5, "line2": 23.5, "prediction1": 18.46, "prediction2": 21.57, "side1": "over", "side2": "under", "recommendation": 0, "ev": 7.93, "kelly": 0.04, "sigma1": "High", "sigma2": "High", "prob1": 0.607, "prob2": 0.605, "hitRate1": 80.7, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 45.5, "l5_2": 0.4, "l15_2": 0.33},
    {"name1": "Josh Hart", "name2": "Simone Fontecchio", "line1": 12.5, "line2": 11.5, "prediction1": 11.05, "prediction2": 13.16, "side1": "under", "side2": "over", "recommendation": 0, "ev": 4.97, "kelly": 0.025, "sigma1": "Med", "sigma2": "High", "prob1": 0.596, "prob2": 0.599, "hitRate1": 60.0, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 31.7, "l5_2": 0.2, "l15_2": 0.33},
    {"name1": "Gradey Dick", "name2": "Klay Thompson", "line1": 8.5, "line2": 10.5, "prediction1": 9.78, "prediction2": 12.01, "side1": "over", "side2": "over", "recommendation": 0, "ev": 2.1, "kelly": 0.011, "sigma1": "Med", "sigma2": "High", "prob1": 0.591, "prob2": 0.587, "hitRate1": 33.0, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 75.7, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Donovan Mitchell", "name2": "Tyrese Martin", "line1": 30.5, "line2": 8.5, "prediction1": 29.14, "prediction2": 9.83, "side1": "under", "side2": "over", "recommendation": 0, "ev": -1.31, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.573, "prob2": 0.586, "hitRate1": 46.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 61.6, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Cade Cunningham", "name2": "Bam Adebayo", "line1": 26.5, "line2": 21.5, "prediction1": 27.43, "prediction2": 20.48, "side1": "over", "side2": "under", "recommendation": 0, "ev": -7.96, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.56, "prob2": 0.559, "hitRate1": 80.6, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 71.0, "l5_2": 0.2, "l15_2": 0.27},
    {"name1": "Day'Ron Sharpe", "name2": "D'Angelo Russell", "line1": 6.5, "line2": 12.5, "prediction1": 7.11, "prediction2": 13.42, "side1": "over", "side2": "over", "recommendation": 0, "ev": -11.31, "kelly": 0.0, "sigma1": "Low", "sigma2": "High", "prob1": 0.548, "prob2": 0.55, "hitRate1": 58.0, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 48.9, "l5_2": 0.4, "l15_2": 0.33},
];const underdogTriosData = [
    {"name1": "Evan Mobley", "name2": "Jaden Ivey", "name3": "Pelle Larsson", "line1": 21.5, "line2": 8.5, "line3": 12.5, "prediction1": 18.18, "prediction2": 11.83, "prediction3": 15.66, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 88.35, "kelly": 0.177, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "prob1": 0.679, "prob2": 0.756, "prob3": 0.68, "hitRate1": 81.8, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 95.8, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 30.6, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Bennedict Mathurin", "name2": "Miles McBride", "name3": "Daniel Gafford", "line1": 21.5, "line2": 10.5, "line3": 10.5, "prediction1": 24.01, "prediction2": 13.76, "prediction3": 8.78, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 44.88, "kelly": 0.09, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.636, "prob2": 0.677, "prob3": 0.623, "hitRate1": 83.9, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 50.6, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 54.1, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Duncan Robinson", "name2": "Karl-Anthony Towns", "name3": "Davion Mitchell", "line1": 10.5, "line2": 23.5, "line3": 10.5, "prediction1": 12.75, "prediction2": 21.57, "prediction3": 12.29, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 27.1, "kelly": 0.054, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.635, "prob2": 0.605, "prob3": 0.613, "hitRate1": 90.0, "l5_1": 1.0, "l15_1": 0.73, "hitRate2": 45.5, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 63.1, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Andrew Nembhard", "name2": "Josh Hart", "name3": "Simone Fontecchio", "line1": 16.5, "line2": 12.5, "line3": 11.5, "prediction1": 18.46, "prediction2": 11.05, "prediction3": 13.16, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 16.95, "kelly": 0.034, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.607, "prob2": 0.596, "prob3": 0.599, "hitRate1": 80.7, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 60.0, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 31.7, "l5_3": 0.2, "l15_3": 0.33},
    {"name1": "Gradey Dick", "name2": "Tyrese Martin", "name3": "Klay Thompson", "line1": 8.5, "line2": 8.5, "line3": 10.5, "prediction1": 9.78, "prediction2": 9.83, "prediction3": 12.01, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.87, "kelly": 0.02, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.591, "prob2": 0.586, "prob3": 0.587, "hitRate1": 33.0, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 61.6, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 75.7, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Donovan Mitchell", "name2": "Cade Cunningham", "name3": "Bam Adebayo", "line1": 30.5, "line2": 26.5, "line3": 21.5, "prediction1": 29.14, "prediction2": 27.43, "prediction3": 20.48, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": -3.13, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.573, "prob2": 0.56, "prob3": 0.559, "hitRate1": 46.3, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 80.6, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 71.0, "l5_3": 0.2, "l15_3": 0.27},
    {"name1": "Ja'Kobe Walter", "name2": "Day'Ron Sharpe", "name3": "D'Angelo Russell", "line1": 7.5, "line2": 6.5, "line3": 12.5, "prediction1": 6.92, "prediction2": 7.11, "prediction3": 13.42, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": -10.96, "kelly": 0.0, "sigma1": "Low", "sigma2": "Low", "sigma3": "High", "prob1": 0.547, "prob2": 0.548, "prob3": 0.55, "hitRate1": 78.8, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 58.0, "l5_2": 0.2, "l15_2": 0.33, "hitRate3": 48.9, "l5_3": 0.4, "l15_3": 0.33},
    {"name1": "Brandon Ingram", "name2": "Pascal Siakam", "name3": "Mikal Bridges", "line1": 22.5, "line2": 23.5, "line3": 16.5, "prediction1": 21.83, "prediction2": 24.28, "prediction3": 15.83, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": -15.63, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.537, "prob2": 0.541, "prob3": 0.538, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 63.6, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 50.8, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Scottie Barnes", "name2": "Terance Mann", "name3": "Drake Powell", "line1": 19.5, "line2": 8.5, "line3": 6.5, "prediction1": 19.1, "prediction2": 8.86, "prediction3": 6.08, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": -21.03, "kelly": 0.0, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.523, "prob2": 0.525, "prob3": 0.533, "hitRate1": 51.1, "l5_1": 0.2, "l15_1": 0.33, "hitRate2": 20.1, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 44.0, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Jakob Poeltl", "name2": "Jalen Brunson", "name3": "Brandon Williams", "line1": 12.5, "line2": 27.5, "line3": 13.5, "prediction1": 12.72, "prediction2": 28.05, "prediction3": 13.26, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": -24.4, "kelly": 0.0, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.515, "prob2": 0.528, "prob3": 0.515, "hitRate1": 70.7, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 36.2, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 71.8, "l5_3": 0.6, "l15_3": 0.47},
];const prizepicksPointsHitRates = [
    {"name": "Jaden Ivey", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.958, "underPct": 0.042},
    {"name": "Tre Jones", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.862, "underPct": 0.138},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.839, "underPct": 0.161},
    {"name": "Sandro Mamukelashvili", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.823, "underPct": 0.177},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.807, "underPct": 0.193},
    {"name": "Keegan Murray", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.79, "underPct": 0.21},
    {"name": "Cooper Flagg", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.776, "underPct": 0.224},
    {"name": "Dillon Brooks", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.766, "underPct": 0.234},
    {"name": "Tobias Harris", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.763, "underPct": 0.237},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.757, "underPct": 0.243},
    {"name": "Cade Cunningham", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.751, "underPct": 0.249},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.725, "underPct": 0.275},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.716, "underPct": 0.284},
    {"name": "Pascal Siakam", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.709, "underPct": 0.291},
    {"name": "Naz Reid", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.708, "underPct": 0.292},
    {"name": "Jakob Poeltl", "line": 12.0, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.707, "underPct": 0.293},
    {"name": "Jaden McDaniels", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.703, "underPct": 0.297},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.688, "underPct": 0.312},
    {"name": "Kevin Huerter", "line": 9.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.684, "underPct": 0.316},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.679, "underPct": 0.321},
    {"name": "Ace Bailey", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.679, "underPct": 0.321},
    {"name": "Ayo Dosunmu", "line": 14.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.644, "underPct": 0.356},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.639, "underPct": 0.361},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.639, "underPct": 0.361},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.631, "underPct": 0.369},
    {"name": "Stephen Curry", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.628, "underPct": 0.372},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.617, "underPct": 0.383},
    {"name": "Tyrese Martin", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.616, "underPct": 0.384},
    {"name": "Kel'el Ware", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.607, "underPct": 0.393},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.581, "underPct": 0.419},
    {"name": "Day'Ron Sharpe", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.58, "underPct": 0.42},
    {"name": "Saddiq Bey", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.574, "underPct": 0.426},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.568, "underPct": 0.432},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.561, "underPct": 0.439},
    {"name": "Karl-Anthony Towns", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.545, "underPct": 0.455},
    {"name": "Jarace Walker", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.524, "underPct": 0.476},
    {"name": "Derik Queen", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.512, "underPct": 0.488},
    {"name": "Brice Sensabaugh", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.508, "underPct": 0.492},
    {"name": "Miles McBride", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.506, "underPct": 0.494},
    {"name": "Brandin Podziemski", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.505, "underPct": 0.495},
    {"name": "Ausar Thompson", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.502, "underPct": 0.498},
    {"name": "Mikal Bridges", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.492, "underPct": 0.508},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Noah Clowney", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.475, "underPct": 0.525},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.462, "underPct": 0.538},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.459, "underPct": 0.541},
    {"name": "Moses Moody", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.457, "underPct": 0.543},
    {"name": "Jay Huff", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.457, "underPct": 0.543},
    {"name": "Cam Spencer", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.447, "underPct": 0.553},
    {"name": "Kyle Filipowski", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.438, "underPct": 0.562},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.435, "underPct": 0.565},
    {"name": "Will Richard", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Jose Alvarado", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.429, "underPct": 0.571},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.428, "underPct": 0.572},
    {"name": "Kyle Kuzma", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.428, "underPct": 0.572},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.426, "underPct": 0.574},
    {"name": "Jaylen Wells", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.416, "underPct": 0.584},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.414, "underPct": 0.586},
    {"name": "Cedric Coward", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.414, "underPct": 0.586},
    {"name": "Rob Dillingham", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.41, "underPct": 0.59},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Murray-Boyles", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.391, "underPct": 0.609},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.388, "underPct": 0.612},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.382, "underPct": 0.618},
    {"name": "Ziaire Williams", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.382, "underPct": 0.618},
    {"name": "Brandon Ingram", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.38, "underPct": 0.62},
    {"name": "Jordan Goodwin", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.378, "underPct": 0.622},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.369, "underPct": 0.631},
    {"name": "Alperen Sengun", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.365, "underPct": 0.635},
    {"name": "Jalen Brunson", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.362, "underPct": 0.638},
    {"name": "Zach LaVine", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.355, "underPct": 0.645},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.355, "underPct": 0.645},
    {"name": "Jaylon Tyson", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.353, "underPct": 0.647},
    {"name": "Josh Giddey", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.331, "underPct": 0.669},
    {"name": "Jordan Clarkson", "line": 13.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.331, "underPct": 0.669},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.33, "underPct": 0.67},
    {"name": "Ryan Rollins", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.322, "underPct": 0.678},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.321, "underPct": 0.679},
    {"name": "Simone Fontecchio", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.317, "underPct": 0.683},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.314, "underPct": 0.686},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.306, "underPct": 0.694},
    {"name": "Myles Turner", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.302, "underPct": 0.698},
    {"name": "Peyton Watson", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.293, "underPct": 0.707},
    {"name": "Devin Booker", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.289, "underPct": 0.711},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.282, "underPct": 0.718},
    {"name": "Lonzo Ball", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.274, "underPct": 0.726},
    {"name": "Reed Sheppard", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.269, "underPct": 0.731},
    {"name": "Coby White", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.259, "underPct": 0.741},
    {"name": "Zion Williamson", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.257, "underPct": 0.743},
    {"name": "Sidy Cissoko", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.214, "underPct": 0.786},
    {"name": "Kris Murray", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.213, "underPct": 0.787},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.212, "underPct": 0.788},
    {"name": "Ja'Kobe Walter", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.212, "underPct": 0.788},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.205, "underPct": 0.795},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.202, "underPct": 0.798},
    {"name": "Terance Mann", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.201, "underPct": 0.799},
    {"name": "Amen Thompson", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.201, "underPct": 0.799},
    {"name": "Dean Wade", "line": 6.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.182, "underPct": 0.818},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.165, "underPct": 0.835},
    {"name": "Buddy Hield", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.134, "underPct": 0.866},
    {"name": "Josh Okogie", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.074, "underPct": 0.926},
    {"name": "Cole Anthony", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.037, "underPct": 0.963},
    {"name": "Nick Richards", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.025, "underPct": 0.975},
    {"name": "Trayce Jackson-Davis", "line": 9.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.021, "underPct": 0.979},
];const prizepicksAssistsHitRates = [
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.703, "underPct": 0.297},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.691, "underPct": 0.309},
    {"name": "Duncan Robinson", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.679, "underPct": 0.321},
    {"name": "Pelle Larsson", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.646, "underPct": 0.354},
    {"name": "Kyle Filipowski", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.632, "underPct": 0.368},
    {"name": "Isaiah Collier", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.604, "underPct": 0.396},
    {"name": "Davion Mitchell", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.604, "underPct": 0.396},
    {"name": "Terance Mann", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.603, "underPct": 0.397},
    {"name": "Jamal Shead", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.597, "underPct": 0.403},
    {"name": "Coby White", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.568, "underPct": 0.432},
    {"name": "Jamal Murray", "line": 7.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.538, "underPct": 0.462},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.529, "underPct": 0.471},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.52, "underPct": 0.48},
    {"name": "Cole Anthony", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.516, "underPct": 0.484},
    {"name": "Oso Ighodaro", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.506, "underPct": 0.494},
    {"name": "Tre Jones", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.499, "underPct": 0.501},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.484, "underPct": 0.516},
    {"name": "Kyle Kuzma", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.476, "underPct": 0.524},
    {"name": "Brandon Williams", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.462, "underPct": 0.538},
    {"name": "Zion Williamson", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.441, "underPct": 0.559},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.438, "underPct": 0.562},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.431, "underPct": 0.569},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.421, "underPct": 0.579},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.419, "underPct": 0.581},
    {"name": "D'Angelo Russell", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.417, "underPct": 0.583},
    {"name": "Alperen Sengun", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.411, "underPct": 0.589},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.391, "underPct": 0.609},
    {"name": "Devin Booker", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.355, "underPct": 0.645},
    {"name": "Pascal Siakam", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.337, "underPct": 0.663},
    {"name": "Lonzo Ball", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.335, "underPct": 0.665},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.277, "underPct": 0.723},
    {"name": "Sidy Cissoko", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.254, "underPct": 0.746},
    {"name": "Cooper Flagg", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.246, "underPct": 0.754},
    {"name": "Will Richard", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.24, "underPct": 0.76},
    {"name": "Deni Avdija", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.228, "underPct": 0.772},
    {"name": "Brandin Podziemski", "line": 3.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.215, "underPct": 0.785},
    {"name": "Reed Sheppard", "line": 4.0, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.177, "underPct": 0.823},
    {"name": "Stephen Curry", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.099, "underPct": 0.901},
];const prizepicksReboundsHitRates = [
    {"name": "Donovan Mitchell", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.837, "underPct": 0.163},
    {"name": "Saddiq Bey", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.723, "underPct": 0.277},
    {"name": "Kel'el Ware", "line": 11.0, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.699, "underPct": 0.301},
    {"name": "Keegan Murray", "line": 5.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.65, "underPct": 0.35},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.641, "underPct": 0.359},
    {"name": "Zach Edey", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.623, "underPct": 0.377},
    {"name": "Cedric Coward", "line": 5.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.599, "underPct": 0.401},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.591, "underPct": 0.409},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.587, "underPct": 0.413},
    {"name": "Matas Buzelis", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.585, "underPct": 0.415},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.582, "underPct": 0.418},
    {"name": "Jalen Duren", "line": 12.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.576, "underPct": 0.424},
    {"name": "Bennedict Mathurin", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.568, "underPct": 0.432},
    {"name": "Josh Giddey", "line": 9.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.557, "underPct": 0.443},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.55, "underPct": 0.45},
    {"name": "Daniel Gafford", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.537, "underPct": 0.463},
    {"name": "Jeremiah Fears", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.535, "underPct": 0.465},
    {"name": "Tobias Harris", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.525, "underPct": 0.475},
    {"name": "Naz Reid", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.511, "underPct": 0.489},
    {"name": "Toumani Camara", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.505, "underPct": 0.495},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Karl-Anthony Towns", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Pelle Larsson", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.48, "underPct": 0.52},
    {"name": "Cade Cunningham", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.48, "underPct": 0.52},
    {"name": "Lauri Markkanen", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.478, "underPct": 0.522},
    {"name": "Cooper Flagg", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.477, "underPct": 0.523},
    {"name": "Zach LaVine", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.469, "underPct": 0.531},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.468, "underPct": 0.532},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.454, "underPct": 0.546},
    {"name": "Zion Williamson", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.449, "underPct": 0.551},
    {"name": "Rudy Gobert", "line": 11.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.448, "underPct": 0.552},
    {"name": "Jaden McDaniels", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.446, "underPct": 0.554},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.44, "underPct": 0.56},
    {"name": "Isaiah Jackson", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Svi Mykhailiuk", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.422, "underPct": 0.578},
    {"name": "Alperen Sengun", "line": 10.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.422, "underPct": 0.578},
    {"name": "Ryan Rollins", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.42, "underPct": 0.58},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.419, "underPct": 0.581},
    {"name": "Derik Queen", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.415, "underPct": 0.585},
    {"name": "Deni Avdija", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.408, "underPct": 0.592},
    {"name": "Kyle Kuzma", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.407, "underPct": 0.593},
    {"name": "Day'Ron Sharpe", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.395, "underPct": 0.605},
    {"name": "Collin Gillespie", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.392, "underPct": 0.608},
    {"name": "Pascal Siakam", "line": 6.0, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.391, "underPct": 0.609},
    {"name": "Sandro Mamukelashvili", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.375, "underPct": 0.625},
    {"name": "Simone Fontecchio", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.37, "underPct": 0.63},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.358, "underPct": 0.642},
    {"name": "Moses Moody", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.357, "underPct": 0.643},
    {"name": "Ausar Thompson", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.352, "underPct": 0.648},
    {"name": "Peyton Watson", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.344, "underPct": 0.656},
    {"name": "Amen Thompson", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.339, "underPct": 0.661},
    {"name": "Will Richard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.336, "underPct": 0.664},
    {"name": "Dean Wade", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.305, "underPct": 0.695},
    {"name": "Royce O'Neale", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.293, "underPct": 0.707},
    {"name": "Bruce Brown", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.289, "underPct": 0.711},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.281, "underPct": 0.719},
    {"name": "Cameron Johnson", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.278, "underPct": 0.722},
    {"name": "Ace Bailey", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.275, "underPct": 0.725},
    {"name": "Stephen Curry", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.245, "underPct": 0.755},
    {"name": "Bobby Portis", "line": 8.0, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.2, "underPct": 0.8},
    {"name": "Noah Clowney", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.188, "underPct": 0.812},
    {"name": "Quinten Post", "line": 5.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.175, "underPct": 0.825},
    {"name": "Nick Richards", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.071, "underPct": 0.929},
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
    {"name": "Isaiah Jackson", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.676, "underPct": 0.324},
    {"name": "Tobias Harris", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.52, "underPct": 0.48},
    {"name": "Jordan Clarkson", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.378, "underPct": 0.622},
    {"name": "Terance Mann", "line": 0.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.285, "underPct": 0.715},
    {"name": "Ziaire Williams", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.406, "underPct": 0.594},
    {"name": "Daniel Gafford", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.586, "underPct": 0.414},
    {"name": "Max Christie", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.542, "underPct": 0.458},
    {"name": "D'Angelo Russell", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.396, "underPct": 0.604},
    {"name": "Bruce Brown", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.637, "underPct": 0.363},
    {"name": "Cedric Coward", "line": 0.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.367, "underPct": 0.633},
    {"name": "Spencer Jones", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.402, "underPct": 0.598},
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.538, "underPct": 0.462},
    {"name": "Oso Ighodaro", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.512, "underPct": 0.488},
    {"name": "Isaiah Collier", "line": 0.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.314, "underPct": 0.686},
    {"name": "Rudy Gobert", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.408, "underPct": 0.592},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Keyonte George", "line": 30.5, "l5": 1.0, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dillon Brooks", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Filipowski", "line": 16.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Goodwin", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandon Williams", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandin Podziemski", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naji Marshall", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pelle Larsson", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dru Smith", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tre Jones", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 34.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alperen Sengun", "line": 41.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 24.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 36.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 33.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 41.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ausar Thompson", "line": 19.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Noah Clowney", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Murray-Boyles", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 12.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ziaire Williams", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lonzo Ball", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Edwards", "line": 38.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 34.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bobby Portis", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sidy Cissoko", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brice Sensabaugh", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 26.0, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Shead", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylon Tyson", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 22.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Spencer", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Terance Mann", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles McBride", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 31.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 29.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Huerter", "line": 15.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Drew Eubanks", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Day'Ron Sharpe", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Aaron Holiday", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Josh Okogie", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Simone Fontecchio", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zion Williamson", "line": 35.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Jackson", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cole Anthony", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 37.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Royce O'Neale", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaden Ivey", "line": 14.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Scottie Barnes", "line": 33.0, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 32.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const prizepicksPRHitRates = [
    {"name": "Duncan Robinson", "line": 13.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tobias Harris", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Brunson", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Stephen Curry", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Goodwin", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Davion Mitchell", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Jones", "line": 11.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Saddiq Bey", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 14.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Derik Queen", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naji Marshall", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Clingan", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 22.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Gillespie", "line": 18.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 30.0, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 30.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 16.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Murray-Boyles", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lonzo Ball", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Oso Ighodaro", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ace Bailey", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Collier", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brice Sensabaugh", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylon Tyson", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 24.0, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "DeMar DeRozan", "line": 21.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jay Huff", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Evan Mobley", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "D'Angelo Russell", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kevin Huerter", "line": 13.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pelle Larsson", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 19.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Day'Ron Sharpe", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trayce Jackson-Davis", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Keegan Murray", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Simone Fontecchio", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Edwards", "line": 34.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zion Williamson", "line": 31.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drew Eubanks", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dean Wade", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Zach Edey", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Isaiah Jackson", "line": 13.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cole Anthony", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sidy Cissoko", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 30.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Amen Thompson", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Okogie", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nick Richards", "line": 15.5, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.0, "underPct": 1.0},
    {"name": "Scottie Barnes", "line": 28.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kentavious Caldwell-Pope", "line": 10.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylen Wells", "line": 14.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPAHitRates = [
    {"name": "Duncan Robinson", "line": 12.0, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Duren", "line": 21.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Stephen Curry", "line": 34.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kel'el Ware", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Collin Gillespie", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Goodwin", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Davion Mitchell", "line": 17.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jamal Murray", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Jones", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "P.J. Washington", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 13.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Holiday", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "D'Angelo Russell", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 36.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 23.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 37.0, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tobias Harris", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles McBride", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Martin", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gradey Dick", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ja'Kobe Walter", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drew Eubanks", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Oso Ighodaro", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sidy Cissoko", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bobby Portis", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 29.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Williams", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pelle Larsson", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jay Huff", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cedric Coward", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keegan Murray", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kevin Huerter", "line": 12.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Okogie", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden Ivey", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ben Sheppard", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Jackson", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Simone Fontecchio", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Wells", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cole Anthony", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 34.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
];const prizepicksRAHitRates = [
    {"name": "Josh Giddey", "line": 17.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 8.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cole Anthony", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lonzo Ball", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 10.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alperen Sengun", "line": 17.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ben Sheppard", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Devin Booker", "line": 11.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Daniel Gafford", "line": 9.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tre Jones", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cameron Johnson", "line": 6.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 5.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mikal Bridges", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 15.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 7.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Stephen Curry", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 8.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 14.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Brunson", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 10.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Okogie", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Reed Sheppard", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandin Podziemski", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zion Williamson", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Amen Thompson", "line": 13.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 9.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Edwards", "line": 10.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ayo Dosunmu", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sidy Cissoko", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Zach LaVine", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jakob Poeltl", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Royce O'Neale", "line": 8.0, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drew Eubanks", "line": 5.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const prizepicksTurnoversHitRates = [
    {"name": "Jay Huff", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mike Conley", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 1.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ace Bailey", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 1.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Royce O'Neale", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
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
    {"name": "Jaden Ivey", "line": 8.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.958, "underPct": 0.042},
    {"name": "Duncan Robinson", "line": 10.5, "l5": 1.0, "l10": 0.8, "l15": 0.73, "overPct": 0.9, "underPct": 0.1},
    {"name": "Bennedict Mathurin", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.839, "underPct": 0.161},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.807, "underPct": 0.193},
    {"name": "Cade Cunningham", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.806, "underPct": 0.194},
    {"name": "Keyonte George", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.793, "underPct": 0.207},
    {"name": "Dillon Brooks", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.766, "underPct": 0.234},
    {"name": "Lauri Markkanen", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.765, "underPct": 0.235},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.757, "underPct": 0.243},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.716, "underPct": 0.284},
    {"name": "Naz Reid", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.708, "underPct": 0.292},
    {"name": "Jakob Poeltl", "line": 12.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.707, "underPct": 0.293},
    {"name": "Keegan Murray", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.704, "underPct": 0.296},
    {"name": "Jaden McDaniels", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.703, "underPct": 0.297},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.688, "underPct": 0.312},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.679, "underPct": 0.321},
    {"name": "Ayo Dosunmu", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.644, "underPct": 0.356},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.639, "underPct": 0.361},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.639, "underPct": 0.361},
    {"name": "Pascal Siakam", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.636, "underPct": 0.364},
    {"name": "Davion Mitchell", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.631, "underPct": 0.369},
    {"name": "Tyrese Martin", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.616, "underPct": 0.384},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.591, "underPct": 0.409},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.581, "underPct": 0.419},
    {"name": "Day'Ron Sharpe", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.58, "underPct": 0.42},
    {"name": "Saddiq Bey", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.574, "underPct": 0.426},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.568, "underPct": 0.432},
    {"name": "Jeremiah Fears", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.568, "underPct": 0.432},
    {"name": "Moses Moody", "line": 12.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.567, "underPct": 0.433},
    {"name": "Kevin Huerter", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.67, "overPct": 0.567, "underPct": 0.433},
    {"name": "Drake Powell", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.56, "underPct": 0.44},
    {"name": "Stephen Curry", "line": 29.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.556, "underPct": 0.444},
    {"name": "Karl-Anthony Towns", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.545, "underPct": 0.455},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.541, "underPct": 0.459},
    {"name": "Donovan Mitchell", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.537, "underPct": 0.463},
    {"name": "Derik Queen", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.512, "underPct": 0.488},
    {"name": "Brice Sensabaugh", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.508, "underPct": 0.492},
    {"name": "Miles McBride", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.506, "underPct": 0.494},
    {"name": "Mikal Bridges", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.492, "underPct": 0.508},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.489, "underPct": 0.511},
    {"name": "Daniel Gafford", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.459, "underPct": 0.541},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.435, "underPct": 0.565},
    {"name": "Will Richard", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.43, "underPct": 0.57},
    {"name": "Kyle Kuzma", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.428, "underPct": 0.572},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.428, "underPct": 0.572},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "Jaylen Wells", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.416, "underPct": 0.584},
    {"name": "Cedric Coward", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.414, "underPct": 0.586},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.414, "underPct": 0.586},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.388, "underPct": 0.612},
    {"name": "Drew Eubanks", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.382, "underPct": 0.618},
    {"name": "Brandon Ingram", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.38, "underPct": 0.62},
    {"name": "Jordan Goodwin", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.378, "underPct": 0.622},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.369, "underPct": 0.631},
    {"name": "Alperen Sengun", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.365, "underPct": 0.635},
    {"name": "Jalen Brunson", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.362, "underPct": 0.638},
    {"name": "Zach LaVine", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.355, "underPct": 0.645},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.355, "underPct": 0.645},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.33, "underPct": 0.67},
    {"name": "Deni Avdija", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.329, "underPct": 0.671},
    {"name": "Ryan Rollins", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.322, "underPct": 0.678},
    {"name": "Simone Fontecchio", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.317, "underPct": 0.683},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.309, "underPct": 0.691},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.306, "underPct": 0.694},
    {"name": "Royce O'Neale", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.306, "underPct": 0.694},
    {"name": "Myles Turner", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.302, "underPct": 0.698},
    {"name": "Bam Adebayo", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.29, "underPct": 0.71},
    {"name": "Devin Booker", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.289, "underPct": 0.711},
    {"name": "Oso Ighodaro", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.288, "underPct": 0.712},
    {"name": "Brandon Williams", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.282, "underPct": 0.718},
    {"name": "DeMar DeRozan", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.8, "overPct": 0.279, "underPct": 0.721},
    {"name": "Lonzo Ball", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.274, "underPct": 0.726},
    {"name": "Reed Sheppard", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.269, "underPct": 0.731},
    {"name": "Coby White", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.259, "underPct": 0.741},
    {"name": "Zion Williamson", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.257, "underPct": 0.743},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.253, "underPct": 0.747},
    {"name": "Sidy Cissoko", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.214, "underPct": 0.786},
    {"name": "Kris Murray", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.213, "underPct": 0.787},
    {"name": "Ja'Kobe Walter", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.212, "underPct": 0.788},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.212, "underPct": 0.788},
    {"name": "Amen Thompson", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.201, "underPct": 0.799},
    {"name": "Terance Mann", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.201, "underPct": 0.799},
    {"name": "Kentavious Caldwell-Pope", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.186, "underPct": 0.814},
    {"name": "Evan Mobley", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.182, "underPct": 0.818},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.165, "underPct": 0.835},
    {"name": "Jerami Grant", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.123, "underPct": 0.877},
    {"name": "Spencer Jones", "line": 4.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.062, "underPct": 0.938},
    {"name": "Cole Anthony", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.037, "underPct": 0.963},
];const underdogAssistsHitRates = [
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.691, "underPct": 0.309},
    {"name": "Kyle Filipowski", "line": 1.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.632, "underPct": 0.368},
    {"name": "Isaiah Collier", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.604, "underPct": 0.396},
    {"name": "Davion Mitchell", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.604, "underPct": 0.396},
    {"name": "Terance Mann", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.603, "underPct": 0.397},
    {"name": "Cade Cunningham", "line": 9.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.551, "underPct": 0.449},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.549, "underPct": 0.451},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.529, "underPct": 0.471},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.52, "underPct": 0.48},
    {"name": "Cole Anthony", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.516, "underPct": 0.484},
    {"name": "Jaden McDaniels", "line": 2.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.491, "underPct": 0.509},
    {"name": "Kyle Kuzma", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.476, "underPct": 0.524},
    {"name": "T.J. McConnell", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.444, "underPct": 0.556},
    {"name": "Zion Williamson", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.441, "underPct": 0.559},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.431, "underPct": 0.569},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.419, "underPct": 0.581},
    {"name": "D'Angelo Russell", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.417, "underPct": 0.583},
    {"name": "Sidy Cissoko", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.254, "underPct": 0.746},
    {"name": "Will Richard", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.24, "underPct": 0.76},
    {"name": "Brandin Podziemski", "line": 3.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.215, "underPct": 0.785},
];const underdogReboundsHitRates = [
    {"name": "Donovan Mitchell", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.837, "underPct": 0.163},
    {"name": "Saddiq Bey", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.723, "underPct": 0.277},
    {"name": "Naji Marshall", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.692, "underPct": 0.308},
    {"name": "Keyonte George", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.641, "underPct": 0.359},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.587, "underPct": 0.413},
    {"name": "Kevin Huerter", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.564, "underPct": 0.436},
    {"name": "Tyrese Martin", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.55, "underPct": 0.45},
    {"name": "Daniel Gafford", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.537, "underPct": 0.463},
    {"name": "Jeremiah Fears", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.535, "underPct": 0.465},
    {"name": "Trey Murphy III", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Karl-Anthony Towns", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Bam Adebayo", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.475, "underPct": 0.525},
    {"name": "Zion Williamson", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.449, "underPct": 0.551},
    {"name": "Precious Achiuwa", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.432, "underPct": 0.568},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.419, "underPct": 0.581},
    {"name": "Kyle Kuzma", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.407, "underPct": 0.593},
    {"name": "Moses Moody", "line": 3.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.357, "underPct": 0.643},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.343, "underPct": 0.657},
    {"name": "Will Richard", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.336, "underPct": 0.664},
    {"name": "Collin Murray-Boyles", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.326, "underPct": 0.674},
    {"name": "Andrew Nembhard", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.307, "underPct": 0.693},
    {"name": "Ja'Kobe Walter", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.167, "underPct": 0.833},
    {"name": "Reed Sheppard", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.159, "underPct": 0.841},
];const underdogBlocksHitRates = [
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.465, "underPct": 0.535},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.515, "underPct": 0.485},
];const underdogStealsHitRates = [
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Keyonte George", "line": 29.5, "l5": 1.0, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dillon Brooks", "line": 25.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Goodwin", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 38.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Duren", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Brandon Williams", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 37.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 41.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "P.J. Washington", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derik Queen", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Mitchell", "line": 40.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alperen Sengun", "line": 41.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 33.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "D'Angelo Russell", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Murray", "line": 34.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Murray-Boyles", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 30.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Clarkson", "line": 18.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Martin", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 42.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Brunson", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Drake Powell", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 34.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sidy Cissoko", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Murray", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ben Sheppard", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Reed Sheppard", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Karl-Anthony Towns", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 38.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bobby Portis", "line": 25.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kyle Kuzma", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Noah Clowney", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 34.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 32.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles McBride", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jose Alvarado", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 29.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brice Sensabaugh", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Spencer", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mike Conley", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bam Adebayo", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Holiday", "line": 11.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Royce O'Neale", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarace Walker", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 38.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cole Anthony", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach Edey", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Zion Williamson", "line": 35.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Simone Fontecchio", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Day'Ron Sharpe", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden Ivey", "line": 14.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Kentavious Caldwell-Pope", "line": 13.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
];const underdogPRHitRates = [
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dillon Brooks", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Stephen Curry", "line": 33.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 21.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donovan Mitchell", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 23.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 34.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach Edey", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Keegan Murray", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zion Williamson", "line": 31.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 28.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const underdogPAHitRates = [
    {"name": "Jalen Duren", "line": 21.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Dillon Brooks", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cade Cunningham", "line": 36.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Andrew Nembhard", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keyonte George", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Stephen Curry", "line": 35.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Gillespie", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lauri Markkanen", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cooper Flagg", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Reed Sheppard", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Williams", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Coby White", "line": 28.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jerami Grant", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Booker", "line": 35.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const underdogRAHitRates = [
    {"name": "Josh Giddey", "line": 17.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Kel'el Ware", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Lonzo Ball", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cole Anthony", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pelle Larsson", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden McDaniels", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
];const underdogTurnoversHitRates = [
    {"name": "Donovan Mitchell", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alperen Sengun", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Collier", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Julius Randle", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
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

