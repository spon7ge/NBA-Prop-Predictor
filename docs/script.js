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
    {"name1": "Jared McCain", "name2": "LeBron James", "line1": 13.5, "line2": 20.5, "prediction1": 9.13, "prediction2": 15.88, "side1": "under", "side2": "under", "recommendation": 1, "ev": 82.22, "kelly": 0.411, "sigma1": "Med", "sigma2": "Med", "prob1": 0.785, "prob2": 0.789, "hitRate1": 100.0, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 36.8, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Tyrese Maxey", "name2": "Jake LaRavia", "line1": 32.5, "line2": 7.5, "prediction1": 26.23, "prediction2": 10.83, "side1": "under", "side2": "over", "recommendation": 0, "ev": 59.38, "kelly": 0.297, "sigma1": "High", "sigma2": "High", "prob1": 0.781, "prob2": 0.694, "hitRate1": 35.0, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 51.1, "l5_2": 0.2, "l15_2": 0.53},
    {"name1": "Alex Sarr", "name2": "Rui Hachimura", "line1": 17.5, "line2": 11.5, "prediction1": 20.33, "prediction2": 14.58, "side1": "over", "side2": "over", "recommendation": 0, "ev": 30.92, "kelly": 0.155, "sigma1": "High", "sigma2": "High", "prob1": 0.652, "prob2": 0.683, "hitRate1": 60.1, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 73.4, "l5_2": 0.8, "l15_2": 0.73},
    {"name1": "Tristan da Silva", "name2": "Brook Lopez", "line1": 12.5, "line2": 5.5, "prediction1": 14.88, "prediction2": 8.07, "side1": "over", "side2": "over", "recommendation": 0, "ev": 24.77, "kelly": 0.124, "sigma1": "High", "sigma2": "Med", "prob1": 0.632, "prob2": 0.672, "hitRate1": 44.6, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 41.4, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Goga Bitadze", "name2": "John Collins", "line1": 5.5, "line2": 11.5, "prediction1": 6.74, "prediction2": 13.91, "side1": "over", "side2": "over", "recommendation": 0, "ev": 14.45, "kelly": 0.072, "sigma1": "Low", "sigma2": "High", "prob1": 0.605, "prob2": 0.643, "hitRate1": 63.1, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 28.4, "l5_2": 0.2, "l15_2": 0.47},
    {"name1": "Anthony Black", "name2": "Ivica Zubac", "line1": 12.5, "line2": 16.5, "prediction1": 14.09, "prediction2": 18.89, "side1": "over", "side2": "over", "recommendation": 0, "ev": 10.86, "kelly": 0.054, "sigma1": "High", "sigma2": "High", "prob1": 0.593, "prob2": 0.635, "hitRate1": 54.8, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 75.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Andre Drummond", "name2": "Austin Reaves", "line1": 11.5, "line2": 22.5, "prediction1": 10.35, "prediction2": 24.98, "side1": "under", "side2": "over", "recommendation": 0, "ev": 5.38, "kelly": 0.027, "sigma1": "High", "sigma2": "High", "prob1": 0.572, "prob2": 0.627, "hitRate1": 53.2, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 62.1, "l5_2": 0.6, "l15_2": 0.67},
    {"name1": "Trendon Watford", "name2": "Kris Dunn", "line1": 9.5, "line2": 6.5, "prediction1": 8.48, "prediction2": 8.16, "side1": "under", "side2": "over", "recommendation": 0, "ev": 4.03, "kelly": 0.02, "sigma1": "Med", "sigma2": "Med", "prob1": 0.57, "prob2": 0.62, "hitRate1": 70.8, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 50.1, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Zaccharie Risacher", "name2": "Marcus Smart", "line1": 11.5, "line2": 6.5, "prediction1": 12.6, "prediction2": 8.19, "side1": "over", "side2": "over", "recommendation": 0, "ev": 0.57, "kelly": 0.003, "sigma1": "High", "sigma2": "High", "prob1": 0.561, "prob2": 0.61, "hitRate1": 32.1, "l5_1": 0.2, "l15_1": 0.47, "hitRate2": 82.9, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Nickeil Alexander-Walker", "name2": "Kawhi Leonard", "line1": 18.5, "line2": 20.5, "prediction1": 19.61, "prediction2": 22.5, "side1": "over", "side2": "over", "recommendation": 0, "ev": -0.53, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.559, "prob2": 0.605, "hitRate1": 67.6, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 73.2, "l5_2": 0.6, "l15_2": 0.27},
];const prizepicksTriosData = [
    {"name1": "Tyrese Maxey", "name2": "Jared McCain", "name3": "LeBron James", "line1": 32.5, "line2": 13.5, "line3": 20.5, "prediction1": 26.23, "prediction2": 9.13, "prediction3": 15.88, "side1": "under", "side2": "under", "side3": "under", "recommendation": 1, "ev": 161.34, "kelly": 0.323, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.781, "prob2": 0.785, "prob3": 0.789, "hitRate1": 35.0, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 100.0, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 36.8, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Alex Sarr", "name2": "Rui Hachimura", "name3": "Jake LaRavia", "line1": 17.5, "line2": 11.5, "line3": 7.5, "prediction1": 20.33, "prediction2": 14.58, "prediction3": 10.83, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 66.95, "kelly": 0.134, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.652, "prob2": 0.683, "prob3": 0.694, "hitRate1": 60.1, "l5_1": 0.2, "l15_1": 0.4, "hitRate2": 73.4, "l5_2": 0.8, "l15_2": 0.73, "hitRate3": 51.1, "l5_3": 0.2, "l15_3": 0.53},
    {"name1": "Tristan da Silva", "name2": "John Collins", "name3": "Brook Lopez", "line1": 12.5, "line2": 11.5, "line3": 5.5, "prediction1": 14.88, "prediction2": 13.91, "prediction3": 8.07, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 47.34, "kelly": 0.095, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.632, "prob2": 0.643, "prob3": 0.672, "hitRate1": 44.6, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 28.4, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 41.4, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Goga Bitadze", "name2": "Ivica Zubac", "name3": "Kris Dunn", "line1": 5.5, "line2": 16.5, "line3": 6.5, "prediction1": 6.74, "prediction2": 18.89, "prediction3": 8.16, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 28.87, "kelly": 0.058, "sigma1": "Low", "sigma2": "High", "sigma3": "Med", "prob1": 0.605, "prob2": 0.635, "prob3": 0.62, "hitRate1": 63.1, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 75.5, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 50.1, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Anthony Black", "name2": "Austin Reaves", "name3": "Marcus Smart", "line1": 12.5, "line2": 22.5, "line3": 6.5, "prediction1": 14.09, "prediction2": 24.98, "prediction3": 8.19, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 22.44, "kelly": 0.045, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.593, "prob2": 0.627, "prob3": 0.61, "hitRate1": 54.8, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 62.1, "l5_2": 0.6, "l15_2": 0.67, "hitRate3": 82.9, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Andre Drummond", "name2": "James Harden", "name3": "Kawhi Leonard", "line1": 11.5, "line2": 24.5, "line3": 20.5, "prediction1": 10.35, "prediction2": 25.99, "prediction3": 22.5, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 10.29, "kelly": 0.021, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.572, "prob2": 0.59, "prob3": 0.605, "hitRate1": 53.2, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 87.3, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 73.2, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Trendon Watford", "name2": "Luka Don\u010di\u0107", "name3": "Jaxson Hayes", "line1": 9.5, "line2": 32.5, "line3": 7.5, "prediction1": 8.48, "prediction2": 31.27, "prediction3": 6.68, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": -0.05, "kelly": 0.0, "sigma1": "Med", "sigma2": "High", "sigma3": "Low", "prob1": 0.57, "prob2": 0.568, "prob3": 0.571, "hitRate1": 70.8, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 87.9, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Nickeil Alexander-Walker", "name2": "Zaccharie Risacher", "name3": "Quentin Grimes", "line1": 18.5, "line2": 11.5, "line3": 17.5, "prediction1": 19.61, "prediction2": 12.6, "prediction3": 16.47, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": -6.1, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.559, "prob2": 0.561, "prob3": 0.554, "hitRate1": 67.6, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 32.1, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 67.9, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Bilal Coulibaly", "name2": "Khris Middleton", "name3": "Jalen Suggs", "line1": 10.5, "line2": 9.0, "line3": 16.5, "prediction1": 9.66, "prediction2": 9.79, "prediction3": 15.85, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": -10.24, "kelly": 0.0, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.556, "prob2": 0.551, "prob3": 0.542, "hitRate1": 69.2, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 24.6, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 46.1, "l5_3": 0.4, "l15_3": 0.2},
    {"name1": "Kristaps Porzi\u0146\u0123is", "name2": "Bogdan Bogdanovi\u0107", "name3": "Kobe Sanders", "line1": 18.5, "line2": 8.5, "line3": 7.5, "prediction1": 17.54, "prediction2": 9.01, "prediction3": 7.19, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": -17.04, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.554, "prob2": 0.532, "prob3": 0.522, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 38.5, "l5_3": 0.6, "l15_3": 0.27},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Tyrese Maxey", "name2": "LeBron James", "line1": 32.5, "line2": 20.5, "prediction1": 26.23, "prediction2": 15.88, "side1": "under", "side2": "under", "recommendation": 1, "ev": 81.15, "kelly": 0.406, "sigma1": "High", "sigma2": "Med", "prob1": 0.781, "prob2": 0.789, "hitRate1": 35.0, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 36.8, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Anthony Black", "name2": "Rui Hachimura", "line1": 12.5, "line2": 11.5, "prediction1": 14.09, "prediction2": 14.58, "side1": "over", "side2": "over", "recommendation": 0, "ev": 19.21, "kelly": 0.096, "sigma1": "High", "sigma2": "High", "prob1": 0.593, "prob2": 0.683, "hitRate1": 62.5, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 73.4, "l5_2": 0.8, "l15_2": 0.73},
    {"name1": "Trendon Watford", "name2": "Brook Lopez", "line1": 9.5, "line2": 5.5, "prediction1": 8.48, "prediction2": 8.07, "side1": "under", "side2": "over", "recommendation": 0, "ev": 12.66, "kelly": 0.063, "sigma1": "Med", "sigma2": "Med", "prob1": 0.57, "prob2": 0.672, "hitRate1": 70.8, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 41.4, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Zaccharie Risacher", "name2": "Ivica Zubac", "line1": 11.5, "line2": 16.5, "prediction1": 12.6, "prediction2": 18.89, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.81, "kelly": 0.024, "sigma1": "High", "sigma2": "High", "prob1": 0.561, "prob2": 0.635, "hitRate1": 32.1, "l5_1": 0.2, "l15_1": 0.47, "hitRate2": 75.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Nickeil Alexander-Walker", "name2": "Austin Reaves", "line1": 18.5, "line2": 22.5, "prediction1": 19.61, "prediction2": 24.98, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.02, "kelly": 0.015, "sigma1": "High", "sigma2": "High", "prob1": 0.559, "prob2": 0.627, "hitRate1": 67.6, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 62.1, "l5_2": 0.6, "l15_2": 0.67},
    {"name1": "Bilal Coulibaly", "name2": "Kris Dunn", "line1": 10.5, "line2": 6.5, "prediction1": 9.66, "prediction2": 8.16, "side1": "under", "side2": "over", "recommendation": 0, "ev": 1.47, "kelly": 0.007, "sigma1": "Med", "sigma2": "Med", "prob1": 0.556, "prob2": 0.62, "hitRate1": 69.2, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 44.0, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Quentin Grimes", "name2": "Marcus Smart", "line1": 17.5, "line2": 6.5, "prediction1": 16.47, "prediction2": 8.19, "side1": "under", "side2": "over", "recommendation": 0, "ev": -0.62, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.554, "prob2": 0.61, "hitRate1": 67.9, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 82.9, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Kristaps Porzi\u0146\u0123is", "name2": "Kawhi Leonard", "line1": 18.5, "line2": 20.5, "prediction1": 17.54, "prediction2": 22.5, "side1": "under", "side2": "over", "recommendation": 0, "ev": -1.49, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.554, "prob2": 0.605, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 73.2, "l5_2": 0.6, "l15_2": 0.27},
    {"name1": "Jalen Suggs", "name2": "James Harden", "line1": 16.5, "line2": 24.5, "prediction1": 15.85, "prediction2": 25.99, "side1": "under", "side2": "over", "recommendation": 0, "ev": -6.01, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.542, "prob2": 0.59, "hitRate1": 51.1, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 91.0, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Marvin Bagley III", "name2": "Jaxson Hayes", "line1": 7.5, "line2": 7.5, "prediction1": 8.01, "prediction2": 6.68, "side1": "over", "side2": "under", "recommendation": 0, "ev": -9.56, "kelly": 0.0, "sigma1": "Med", "sigma2": "Low", "prob1": 0.539, "prob2": 0.571, "hitRate1": 52.0, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 90.0, "l5_2": 0.4, "l15_2": 0.27},
];const underdogTriosData = [
    {"name1": "Tyrese Maxey", "name2": "LeBron James", "name3": "Rui Hachimura", "line1": 32.5, "line2": 20.5, "line3": 11.5, "prediction1": 26.23, "prediction2": 15.88, "prediction3": 14.58, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 127.37, "kelly": 0.255, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.781, "prob2": 0.789, "prob3": 0.683, "hitRate1": 35.0, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 36.8, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 73.4, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Anthony Black", "name2": "Ivica Zubac", "name3": "Brook Lopez", "line1": 12.5, "line2": 16.5, "line3": 5.5, "prediction1": 14.09, "prediction2": 18.89, "prediction3": 8.07, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 36.77, "kelly": 0.074, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.593, "prob2": 0.635, "prob3": 0.672, "hitRate1": 62.5, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 75.5, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 41.4, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Trendon Watford", "name2": "Austin Reaves", "name3": "Marcus Smart", "line1": 9.5, "line2": 22.5, "line3": 6.5, "prediction1": 8.48, "prediction2": 24.98, "prediction3": 8.19, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 17.72, "kelly": 0.035, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.57, "prob2": 0.627, "prob3": 0.61, "hitRate1": 70.8, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 62.1, "l5_2": 0.6, "l15_2": 0.67, "hitRate3": 82.9, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Zaccharie Risacher", "name2": "Kawhi Leonard", "name3": "Kris Dunn", "line1": 11.5, "line2": 20.5, "line3": 6.5, "prediction1": 12.6, "prediction2": 22.5, "prediction3": 8.16, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 13.71, "kelly": 0.027, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.561, "prob2": 0.605, "prob3": 0.62, "hitRate1": 32.1, "l5_1": 0.2, "l15_1": 0.47, "hitRate2": 73.2, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 44.0, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Nickeil Alexander-Walker", "name2": "Quentin Grimes", "name3": "James Harden", "line1": 18.5, "line2": 17.5, "line3": 24.5, "prediction1": 19.61, "prediction2": 16.47, "prediction3": 25.99, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": -1.23, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.559, "prob2": 0.554, "prob3": 0.59, "hitRate1": 67.6, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 67.9, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 91.0, "l5_3": 0.8, "l15_3": 0.6},
    {"name1": "Bilal Coulibaly", "name2": "Luka Don\u010di\u0107", "name3": "Jaxson Hayes", "line1": 10.5, "line2": 32.5, "line3": 7.5, "prediction1": 9.66, "prediction2": 31.27, "prediction3": 6.68, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": -2.51, "kelly": 0.0, "sigma1": "Med", "sigma2": "High", "sigma3": "Low", "prob1": 0.556, "prob2": 0.568, "prob3": 0.571, "hitRate1": 69.2, "l5_1": 0.2, "l15_1": 0.2, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 90.0, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Kristaps Porzi\u0146\u0123is", "name2": "Jalen Suggs", "name3": "Bogdan Bogdanovi\u0107", "line1": 18.5, "line2": 16.5, "line3": 8.5, "prediction1": 17.54, "prediction2": 15.85, "prediction3": 9.01, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": -13.91, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.554, "prob2": 0.542, "prob3": 0.532, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 51.1, "l5_2": 0.4, "l15_2": 0.2, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Marvin Bagley III", "name2": "Nicolas Batum", "name3": "Kobe Sanders", "line1": 7.5, "line2": 5.5, "line3": 7.5, "prediction1": 8.01, "prediction2": 5.15, "prediction3": 7.19, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": -19.85, "kelly": 0.0, "sigma1": "Med", "sigma2": "Med", "sigma3": "Med", "prob1": 0.539, "prob2": 0.528, "prob3": 0.522, "hitRate1": 52.0, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 41.5, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 38.5, "l5_3": 0.6, "l15_3": 0.27},
    {"name1": "Desmond Bane", "name2": "Franz Wagner", "name3": "Gabe Vincent", "line1": 20.5, "line2": 24.5, "line3": 4.5, "prediction1": 20.1, "prediction2": 24.15, "prediction3": 4.77, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": -23.72, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.521, "prob2": 0.518, "prob3": 0.523, "hitRate1": 42.6, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 51.7, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 36.8, "l5_3": 0.4, "l15_3": 0.13},
];const prizepicksPointsHitRates = [
    {"name": "James Harden", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.873, "underPct": 0.127},
    {"name": "Onyeka Okongwu", "line": 14.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.844, "underPct": 0.156},
    {"name": "Marcus Smart", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.829, "underPct": 0.171},
    {"name": "Ivica Zubac", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.755, "underPct": 0.245},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.734, "underPct": 0.266},
    {"name": "Kawhi Leonard", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.732, "underPct": 0.268},
    {"name": "Nickeil Alexander-Walker", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.676, "underPct": 0.324},
    {"name": "Tyrese Maxey", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.65, "underPct": 0.35},
    {"name": "Cam Whitmore", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.646, "underPct": 0.354},
    {"name": "LeBron James", "line": 20.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.632, "underPct": 0.368},
    {"name": "Goga Bitadze", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.631, "underPct": 0.369},
    {"name": "Austin Reaves", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.621, "underPct": 0.379},
    {"name": "Kobe Sanders", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.615, "underPct": 0.385},
    {"name": "Alex Sarr", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.601, "underPct": 0.399},
    {"name": "Jalen Johnson", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.59, "underPct": 0.41},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.574, "underPct": 0.426},
    {"name": "Anthony Black", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.548, "underPct": 0.452},
    {"name": "Jalen Suggs", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.539, "underPct": 0.461},
    {"name": "Marvin Bagley III", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.52, "underPct": 0.48},
    {"name": "Jake LaRavia", "line": 7.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.511, "underPct": 0.489},
    {"name": "Kris Dunn", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Franz Wagner", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.483, "underPct": 0.517},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.468, "underPct": 0.532},
    {"name": "Tristan da Silva", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Brook Lopez", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.414, "underPct": 0.586},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.397, "underPct": 0.603},
    {"name": "Corey Kispert", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.366, "underPct": 0.634},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.321, "underPct": 0.679},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.321, "underPct": 0.679},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.308, "underPct": 0.692},
    {"name": "Trendon Watford", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.292, "underPct": 0.708},
    {"name": "John Collins", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.284, "underPct": 0.716},
    {"name": "Khris Middleton", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.246, "underPct": 0.754},
    {"name": "Jaxson Hayes", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.121, "underPct": 0.879},
    {"name": "Jared McCain", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.0, "underPct": 1.0},
];const prizepicksAssistsHitRates = [
    {"name": "Gabe Vincent", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.793, "underPct": 0.207},
    {"name": "Kyshawn George", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.644, "underPct": 0.356},
    {"name": "LeBron James", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.61, "underPct": 0.39},
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.543, "underPct": 0.457},
    {"name": "Jalen Johnson", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.51, "underPct": 0.49},
    {"name": "Desmond Bane", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.506, "underPct": 0.494},
    {"name": "Tyrese Maxey", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
    {"name": "Franz Wagner", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.447, "underPct": 0.553},
    {"name": "Jaxson Hayes", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.422, "underPct": 0.578},
    {"name": "Jalen Suggs", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.383, "underPct": 0.617},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.296, "underPct": 0.704},
    {"name": "James Harden", "line": 8.0, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.287, "underPct": 0.713},
];const prizepicksReboundsHitRates = [
    {"name": "Ivica Zubac", "line": 11.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.67, "underPct": 0.33},
    {"name": "Austin Reaves", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.655, "underPct": 0.345},
    {"name": "Kawhi Leonard", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.613, "underPct": 0.387},
    {"name": "Zaccharie Risacher", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.606, "underPct": 0.394},
    {"name": "Jalen Johnson", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.603, "underPct": 0.397},
    {"name": "James Harden", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.555, "underPct": 0.445},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.8, "l10": 0.4, "l15": 0.6, "overPct": 0.521, "underPct": 0.479},
    {"name": "LeBron James", "line": 7.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.517, "underPct": 0.483},
    {"name": "Goga Bitadze", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.513, "underPct": 0.487},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.507, "underPct": 0.493},
    {"name": "Franz Wagner", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.494, "underPct": 0.506},
    {"name": "Alex Sarr", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.465, "underPct": 0.535},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.464, "underPct": 0.536},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.444, "underPct": 0.556},
    {"name": "Jake LaRavia", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.41, "underPct": 0.59},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.397, "underPct": 0.603},
    {"name": "Rui Hachimura", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.377, "underPct": 0.623},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.373, "underPct": 0.627},
    {"name": "Tristan da Silva", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.349, "underPct": 0.651},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.347, "underPct": 0.653},
    {"name": "John Collins", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.312, "underPct": 0.688},
    {"name": "Bilal Coulibaly", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.281, "underPct": 0.719},
    {"name": "Khris Middleton", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.261, "underPct": 0.739},
    {"name": "Anthony Black", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.256, "underPct": 0.744},
    {"name": "Onyeka Okongwu", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.237, "underPct": 0.763},
    {"name": "Jaxson Hayes", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.199, "underPct": 0.801},
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
    {"name": "Jalen Johnson", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.531, "underPct": 0.469},
    {"name": "Cam Whitmore", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.418, "underPct": 0.582},
    {"name": "Kawhi Leonard", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.726, "underPct": 0.274},
    {"name": "Ivica Zubac", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.413, "underPct": 0.587},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Onyeka Okongwu", "line": 24.5, "l5": 1.0, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 37.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Desmond Bane", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alex Sarr", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kawhi Leonard", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kobe Sanders", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "John Collins", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brook Lopez", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 42.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 44.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Corey Kispert", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 15.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Whitmore", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Khris Middleton", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gabe Vincent", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Trendon Watford", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jake LaRavia", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Justin Edwards", "line": 15.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LeBron James", "line": 34.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Quentin Grimes", "line": 25.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
];const prizepicksPRHitRates = [
    {"name": "Nicolas Batum", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "James Harden", "line": 28.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kobe Sanders", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 18.0, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Marvin Bagley III", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 13.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "John Collins", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Whitmore", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Corey Kispert", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Khris Middleton", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 21.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Justin Edwards", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bilal Coulibaly", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaxson Hayes", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jake LaRavia", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 27.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPAHitRates = [
    {"name": "Onyeka Okongwu", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rui Hachimura", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Marvin Bagley III", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kobe Sanders", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tristan da Silva", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nicolas Batum", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 16.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaxson Hayes", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Sarr", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Whitmore", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Khris Middleton", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 39.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Corey Kispert", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trendon Watford", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jake LaRavia", "line": 8.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Justin Edwards", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "LeBron James", "line": 27.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksRAHitRates = [
    {"name": "Ivica Zubac", "line": 13.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kawhi Leonard", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Sarr", "line": 12.0, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 11.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bilal Coulibaly", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 7.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "LeBron James", "line": 14.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "James Harden", "line": 13.0, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 8.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
];const prizepicksTurnoversHitRates = [
    {"name": "Khris Middleton", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andre Drummond", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Rui Hachimura", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bilal Coulibaly", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brook Lopez", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
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
    {"name": "James Harden", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.91, "underPct": 0.09},
    {"name": "Onyeka Okongwu", "line": 14.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.877, "underPct": 0.123},
    {"name": "Marcus Smart", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.829, "underPct": 0.171},
    {"name": "Ivica Zubac", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.755, "underPct": 0.245},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.734, "underPct": 0.266},
    {"name": "Kawhi Leonard", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.732, "underPct": 0.268},
    {"name": "Nickeil Alexander-Walker", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.676, "underPct": 0.324},
    {"name": "Tyrese Maxey", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.65, "underPct": 0.35},
    {"name": "LeBron James", "line": 20.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.632, "underPct": 0.368},
    {"name": "Anthony Black", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.625, "underPct": 0.375},
    {"name": "Austin Reaves", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.621, "underPct": 0.379},
    {"name": "Kobe Sanders", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.615, "underPct": 0.385},
    {"name": "Jalen Johnson", "line": 23.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.585, "underPct": 0.415},
    {"name": "Desmond Bane", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.574, "underPct": 0.426},
    {"name": "Marvin Bagley III", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.52, "underPct": 0.48},
    {"name": "Jalen Suggs", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.489, "underPct": 0.511},
    {"name": "Franz Wagner", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.483, "underPct": 0.517},
    {"name": "Kris Dunn", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.44, "underPct": 0.56},
    {"name": "Brook Lopez", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.414, "underPct": 0.586},
    {"name": "Gabe Vincent", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.368, "underPct": 0.632},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.321, "underPct": 0.679},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.321, "underPct": 0.679},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.308, "underPct": 0.692},
    {"name": "Trendon Watford", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.292, "underPct": 0.708},
    {"name": "Jaxson Hayes", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const underdogAssistsHitRates = [
    {"name": "Desmond Bane", "line": 4.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.506, "underPct": 0.494},
    {"name": "Tyrese Maxey", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.451, "underPct": 0.549},
];const underdogReboundsHitRates = [
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.8, "l10": 0.4, "l15": 0.6, "overPct": 0.56, "underPct": 0.44},
    {"name": "Goga Bitadze", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.513, "underPct": 0.487},
    {"name": "Alex Sarr", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.465, "underPct": 0.535},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.444, "underPct": 0.556},
    {"name": "Jake LaRavia", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.41, "underPct": 0.59},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.373, "underPct": 0.627},
    {"name": "Trendon Watford", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.355, "underPct": 0.645},
    {"name": "Anthony Black", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.284, "underPct": 0.716},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
    {"name": "Jalen Johnson", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.531, "underPct": 0.469},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Onyeka Okongwu", "line": 24.5, "l5": 1.0, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rui Hachimura", "line": 16.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dyson Daniels", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 37.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Austin Reaves", "line": 31.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alex Sarr", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kobe Sanders", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Marcus Smart", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ivica Zubac", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Stephen Curry", "line": 38.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 43.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 42.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cam Whitmore", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Corey Kispert", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 15.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trendon Watford", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bilal Coulibaly", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jake LaRavia", "line": 12.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaxson Hayes", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 34.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Justin Edwards", "line": 15.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quentin Grimes", "line": 25.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
];const underdogPRHitRates = [
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 1.0, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "James Harden", "line": 29.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alex Sarr", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 27.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPAHitRates = [
    {"name": "Jalen Johnson", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alex Sarr", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 40.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 27.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogRAHitRates = [
    {"name": "Ivica Zubac", "line": 13.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trendon Watford", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const underdogTurnoversHitRates = [
    {"name": "Austin Reaves", "line": 2.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jalen Suggs", "line": 2.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Quentin Grimes", "line": 2.5, "l5": 0.0, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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

