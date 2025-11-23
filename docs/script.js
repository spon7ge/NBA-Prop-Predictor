const prizepicksSinglesData = [
    {"name": "Tyrese Maxey", "bookmaker": "BetRivers", "line": 30.5, "prediction": 25.45, "side": "Under", "odds": 107, "recommendation": 1, "ev": 66.84, "kelly": 0.625, "sigma": "Med"},
    {"name": "Brandon Miller", "bookmaker": "BetMGM", "line": 18.5, "prediction": 14.01, "side": "Under", "odds": -110, "recommendation": 1, "ev": 57.99, "kelly": 0.638, "sigma": "Low"},
    {"name": "Bam Adebayo", "bookmaker": "BetRivers", "line": 19.5, "prediction": 15.79, "side": "Under", "odds": 120, "recommendation": 0, "ev": 57.02, "kelly": 0.475, "sigma": "High"},
    {"name": "Kristaps Porzingis", "bookmaker": "BetRivers", "line": 18.5, "prediction": 15.09, "side": "Under", "odds": 110, "recommendation": 0, "ev": 51.44, "kelly": 0.468, "sigma": "Med"},
    {"name": "Rui Hachimura", "bookmaker": "BetRivers", "line": 12.5, "prediction": 15.43, "side": "Over", "odds": 120, "recommendation": 0, "ev": 48.47, "kelly": 0.404, "sigma": "High"},
    {"name": "Dillon Brooks", "bookmaker": "DraftKings", "line": 18.5, "prediction": 23.81, "side": "Over", "odds": -114, "recommendation": 1, "ev": 44.49, "kelly": 0.507, "sigma": "High"},
    {"name": "James Harden", "bookmaker": "DraftKings", "line": 24.5, "prediction": 20.45, "side": "Under", "odds": -122, "recommendation": 1, "ev": 36.84, "kelly": 0.449, "sigma": "Med"},
    {"name": "Austin Reaves", "bookmaker": "BetMGM", "line": 22.5, "prediction": 26.89, "side": "Over", "odds": -115, "recommendation": 1, "ev": 36.43, "kelly": 0.419, "sigma": "High"},
    {"name": "Deni Avdija", "bookmaker": "BetMGM", "line": 24.5, "prediction": 26.88, "side": "Over", "odds": 110, "recommendation": 0, "ev": 34.84, "kelly": 0.317, "sigma": "High"},
    {"name": "Dyson Daniels", "bookmaker": "BetMGM", "line": 12.5, "prediction": 9.64, "side": "Under", "odds": -118, "recommendation": 0, "ev": 34.64, "kelly": 0.409, "sigma": "Low"},
];const prizepicksPairsData = [
    {"name1": "Tyrese Maxey", "name2": "Kristaps Porzi\u0146\u0123is", "line1": 31.5, "line2": 19.5, "prediction1": 25.45, "prediction2": 15.09, "side1": "under", "side2": "under", "recommendation": 1, "ev": 93.75, "kelly": 0.469, "sigma1": "Med", "sigma2": "Med", "prob1": 0.849, "prob2": 0.776, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Bam Adebayo", "name2": "Brandon Miller", "line1": 20.5, "line2": 17.5, "prediction1": 15.79, "prediction2": 14.01, "side1": "under", "side2": "under", "recommendation": 0, "ev": 72.46, "kelly": 0.362, "sigma1": "High", "sigma2": "Low", "prob1": 0.763, "prob2": 0.769, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Tristan da Silva", "name2": "Austin Reaves", "line1": 12.5, "line2": 22.5, "prediction1": 15.52, "prediction2": 26.89, "side1": "over", "side2": "over", "recommendation": 0, "ev": 45.41, "kelly": 0.227, "sigma1": "High", "sigma2": "High", "prob1": 0.678, "prob2": 0.73, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Dyson Daniels", "name2": "Kevin Love", "line1": 11.5, "line2": 5.0, "prediction1": 9.64, "prediction2": 7.45, "side1": "under", "side2": "over", "recommendation": 0, "ev": 32.9, "kelly": 0.165, "sigma1": "Low", "sigma2": "Low", "prob1": 0.654, "prob2": 0.691, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Andre Drummond", "name2": "Keyonte George", "line1": 11.5, "line2": 20.5, "prediction1": 9.31, "prediction2": 23.03, "side1": "under", "side2": "over", "recommendation": 0, "ev": 23.56, "kelly": 0.118, "sigma1": "Med", "sigma2": "High", "prob1": 0.645, "prob2": 0.651, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Anthony Black", "name2": "Marcus Smart", "line1": 13.5, "line2": 6.5, "prediction1": 15.81, "prediction2": 8.76, "side1": "over", "side2": "over", "recommendation": 0, "ev": 21.23, "kelly": 0.106, "sigma1": "High", "sigma2": "Med", "prob1": 0.637, "prob2": 0.647, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Justin Edwards", "name2": "Noah Clowney", "line1": 9.5, "line2": 12.5, "prediction1": 11.59, "prediction2": 14.66, "side1": "over", "side2": "over", "recommendation": 0, "ev": 17.71, "kelly": 0.089, "sigma1": "High", "sigma2": "High", "prob1": 0.636, "prob2": 0.63, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Davion Mitchell", "name2": "Jerami Grant", "line1": 9.5, "line2": 18.5, "prediction1": 11.48, "prediction2": 16.45, "side1": "over", "side2": "under", "recommendation": 0, "ev": 13.5, "kelly": 0.067, "sigma1": "High", "sigma2": "High", "prob1": 0.624, "prob2": 0.619, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Dru Smith", "name2": "Darius Garland", "line1": 5.5, "line2": 17.5, "prediction1": 6.94, "prediction2": 15.51, "side1": "over", "side2": "under", "recommendation": 0, "ev": 12.02, "kelly": 0.06, "sigma1": "Low", "sigma2": "High", "prob1": 0.618, "prob2": 0.616, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "Zaccharie Risacher", "name2": "Dean Wade", "line1": 12.5, "line2": 6.0, "prediction1": 14.34, "prediction2": 4.67, "side1": "over", "side2": "under", "recommendation": 0, "ev": 9.96, "kelly": 0.05, "sigma1": "High", "sigma2": "Low", "prob1": 0.616, "prob2": 0.607, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
];const prizepicksTriosData = [
    {"name1": "Tyrese Maxey", "name2": "Kristaps Porzi\u0146\u0123is", "name3": "Austin Reaves", "line1": 31.5, "line2": 19.5, "line3": 22.5, "prediction1": 25.45, "prediction2": 15.09, "prediction3": 26.89, "side1": "under", "side2": "under", "side3": "over", "recommendation": 1, "ev": 159.69, "kelly": 0.319, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.849, "prob2": 0.776, "prob3": 0.73, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Bam Adebayo", "name2": "Brandon Miller", "name3": "Kevin Love", "line1": 20.5, "line2": 17.5, "line3": 5.0, "prediction1": 15.79, "prediction2": 14.01, "prediction3": 7.45, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 118.95, "kelly": 0.238, "sigma1": "High", "sigma2": "Low", "sigma3": "Low", "prob1": 0.763, "prob2": 0.769, "prob3": 0.691, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Dyson Daniels", "name2": "Tristan da Silva", "name3": "Keyonte George", "line1": 11.5, "line2": 12.5, "line3": 20.5, "prediction1": 9.64, "prediction2": 15.52, "prediction3": 23.03, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 55.87, "kelly": 0.112, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.654, "prob2": 0.678, "prob3": 0.651, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Andre Drummond", "name2": "Anthony Black", "name3": "Marcus Smart", "line1": 11.5, "line2": 13.5, "line3": 6.5, "prediction1": 9.31, "prediction2": 15.81, "prediction3": 8.76, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 43.7, "kelly": 0.087, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.645, "prob2": 0.637, "prob3": 0.647, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Justin Edwards", "name2": "Noah Clowney", "name3": "Jerami Grant", "line1": 9.5, "line2": 12.5, "line3": 18.5, "prediction1": 11.59, "prediction2": 14.66, "prediction3": 16.45, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 33.75, "kelly": 0.067, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.636, "prob2": 0.63, "prob3": 0.619, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Davion Mitchell", "name2": "Dru Smith", "name3": "Darius Garland", "line1": 9.5, "line2": 5.5, "line3": 17.5, "prediction1": 11.48, "prediction2": 6.94, "prediction3": 15.51, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 28.39, "kelly": 0.057, "sigma1": "High", "sigma2": "Low", "sigma3": "High", "prob1": 0.624, "prob2": 0.618, "prob3": 0.616, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Nickeil Alexander-Walker", "name2": "Zaccharie Risacher", "name3": "Dean Wade", "line1": 18.5, "line2": 12.5, "line3": 6.0, "prediction1": 20.39, "prediction2": 14.34, "prediction3": 4.67, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 23.3, "kelly": 0.047, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.61, "prob2": 0.616, "prob3": 0.607, "hitRate1": 74.9, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Simone Fontecchio", "name2": "Miles Bridges", "name3": "Day'Ron Sharpe", "line1": 10.5, "line2": 19.5, "line3": 6.0, "prediction1": 11.86, "prediction2": 21.25, "prediction3": 7.27, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 18.25, "kelly": 0.036, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.598, "prob2": 0.614, "prob3": 0.597, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "LaMelo Ball", "name2": "Goga Bitadze", "name3": "Deandre Ayton", "line1": 19.5, "line2": 5.5, "line3": 14.5, "prediction1": 18.16, "prediction2": 6.5, "prediction3": 15.99, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 13.49, "kelly": 0.027, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.596, "prob2": 0.593, "prob3": 0.595, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Norman Powell", "name2": "Desmond Bane", "name3": "Kyle Filipowski", "line1": 23.5, "line2": 20.5, "line3": 8.5, "prediction1": 24.86, "prediction2": 21.96, "prediction3": 9.62, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.84, "kelly": 0.014, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.583, "prob2": 0.581, "prob3": 0.584, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 44.0, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Tyrese Maxey", "name2": "Dillon Brooks", "line1": 31.5, "line2": 18.5, "prediction1": 25.45, "prediction2": 23.81, "side1": "under", "side2": "over", "recommendation": 1, "ev": 92.22, "kelly": 0.461, "sigma1": "Med", "sigma2": "High", "prob1": 0.849, "prob2": 0.77, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 74.6, "l5_2": 0.6, "l15_2": 0.33},
    {"name1": "Bam Adebayo", "name2": "Brandon Miller", "line1": 21.5, "line2": 17.5, "prediction1": 15.79, "prediction2": 14.01, "side1": "under", "side2": "under", "recommendation": 0, "ev": 82.48, "kelly": 0.412, "sigma1": "High", "sigma2": "Low", "prob1": 0.808, "prob2": 0.769, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
    {"name1": "James Harden", "name2": "Rui Hachimura", "line1": 24.5, "line2": 11.5, "prediction1": 20.45, "prediction2": 15.43, "side1": "under", "side2": "over", "recommendation": 0, "ev": 61.06, "kelly": 0.305, "sigma1": "Med", "sigma2": "High", "prob1": 0.752, "prob2": 0.728, "hitRate1": 21.4, "l5_1": 0.8, "l15_1": 0.53, "hitRate2": 80.2, "l5_2": 0.6, "l15_2": 0.67},
    {"name1": "Deni Avdija", "name2": "Jake LaRavia", "line1": 23.5, "line2": 7.5, "prediction1": 26.88, "prediction2": 11.15, "side1": "over", "side2": "over", "recommendation": 0, "ev": 47.46, "kelly": 0.237, "sigma1": "High", "sigma2": "High", "prob1": 0.697, "prob2": 0.719, "hitRate1": 78.5, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 62.5, "l5_2": 0.2, "l15_2": 0.53},
    {"name1": "Brandon Ingram", "name2": "Donovan Clingan", "line1": 20.5, "line2": 9.5, "prediction1": 23.75, "prediction2": 12.47, "side1": "over", "side2": "over", "recommendation": 0, "ev": 38.94, "kelly": 0.195, "sigma1": "High", "sigma2": "Med", "prob1": 0.682, "prob2": 0.693, "hitRate1": 63.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 73.5, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Shai Gilgeous-Alexander", "name2": "Keyonte George", "line1": 31.5, "line2": 20.5, "prediction1": 29.12, "prediction2": 23.03, "side1": "under", "side2": "over", "recommendation": 0, "ev": 24.35, "kelly": 0.122, "sigma1": "High", "sigma2": "High", "prob1": 0.65, "prob2": 0.651, "hitRate1": 57.3, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 77.4, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Anthony Black", "name2": "Ziaire Williams", "line1": 13.5, "line2": 10.5, "prediction1": 15.81, "prediction2": 8.37, "side1": "over", "side2": "under", "recommendation": 0, "ev": 19.45, "kelly": 0.097, "sigma1": "High", "sigma2": "High", "prob1": 0.637, "prob2": 0.638, "hitRate1": 22.8, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 75.0, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Justin Edwards", "name2": "Kawhi Leonard", "line1": 9.5, "line2": 18.5, "prediction1": 11.59, "prediction2": 20.38, "side1": "over", "side2": "over", "recommendation": 0, "ev": 19.03, "kelly": 0.095, "sigma1": "High", "sigma2": "Med", "prob1": 0.636, "prob2": 0.637, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 90.3, "l5_2": 0.8, "l15_2": 0.27},
    {"name1": "Noah Clowney", "name2": "Evan Mobley", "line1": 12.5, "line2": 19.5, "prediction1": 14.66, "prediction2": 17.57, "side1": "over", "side2": "under", "recommendation": 0, "ev": 16.14, "kelly": 0.081, "sigma1": "High", "sigma2": "Med", "prob1": 0.63, "prob2": 0.627, "hitRate1": 60.7, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 56.9, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Davion Mitchell", "name2": "Onyeka Okongwu", "line1": 9.5, "line2": 18.5, "prediction1": 11.48, "prediction2": 16.45, "side1": "over", "side2": "under", "recommendation": 0, "ev": 14.77, "kelly": 0.074, "sigma1": "High", "sigma2": "High", "prob1": 0.624, "prob2": 0.626, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0},
];const underdogTriosData = [
    {"name1": "Tyrese Maxey", "name2": "Brandon Miller", "name3": "Dillon Brooks", "line1": 31.5, "line2": 17.5, "line3": 18.5, "prediction1": 25.45, "prediction2": 14.01, "prediction3": 23.81, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 171.37, "kelly": 0.343, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.849, "prob2": 0.769, "prob3": 0.77, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 74.6, "l5_3": 0.6, "l15_3": 0.33},
    {"name1": "Bam Adebayo", "name2": "James Harden", "name3": "Rui Hachimura", "line1": 21.5, "line2": 24.5, "line3": 11.5, "prediction1": 15.79, "prediction2": 20.45, "prediction3": 15.43, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": 138.87, "kelly": 0.278, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.808, "prob2": 0.752, "prob3": 0.728, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 21.4, "l5_2": 0.8, "l15_2": 0.53, "hitRate3": 80.2, "l5_3": 0.6, "l15_3": 0.67},
    {"name1": "Deni Avdija", "name2": "Donovan Clingan", "name3": "Jake LaRavia", "line1": 23.5, "line2": 9.5, "line3": 7.5, "prediction1": 26.88, "prediction2": 12.47, "prediction3": 11.15, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 87.57, "kelly": 0.175, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.697, "prob2": 0.693, "prob3": 0.719, "hitRate1": 78.5, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 73.5, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 62.5, "l5_3": 0.2, "l15_3": 0.53},
    {"name1": "Brandon Ingram", "name2": "Shai Gilgeous-Alexander", "name3": "Keyonte George", "line1": 20.5, "line2": 31.5, "line3": 20.5, "prediction1": 23.75, "prediction2": 29.12, "prediction3": 23.03, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 55.87, "kelly": 0.112, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.682, "prob2": 0.65, "prob3": 0.651, "hitRate1": 63.7, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 57.3, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 77.4, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Anthony Black", "name2": "Ziaire Williams", "name3": "Kawhi Leonard", "line1": 13.5, "line2": 10.5, "line3": 18.5, "prediction1": 15.81, "prediction2": 8.37, "prediction3": 20.38, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 39.7, "kelly": 0.079, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.637, "prob2": 0.638, "prob3": 0.637, "hitRate1": 22.8, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 75.0, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 90.3, "l5_3": 0.8, "l15_3": 0.27},
    {"name1": "Justin Edwards", "name2": "Noah Clowney", "name3": "Evan Mobley", "line1": 9.5, "line2": 12.5, "line3": 19.5, "prediction1": 11.59, "prediction2": 14.66, "prediction3": 17.57, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 35.63, "kelly": 0.071, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.636, "prob2": 0.63, "prob3": 0.627, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 60.7, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 56.9, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Davion Mitchell", "name2": "Onyeka Okongwu", "name3": "Jerami Grant", "line1": 9.5, "line2": 18.5, "line3": 18.5, "prediction1": 11.48, "prediction2": 16.45, "prediction3": 16.45, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 30.41, "kelly": 0.061, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.624, "prob2": 0.626, "prob3": 0.619, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 68.1, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Dru Smith", "name2": "Jamal Shead", "name3": "Darius Garland", "line1": 5.5, "line2": 6.5, "line3": 17.5, "prediction1": 6.94, "prediction2": 5.08, "prediction3": 15.51, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 27.12, "kelly": 0.054, "sigma1": "Low", "sigma2": "Low", "sigma3": "High", "prob1": 0.618, "prob2": 0.618, "prob3": 0.616, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 69.1, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 75.2, "l5_3": 0.4, "l15_3": 0.13},
    {"name1": "Simone Fontecchio", "name2": "LaMelo Ball", "name3": "Miles Bridges", "line1": 10.5, "line2": 19.5, "line3": 19.5, "prediction1": 11.86, "prediction2": 18.16, "prediction3": 21.25, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 18.03, "kelly": 0.036, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.598, "prob2": 0.596, "prob3": 0.614, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 0.0, "l5_3": 0.0, "l15_3": 0.0},
    {"name1": "Norman Powell", "name2": "Donovan Mitchell", "name3": "Kyle Filipowski", "line1": 23.5, "line2": 27.5, "line3": 8.5, "prediction1": 24.86, "prediction2": 29.0, "prediction3": 9.62, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.12, "kelly": 0.016, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.583, "prob2": 0.588, "prob3": 0.584, "hitRate1": 0.0, "l5_1": 0.0, "l15_1": 0.0, "hitRate2": 63.5, "l5_2": 0.8, "l15_2": 0.73, "hitRate3": 59.4, "l5_3": 0.4, "l15_3": 0.4},
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
    {"name": "Kawhi Leonard", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.903, "underPct": 0.097},
    {"name": "Rui Hachimura", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.802, "underPct": 0.198},
    {"name": "James Harden", "line": 24.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.786, "underPct": 0.214},
    {"name": "Deni Avdija", "line": 23.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.785, "underPct": 0.215},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.774, "underPct": 0.226},
    {"name": "Lauri Markkanen", "line": 26.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.757, "underPct": 0.243},
    {"name": "Dillon Brooks", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.746, "underPct": 0.254},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.735, "underPct": 0.265},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.694, "underPct": 0.306},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.678, "underPct": 0.322},
    {"name": "Jordan Goodwin", "line": 8.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.64, "underPct": 0.36},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.638, "underPct": 0.362},
    {"name": "Payton Pritchard", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.638, "underPct": 0.362},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.637, "underPct": 0.363},
    {"name": "Donovan Mitchell", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.635, "underPct": 0.365},
    {"name": "LeBron James", "line": 18.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.627, "underPct": 0.373},
    {"name": "Jake LaRavia", "line": 7.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.625, "underPct": 0.375},
    {"name": "Noah Clowney", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.607, "underPct": 0.393},
    {"name": "Kevin Love", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.601, "underPct": 0.399},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.594, "underPct": 0.406},
    {"name": "Derrick White", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.586, "underPct": 0.414},
    {"name": "Jalen Johnson", "line": 22.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.576, "underPct": 0.424},
    {"name": "Harrison Barnes", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.566, "underPct": 0.434},
    {"name": "Onyeka Okongwu", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.553, "underPct": 0.447},
    {"name": "Brice Sensabaugh", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.53, "underPct": 0.47},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.485, "underPct": 0.515},
    {"name": "Isaiah Joe", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.476, "underPct": 0.524},
    {"name": "Evan Mobley", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.431, "underPct": 0.569},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.427, "underPct": 0.573},
    {"name": "Ajay Mitchell", "line": 15.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.426, "underPct": 0.574},
    {"name": "Royce O'Neale", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.397, "underPct": 0.603},
    {"name": "Collin Gillespie", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.349, "underPct": 0.651},
    {"name": "De'Aaron Fox", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.344, "underPct": 0.656},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.319, "underPct": 0.681},
    {"name": "Franz Wagner", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.317, "underPct": 0.683},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.309, "underPct": 0.691},
    {"name": "Luka Doncic", "line": 31.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.298, "underPct": 0.702},
    {"name": "Desmond Bane", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.275, "underPct": 0.725},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.253, "underPct": 0.747},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.25, "underPct": 0.75},
    {"name": "Darius Garland", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.248, "underPct": 0.752},
    {"name": "Anthony Black", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.228, "underPct": 0.772},
    {"name": "Jeremy Sochan", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.225, "underPct": 0.775},
    {"name": "Brook Lopez", "line": 6.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.213, "underPct": 0.787},
    {"name": "Devin Booker", "line": 29.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.18, "underPct": 0.82},
    {"name": "Terance Mann", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.146, "underPct": 0.854},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.087, "underPct": 0.913},
];const underdogAssistsHitRates = [
    {"name": "Jamal Shead", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.66, "underPct": 0.34},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.513, "underPct": 0.487},
    {"name": "Jordan Goodwin", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.502, "underPct": 0.498},
    {"name": "Zaccharie Risacher", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.499, "underPct": 0.501},
    {"name": "Jake LaRavia", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.461, "underPct": 0.539},
    {"name": "Julian Champagnie", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.365, "underPct": 0.635},
    {"name": "Jeremy Sochan", "line": 2.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.351, "underPct": 0.649},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.345, "underPct": 0.655},
];const underdogReboundsHitRates = [
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.704, "underPct": 0.296},
    {"name": "Jalen Johnson", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.559, "underPct": 0.441},
    {"name": "Luka Garza", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.524, "underPct": 0.476},
    {"name": "Collin Gillespie", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.496, "underPct": 0.504},
    {"name": "Dyson Daniels", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.488, "underPct": 0.512},
    {"name": "Luguentz Dort", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Isaiah Hartenstein", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.476, "underPct": 0.524},
    {"name": "Day'Ron Sharpe", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.455, "underPct": 0.545},
    {"name": "Neemias Queta", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.425, "underPct": 0.575},
    {"name": "Alex Caruso", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.377, "underPct": 0.623},
    {"name": "Chet Holmgren", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.364, "underPct": 0.636},
    {"name": "Gradey Dick", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.304, "underPct": 0.696},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.255, "underPct": 0.745},
    {"name": "Luke Kornet", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.243, "underPct": 0.757},
    {"name": "Julian Champagnie", "line": 4.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.134, "underPct": 0.866},
];const underdogBlocksHitRates = [
];const underdogStealsHitRates = [
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.472, "underPct": 0.528},
    {"name": "Anthony Black", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.447, "underPct": 0.553},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.581, "underPct": 0.419},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Jalen Johnson", "line": 40.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyle Filipowski", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Love", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 34.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 30.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Caruso", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 37.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 38.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kawhi Leonard", "line": 25.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Goodwin", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 20.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deandre Ayton", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 43.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 38.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylin Williams", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anfernee Simons", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luka Garza", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rui Hachimura", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Dunn", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brice Sensabaugh", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ziaire Williams", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Gillespie", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Immanuel Quickley", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keldon Johnson", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dillon Brooks", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Harrison Barnes", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Payton Pritchard", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 36.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Royce O'Neale", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Marcus Smart", "line": 10.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Dunn", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 37.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 26.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Evan Mobley", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 30.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Terance Mann", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Day'Ron Sharpe", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luguentz Dort", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Murray", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sidy Cissoko", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ajay Mitchell", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brook Lopez", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Devin Booker", "line": 41.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 49.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jake LaRavia", "line": 12.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Mark Williams", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 32.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogPRHitRates = [
    {"name": "Jalen Johnson", "line": 32.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 23.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 27.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Donovan Clingan", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 32.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Lauri Markkanen", "line": 32.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 29.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Deni Avdija", "line": 30.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jaylen Brown", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keldon Johnson", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Hartenstein", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandon Ingram", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luka Doncic", "line": 39.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 33.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mark Williams", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 25.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.0, "l10": 0.0, "l15": 0.13, "overPct": 0.0, "underPct": 1.0},
];const underdogPAHitRates = [
    {"name": "Jalen Johnson", "line": 30.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dillon Brooks", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 21.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "James Harden", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 32.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Austin Reaves", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Chet Holmgren", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 38.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Gillespie", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 24.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luka Doncic", "line": 40.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Devin Booker", "line": 37.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "LeBron James", "line": 26.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogRAHitRates = [
    {"name": "Keyonte George", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremy Sochan", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Nickeil Alexander-Walker", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "James Harden", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "LeBron James", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mark Williams", "line": 9.5, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
];const underdogTurnoversHitRates = [
    {"name": "Desmond Bane", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Booker", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Darius Garland", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 4.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Sexton", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luka Doncic", "line": 3.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const underdogBlocksStealsHitRates = [
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
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

