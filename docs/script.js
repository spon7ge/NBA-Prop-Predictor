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
    {"name1": "Steven Adams", "name2": "Drew Eubanks", "line1": 6.5, "line2": 5.5, "prediction1": 4.33, "prediction2": 8.04, "side1": "under", "side2": "over", "recommendation": 0, "ev": -10.31, "kelly": 0.0, "sigma1": "Low", "sigma2": "Low", "prob1": 0.565, "prob2": 0.54, "hitRate1": 63.6, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 44.7, "l5_2": 0.4, "l15_2": 0.6},
    {"name1": "Sam Hauser", "name2": "Jeremy Sochan", "line1": 6.0, "line2": 6.5, "prediction1": 9.85, "prediction2": 11.72, "side1": "over", "side2": "over", "recommendation": 0, "ev": -15.82, "kelly": 0.0, "sigma1": "Med", "sigma2": "High", "prob1": 0.536, "prob2": 0.535, "hitRate1": 47.9, "l5_1": 0.6, "l15_1": 0.27, "hitRate2": 45.6, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Jordan Walsh", "name2": "Alex Caruso", "line1": 5.5, "line2": 5.5, "prediction1": 8.21, "prediction2": 7.8, "side1": "over", "side2": "over", "recommendation": 0, "ev": -16.37, "kelly": 0.0, "sigma1": "Low", "sigma2": "Low", "prob1": 0.535, "prob2": 0.532, "hitRate1": 42.3, "l5_1": 0.4, "l15_1": 0.53, "hitRate2": 40.2, "l5_2": 0.2, "l15_2": 0.47},
    {"name1": "Josh Okogie", "name2": "Jared McCain", "line1": 6.5, "line2": 11.5, "prediction1": 8.89, "prediction2": 7.88, "side1": "over", "side2": "under", "recommendation": 0, "ev": -17.48, "kelly": 0.0, "sigma1": "Low", "sigma2": "Med", "prob1": 0.53, "prob2": 0.529, "hitRate1": 32.4, "l5_1": 0.2, "l15_1": 0.6, "hitRate2": 83.2, "l5_2": 0.6, "l15_2": 0.2},
    {"name1": "Luguentz Dort", "name2": "Harrison Barnes", "line1": 7.5, "line2": 12.5, "prediction1": 11.97, "prediction2": 19.25, "side1": "over", "side2": "over", "recommendation": 0, "ev": -18.96, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.525, "prob2": 0.525, "hitRate1": 60.0, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 71.3, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Kris Murray", "name2": "Jock Landale", "line1": 5.5, "line2": 8.5, "prediction1": 6.89, "prediction2": 12.65, "side1": "over", "side2": "over", "recommendation": 0, "ev": -19.35, "kelly": 0.0, "sigma1": "Low", "sigma2": "High", "prob1": 0.524, "prob2": 0.524, "hitRate1": 51.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 82.9, "l5_2": 0.8, "l15_2": 0.73},
    {"name1": "Quentin Grimes", "name2": "Cam Spencer", "line1": 17.5, "line2": 9.5, "prediction1": 11.95, "prediction2": 14.12, "side1": "under", "side2": "over", "recommendation": 0, "ev": -20.06, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.521, "prob2": 0.522, "hitRate1": 61.6, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 59.7, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Mouhamed Gueye", "name2": "Cason Wallace", "line1": 6.5, "line2": 8.5, "prediction1": 8.27, "prediction2": 11.29, "side1": "over", "side2": "over", "recommendation": 0, "ev": -20.73, "kelly": 0.0, "sigma1": "Low", "sigma2": "Med", "prob1": 0.52, "prob2": 0.518, "hitRate1": 30.0, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 61.7, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Isaiah Collier", "name2": "Luke Kennard", "line1": 6.5, "line2": 6.5, "prediction1": 7.91, "prediction2": 7.97, "side1": "over", "side2": "over", "recommendation": 0, "ev": -21.41, "kelly": 0.0, "sigma1": "Low", "sigma2": "Med", "prob1": 0.517, "prob2": 0.517, "hitRate1": 77.1, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 49.2, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Neemias Queta", "name2": "Dylan Harper", "line1": 9.5, "line2": 10.5, "prediction1": 12.64, "prediction2": 14.07, "side1": "over", "side2": "over", "recommendation": 0, "ev": -21.57, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.516, "prob2": 0.517, "hitRate1": 70.9, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 68.1, "l5_2": 0.8, "l15_2": 0.47},
];const prizepicksTriosData = [
    {"name1": "Steven Adams", "name2": "Sam Hauser", "name3": "Drew Eubanks", "line1": 6.5, "line2": 6.0, "line3": 5.5, "prediction1": 4.33, "prediction2": 9.85, "prediction3": 8.04, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": -11.77, "kelly": 0.0, "sigma1": "Low", "sigma2": "Med", "sigma3": "Low", "prob1": 0.565, "prob2": 0.536, "prob3": 0.54, "hitRate1": 63.6, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 47.9, "l5_2": 0.6, "l15_2": 0.27, "hitRate3": 44.7, "l5_3": 0.4, "l15_3": 0.6},
    {"name1": "Jordan Walsh", "name2": "Alex Caruso", "name3": "Jeremy Sochan", "line1": 5.5, "line2": 5.5, "line3": 6.5, "prediction1": 8.21, "prediction2": 7.8, "prediction3": 11.72, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": -17.88, "kelly": 0.0, "sigma1": "Low", "sigma2": "Low", "sigma3": "High", "prob1": 0.535, "prob2": 0.532, "prob3": 0.535, "hitRate1": 42.3, "l5_1": 0.4, "l15_1": 0.53, "hitRate2": 40.2, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 45.6, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Josh Okogie", "name2": "Jared McCain", "name3": "Luguentz Dort", "line1": 6.5, "line2": 11.5, "line3": 7.5, "prediction1": 8.89, "prediction2": 7.88, "prediction3": 11.97, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": -20.4, "kelly": 0.0, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "prob1": 0.53, "prob2": 0.529, "prob3": 0.525, "hitRate1": 32.4, "l5_1": 0.2, "l15_1": 0.6, "hitRate2": 83.2, "l5_2": 0.6, "l15_2": 0.2, "hitRate3": 60.0, "l5_3": 0.4, "l15_3": 0.27},
    {"name1": "Kris Murray", "name2": "Harrison Barnes", "name3": "Jock Landale", "line1": 5.5, "line2": 12.5, "line3": 8.5, "prediction1": 6.89, "prediction2": 19.25, "prediction3": 12.65, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": -22.25, "kelly": 0.0, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.524, "prob2": 0.525, "prob3": 0.524, "hitRate1": 51.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 71.3, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 82.9, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Quentin Grimes", "name2": "Cason Wallace", "name3": "Cam Spencer", "line1": 17.5, "line2": 8.5, "line3": 9.5, "prediction1": 11.95, "prediction2": 11.29, "prediction3": 14.12, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": -23.95, "kelly": 0.0, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.521, "prob2": 0.518, "prob3": 0.522, "hitRate1": 61.6, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 61.7, "l5_2": 0.8, "l15_2": 0.6, "hitRate3": 59.7, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Isaiah Collier", "name2": "Mouhamed Gueye", "name3": "Dylan Harper", "line1": 6.5, "line2": 6.5, "line3": 10.5, "prediction1": 7.91, "prediction2": 8.27, "prediction3": 14.07, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": -24.93, "kelly": 0.0, "sigma1": "Low", "sigma2": "Low", "sigma3": "High", "prob1": 0.517, "prob2": 0.52, "prob3": 0.517, "hitRate1": 77.1, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 30.0, "l5_2": 0.4, "l15_2": 0.33, "hitRate3": 68.1, "l5_3": 0.8, "l15_3": 0.47},
    {"name1": "Luke Kennard", "name2": "Neemias Queta", "name3": "Immanuel Quickley", "line1": 6.5, "line2": 9.5, "line3": 17.5, "prediction1": 7.97, "prediction2": 12.64, "prediction3": 13.19, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": -25.68, "kelly": 0.0, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.517, "prob2": 0.516, "prob3": 0.516, "hitRate1": 49.2, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 70.9, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 67.0, "l5_3": 0.2, "l15_3": 0.4},
    {"name1": "Nickeil Alexander-Walker", "name2": "Darius Garland", "name3": "Zach LaVine", "line1": 18.5, "line2": 16.5, "line3": 20.5, "prediction1": 14.33, "prediction2": 12.52, "prediction3": 15.69, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": -26.01, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.515, "prob2": 0.516, "prob3": 0.515, "hitRate1": 25.2, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 78.4, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 72.8, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Brice Sensabaugh", "name2": "Paul George", "name3": "Dean Wade", "line1": 8.5, "line2": 14.5, "line3": 5.5, "prediction1": 10.94, "prediction2": 11.19, "prediction3": 6.27, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": -26.14, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.515, "prob2": 0.515, "prob3": 0.515, "hitRate1": 73.1, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 57.3, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 33.1, "l5_3": 0.0, "l15_3": 0.27},
    {"name1": "Onyeka Okongwu", "name2": "Scottie Barnes", "name3": "Jaden McDaniels", "line1": 18.5, "line2": 19.5, "line3": 12.5, "prediction1": 14.65, "prediction2": 15.26, "prediction3": 10.28, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": -26.32, "kelly": 0.0, "sigma1": "Med", "sigma2": "High", "sigma3": "Med", "prob1": 0.515, "prob2": 0.515, "prob3": 0.515, "hitRate1": 41.1, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 41.9, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 56.3, "l5_3": 0.4, "l15_3": 0.8},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Steven Adams", "name2": "Drew Eubanks", "line1": 6.5, "line2": 5.5, "prediction1": 4.33, "prediction2": 8.04, "side1": "under", "side2": "over", "recommendation": 0, "ev": -10.31, "kelly": 0.0, "sigma1": "Low", "sigma2": "Low", "prob1": 0.565, "prob2": 0.54, "hitRate1": 63.6, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 44.7, "l5_2": 0.4, "l15_2": 0.6},
    {"name1": "Jordan Walsh", "name2": "Alex Caruso", "line1": 5.5, "line2": 5.5, "prediction1": 8.21, "prediction2": 7.8, "side1": "over", "side2": "over", "recommendation": 0, "ev": -16.37, "kelly": 0.0, "sigma1": "Low", "sigma2": "Low", "prob1": 0.535, "prob2": 0.532, "hitRate1": 42.3, "l5_1": 0.4, "l15_1": 0.53, "hitRate2": 40.2, "l5_2": 0.2, "l15_2": 0.47},
    {"name1": "Luguentz Dort", "name2": "Harrison Barnes", "line1": 7.5, "line2": 12.5, "prediction1": 11.97, "prediction2": 19.25, "side1": "over", "side2": "over", "recommendation": 0, "ev": -18.96, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.525, "prob2": 0.525, "hitRate1": 60.0, "l5_1": 0.4, "l15_1": 0.27, "hitRate2": 71.3, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Kevin Durant", "name2": "Jock Landale", "line1": 25.5, "line2": 8.5, "prediction1": 16.92, "prediction2": 12.65, "side1": "under", "side2": "over", "recommendation": 0, "ev": -19.23, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.524, "prob2": 0.524, "hitRate1": 57.7, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 82.9, "l5_2": 0.8, "l15_2": 0.73},
    {"name1": "Kris Murray", "name2": "Anthony Edwards", "line1": 5.5, "line2": 28.5, "prediction1": 6.89, "prediction2": 19.23, "side1": "over", "side2": "under", "recommendation": 0, "ev": -19.4, "kelly": 0.0, "sigma1": "Low", "sigma2": "High", "prob1": 0.524, "prob2": 0.523, "hitRate1": 51.1, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 43.1, "l5_2": 0.8, "l15_2": 0.6},
    {"name1": "Donovan Mitchell", "name2": "Cam Spencer", "line1": 28.5, "line2": 9.5, "prediction1": 20.21, "prediction2": 14.12, "side1": "under", "side2": "over", "recommendation": 0, "ev": -19.98, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.521, "prob2": 0.522, "hitRate1": 36.0, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 59.7, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Mouhamed Gueye", "name2": "Brandon Ingram", "line1": 6.5, "line2": 23.5, "prediction1": 8.27, "prediction2": 17.18, "side1": "over", "side2": "under", "recommendation": 0, "ev": -20.78, "kelly": 0.0, "sigma1": "Low", "sigma2": "High", "prob1": 0.52, "prob2": 0.518, "hitRate1": 30.0, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 46.3, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Isaiah Collier", "name2": "Nickeil Alexander-Walker", "line1": 6.5, "line2": 19.5, "prediction1": 7.91, "prediction2": 14.33, "side1": "over", "side2": "under", "recommendation": 0, "ev": -21.17, "kelly": 0.0, "sigma1": "Low", "sigma2": "High", "prob1": 0.517, "prob2": 0.519, "hitRate1": 77.1, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 32.9, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Onyeka Okongwu", "name2": "Dylan Harper", "line1": 19.5, "line2": 10.5, "prediction1": 14.65, "prediction2": 14.07, "side1": "under", "side2": "over", "recommendation": 0, "ev": -21.3, "kelly": 0.0, "sigma1": "Med", "sigma2": "High", "prob1": 0.518, "prob2": 0.517, "hitRate1": 43.2, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 68.1, "l5_2": 0.8, "l15_2": 0.47},
    {"name1": "Tyrese Maxey", "name2": "Immanuel Quickley", "line1": 29.5, "line2": 17.5, "prediction1": 22.11, "prediction2": 13.19, "side1": "under", "side2": "under", "recommendation": 0, "ev": -21.62, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "prob1": 0.517, "prob2": 0.516, "hitRate1": 32.2, "l5_1": 0.2, "l15_1": 0.47, "hitRate2": 67.0, "l5_2": 0.2, "l15_2": 0.4},
];const underdogTriosData = [
    {"name1": "Steven Adams", "name2": "Jordan Walsh", "name3": "Drew Eubanks", "line1": 6.5, "line2": 5.5, "line3": 5.5, "prediction1": 4.33, "prediction2": 8.21, "prediction3": 8.04, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": -11.92, "kelly": 0.0, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.565, "prob2": 0.535, "prob3": 0.54, "hitRate1": 63.6, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 42.3, "l5_2": 0.4, "l15_2": 0.53, "hitRate3": 44.7, "l5_3": 0.4, "l15_3": 0.6},
    {"name1": "Alex Caruso", "name2": "Harrison Barnes", "name3": "Jock Landale", "line1": 5.5, "line2": 12.5, "line3": 8.5, "prediction1": 7.8, "prediction2": 19.25, "prediction3": 12.65, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": -21.01, "kelly": 0.0, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.532, "prob2": 0.525, "prob3": 0.524, "hitRate1": 40.2, "l5_1": 0.2, "l15_1": 0.47, "hitRate2": 71.3, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 82.9, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Mouhamed Gueye", "name2": "Luguentz Dort", "name3": "Cam Spencer", "line1": 6.5, "line2": 7.5, "line3": 9.5, "prediction1": 8.27, "prediction2": 11.97, "prediction3": 14.12, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": -22.94, "kelly": 0.0, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.52, "prob2": 0.525, "prob3": 0.522, "hitRate1": 30.0, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 60.0, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 59.7, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Nickeil Alexander-Walker", "name2": "Brandon Ingram", "name3": "Kris Murray", "line1": 19.5, "line2": 23.5, "line3": 5.5, "prediction1": 14.33, "prediction2": 17.18, "prediction3": 6.89, "side1": "under", "side2": "under", "side3": "over", "recommendation": 0, "ev": -24.09, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "prob1": 0.519, "prob2": 0.518, "prob3": 0.524, "hitRate1": 32.9, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 46.3, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 51.1, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Isaiah Collier", "name2": "Onyeka Okongwu", "name3": "Dylan Harper", "line1": 6.5, "line2": 19.5, "line3": 10.5, "prediction1": 7.91, "prediction2": 14.65, "prediction3": 14.07, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": -25.25, "kelly": 0.0, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "prob1": 0.517, "prob2": 0.518, "prob3": 0.517, "hitRate1": 77.1, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 43.2, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 68.1, "l5_3": 0.8, "l15_3": 0.47},
    {"name1": "Luke Kennard", "name2": "Dean Wade", "name3": "Immanuel Quickley", "line1": 6.5, "line2": 5.5, "line3": 17.5, "prediction1": 7.97, "prediction2": 6.27, "prediction3": 13.19, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": -25.82, "kelly": 0.0, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.517, "prob2": 0.515, "prob3": 0.516, "hitRate1": 49.2, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 33.1, "l5_2": 0.0, "l15_2": 0.27, "hitRate3": 67.0, "l5_3": 0.2, "l15_3": 0.4},
    {"name1": "Paul George", "name2": "Jaylon Tyson", "name3": "Zach LaVine", "line1": 14.5, "line2": 12.5, "line3": 20.5, "prediction1": 11.19, "prediction2": 10.14, "prediction3": 15.69, "side1": "under", "side2": "under", "side3": "under", "recommendation": 0, "ev": -26.16, "kelly": 0.0, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.515, "prob2": 0.515, "prob3": 0.515, "hitRate1": 34.3, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 56.3, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 72.8, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Brice Sensabaugh", "name2": "Scottie Barnes", "name3": "Jaden McDaniels", "line1": 8.5, "line2": 19.5, "line3": 12.5, "prediction1": 10.94, "prediction2": 15.26, "prediction3": 10.28, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": -26.28, "kelly": 0.0, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.515, "prob2": 0.515, "prob3": 0.515, "hitRate1": 73.1, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 41.9, "l5_2": 0.6, "l15_2": 0.33, "hitRate3": 56.3, "l5_3": 0.4, "l15_3": 0.8},
    {"name1": "Dyson Daniels", "name2": "Julian Champagnie", "name3": "Santi Aldama", "line1": 12.5, "line2": 12.5, "line3": 12.5, "prediction1": 10.08, "prediction2": 16.47, "prediction3": 16.19, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": -26.55, "kelly": 0.0, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.515, "prob2": 0.514, "prob3": 0.514, "hitRate1": 51.5, "l5_1": 0.4, "l15_1": 0.33, "hitRate2": 62.3, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 72.6, "l5_3": 0.8, "l15_3": 0.6},
    {"name1": "Jalen Johnson", "name2": "Precious Achiuwa", "name3": "Zion Williamson", "line1": 23.5, "line2": 6.5, "line3": 22.5, "prediction1": 18.59, "prediction2": 7.32, "prediction3": 18.36, "side1": "under", "side2": "over", "side3": "under", "recommendation": 0, "ev": -26.96, "kelly": 0.0, "sigma1": "High", "sigma2": "Low", "sigma3": "Med", "prob1": 0.514, "prob2": 0.513, "prob3": 0.513, "hitRate1": 57.1, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 53.9, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 51.0, "l5_3": 0.4, "l15_3": 0.33},
];const prizepicksPointsHitRates = [
    {"name": "Shaedon Sharpe", "line": 18.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.897, "underPct": 0.103},
    {"name": "Keyonte George", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.887, "underPct": 0.113},
    {"name": "Aaron Wiggins", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.842, "underPct": 0.158},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.841, "underPct": 0.159},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.829, "underPct": 0.171},
    {"name": "LeBron James", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.809, "underPct": 0.191},
    {"name": "Jordan Clarkson", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.774, "underPct": 0.226},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.771, "underPct": 0.229},
    {"name": "Isaiah Collier", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.771, "underPct": 0.229},
    {"name": "Jerami Grant", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.757, "underPct": 0.243},
    {"name": "Nickeil Alexander-Walker", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.748, "underPct": 0.252},
    {"name": "Austin Reaves", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.746, "underPct": 0.254},
    {"name": "Miles McBride", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.738, "underPct": 0.262},
    {"name": "Brice Sensabaugh", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.731, "underPct": 0.269},
    {"name": "Jaylen Brown", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.731, "underPct": 0.269},
    {"name": "Santi Aldama", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.726, "underPct": 0.274},
    {"name": "De'Aaron Fox", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.723, "underPct": 0.277},
    {"name": "Payton Pritchard", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.717, "underPct": 0.283},
    {"name": "Jalen Williams", "line": 16.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.715, "underPct": 0.285},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.713, "underPct": 0.287},
    {"name": "Neemias Queta", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.709, "underPct": 0.291},
    {"name": "Ajay Mitchell", "line": 10.5, "l5": 0.8, "l10": 0.9, "l15": 0.8, "overPct": 0.704, "underPct": 0.296},
    {"name": "Reed Sheppard", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.693, "underPct": 0.307},
    {"name": "Jalen Brunson", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.692, "underPct": 0.308},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.69, "underPct": 0.31},
    {"name": "Dylan Harper", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.681, "underPct": 0.319},
    {"name": "Tyrese Maxey", "line": 29.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.678, "underPct": 0.322},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.673, "underPct": 0.327},
    {"name": "Aaron Holiday", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.672, "underPct": 0.328},
    {"name": "Deni Avdija", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.668, "underPct": 0.332},
    {"name": "Keegan Murray", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.663, "underPct": 0.337},
    {"name": "Lauri Markkanen", "line": 26.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.652, "underPct": 0.348},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.64, "underPct": 0.36},
    {"name": "Karl-Anthony Towns", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.639, "underPct": 0.361},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.632, "underPct": 0.368},
    {"name": "Keldon Johnson", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.632, "underPct": 0.368},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.631, "underPct": 0.369},
    {"name": "Julian Champagnie", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.623, "underPct": 0.377},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.617, "underPct": 0.383},
    {"name": "DeMar DeRozan", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.616, "underPct": 0.384},
    {"name": "Josh Hart", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.616, "underPct": 0.384},
    {"name": "Donte DiVincenzo", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.615, "underPct": 0.385},
    {"name": "Derrick White", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.612, "underPct": 0.388},
    {"name": "Luguentz Dort", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Spencer", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.597, "underPct": 0.403},
    {"name": "Onyeka Okongwu", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.589, "underPct": 0.411},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.581, "underPct": 0.419},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.579, "underPct": 0.421},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.569, "underPct": 0.431},
    {"name": "Jeremiah Fears", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.559, "underPct": 0.441},
    {"name": "Amen Thompson", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.558, "underPct": 0.442},
    {"name": "Alperen Sengun", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.554, "underPct": 0.446},
    {"name": "Derik Queen", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.551, "underPct": 0.449},
    {"name": "Jaylon Tyson", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.551, "underPct": 0.449},
    {"name": "Sandro Mamukelashvili", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.55, "underPct": 0.45},
    {"name": "Jakob Poeltl", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.545, "underPct": 0.455},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.539, "underPct": 0.461},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.535, "underPct": 0.465},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.532, "underPct": 0.468},
    {"name": "Rui Hachimura", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.527, "underPct": 0.473},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.525, "underPct": 0.475},
    {"name": "Kris Murray", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.511, "underPct": 0.489},
    {"name": "Toumani Camara", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.493, "underPct": 0.507},
    {"name": "Luke Kennard", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.492, "underPct": 0.508},
    {"name": "Anfernee Simons", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.491, "underPct": 0.509},
    {"name": "Zion Williamson", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.49, "underPct": 0.51},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.485, "underPct": 0.515},
    {"name": "Sam Hauser", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.479, "underPct": 0.521},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.478, "underPct": 0.522},
    {"name": "Cedric Coward", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.476, "underPct": 0.524},
    {"name": "Deandre Ayton", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.472, "underPct": 0.528},
    {"name": "Jeremy Sochan", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.456, "underPct": 0.544},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.456, "underPct": 0.544},
    {"name": "De'Andre Hunter", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.45, "underPct": 0.55},
    {"name": "Drew Eubanks", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.447, "underPct": 0.553},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.439, "underPct": 0.561},
    {"name": "Jaden McDaniels", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.437, "underPct": 0.563},
    {"name": "Jalen Johnson", "line": 23.0, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.429, "underPct": 0.571},
    {"name": "Paul George", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.427, "underPct": 0.573},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.423, "underPct": 0.577},
    {"name": "Kevin Durant", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.423, "underPct": 0.577},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.41, "underPct": 0.59},
    {"name": "Julius Randle", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.404, "underPct": 0.596},
    {"name": "Alex Caruso", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.402, "underPct": 0.598},
    {"name": "Chet Holmgren", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.393, "underPct": 0.607},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.384, "underPct": 0.616},
    {"name": "Russell Westbrook", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.377, "underPct": 0.623},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.375, "underPct": 0.625},
    {"name": "Steven Adams", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.364, "underPct": 0.636},
    {"name": "VJ Edgecombe", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.36, "underPct": 0.64},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.331, "underPct": 0.669},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.33, "underPct": 0.67},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.329, "underPct": 0.671},
    {"name": "Josh Okogie", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.324, "underPct": 0.676},
    {"name": "Andre Drummond", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.308, "underPct": 0.692},
    {"name": "Mouhamed Gueye", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.274, "underPct": 0.726},
    {"name": "Zach LaVine", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.272, "underPct": 0.728},
    {"name": "Ja'Kobe Walter", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.267, "underPct": 0.733},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.233, "underPct": 0.767},
    {"name": "Saddiq Bey", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.22, "underPct": 0.78},
    {"name": "Mitchell Robinson", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.22, "underPct": 0.78},
    {"name": "Darius Garland", "line": 16.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.216, "underPct": 0.784},
    {"name": "Jared McCain", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.168, "underPct": 0.832},
];const prizepicksAssistsHitRates = [
    {"name": "De'Aaron Fox", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.684, "underPct": 0.316},
    {"name": "Josh Hart", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.65, "underPct": 0.35},
    {"name": "Jalen Williams", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.616, "underPct": 0.384},
    {"name": "Evan Mobley", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.611, "underPct": 0.389},
    {"name": "Keegan Murray", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.595, "underPct": 0.405},
    {"name": "Dyson Daniels", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.591, "underPct": 0.409},
    {"name": "Jeremy Sochan", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.584, "underPct": 0.416},
    {"name": "Keyonte George", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.569, "underPct": 0.431},
    {"name": "Payton Pritchard", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.567, "underPct": 0.433},
    {"name": "Deni Avdija", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.566, "underPct": 0.434},
    {"name": "Kentavious Caldwell-Pope", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.564, "underPct": 0.436},
    {"name": "Jaylen Brown", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.564, "underPct": 0.436},
    {"name": "Devin Vassell", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.539, "underPct": 0.461},
    {"name": "Jamal Shead", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.539, "underPct": 0.461},
    {"name": "Jalen Johnson", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.529, "underPct": 0.471},
    {"name": "Donovan Mitchell", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.525, "underPct": 0.475},
    {"name": "Neemias Queta", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.522, "underPct": 0.478},
    {"name": "LeBron James", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.521, "underPct": 0.479},
    {"name": "Derrick White", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.517, "underPct": 0.483},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.503, "underPct": 0.497},
    {"name": "Jalen Brunson", "line": 6.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.499, "underPct": 0.501},
    {"name": "VJ Edgecombe", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.484, "underPct": 0.516},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.478, "underPct": 0.522},
    {"name": "Alperen Sengun", "line": 6.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.468, "underPct": 0.532},
    {"name": "Shai Gilgeous-Alexander", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.468, "underPct": 0.532},
    {"name": "Derik Queen", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.465, "underPct": 0.535},
    {"name": "Luguentz Dort", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.465, "underPct": 0.535},
    {"name": "Isaiah Collier", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.454, "underPct": 0.546},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.449, "underPct": 0.551},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Drew Eubanks", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.41, "underPct": 0.59},
    {"name": "Steven Adams", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.395, "underPct": 0.605},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.393, "underPct": 0.607},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.383, "underPct": 0.617},
    {"name": "Julius Randle", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.383, "underPct": 0.617},
    {"name": "Anthony Edwards", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.369, "underPct": 0.631},
    {"name": "Tyrese Maxey", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.36, "underPct": 0.64},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.357, "underPct": 0.643},
    {"name": "Darius Garland", "line": 7.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.357, "underPct": 0.643},
    {"name": "Nickeil Alexander-Walker", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.321, "underPct": 0.679},
    {"name": "Zion Williamson", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.316, "underPct": 0.684},
    {"name": "Amen Thompson", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.307, "underPct": 0.693},
    {"name": "Cam Spencer", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.282, "underPct": 0.718},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.226, "underPct": 0.774},
    {"name": "Reed Sheppard", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.213, "underPct": 0.787},
    {"name": "Austin Reaves", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.198, "underPct": 0.802},
];const prizepicksReboundsHitRates = [
    {"name": "Shaedon Sharpe", "line": 3.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.781, "underPct": 0.219},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.751, "underPct": 0.249},
    {"name": "Jock Landale", "line": 4.5, "l5": 0.6, "l10": 0.8, "l15": 0.8, "overPct": 0.671, "underPct": 0.329},
    {"name": "Jerami Grant", "line": 3.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.644, "underPct": 0.356},
    {"name": "Dylan Harper", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.633, "underPct": 0.367},
    {"name": "Austin Reaves", "line": 5.0, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.63, "underPct": 0.37},
    {"name": "VJ Edgecombe", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.614, "underPct": 0.386},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.606, "underPct": 0.394},
    {"name": "Amen Thompson", "line": 6.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.604, "underPct": 0.396},
    {"name": "Keldon Johnson", "line": 6.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.602, "underPct": 0.398},
    {"name": "Saddiq Bey", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.602, "underPct": 0.398},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.595, "underPct": 0.405},
    {"name": "Jakob Poeltl", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.578, "underPct": 0.422},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.577, "underPct": 0.423},
    {"name": "Kris Murray", "line": 4.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.573, "underPct": 0.427},
    {"name": "Cedric Coward", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.568, "underPct": 0.432},
    {"name": "Immanuel Quickley", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.562, "underPct": 0.438},
    {"name": "Jaylon Tyson", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.554, "underPct": 0.446},
    {"name": "Scottie Barnes", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.551, "underPct": 0.449},
    {"name": "Payton Pritchard", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.55, "underPct": 0.45},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.55, "underPct": 0.45},
    {"name": "Deni Avdija", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.549, "underPct": 0.451},
    {"name": "Paul George", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.546, "underPct": 0.454},
    {"name": "Keegan Murray", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.532, "underPct": 0.468},
    {"name": "Steven Adams", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.522, "underPct": 0.478},
    {"name": "Jaylen Brown", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.519, "underPct": 0.481},
    {"name": "Dyson Daniels", "line": 7.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.516, "underPct": 0.484},
    {"name": "Donovan Mitchell", "line": 4.0, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.512, "underPct": 0.488},
    {"name": "Evan Mobley", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.511, "underPct": 0.489},
    {"name": "Santi Aldama", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.507, "underPct": 0.493},
    {"name": "Harrison Barnes", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.507, "underPct": 0.493},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.505, "underPct": 0.495},
    {"name": "Derrick White", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.503, "underPct": 0.497},
    {"name": "Naz Reid", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.498, "underPct": 0.502},
    {"name": "Neemias Queta", "line": 9.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.478, "underPct": 0.522},
    {"name": "De'Aaron Fox", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.47, "underPct": 0.53},
    {"name": "Alperen Sengun", "line": 9.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.467, "underPct": 0.533},
    {"name": "Shai Gilgeous-Alexander", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.464, "underPct": 0.536},
    {"name": "Dean Wade", "line": 4.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.462, "underPct": 0.538},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.461, "underPct": 0.539},
    {"name": "Jaylen Wells", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.458, "underPct": 0.542},
    {"name": "Alex Caruso", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.456, "underPct": 0.544},
    {"name": "Lauri Markkanen", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.45, "underPct": 0.55},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.447, "underPct": 0.553},
    {"name": "Luke Kornet", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.43, "underPct": 0.57},
    {"name": "Sandro Mamukelashvili", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.427, "underPct": 0.573},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.424, "underPct": 0.576},
    {"name": "Josh Hart", "line": 9.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.423, "underPct": 0.577},
    {"name": "Zion Williamson", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.423, "underPct": 0.577},
    {"name": "Jaden McDaniels", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.418, "underPct": 0.582},
    {"name": "LeBron James", "line": 6.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.414, "underPct": 0.586},
    {"name": "Rui Hachimura", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.408, "underPct": 0.592},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.407, "underPct": 0.593},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.388, "underPct": 0.612},
    {"name": "Deandre Ayton", "line": 9.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.383, "underPct": 0.617},
    {"name": "Jalen Johnson", "line": 10.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.379, "underPct": 0.621},
    {"name": "Julian Champagnie", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.368, "underPct": 0.632},
    {"name": "Jalen Williams", "line": 4.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.362, "underPct": 0.638},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.355, "underPct": 0.645},
    {"name": "Brandon Ingram", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.351, "underPct": 0.649},
    {"name": "Drew Eubanks", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.338, "underPct": 0.662},
    {"name": "Derik Queen", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.304, "underPct": 0.696},
    {"name": "Jordan Walsh", "line": 5.0, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.277, "underPct": 0.723},
    {"name": "Trey Murphy III", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.275, "underPct": 0.725},
    {"name": "De'Andre Hunter", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.253, "underPct": 0.747},
    {"name": "Chet Holmgren", "line": 9.0, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.248, "underPct": 0.752},
    {"name": "Kevin Durant", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.237, "underPct": 0.763},
    {"name": "Mouhamed Gueye", "line": 5.0, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.186, "underPct": 0.814},
    {"name": "Ace Bailey", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.163, "underPct": 0.837},
    {"name": "Quentin Grimes", "line": 4.0, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.147, "underPct": 0.853},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.128, "underPct": 0.872},
];const prizepicksBlocksHitRates = [
    {"name": "Kyle Filipowski", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.531, "underPct": 0.469},
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.513, "underPct": 0.487},
    {"name": "Jakob Poeltl", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.504, "underPct": 0.496},
    {"name": "Karl-Anthony Towns", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.503, "underPct": 0.497},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.409, "underPct": 0.591},
    {"name": "Shai Gilgeous-Alexander", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.316, "underPct": 0.684},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.32, "underPct": 0.68},
    {"name": "Derik Queen", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.634, "underPct": 0.366},
];const prizepicksStealsHitRates = [
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.519, "underPct": 0.481},
    {"name": "Reed Sheppard", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.58, "underPct": 0.42},
    {"name": "Brice Sensabaugh", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.477, "underPct": 0.523},
    {"name": "Dean Wade", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.529, "underPct": 0.471},
    {"name": "Donovan Clingan", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.378, "underPct": 0.622},
    {"name": "Kentavious Caldwell-Pope", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.499, "underPct": 0.501},
    {"name": "Malik Monk", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.487, "underPct": 0.513},
    {"name": "Zach Edey", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.586, "underPct": 0.414},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.527, "underPct": 0.473},
    {"name": "Trey Murphy III", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Keldon Johnson", "line": 20.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keyonte George", "line": 30.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Santi Aldama", "line": 22.5, "l5": 1.0, "l10": 0.9, "l15": 0.6, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Shai Gilgeous-Alexander", "line": 43.0, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jakob Poeltl", "line": 19.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jock Landale", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julian Champagnie", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deandre Ayton", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Anthony Edwards", "line": 37.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Saddiq Bey", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Caruso", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles McBride", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Murray", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremy Sochan", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dean Wade", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach Edey", "line": 25.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Holiday", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Austin Reaves", "line": 36.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drew Eubanks", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nickeil Alexander-Walker", "line": 27.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Reed Sheppard", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kennard", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 38.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jaylen Brown", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Wells", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 42.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dylan Harper", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 37.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brice Sensabaugh", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mouhamed Gueye", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 25.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Aaron Fox", "line": 34.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ja'Kobe Walter", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 34.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 38.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 41.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 30.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 32.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 33.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quentin Grimes", "line": 26.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Paul George", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Chet Holmgren", "line": 29.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rui Hachimura", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 30.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Williams", "line": 25.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Keldon Johnson", "line": 18.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 30.5, "l5": 1.0, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 36.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 23.0, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Donovan Clingan", "line": 19.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 33.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 27.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Saddiq Bey", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deandre Ayton", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Holiday", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keyonte George", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jock Landale", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach Edey", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 17.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 18.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Murray", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Drew Eubanks", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 21.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Wells", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keegan Murray", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rudy Gobert", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Hart", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremy Sochan", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rui Hachimura", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 19.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 23.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 32.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Isaiah Collier", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Steven Adams", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dyson Daniels", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Payton Pritchard", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 30.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sam Hauser", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylon Tyson", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dylan Harper", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ace Bailey", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 31.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden McDaniels", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derrick White", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ja'Kobe Walter", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zaccharie Risacher", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Walsh", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mouhamed Gueye", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Luke Kennard", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Caruso", "line": 8.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cason Wallace", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.47, "overPct": 0.2, "underPct": 0.8},
    {"name": "De'Andre Hunter", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 35.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Luguentz Dort", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 22.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 33.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Luke Kornet", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Williams", "line": 21.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Onyeka Okongwu", "line": 27.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const prizepicksPAHitRates = [
    {"name": "Keyonte George", "line": 27.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Santi Aldama", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sandro Mamukelashvili", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Wells", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 37.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Shaedon Sharpe", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylen Brown", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keegan Murray", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Deandre Ayton", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Dean Wade", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 35.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Walsh", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Scottie Barnes", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dylan Harper", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Austin Reaves", "line": 31.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 32.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Collier", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Onyeka Okongwu", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach Edey", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "VJ Edgecombe", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Spencer", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kennard", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Darius Garland", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derrick White", "line": 20.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Devin Vassell", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 15.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Russell Westbrook", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden McDaniels", "line": 14.5, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 31.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lauri Markkanen", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Filipowski", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dyson Daniels", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zaccharie Risacher", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mouhamed Gueye", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alex Caruso", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Chet Holmgren", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ja'Kobe Walter", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cedric Coward", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Naz Reid", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Tyrese Maxey", "line": 36.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Steven Adams", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Fears", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 17.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jalen Williams", "line": 21.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Rui Hachimura", "line": 14.5, "l5": 0.0, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
];const prizepicksRAHitRates = [
    {"name": "Keyonte George", "line": 9.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jeremiah Fears", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "VJ Edgecombe", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 6.0, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rudy Gobert", "line": 12.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julian Champagnie", "line": 7.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jakob Poeltl", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Brown", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Harrison Barnes", "line": 5.5, "l5": 0.8, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lauri Markkanen", "line": 8.0, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Johnson", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Wells", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Steven Adams", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 12.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 9.0, "l5": 0.6, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Austin Reaves", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaden McDaniels", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Murray", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 8.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cam Spencer", "line": 7.0, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Payton Pritchard", "line": 8.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Santi Aldama", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alex Caruso", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kentavious Caldwell-Pope", "line": 4.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "LeBron James", "line": 13.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Russell Westbrook", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zion Williamson", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dylan Harper", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 15.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Paul George", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jerami Grant", "line": 6.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 10.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donovan Mitchell", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mikal Bridges", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Reed Sheppard", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Chet Holmgren", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ace Bailey", "line": 6.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 14.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Onyeka Okongwu", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Aaron Holiday", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cason Wallace", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cedric Coward", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Naz Reid", "line": 7.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 9.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 9.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Williams", "line": 9.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Miles McBride", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 4.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
];const prizepicksTurnoversHitRates = [
    {"name": "Jaden McDaniels", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Walsh", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deandre Ayton", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyle Filipowski", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mouhamed Gueye", "line": 0.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Zion Williamson", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Derik Queen", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Nickeil Alexander-Walker", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luguentz Dort", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alex Caruso", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Steven Adams", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Williams", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Shai Gilgeous-Alexander", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
];const underdogPointsHitRates = [
    {"name": "Keyonte George", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.887, "underPct": 0.113},
    {"name": "Shaedon Sharpe", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.852, "underPct": 0.148},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.841, "underPct": 0.159},
    {"name": "Jock Landale", "line": 8.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.829, "underPct": 0.171},
    {"name": "LeBron James", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.809, "underPct": 0.191},
    {"name": "Jordan Clarkson", "line": 9.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.774, "underPct": 0.226},
    {"name": "Isaiah Collier", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.771, "underPct": 0.229},
    {"name": "Jerami Grant", "line": 17.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.757, "underPct": 0.243},
    {"name": "Austin Reaves", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.746, "underPct": 0.254},
    {"name": "Deni Avdija", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.739, "underPct": 0.261},
    {"name": "Miles McBride", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.738, "underPct": 0.262},
    {"name": "Brice Sensabaugh", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.731, "underPct": 0.269},
    {"name": "Jaylen Brown", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.731, "underPct": 0.269},
    {"name": "Santi Aldama", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.726, "underPct": 0.274},
    {"name": "De'Aaron Fox", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.723, "underPct": 0.277},
    {"name": "Lauri Markkanen", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.72, "underPct": 0.28},
    {"name": "Payton Pritchard", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.717, "underPct": 0.283},
    {"name": "Jalen Williams", "line": 16.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.715, "underPct": 0.285},
    {"name": "Shai Gilgeous-Alexander", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.715, "underPct": 0.285},
    {"name": "Harrison Barnes", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.713, "underPct": 0.287},
    {"name": "Reed Sheppard", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.693, "underPct": 0.307},
    {"name": "Jalen Brunson", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.692, "underPct": 0.308},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.69, "underPct": 0.31},
    {"name": "Dylan Harper", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.681, "underPct": 0.319},
    {"name": "Tyrese Maxey", "line": 29.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.678, "underPct": 0.322},
    {"name": "Aaron Holiday", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.672, "underPct": 0.328},
    {"name": "Nickeil Alexander-Walker", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.671, "underPct": 0.329},
    {"name": "Keegan Murray", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.663, "underPct": 0.337},
    {"name": "Paul George", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.657, "underPct": 0.343},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.64, "underPct": 0.36},
    {"name": "Karl-Anthony Towns", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.639, "underPct": 0.361},
    {"name": "Jaylen Wells", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.632, "underPct": 0.368},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.631, "underPct": 0.369},
    {"name": "Julian Champagnie", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.623, "underPct": 0.377},
    {"name": "Donte DiVincenzo", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.615, "underPct": 0.385},
    {"name": "Derrick White", "line": 16.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.612, "underPct": 0.388},
    {"name": "Luguentz Dort", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cam Spencer", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.597, "underPct": 0.403},
    {"name": "Scottie Barnes", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.581, "underPct": 0.419},
    {"name": "Devin Vassell", "line": 16.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.579, "underPct": 0.421},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.569, "underPct": 0.431},
    {"name": "Onyeka Okongwu", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.568, "underPct": 0.432},
    {"name": "Jeremiah Fears", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.559, "underPct": 0.441},
    {"name": "Amen Thompson", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.558, "underPct": 0.442},
    {"name": "Alperen Sengun", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.554, "underPct": 0.446},
    {"name": "Derik Queen", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.551, "underPct": 0.449},
    {"name": "VJ Edgecombe", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.6, "overPct": 0.547, "underPct": 0.453},
    {"name": "Jakob Poeltl", "line": 10.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.545, "underPct": 0.455},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.539, "underPct": 0.461},
    {"name": "Brandon Ingram", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.537, "underPct": 0.463},
    {"name": "Ace Bailey", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.535, "underPct": 0.465},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.532, "underPct": 0.468},
    {"name": "Rui Hachimura", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.527, "underPct": 0.473},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.525, "underPct": 0.475},
    {"name": "Kris Murray", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.511, "underPct": 0.489},
    {"name": "Luke Kennard", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.492, "underPct": 0.508},
    {"name": "Zion Williamson", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.49, "underPct": 0.51},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.486, "underPct": 0.514},
    {"name": "Dyson Daniels", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.33, "overPct": 0.485, "underPct": 0.515},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.478, "underPct": 0.522},
    {"name": "Deandre Ayton", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.464, "underPct": 0.536},
    {"name": "De'Andre Hunter", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.45, "underPct": 0.55},
    {"name": "Drew Eubanks", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.447, "underPct": 0.553},
    {"name": "Zaccharie Risacher", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.439, "underPct": 0.561},
    {"name": "Jaylon Tyson", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.437, "underPct": 0.563},
    {"name": "Jaden McDaniels", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.437, "underPct": 0.563},
    {"name": "Jalen Johnson", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.429, "underPct": 0.571},
    {"name": "Kevin Durant", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.423, "underPct": 0.577},
    {"name": "Jordan Walsh", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.423, "underPct": 0.577},
    {"name": "Svi Mykhailiuk", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.41, "underPct": 0.59},
    {"name": "Julius Randle", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.404, "underPct": 0.596},
    {"name": "Alex Caruso", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.402, "underPct": 0.598},
    {"name": "Russell Westbrook", "line": 14.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.377, "underPct": 0.623},
    {"name": "Trey Murphy III", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.375, "underPct": 0.625},
    {"name": "Cedric Coward", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.367, "underPct": 0.633},
    {"name": "Steven Adams", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.364, "underPct": 0.636},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.331, "underPct": 0.669},
    {"name": "Immanuel Quickley", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.33, "underPct": 0.67},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.329, "underPct": 0.671},
    {"name": "Andre Drummond", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.308, "underPct": 0.692},
    {"name": "Mouhamed Gueye", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Darius Garland", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.298, "underPct": 0.702},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.274, "underPct": 0.726},
    {"name": "Zach LaVine", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.272, "underPct": 0.728},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.233, "underPct": 0.767},
    {"name": "Saddiq Bey", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.22, "underPct": 0.78},
];const underdogAssistsHitRates = [
    {"name": "Josh Hart", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.65, "underPct": 0.35},
    {"name": "Evan Mobley", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.611, "underPct": 0.389},
    {"name": "Keegan Murray", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.595, "underPct": 0.405},
    {"name": "Jeremy Sochan", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.584, "underPct": 0.416},
    {"name": "Keyonte George", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.569, "underPct": 0.431},
    {"name": "Jaylen Brown", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.564, "underPct": 0.436},
    {"name": "Kentavious Caldwell-Pope", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.564, "underPct": 0.436},
    {"name": "Devin Vassell", "line": 2.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.539, "underPct": 0.461},
    {"name": "Neemias Queta", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.522, "underPct": 0.478},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.503, "underPct": 0.497},
    {"name": "Shai Gilgeous-Alexander", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.468, "underPct": 0.532},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Steven Adams", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.395, "underPct": 0.605},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.226, "underPct": 0.774},
];const underdogReboundsHitRates = [
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.751, "underPct": 0.249},
    {"name": "Amen Thompson", "line": 6.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.604, "underPct": 0.396},
    {"name": "Keldon Johnson", "line": 6.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.602, "underPct": 0.398},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.595, "underPct": 0.405},
    {"name": "Russell Westbrook", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.577, "underPct": 0.423},
    {"name": "Cedric Coward", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.568, "underPct": 0.432},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.55, "underPct": 0.45},
    {"name": "Deni Avdija", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.549, "underPct": 0.451},
    {"name": "Keegan Murray", "line": 7.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.532, "underPct": 0.468},
    {"name": "Steven Adams", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.522, "underPct": 0.478},
    {"name": "Harrison Barnes", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.507, "underPct": 0.493},
    {"name": "Brice Sensabaugh", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.475, "underPct": 0.525},
    {"name": "De'Aaron Fox", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.47, "underPct": 0.53},
    {"name": "Jaylen Wells", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.458, "underPct": 0.542},
    {"name": "Alex Caruso", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.456, "underPct": 0.544},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.447, "underPct": 0.553},
    {"name": "Deandre Ayton", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.435, "underPct": 0.565},
    {"name": "Jamal Shead", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.429, "underPct": 0.571},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.424, "underPct": 0.576},
    {"name": "Jalen Williams", "line": 4.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.362, "underPct": 0.638},
    {"name": "Zach Edey", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.355, "underPct": 0.645},
    {"name": "Onyeka Okongwu", "line": 8.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.16, "underPct": 0.84},
];const underdogBlocksHitRates = [
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.409, "underPct": 0.591},
];const underdogStealsHitRates = [
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.519, "underPct": 0.481},
    {"name": "Trey Murphy III", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Keyonte George", "line": 30.5, "l5": 1.0, "l10": 0.8, "l15": 0.8, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Keldon Johnson", "line": 20.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Santi Aldama", "line": 22.5, "l5": 1.0, "l10": 0.9, "l15": 0.6, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dyson Daniels", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jakob Poeltl", "line": 19.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Anthony Edwards", "line": 37.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shaedon Sharpe", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Deandre Ayton", "line": 25.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Saddiq Bey", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 20.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jock Landale", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach Edey", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Brunson", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keegan Murray", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Ingram", "line": 33.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 32.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Neemias Queta", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles McBride", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Murray", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dean Wade", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ace Bailey", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Reed Sheppard", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Austin Reaves", "line": 36.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Svi Mykhailiuk", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Wells", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremy Sochan", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Johnson", "line": 42.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kennard", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cam Spencer", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 38.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Darius Garland", "line": 26.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alex Caruso", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dylan Harper", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 33.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaden McDaniels", "line": 19.5, "l5": 0.4, "l10": 0.7, "l15": 0.8, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kentavious Caldwell-Pope", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cedric Coward", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Precious Achiuwa", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Harrison Barnes", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Walsh", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 37.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyle Filipowski", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mouhamed Gueye", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Payton Pritchard", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derrick White", "line": 25.5, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ja'Kobe Walter", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 34.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lauri Markkanen", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Clarkson", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "LeBron James", "line": 32.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Julius Randle", "line": 33.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 30.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Steven Adams", "line": 17.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Rui Hachimura", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 32.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 41.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Paul George", "line": 21.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Karl-Anthony Towns", "line": 38.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luke Kornet", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 28.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 30.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Williams", "line": 25.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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
    {"name": "Jakob Poeltl", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Filipowski", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "VJ Edgecombe", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keldon Johnson", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julian Champagnie", "line": 7.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylen Brown", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Josh Hart", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach Edey", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Steven Adams", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Neemias Queta", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Aaron Fox", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Santi Aldama", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deandre Ayton", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alperen Sengun", "line": 15.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 10.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Collier", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
];const underdogTurnoversHitRates = [
    {"name": "Kevin Durant", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Johnson", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Brown", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 2.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Austin Reaves", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
];const underdogBlocksStealsHitRates = [
    {"name": "Scottie Barnes", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shai Gilgeous-Alexander", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
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

