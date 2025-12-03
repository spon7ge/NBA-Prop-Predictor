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
    {"name1": "Patrick Williams", "name2": "Drew Eubanks", "line1": 8.5, "line2": 4.5, "prediction1": 16.95, "prediction2": 11.79, "side1": "over", "side2": "over", "recommendation": 1, "ev": 151.07, "kelly": 0.755, "sigma1": "Med", "sigma2": "Low", "prob1": 0.925, "prob2": 0.923, "hitRate1": 13.1, "l5_1": 0.0, "l15_1": 0.2, "hitRate2": 57.6, "l5_2": 0.2, "l15_2": 0.6},
    {"name1": "Jeremy Sochan", "name2": "Russell Westbrook", "line1": 5.5, "line2": 13.0, "prediction1": 11.98, "prediction2": 23.9, "side1": "over", "side2": "over", "recommendation": 1, "ev": 133.52, "kelly": 0.668, "sigma1": "Med", "sigma2": "High", "prob1": 0.874, "prob2": 0.908, "hitRate1": 54.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 44.0, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Jonathan Isaac", "name2": "Steven Adams", "line1": 2.5, "line2": 5.5, "prediction1": 6.24, "prediction2": 9.76, "side1": "over", "side2": "over", "recommendation": 0, "ev": 115.16, "kelly": 0.576, "sigma1": "Low", "sigma2": "Low", "prob1": 0.846, "prob2": 0.865, "hitRate1": 62.2, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 54.4, "l5_2": 0.6, "l15_2": 0.4},
    {"name1": "Ben Sheppard", "name2": "Luke Kornet", "line1": 5.5, "line2": 6.5, "prediction1": 10.52, "prediction2": 11.69, "side1": "over", "side2": "over", "recommendation": 1, "ev": 106.26, "kelly": 0.531, "sigma1": "Low", "sigma2": "Low", "prob1": 0.837, "prob2": 0.839, "hitRate1": 50.4, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 40.1, "l5_2": 0.4, "l15_2": 0.4},
    {"name1": "Ausar Thompson", "name2": "Ryan Nembhard", "line1": 10.5, "line2": 8.5, "prediction1": 17.47, "prediction2": 14.78, "side1": "over", "side2": "over", "recommendation": 1, "ev": 98.22, "kelly": 0.491, "sigma1": "High", "sigma2": "High", "prob1": 0.834, "prob2": 0.809, "hitRate1": 27.9, "l5_1": 0.2, "l15_1": 0.47, "hitRate2": 46.1, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Kobe Brown", "name2": "Ryan Rollins", "line1": 4.0, "line2": 14.5, "prediction1": 7.43, "prediction2": 21.57, "side1": "over", "side2": "over", "recommendation": 0, "ev": 86.61, "kelly": 0.433, "sigma1": "Low", "sigma2": "High", "prob1": 0.8, "prob2": 0.794, "hitRate1": 38.7, "l5_1": 0.6, "l15_1": 0.33, "hitRate2": 88.9, "l5_2": 0.2, "l15_2": 0.53},
    {"name1": "Liam McNeeley", "name2": "Jalen Duren", "line1": 4.5, "line2": 17.5, "prediction1": 8.09, "prediction2": 23.79, "side1": "over", "side2": "over", "recommendation": 0, "ev": 79.72, "kelly": 0.399, "sigma1": "Low", "sigma2": "High", "prob1": 0.783, "prob2": 0.781, "hitRate1": 42.7, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 71.1, "l5_2": 0.4, "l15_2": 0.67},
    {"name1": "Julian Champagnie", "name2": "Keegan Murray", "line1": 9.5, "line2": 16.5, "prediction1": 15.02, "prediction2": 22.85, "side1": "over", "side2": "over", "recommendation": 1, "ev": 75.12, "kelly": 0.376, "sigma1": "High", "sigma2": "High", "prob1": 0.769, "prob2": 0.775, "hitRate1": 75.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 64.2, "l5_2": 0.8, "l15_2": 0.27},
    {"name1": "Darius Garland", "name2": "Harrison Barnes", "line1": 18.5, "line2": 11.5, "prediction1": 24.68, "prediction2": 17.36, "side1": "over", "side2": "over", "recommendation": 1, "ev": 71.82, "kelly": 0.359, "sigma1": "High", "sigma2": "High", "prob1": 0.761, "prob2": 0.768, "hitRate1": 16.5, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 85.8, "l5_2": 0.4, "l15_2": 0.6},
    {"name1": "Tyrese Martin", "name2": "Zach LaVine", "line1": 9.5, "line2": 18.5, "prediction1": 14.89, "prediction2": 24.05, "side1": "over", "side2": "over", "recommendation": 1, "ev": 69.67, "kelly": 0.348, "sigma1": "High", "sigma2": "High", "prob1": 0.76, "prob2": 0.76, "hitRate1": 59.2, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 41.6, "l5_2": 0.4, "l15_2": 0.47},
];const prizepicksTriosData = [
    {"name1": "Jeremy Sochan", "name2": "Patrick Williams", "name3": "Russell Westbrook", "line1": 5.5, "line2": 8.5, "line3": 13.0, "prediction1": 11.98, "prediction2": 16.95, "prediction3": 23.9, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 296.75, "kelly": 0.594, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "prob1": 0.874, "prob2": 0.925, "prob3": 0.908, "hitRate1": 54.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 13.1, "l5_2": 0.0, "l15_2": 0.2, "hitRate3": 44.0, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Ben Sheppard", "name2": "Luke Kornet", "name3": "Steven Adams", "line1": 5.5, "line2": 6.5, "line3": 5.5, "prediction1": 10.52, "prediction2": 11.69, "prediction3": 9.76, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 227.73, "kelly": 0.455, "sigma1": "Low", "sigma2": "Low", "sigma3": "Low", "prob1": 0.837, "prob2": 0.839, "prob3": 0.865, "hitRate1": 50.4, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 40.1, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 54.4, "l5_3": 0.6, "l15_3": 0.4},
    {"name1": "Keegan Murray", "name2": "Ausar Thompson", "name3": "Ryan Nembhard", "line1": 16.5, "line2": 10.5, "line3": 8.5, "prediction1": 22.85, "prediction2": 17.47, "prediction3": 14.78, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 182.17, "kelly": 0.364, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.775, "prob2": 0.834, "prob3": 0.809, "hitRate1": 64.2, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 27.9, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 46.1, "l5_3": 0.4, "l15_3": 0.13},
    {"name1": "Darius Garland", "name2": "Julian Champagnie", "name3": "Ryan Rollins", "line1": 18.5, "line2": 9.5, "line3": 14.5, "prediction1": 24.68, "prediction2": 15.02, "prediction3": 21.57, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 150.74, "kelly": 0.301, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.761, "prob2": 0.769, "prob3": 0.794, "hitRate1": 16.5, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 75.9, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 88.9, "l5_3": 0.2, "l15_3": 0.53},
    {"name1": "Harrison Barnes", "name2": "Zach LaVine", "name3": "Jalen Duren", "line1": 11.5, "line2": 18.5, "line3": 17.5, "prediction1": 17.36, "prediction2": 24.05, "prediction3": 23.79, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 145.94, "kelly": 0.292, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.768, "prob2": 0.76, "prob3": 0.781, "hitRate1": 85.8, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 41.6, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 71.1, "l5_3": 0.4, "l15_3": 0.67},
    {"name1": "Tyrese Martin", "name2": "Alperen Sengun", "name3": "Cooper Flagg", "line1": 9.5, "line2": 23.0, "line3": 17.5, "prediction1": 14.89, "prediction2": 27.87, "prediction3": 22.91, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 133.64, "kelly": 0.267, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.76, "prob2": 0.756, "prob3": 0.754, "hitRate1": 59.2, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 57.6, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 64.0, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Dean Wade", "name2": "Jonas Valan\u010di\u016bnas", "name3": "Sion James", "line1": 5.5, "line2": 8.0, "line3": 6.5, "prediction1": 8.35, "prediction2": 12.14, "prediction3": 9.91, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 121.83, "kelly": 0.244, "sigma1": "Low", "sigma2": "Med", "sigma3": "Low", "prob1": 0.74, "prob2": 0.752, "prob3": 0.739, "hitRate1": 39.7, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 20.6, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Ryan Kalkbrenner", "name2": "Danny Wolf", "name3": "Myles Turner", "line1": 7.5, "line2": 8.5, "line3": 11.5, "prediction1": 10.33, "prediction2": 12.25, "prediction3": 16.12, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 110.11, "kelly": 0.22, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "prob1": 0.733, "prob2": 0.729, "prob3": 0.728, "hitRate1": 38.5, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 19.4, "l5_2": 0.4, "l15_2": 0.13, "hitRate3": 65.1, "l5_3": 0.4, "l15_3": 0.6},
    {"name1": "Toumani Camara", "name2": "Nikola Vu\u010devi\u0107", "name3": "Andrew Wiggins", "line1": 13.5, "line2": 18.5, "line3": 16.5, "prediction1": 18.37, "prediction2": 23.05, "prediction3": 20.94, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 102.92, "kelly": 0.206, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.726, "prob2": 0.728, "prob3": 0.711, "hitRate1": 59.7, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 0.0, "l5_2": 0.0, "l15_2": 0.0, "hitRate3": 52.1, "l5_3": 0.6, "l15_3": 0.67},
    {"name1": "Evan Mobley", "name2": "Ziaire Williams", "name3": "Tobias Harris", "line1": 18.5, "line2": 8.5, "line3": 13.5, "prediction1": 23.2, "prediction2": 11.74, "prediction3": 18.04, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 90.31, "kelly": 0.181, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "prob1": 0.709, "prob2": 0.706, "prob3": 0.705, "hitRate1": 62.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 39.5, "l5_2": 0.4, "l15_2": 0.53, "hitRate3": 69.1, "l5_3": 0.4, "l15_3": 0.33},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Jeremy Sochan", "name2": "Russell Westbrook", "line1": 5.5, "line2": 12.5, "prediction1": 11.98, "prediction2": 23.9, "side1": "over", "side2": "over", "recommendation": 1, "ev": 136.12, "kelly": 0.681, "sigma1": "Med", "sigma2": "High", "prob1": 0.874, "prob2": 0.919, "hitRate1": 54.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 55.0, "l5_2": 0.6, "l15_2": 0.6},
    {"name1": "Ben Sheppard", "name2": "Josh Okogie", "line1": 5.5, "line2": 6.5, "prediction1": 10.52, "prediction2": 13.43, "side1": "over", "side2": "over", "recommendation": 1, "ev": 119.96, "kelly": 0.6, "sigma1": "Low", "sigma2": "Med", "prob1": 0.837, "prob2": 0.894, "hitRate1": 50.4, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 20.9, "l5_2": 0.4, "l15_2": 0.53},
    {"name1": "Steven Adams", "name2": "Ausar Thompson", "line1": 5.5, "line2": 10.5, "prediction1": 9.76, "prediction2": 17.47, "side1": "over", "side2": "over", "recommendation": 1, "ev": 112.03, "kelly": 0.56, "sigma1": "Low", "sigma2": "High", "prob1": 0.865, "prob2": 0.834, "hitRate1": 54.4, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 27.9, "l5_2": 0.2, "l15_2": 0.47},
    {"name1": "Ryan Rollins", "name2": "Ryan Nembhard", "line1": 13.5, "line2": 8.5, "prediction1": 21.57, "prediction2": 14.78, "side1": "over", "side2": "over", "recommendation": 1, "ev": 96.73, "kelly": 0.484, "sigma1": "High", "sigma2": "High", "prob1": 0.827, "prob2": 0.809, "hitRate1": 93.0, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 46.1, "l5_2": 0.4, "l15_2": 0.13},
    {"name1": "Onyeka Okongwu", "name2": "Liam McNeeley", "line1": 20.5, "line2": 4.5, "prediction1": 16.13, "prediction2": 8.09, "side1": "under", "side2": "over", "recommendation": 0, "ev": 82.91, "kelly": 0.415, "sigma1": "Med", "sigma2": "Low", "prob1": 0.795, "prob2": 0.783, "hitRate1": 79.7, "l5_1": 0.0, "l15_1": 0.27, "hitRate2": 42.7, "l5_2": 0.6, "l15_2": 0.53},
    {"name1": "Alperen Sengun", "name2": "Jalen Duren", "line1": 22.5, "line2": 17.5, "prediction1": 27.87, "prediction2": 23.79, "side1": "over", "side2": "over", "recommendation": 1, "ev": 79.03, "kelly": 0.395, "sigma1": "High", "sigma2": "High", "prob1": 0.78, "prob2": 0.781, "hitRate1": 65.4, "l5_1": 0.4, "l15_1": 0.53, "hitRate2": 71.1, "l5_2": 0.4, "l15_2": 0.67},
    {"name1": "Julian Champagnie", "name2": "Keegan Murray", "line1": 9.5, "line2": 16.5, "prediction1": 15.02, "prediction2": 22.85, "side1": "over", "side2": "over", "recommendation": 1, "ev": 75.12, "kelly": 0.376, "sigma1": "High", "sigma2": "High", "prob1": 0.769, "prob2": 0.775, "hitRate1": 75.9, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 64.2, "l5_2": 0.8, "l15_2": 0.27},
    {"name1": "Darius Garland", "name2": "Harrison Barnes", "line1": 18.5, "line2": 11.5, "prediction1": 24.68, "prediction2": 17.36, "side1": "over", "side2": "over", "recommendation": 1, "ev": 71.82, "kelly": 0.359, "sigma1": "High", "sigma2": "High", "prob1": 0.761, "prob2": 0.768, "hitRate1": 16.5, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 85.8, "l5_2": 0.4, "l15_2": 0.6},
    {"name1": "Tyrese Martin", "name2": "Zach LaVine", "line1": 9.5, "line2": 18.5, "prediction1": 14.89, "prediction2": 24.05, "side1": "over", "side2": "over", "recommendation": 1, "ev": 69.67, "kelly": 0.348, "sigma1": "High", "sigma2": "High", "prob1": 0.76, "prob2": 0.76, "hitRate1": 59.2, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 41.6, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Dean Wade", "name2": "Cooper Flagg", "line1": 5.5, "line2": 17.5, "prediction1": 8.35, "prediction2": 22.91, "side1": "over", "side2": "over", "recommendation": 0, "ev": 63.98, "kelly": 0.32, "sigma1": "Low", "sigma2": "High", "prob1": 0.74, "prob2": 0.754, "hitRate1": 39.7, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 64.0, "l5_2": 0.4, "l15_2": 0.4},
];const underdogTriosData = [
    {"name1": "Ben Sheppard", "name2": "Jeremy Sochan", "name3": "Russell Westbrook", "line1": 5.5, "line2": 5.5, "line3": 12.5, "prediction1": 10.52, "prediction2": 11.98, "prediction3": 23.9, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 262.86, "kelly": 0.526, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "prob1": 0.837, "prob2": 0.874, "prob3": 0.919, "hitRate1": 50.4, "l5_1": 0.6, "l15_1": 0.4, "hitRate2": 54.4, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 55.0, "l5_3": 0.6, "l15_3": 0.6},
    {"name1": "Josh Okogie", "name2": "Ausar Thompson", "name3": "Ryan Nembhard", "line1": 6.5, "line2": 10.5, "line3": 8.5, "prediction1": 13.43, "prediction2": 17.47, "prediction3": 14.78, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 225.56, "kelly": 0.451, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "prob1": 0.894, "prob2": 0.834, "prob3": 0.809, "hitRate1": 20.9, "l5_1": 0.4, "l15_1": 0.53, "hitRate2": 27.9, "l5_2": 0.2, "l15_2": 0.47, "hitRate3": 46.1, "l5_3": 0.4, "l15_3": 0.13},
    {"name1": "Onyeka Okongwu", "name2": "Steven Adams", "name3": "Ryan Rollins", "line1": 20.5, "line2": 5.5, "line3": 13.5, "prediction1": 16.13, "prediction2": 9.76, "prediction3": 21.57, "side1": "under", "side2": "over", "side3": "over", "recommendation": 1, "ev": 207.18, "kelly": 0.414, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "prob1": 0.795, "prob2": 0.865, "prob3": 0.827, "hitRate1": 79.7, "l5_1": 0.0, "l15_1": 0.27, "hitRate2": 54.4, "l5_2": 0.6, "l15_2": 0.4, "hitRate3": 93.0, "l5_3": 0.4, "l15_3": 0.6},
    {"name1": "Liam McNeeley", "name2": "Alperen Sengun", "name3": "Jalen Duren", "line1": 4.5, "line2": 22.5, "line3": 17.5, "prediction1": 8.09, "prediction2": 27.87, "prediction3": 23.79, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 157.44, "kelly": 0.315, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "prob1": 0.783, "prob2": 0.78, "prob3": 0.781, "hitRate1": 42.7, "l5_1": 0.6, "l15_1": 0.53, "hitRate2": 65.4, "l5_2": 0.4, "l15_2": 0.53, "hitRate3": 71.1, "l5_3": 0.4, "l15_3": 0.67},
    {"name1": "Darius Garland", "name2": "Julian Champagnie", "name3": "Keegan Murray", "line1": 18.5, "line2": 9.5, "line3": 16.5, "prediction1": 24.68, "prediction2": 15.02, "prediction3": 22.85, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 144.85, "kelly": 0.29, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.761, "prob2": 0.769, "prob3": 0.775, "hitRate1": 16.5, "l5_1": 0.4, "l15_1": 0.2, "hitRate2": 75.9, "l5_2": 0.6, "l15_2": 0.6, "hitRate3": 64.2, "l5_3": 0.8, "l15_3": 0.27},
    {"name1": "Harrison Barnes", "name2": "Tyrese Martin", "name3": "Zach LaVine", "line1": 11.5, "line2": 9.5, "line3": 18.5, "prediction1": 17.36, "prediction2": 14.89, "prediction3": 24.05, "side1": "over", "side2": "over", "side3": "over", "recommendation": 1, "ev": 139.25, "kelly": 0.279, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.768, "prob2": 0.76, "prob3": 0.76, "hitRate1": 85.8, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 59.2, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 41.6, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Dean Wade", "name2": "Sion James", "name3": "Cooper Flagg", "line1": 5.5, "line2": 6.5, "line3": 17.5, "prediction1": 8.35, "prediction2": 9.91, "prediction3": 22.91, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 122.44, "kelly": 0.245, "sigma1": "Low", "sigma2": "Low", "sigma3": "High", "prob1": 0.74, "prob2": 0.739, "prob3": 0.754, "hitRate1": 39.7, "l5_1": 0.2, "l15_1": 0.27, "hitRate2": 20.6, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 64.0, "l5_3": 0.4, "l15_3": 0.4},
    {"name1": "Toumani Camara", "name2": "Kon Knueppel", "name3": "Bobby Portis", "line1": 13.5, "line2": 18.5, "line3": 8.5, "prediction1": 18.37, "prediction2": 22.91, "prediction3": 11.56, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 88.22, "kelly": 0.176, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "prob1": 0.726, "prob2": 0.695, "prob3": 0.69, "hitRate1": 59.7, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 53.2, "l5_2": 0.4, "l15_2": 0.67, "hitRate3": 94.0, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Evan Mobley", "name2": "Daniss Jenkins", "name3": "Max Christie", "line1": 18.5, "line2": 9.5, "line3": 10.5, "prediction1": 23.2, "prediction2": 12.52, "prediction3": 13.68, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 79.82, "kelly": 0.16, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "prob1": 0.709, "prob2": 0.685, "prob3": 0.686, "hitRate1": 62.4, "l5_1": 0.4, "l15_1": 0.47, "hitRate2": 8.0, "l5_2": 0.0, "l15_2": 0.33, "hitRate3": 77.3, "l5_3": 0.4, "l15_3": 0.67},
    {"name1": "De'Andre Hunter", "name2": "Pascal Siakam", "name3": "Tyler Herro", "line1": 15.5, "line2": 22.5, "line3": 23.5, "prediction1": 19.54, "prediction2": 26.45, "prediction3": 21.01, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 70.72, "kelly": 0.141, "sigma1": "High", "sigma2": "High", "sigma3": "High", "prob1": 0.684, "prob2": 0.68, "prob3": 0.679, "hitRate1": 54.2, "l5_1": 0.6, "l15_1": 0.67, "hitRate2": 47.2, "l5_2": 0.8, "l15_2": 0.73, "hitRate3": 29.2, "l5_3": 0.6, "l15_3": 0.2},
];const prizepicksPointsHitRates = [
    {"name": "Bobby Portis", "line": 9.0, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.893, "underPct": 0.107},
    {"name": "Ryan Rollins", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.889, "underPct": 0.111},
    {"name": "Harrison Barnes", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.858, "underPct": 0.142},
    {"name": "Peyton Watson", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.857, "underPct": 0.143},
    {"name": "Kawhi Leonard", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.83, "underPct": 0.17},
    {"name": "Miles McBride", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.809, "underPct": 0.191},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.801, "underPct": 0.199},
    {"name": "Klay Thompson", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.801, "underPct": 0.199},
    {"name": "Devin Vassell", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.793, "underPct": 0.207},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.783, "underPct": 0.217},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.777, "underPct": 0.223},
    {"name": "Max Christie", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.773, "underPct": 0.227},
    {"name": "Kyle Kuzma", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.761, "underPct": 0.239},
    {"name": "Julian Champagnie", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.759, "underPct": 0.241},
    {"name": "Jaylon Tyson", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.749, "underPct": 0.251},
    {"name": "Josh Hart", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.73, "underPct": 0.27},
    {"name": "Kel'el Ware", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.718, "underPct": 0.282},
    {"name": "Precious Achiuwa", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.712, "underPct": 0.288},
    {"name": "Keldon Johnson", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.711, "underPct": 0.289},
    {"name": "Jalen Duren", "line": 17.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.711, "underPct": 0.289},
    {"name": "Shaedon Sharpe", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.71, "underPct": 0.29},
    {"name": "Tyler Herro", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.708, "underPct": 0.292},
    {"name": "Aaron Holiday", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.696, "underPct": 0.304},
    {"name": "DeMar DeRozan", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.691, "underPct": 0.309},
    {"name": "Tobias Harris", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.691, "underPct": 0.309},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.685, "underPct": 0.315},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.673, "underPct": 0.327},
    {"name": "Cade Cunningham", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.655, "underPct": 0.345},
    {"name": "Myles Turner", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.651, "underPct": 0.349},
    {"name": "James Harden", "line": 25.0, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.642, "underPct": 0.358},
    {"name": "Keegan Murray", "line": 16.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.642, "underPct": 0.358},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.64, "underPct": 0.36},
    {"name": "Kris Dunn", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.629, "underPct": 0.371},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.624, "underPct": 0.376},
    {"name": "Jonathan Isaac", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.622, "underPct": 0.378},
    {"name": "Collin Sexton", "line": 15.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.619, "underPct": 0.381},
    {"name": "Cameron Johnson", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.611, "underPct": 0.389},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.597, "underPct": 0.403},
    {"name": "Deni Avdija", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.592, "underPct": 0.408},
    {"name": "Tyrese Martin", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.592, "underPct": 0.408},
    {"name": "Desmond Bane", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.589, "underPct": 0.411},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.588, "underPct": 0.412},
    {"name": "T.J. McConnell", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.584, "underPct": 0.416},
    {"name": "Nae'Qwan Tomlin", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.583, "underPct": 0.417},
    {"name": "Caris LeVert", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.577, "underPct": 0.423},
    {"name": "Alperen Sengun", "line": 23.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.576, "underPct": 0.424},
    {"name": "Drew Eubanks", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.576, "underPct": 0.424},
    {"name": "Noah Clowney", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.565, "underPct": 0.435},
    {"name": "Jamal Murray", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.562, "underPct": 0.438},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.561, "underPct": 0.439},
    {"name": "Malik Monk", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.548, "underPct": 0.452},
    {"name": "Jeremy Sochan", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.544, "underPct": 0.456},
    {"name": "Steven Adams", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.544, "underPct": 0.456},
    {"name": "De'Andre Hunter", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.542, "underPct": 0.458},
    {"name": "Lonzo Ball", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.534, "underPct": 0.466},
    {"name": "Kon Knueppel", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.532, "underPct": 0.468},
    {"name": "Andrew Wiggins", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.521, "underPct": 0.479},
    {"name": "Ben Sheppard", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "De'Aaron Fox", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.482, "underPct": 0.518},
    {"name": "Pascal Siakam", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.472, "underPct": 0.528},
    {"name": "Ryan Nembhard", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.461, "underPct": 0.539},
    {"name": "Goga Bitadze", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.459, "underPct": 0.541},
    {"name": "Bam Adebayo", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.443, "underPct": 0.557},
    {"name": "Russell Westbrook", "line": 13.0, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.44, "underPct": 0.56},
    {"name": "Jalen Brunson", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.432, "underPct": 0.568},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "Liam McNeeley", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "Miles Bridges", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.417, "underPct": 0.583},
    {"name": "Zach LaVine", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Isaiah Jackson", "line": 7.5, "l5": 0.8, "l10": 0.6, "l15": 0.67, "overPct": 0.406, "underPct": 0.594},
    {"name": "Luke Kornet", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.401, "underPct": 0.599},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.397, "underPct": 0.603},
    {"name": "Jay Huff", "line": 11.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.397, "underPct": 0.603},
    {"name": "Ziaire Williams", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.395, "underPct": 0.605},
    {"name": "Kevin Durant", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.394, "underPct": 0.606},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.394, "underPct": 0.606},
    {"name": "Kobe Brown", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.387, "underPct": 0.613},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.385, "underPct": 0.615},
    {"name": "Drake Powell", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.375, "underPct": 0.625},
    {"name": "Amen Thompson", "line": 17.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.358, "underPct": 0.642},
    {"name": "Jalen Suggs", "line": 16.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.341, "underPct": 0.659},
    {"name": "Josh Giddey", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.331, "underPct": 0.669},
    {"name": "Dylan Harper", "line": 13.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.329, "underPct": 0.671},
    {"name": "Kevin Porter Jr.", "line": 15.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.302, "underPct": 0.698},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.301, "underPct": 0.699},
    {"name": "Karl-Anthony Towns", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.297, "underPct": 0.703},
    {"name": "John Collins", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.283, "underPct": 0.717},
    {"name": "Ausar Thompson", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.279, "underPct": 0.721},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.0, "l10": 0.5, "l15": 0.33, "overPct": 0.262, "underPct": 0.738},
    {"name": "Anthony Davis", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.257, "underPct": 0.743},
    {"name": "Spencer Jones", "line": 9.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.252, "underPct": 0.748},
    {"name": "Nicolas Batum", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.25, "underPct": 0.75},
    {"name": "Jaden Ivey", "line": 8.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.245, "underPct": 0.755},
    {"name": "Giannis Antetokounmpo", "line": 31.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.223, "underPct": 0.777},
    {"name": "Sion James", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.206, "underPct": 0.794},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.196, "underPct": 0.804},
    {"name": "Danny Wolf", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.194, "underPct": 0.806},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.188, "underPct": 0.812},
    {"name": "Darius Garland", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.165, "underPct": 0.835},
    {"name": "Tyus Jones", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.148, "underPct": 0.852},
    {"name": "LaMelo Ball", "line": 19.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.134, "underPct": 0.866},
    {"name": "Patrick Williams", "line": 8.5, "l5": 0.0, "l10": 0.0, "l15": 0.2, "overPct": 0.131, "underPct": 0.869},
    {"name": "Matas Buzelis", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.084, "underPct": 0.916},
];const prizepicksAssistsHitRates = [
    {"name": "Ryan Rollins", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.83, "underPct": 0.17},
    {"name": "Jamal Murray", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.657, "underPct": 0.343},
    {"name": "Deni Avdija", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.605, "underPct": 0.395},
    {"name": "T.J. McConnell", "line": 4.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.603, "underPct": 0.397},
    {"name": "Lonzo Ball", "line": 4.0, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.595, "underPct": 0.405},
    {"name": "Davion Mitchell", "line": 7.0, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.59, "underPct": 0.41},
    {"name": "Alperen Sengun", "line": 6.0, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.581, "underPct": 0.419},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.573, "underPct": 0.427},
    {"name": "Goga Bitadze", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.571, "underPct": 0.429},
    {"name": "Desmond Bane", "line": 4.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.552, "underPct": 0.448},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.544, "underPct": 0.456},
    {"name": "Naji Marshall", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.524, "underPct": 0.476},
    {"name": "Darius Garland", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.523, "underPct": 0.477},
    {"name": "Cade Cunningham", "line": 8.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.508, "underPct": 0.492},
    {"name": "Evan Mobley", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.507, "underPct": 0.493},
    {"name": "Giannis Antetokounmpo", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.507, "underPct": 0.493},
    {"name": "Collin Sexton", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.505, "underPct": 0.495},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.501, "underPct": 0.499},
    {"name": "LaMelo Ball", "line": 7.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.473, "underPct": 0.527},
    {"name": "De'Aaron Fox", "line": 6.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.449, "underPct": 0.551},
    {"name": "Jerami Grant", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.436, "underPct": 0.564},
    {"name": "Mikal Bridges", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.428, "underPct": 0.572},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.426, "underPct": 0.574},
    {"name": "Devin Vassell", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.418, "underPct": 0.582},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Tyler Herro", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.409, "underPct": 0.591},
    {"name": "Toumani Camara", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.383, "underPct": 0.617},
    {"name": "Josh Hart", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.324, "underPct": 0.676},
    {"name": "Franz Wagner", "line": 4.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.318, "underPct": 0.682},
    {"name": "James Harden", "line": 9.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.269, "underPct": 0.731},
    {"name": "Ryan Nembhard", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.216, "underPct": 0.784},
    {"name": "Kevin Porter Jr.", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.199, "underPct": 0.801},
    {"name": "Nicolas Batum", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.19, "underPct": 0.81},
];const prizepicksReboundsHitRates = [
    {"name": "Kel'el Ware", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.807, "underPct": 0.193},
    {"name": "Cade Cunningham", "line": 6.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.773, "underPct": 0.227},
    {"name": "Amen Thompson", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.745, "underPct": 0.255},
    {"name": "Tyler Herro", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.73, "underPct": 0.27},
    {"name": "Peyton Watson", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.701, "underPct": 0.299},
    {"name": "Steven Adams", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.663, "underPct": 0.337},
    {"name": "Ziaire Williams", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.65, "underPct": 0.35},
    {"name": "P.J. Washington", "line": 7.0, "l5": 1.0, "l10": 0.6, "l15": 0.6, "overPct": 0.646, "underPct": 0.354},
    {"name": "Bobby Portis", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.643, "underPct": 0.357},
    {"name": "Jaylon Tyson", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.622, "underPct": 0.378},
    {"name": "Deni Avdija", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.61, "underPct": 0.39},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 5.0, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.599, "underPct": 0.401},
    {"name": "Ivica Zubac", "line": 12.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.59, "underPct": 0.41},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.579, "underPct": 0.421},
    {"name": "Jarace Walker", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.556, "underPct": 0.444},
    {"name": "Naji Marshall", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.556, "underPct": 0.444},
    {"name": "Toumani Camara", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.547, "underPct": 0.453},
    {"name": "Jalen Duren", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.541, "underPct": 0.459},
    {"name": "Franz Wagner", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.54, "underPct": 0.46},
    {"name": "Lonzo Ball", "line": 4.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.537, "underPct": 0.463},
    {"name": "Keegan Murray", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.536, "underPct": 0.464},
    {"name": "Kevin Durant", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.533, "underPct": 0.467},
    {"name": "Precious Achiuwa", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.531, "underPct": 0.469},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.526, "underPct": 0.474},
    {"name": "Andrew Wiggins", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.516, "underPct": 0.484},
    {"name": "Josh Giddey", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.507, "underPct": 0.493},
    {"name": "Myles Turner", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.506, "underPct": 0.494},
    {"name": "Keldon Johnson", "line": 6.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.506, "underPct": 0.494},
    {"name": "Ausar Thompson", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.504, "underPct": 0.496},
    {"name": "Bam Adebayo", "line": 9.0, "l5": 0.8, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.494, "underPct": 0.506},
    {"name": "Tristan da Silva", "line": 3.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.477, "underPct": 0.523},
    {"name": "Jeremy Sochan", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.477, "underPct": 0.523},
    {"name": "Donovan Mitchell", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.474, "underPct": 0.526},
    {"name": "Josh Hart", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.457, "underPct": 0.543},
    {"name": "Jalen Suggs", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.452, "underPct": 0.548},
    {"name": "Bennedict Mathurin", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.441, "underPct": 0.559},
    {"name": "Pascal Siakam", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.419, "underPct": 0.581},
    {"name": "Goga Bitadze", "line": 5.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.413, "underPct": 0.587},
    {"name": "De'Andre Hunter", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.41, "underPct": 0.59},
    {"name": "Kyle Kuzma", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.409, "underPct": 0.591},
    {"name": "Cooper Flagg", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.403, "underPct": 0.597},
    {"name": "Desmond Bane", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.403, "underPct": 0.597},
    {"name": "Kawhi Leonard", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.396, "underPct": 0.604},
    {"name": "Kon Knueppel", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.67, "overPct": 0.393, "underPct": 0.607},
    {"name": "Sidy Cissoko", "line": 2.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.391, "underPct": 0.609},
    {"name": "Cameron Johnson", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.386, "underPct": 0.614},
    {"name": "James Harden", "line": 5.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.38, "underPct": 0.62},
    {"name": "Ayo Dosunmu", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.376, "underPct": 0.624},
    {"name": "Tobias Harris", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.376, "underPct": 0.624},
    {"name": "Alperen Sengun", "line": 9.5, "l5": 0.0, "l10": 0.5, "l15": 0.53, "overPct": 0.368, "underPct": 0.632},
    {"name": "Evan Mobley", "line": 10.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.353, "underPct": 0.647},
    {"name": "Jay Huff", "line": 5.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.334, "underPct": 0.666},
    {"name": "John Collins", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.321, "underPct": 0.679},
    {"name": "Zaccharie Risacher", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.301, "underPct": 0.699},
    {"name": "Spencer Jones", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.294, "underPct": 0.706},
    {"name": "Luke Kornet", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.285, "underPct": 0.715},
    {"name": "Anthony Davis", "line": 11.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.281, "underPct": 0.719},
    {"name": "Isaiah Jackson", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.279, "underPct": 0.721},
    {"name": "De'Aaron Fox", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.244, "underPct": 0.756},
    {"name": "Giannis Antetokounmpo", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.228, "underPct": 0.772},
    {"name": "Nae'Qwan Tomlin", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.22, "underPct": 0.78},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.209, "underPct": 0.791},
    {"name": "Karl-Anthony Towns", "line": 12.0, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.184, "underPct": 0.816},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.181, "underPct": 0.819},
    {"name": "Kevin Porter Jr.", "line": 4.0, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.153, "underPct": 0.847},
    {"name": "Onyeka Okongwu", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.095, "underPct": 0.905},
];const prizepicksBlocksHitRates = [
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Anthony Black", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.455, "underPct": 0.545},
    {"name": "Jonathan Isaac", "line": 0.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.43, "underPct": 0.57},
    {"name": "Ryan Kalkbrenner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.335, "underPct": 0.665},
    {"name": "Noah Clowney", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.653, "underPct": 0.347},
    {"name": "Drew Eubanks", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.416, "underPct": 0.584},
    {"name": "Steven Adams", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.532, "underPct": 0.468},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.447, "underPct": 0.553},
];const prizepicksStealsHitRates = [
    {"name": "Donovan Clingan", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.379, "underPct": 0.621},
    {"name": "Nae'Qwan Tomlin", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.548, "underPct": 0.452},
    {"name": "Spencer Jones", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.386, "underPct": 0.614},
    {"name": "Luke Kornet", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.328, "underPct": 0.672},
    {"name": "Goga Bitadze", "line": 0.5, "l5": 1.0, "l10": 0.9, "l15": 0.73, "overPct": 0.589, "underPct": 0.411},
    {"name": "Tyus Jones", "line": 0.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.595, "underPct": 0.405},
    {"name": "Sion James", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.623, "underPct": 0.377},
    {"name": "Ziaire Williams", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.485, "underPct": 0.515},
    {"name": "Aaron Holiday", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.372, "underPct": 0.628},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.501, "underPct": 0.499},
    {"name": "Ryan Rollins", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.391, "underPct": 0.609},
    {"name": "Ryan Nembhard", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Keldon Johnson", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Dunn", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Devin Vassell", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Sexton", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 22.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 13.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Steven Adams", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Holiday", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Russell Westbrook", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 39.0, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 28.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Davis", "line": 37.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Precious Achiuwa", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 30.0, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Kalkbrenner", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyler Kolek", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Liam McNeeley", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "DeMar DeRozan", "line": 24.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keegan Murray", "line": 25.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 35.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 40.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Deni Avdija", "line": 39.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Darius Garland", "line": 28.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Donovan Clingan", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jay Huff", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Jackson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Harrison Barnes", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ivica Zubac", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lonzo Ball", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "John Collins", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cade Cunningham", "line": 41.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tobias Harris", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Myles Turner", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Caris LeVert", "line": 16.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 20.0, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Nembhard", "line": 17.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 31.0, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Spencer Jones", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Suggs", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Drake Powell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sion James", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremy Sochan", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 35.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 15.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 39.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tyler Herro", "line": 32.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Sidy Cissoko", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaden Ivey", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Matas Buzelis", "line": 24.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Brunson", "line": 39.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kobe Brown", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Porter Jr.", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jalen Duren", "line": 31.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 38.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 47.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dylan Harper", "line": 19.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 14.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 32.5, "l5": 0.0, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 28.0, "l5": 0.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
];const prizepicksPRHitRates = [
    {"name": "Keldon Johnson", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Hart", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kris Dunn", "line": 11.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Goga Bitadze", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "P.J. Washington", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keegan Murray", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Holiday", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jay Huff", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Klay Thompson", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 17.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Devin Vassell", "line": 20.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyle Kuzma", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kawhi Leonard", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bobby Portis", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Luke Kornet", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Precious Achiuwa", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Steven Adams", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 20.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 32.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles McBride", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Liam McNeeley", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 20.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Caris LeVert", "line": 12.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cameron Johnson", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 21.5, "l5": 0.6, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 13.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Davis", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Jackson", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaden Ivey", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bam Adebayo", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drake Powell", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Danny Wolf", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Davion Mitchell", "line": 12.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Rollins", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Max Christie", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyler Herro", "line": 28.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Myles Turner", "line": 18.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 41.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 24.0, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "James Harden", "line": 31.0, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "John Collins", "line": 19.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Evan Mobley", "line": 29.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Andre Hunter", "line": 20.0, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Nae'Qwan Tomlin", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Spencer Jones", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Suggs", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Harrison Barnes", "line": 15.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dylan Harper", "line": 15.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremy Sochan", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Nembhard", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sion James", "line": 10.0, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Kalkbrenner", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 20.0, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nicolas Batum", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 29.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 33.0, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lonzo Ball", "line": 11.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 22.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ayo Dosunmu", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Franz Wagner", "line": 31.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ausar Thompson", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 12.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kobe Brown", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Porter Jr.", "line": 19.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 24.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Bennedict Mathurin", "line": 25.5, "l5": 0.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 12.0, "l5": 0.0, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
];const prizepicksPAHitRates = [
    {"name": "Kel'el Ware", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Aaron Holiday", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keldon Johnson", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Black", "line": 18.0, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Devin Vassell", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 20.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles McBride", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarace Walker", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jay Huff", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 14.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Goga Bitadze", "line": 7.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keegan Murray", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cameron Johnson", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyle Kuzma", "line": 11.5, "l5": 0.8, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Bobby Portis", "line": 9.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "P.J. Washington", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Precious Achiuwa", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kris Dunn", "line": 11.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Davion Mitchell", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 21.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Collin Sexton", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 20.0, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Davis", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Liam McNeeley", "line": 5.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tobias Harris", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 35.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "T.J. McConnell", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Desmond Bane", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jerami Grant", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jaylon Tyson", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Spencer Jones", "line": 10.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kobe Brown", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julian Champagnie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 19.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Caris LeVert", "line": 13.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Durant", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Wiggins", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 34.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyus Jones", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Nembhard", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Evan Mobley", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Darius Garland", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Max Christie", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tristan da Silva", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "John Collins", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Alperen Sengun", "line": 29.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ziaire Williams", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dylan Harper", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mouhamed Gueye", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Aaron Fox", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luke Kornet", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drake Powell", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ayo Dosunmu", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Harrison Barnes", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Porter Jr.", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bam Adebayo", "line": 22.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 22.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 36.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Franz Wagner", "line": 28.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 18.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Clarkson", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyler Herro", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden Ivey", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sidy Cissoko", "line": 7.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ausar Thompson", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 14.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Lonzo Ball", "line": 11.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 30.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "LaMelo Ball", "line": 27.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
];const prizepicksRAHitRates = [
    {"name": "Peyton Watson", "line": 7.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 8.0, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julian Champagnie", "line": 7.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 9.0, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.8, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 9.0, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ziaire Williams", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Black", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Keegan Murray", "line": 8.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Harrison Barnes", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Martin", "line": 6.0, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sion James", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 15.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Duren", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ausar Thompson", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "T.J. McConnell", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 12.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Myles Turner", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 19.0, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Nembhard", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 15.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 12.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 14.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tristan da Silva", "line": 5.0, "l5": 0.4, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Aaron Fox", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Suggs", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dean Wade", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jay Huff", "line": 6.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyus Jones", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "James Harden", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kris Dunn", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylon Tyson", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nicolas Batum", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 11.0, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Davis", "line": 14.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Tobias Harris", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 9.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jerami Grant", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 10.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyler Herro", "line": 9.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bennedict Mathurin", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Porter Jr.", "line": 9.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Isaiah Jackson", "line": 5.5, "l5": 0.2, "l10": 0.3, "l15": 0.53, "overPct": 0.3, "underPct": 0.7},
    {"name": "Spencer Jones", "line": 5.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Giannis Antetokounmpo", "line": 17.0, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Devin Vassell", "line": 6.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 15.5, "l5": 0.2, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dylan Harper", "line": 5.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 8.0, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kobe Brown", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 15.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
];const prizepicksTurnoversHitRates = [
    {"name": "Goga Bitadze", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tristan da Silva", "line": 0.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Giannis Antetokounmpo", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyler Kolek", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Precious Achiuwa", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Dylan Harper", "line": 1.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 1.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.1, "underPct": 0.9},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Jalen Suggs", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kobe Brown", "line": 0.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kris Dunn", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyler Kolek", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Josh Giddey", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Noah Clowney", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Mitchell", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.47, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jalen Duren", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaden Ivey", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogPointsHitRates = [
    {"name": "Bobby Portis", "line": 8.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.94, "underPct": 0.06},
    {"name": "Ryan Rollins", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.93, "underPct": 0.07},
    {"name": "Harrison Barnes", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.858, "underPct": 0.142},
    {"name": "P.J. Washington", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.783, "underPct": 0.217},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.777, "underPct": 0.223},
    {"name": "Kawhi Leonard", "line": 24.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.776, "underPct": 0.224},
    {"name": "Max Christie", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.773, "underPct": 0.227},
    {"name": "Kyle Kuzma", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.761, "underPct": 0.239},
    {"name": "Julian Champagnie", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.759, "underPct": 0.241},
    {"name": "James Harden", "line": 25.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.757, "underPct": 0.243},
    {"name": "Jaylon Tyson", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.749, "underPct": 0.251},
    {"name": "Anthony Black", "line": 14.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.737, "underPct": 0.263},
    {"name": "Josh Hart", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.73, "underPct": 0.27},
    {"name": "Kel'el Ware", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.718, "underPct": 0.282},
    {"name": "Collin Sexton", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.711, "underPct": 0.289},
    {"name": "Jalen Duren", "line": 17.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.711, "underPct": 0.289},
    {"name": "Shaedon Sharpe", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.71, "underPct": 0.29},
    {"name": "Tyler Herro", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.708, "underPct": 0.292},
    {"name": "Aaron Holiday", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.696, "underPct": 0.304},
    {"name": "Donovan Mitchell", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.685, "underPct": 0.315},
    {"name": "Cade Cunningham", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.655, "underPct": 0.345},
    {"name": "Alperen Sengun", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.654, "underPct": 0.346},
    {"name": "Keegan Murray", "line": 16.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.642, "underPct": 0.358},
    {"name": "Cooper Flagg", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.64, "underPct": 0.36},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.624, "underPct": 0.376},
    {"name": "Nickeil Alexander-Walker", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.608, "underPct": 0.392},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.4, "l10": 0.3, "l15": 0.53, "overPct": 0.602, "underPct": 0.398},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Toumani Camara", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.597, "underPct": 0.403},
    {"name": "Tyrese Martin", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.592, "underPct": 0.408},
    {"name": "Deni Avdija", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.592, "underPct": 0.408},
    {"name": "Desmond Bane", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.589, "underPct": 0.411},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.588, "underPct": 0.412},
    {"name": "T.J. McConnell", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.584, "underPct": 0.416},
    {"name": "Nae'Qwan Tomlin", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.583, "underPct": 0.417},
    {"name": "Caris LeVert", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.577, "underPct": 0.423},
    {"name": "Jamal Murray", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.562, "underPct": 0.438},
    {"name": "Jerami Grant", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.561, "underPct": 0.439},
    {"name": "Kris Dunn", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.559, "underPct": 0.441},
    {"name": "Russell Westbrook", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.55, "underPct": 0.45},
    {"name": "Steven Adams", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.544, "underPct": 0.456},
    {"name": "Jeremy Sochan", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.544, "underPct": 0.456},
    {"name": "De'Andre Hunter", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.542, "underPct": 0.458},
    {"name": "Kon Knueppel", "line": 18.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.532, "underPct": 0.468},
    {"name": "Cameron Johnson", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.51, "underPct": 0.49},
    {"name": "Ben Sheppard", "line": 5.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
    {"name": "De'Aaron Fox", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.482, "underPct": 0.518},
    {"name": "Pascal Siakam", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.472, "underPct": 0.528},
    {"name": "Ryan Nembhard", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.461, "underPct": 0.539},
    {"name": "Bam Adebayo", "line": 19.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.443, "underPct": 0.557},
    {"name": "Jalen Brunson", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.432, "underPct": 0.568},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "Liam McNeeley", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "Miles Bridges", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.417, "underPct": 0.583},
    {"name": "Zach LaVine", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.416, "underPct": 0.584},
    {"name": "Kevin Porter Jr.", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.398, "underPct": 0.602},
    {"name": "Dean Wade", "line": 5.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.397, "underPct": 0.603},
    {"name": "Ivica Zubac", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.394, "underPct": 0.606},
    {"name": "Kevin Durant", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.394, "underPct": 0.606},
    {"name": "Dyson Daniels", "line": 13.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.348, "underPct": 0.652},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.301, "underPct": 0.699},
    {"name": "Karl-Anthony Towns", "line": 24.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.297, "underPct": 0.703},
    {"name": "Giannis Antetokounmpo", "line": 30.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.282, "underPct": 0.718},
    {"name": "Ausar Thompson", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.279, "underPct": 0.721},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.0, "l10": 0.5, "l15": 0.33, "overPct": 0.262, "underPct": 0.738},
    {"name": "Anthony Davis", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.257, "underPct": 0.743},
    {"name": "Josh Giddey", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.255, "underPct": 0.745},
    {"name": "Josh Okogie", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.209, "underPct": 0.791},
    {"name": "Sion James", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.206, "underPct": 0.794},
    {"name": "Onyeka Okongwu", "line": 20.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.203, "underPct": 0.797},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.188, "underPct": 0.812},
    {"name": "Darius Garland", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.165, "underPct": 0.835},
    {"name": "Tyus Jones", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.148, "underPct": 0.852},
    {"name": "LaMelo Ball", "line": 19.5, "l5": 0.0, "l10": 0.1, "l15": 0.33, "overPct": 0.134, "underPct": 0.866},
    {"name": "Luke Kennard", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.126, "underPct": 0.874},
    {"name": "Zaccharie Risacher", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.103, "underPct": 0.897},
    {"name": "Daniss Jenkins", "line": 9.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.08, "underPct": 0.92},
    {"name": "Matas Buzelis", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.051, "underPct": 0.949},
];const underdogAssistsHitRates = [
    {"name": "T.J. McConnell", "line": 4.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.603, "underPct": 0.397},
    {"name": "Goga Bitadze", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.571, "underPct": 0.429},
    {"name": "Naji Marshall", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.524, "underPct": 0.476},
    {"name": "Darius Garland", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.523, "underPct": 0.477},
    {"name": "Evan Mobley", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.507, "underPct": 0.493},
    {"name": "Collin Sexton", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.505, "underPct": 0.495},
    {"name": "Mikal Bridges", "line": 3.5, "l5": 0.2, "l10": 0.4, "l15": 0.6, "overPct": 0.428, "underPct": 0.572},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.426, "underPct": 0.574},
    {"name": "Devin Vassell", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.418, "underPct": 0.582},
    {"name": "Jarace Walker", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.406, "underPct": 0.594},
    {"name": "Reed Sheppard", "line": 3.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.405, "underPct": 0.595},
    {"name": "Toumani Camara", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.348, "underPct": 0.652},
    {"name": "Ausar Thompson", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.316, "underPct": 0.684},
    {"name": "Ryan Nembhard", "line": 5.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.216, "underPct": 0.784},
    {"name": "Ayo Dosunmu", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.197, "underPct": 0.803},
    {"name": "Mouhamed Gueye", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.161, "underPct": 0.839},
];const underdogReboundsHitRates = [
    {"name": "Kel'el Ware", "line": 9.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.807, "underPct": 0.193},
    {"name": "Lonzo Ball", "line": 3.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.717, "underPct": 0.283},
    {"name": "Ziaire Williams", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.65, "underPct": 0.35},
    {"name": "Bobby Portis", "line": 4.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.643, "underPct": 0.357},
    {"name": "Jaylon Tyson", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.622, "underPct": 0.378},
    {"name": "Deni Avdija", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.61, "underPct": 0.39},
    {"name": "Shaedon Sharpe", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.73, "overPct": 0.575, "underPct": 0.425},
    {"name": "Toumani Camara", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.547, "underPct": 0.453},
    {"name": "Myles Turner", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.506, "underPct": 0.494},
    {"name": "Ausar Thompson", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.504, "underPct": 0.496},
    {"name": "Devin Vassell", "line": 3.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.494, "underPct": 0.506},
    {"name": "Tristan da Silva", "line": 3.5, "l5": 0.2, "l10": 0.6, "l15": 0.53, "overPct": 0.477, "underPct": 0.523},
    {"name": "Jeremy Sochan", "line": 3.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.477, "underPct": 0.523},
    {"name": "Josh Hart", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.457, "underPct": 0.543},
    {"name": "De'Andre Hunter", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.41, "underPct": 0.59},
    {"name": "Kawhi Leonard", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sidy Cissoko", "line": 2.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.391, "underPct": 0.609},
    {"name": "Cameron Johnson", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.386, "underPct": 0.614},
    {"name": "Alperen Sengun", "line": 9.5, "l5": 0.0, "l10": 0.5, "l15": 0.53, "overPct": 0.368, "underPct": 0.632},
    {"name": "Zaccharie Risacher", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.27, "overPct": 0.301, "underPct": 0.699},
    {"name": "Spencer Jones", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.294, "underPct": 0.706},
    {"name": "Giannis Antetokounmpo", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.228, "underPct": 0.772},
    {"name": "Nae'Qwan Tomlin", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.22, "underPct": 0.78},
    {"name": "Noah Clowney", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.209, "underPct": 0.791},
    {"name": "Onyeka Okongwu", "line": 9.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.122, "underPct": 0.878},
];const underdogBlocksHitRates = [
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.477, "underPct": 0.523},
    {"name": "Ryan Kalkbrenner", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.6, "overPct": 0.335, "underPct": 0.665},
    {"name": "Myles Turner", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.447, "underPct": 0.553},
];const underdogStealsHitRates = [
    {"name": "Dyson Daniels", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.439, "underPct": 0.561},
    {"name": "Josh Hart", "line": 1.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.504, "underPct": 0.496},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Keldon Johnson", "line": 20.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Cameron Johnson", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 33.5, "l5": 0.8, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Devin Vassell", "line": 22.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 29.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles McBride", "line": 17.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Martin", "line": 15.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Noah Clowney", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Black", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keegan Murray", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Goga Bitadze", "line": 12.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bobby Portis", "line": 14.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Steven Adams", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Holiday", "line": 13.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 14.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Davis", "line": 37.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kawhi Leonard", "line": 32.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "James Harden", "line": 39.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 28.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 29.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kon Knueppel", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyler Kolek", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyle Kuzma", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Precious Achiuwa", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Mitchell", "line": 38.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles Bridges", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julian Champagnie", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylon Tyson", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nae'Qwan Tomlin", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Deni Avdija", "line": 39.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 24.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Harrison Barnes", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Jackson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 32.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Okogie", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Giannis Antetokounmpo", "line": 46.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cade Cunningham", "line": 43.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tobias Harris", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ausar Thompson", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Duren", "line": 30.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ryan Rollins", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 31.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremy Sochan", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 20.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sion James", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dyson Daniels", "line": 29.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "John Collins", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Aaron Fox", "line": 35.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 10.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Suggs", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ivica Zubac", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 40.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ziaire Williams", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Spencer Jones", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Max Christie", "line": 15.5, "l5": 0.2, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyler Herro", "line": 32.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Dylan Harper", "line": 19.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zaccharie Risacher", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Luke Kennard", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 38.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Porter Jr.", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Daniss Jenkins", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 35.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 25.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sidy Cissoko", "line": 10.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 39.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Onyeka Okongwu", "line": 33.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mouhamed Gueye", "line": 19.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bennedict Mathurin", "line": 28.5, "l5": 0.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
];const underdogPRHitRates = [
    {"name": "P.J. Washington", "line": 22.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 19.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Nickeil Alexander-Walker", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Peyton Watson", "line": 18.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 23.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Devin Vassell", "line": 19.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keegan Murray", "line": 22.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "DeMar DeRozan", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cade Cunningham", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Davis", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kawhi Leonard", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Black", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Desmond Bane", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Aaron Fox", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 32.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Shaedon Sharpe", "line": 21.5, "l5": 0.6, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyler Herro", "line": 28.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bam Adebayo", "line": 28.5, "l5": 0.4, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 28.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Rollins", "line": 17.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Giannis Antetokounmpo", "line": 41.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 27.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 32.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ivica Zubac", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "James Harden", "line": 30.5, "l5": 0.4, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dyson Daniels", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 32.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 31.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Matas Buzelis", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Duren", "line": 29.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 36.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Porter Jr.", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bennedict Mathurin", "line": 25.5, "l5": 0.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Onyeka Okongwu", "line": 30.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogPAHitRates = [
    {"name": "Devin Vassell", "line": 18.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nickeil Alexander-Walker", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 34.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Rollins", "line": 18.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kon Knueppel", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Collin Sexton", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles Bridges", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Cade Cunningham", "line": 35.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kawhi Leonard", "line": 27.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Davis", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 31.5, "l5": 0.6, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shaedon Sharpe", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.73, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Amen Thompson", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dyson Daniels", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Duren", "line": 19.5, "l5": 0.4, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 28.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 29.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 22.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 34.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 27.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ivica Zubac", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Aaron Fox", "line": 31.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Wiggins", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Darius Garland", "line": 25.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Suggs", "line": 21.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kevin Porter Jr.", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jalen Brunson", "line": 36.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Giannis Antetokounmpo", "line": 37.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Giddey", "line": 31.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Franz Wagner", "line": 28.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 22.5, "l5": 0.2, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyler Herro", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 27.5, "l5": 0.0, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Onyeka Okongwu", "line": 24.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogRAHitRates = [
    {"name": "Lonzo Ball", "line": 7.5, "l5": 0.8, "l10": 0.8, "l15": 0.87, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kon Knueppel", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ausar Thompson", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Amen Thompson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Luke Kornet", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.67, "overPct": 0.5, "underPct": 0.5},
    {"name": "James Harden", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Suggs", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Franz Wagner", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Desmond Bane", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylon Tyson", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ryan Nembhard", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Onyeka Okongwu", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Alperen Sengun", "line": 15.5, "l5": 0.2, "l10": 0.6, "l15": 0.73, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
];const underdogTurnoversHitRates = [
    {"name": "Darius Garland", "line": 2.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Giannis Antetokounmpo", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Nickeil Alexander-Walker", "line": 2.5, "l5": 0.4, "l10": 0.2, "l15": 0.47, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Giddey", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
];const underdogBlocksStealsHitRates = [
    {"name": "Jalen Suggs", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Myles Turner", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
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

