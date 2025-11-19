const prizepicksSinglesData = [
    {"name": "Josh Giddey", "bookmaker": "BetRivers", "line": 21.5, "prediction": 25.27, "side": "Over", "odds": 120, "recommendation": 0, "ev": 5.06, "kelly": 0.421, "sigma": "High"},
    {"name": "Dereck Lively II", "bookmaker": "BetMGM", "line": 4.5, "prediction": 7.96, "side": "Over", "odds": -125, "recommendation": 0, "ev": 4.68, "kelly": 0.585, "sigma": "Low"},
    {"name": "Josh Giddey", "bookmaker": "BetRivers", "line": 20.5, "prediction": 25.27, "side": "Over", "odds": 102, "recommendation": 1, "ev": 4.51, "kelly": 0.442, "sigma": "High"},
    {"name": "Pelle Larsson", "bookmaker": "BetRivers", "line": 10.5, "prediction": 13.3, "side": "Over", "odds": 112, "recommendation": 0, "ev": 4.37, "kelly": 0.39, "sigma": "High"},
    {"name": "Landry Shamet", "bookmaker": "BetRivers", "line": 10.5, "prediction": 13.84, "side": "Over", "odds": 100, "recommendation": 0, "ev": 4.2, "kelly": 0.42, "sigma": "High"},
    {"name": "Landry Shamet", "bookmaker": "BetMGM", "line": 10.5, "prediction": 13.84, "side": "Over", "odds": 100, "recommendation": 0, "ev": 4.03, "kelly": 0.403, "sigma": "High"},
    {"name": "Kyshawn George", "bookmaker": "BetRivers", "line": 15.5, "prediction": 17.97, "side": "Over", "odds": 120, "recommendation": 0, "ev": 4.02, "kelly": 0.335, "sigma": "High"},
    {"name": "Aaron Gordon", "bookmaker": "BetRivers", "line": 19.5, "prediction": 22.11, "side": "Over", "odds": 120, "recommendation": 0, "ev": 4.02, "kelly": 0.335, "sigma": "High"},
    {"name": "Josh Giddey", "bookmaker": "BetRivers", "line": 19.5, "prediction": 25.27, "side": "Over", "odds": -122, "recommendation": 1, "ev": 4.0, "kelly": 0.488, "sigma": "High"},
    {"name": "Zion Williamson", "bookmaker": "BetRivers", "line": 20.5, "prediction": 22.59, "side": "Over", "odds": 120, "recommendation": 0, "ev": 3.98, "kelly": 0.332, "sigma": "Med"},
    {"name": "Landry Shamet", "bookmaker": "DraftKings", "line": 9.5, "prediction": 13.84, "side": "Over", "odds": -118, "recommendation": 0, "ev": 3.95, "kelly": 0.466, "sigma": "High"},
    {"name": "Tony Bradley", "bookmaker": "BetMGM", "line": 4.5, "prediction": 7.06, "side": "Over", "odds": -120, "recommendation": 0, "ev": 3.94, "kelly": 0.472, "sigma": "Low"},
    {"name": "Zion Williamson", "bookmaker": "BetRivers", "line": 19.5, "prediction": 22.59, "side": "Over", "odds": -103, "recommendation": 0, "ev": 3.94, "kelly": 0.406, "sigma": "Med"},
    {"name": "Josh Giddey", "bookmaker": "DraftKings", "line": 19.5, "prediction": 25.27, "side": "Over", "odds": -123, "recommendation": 1, "ev": 3.89, "kelly": 0.478, "sigma": "High"},
    {"name": "Zion Williamson", "bookmaker": "BetRivers", "line": 18.5, "prediction": 22.59, "side": "Over", "odds": -124, "recommendation": 0, "ev": 3.88, "kelly": 0.481, "sigma": "Med"},
];const prizepicksPairsData = [
    {"name1": "Dereck Lively II", "name2": "Isaac Okoro", "line1": 4.5, "line2": 5.5, "prediction1": 7.96, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.88, "kelly": 0.344, "sigma1": "Low", "sigma2": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Dereck Lively II", "name2": "Josh Giddey", "line1": 4.5, "line2": 19.5, "prediction1": 7.96, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.82, "kelly": 0.341, "sigma1": "Low", "sigma2": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Dereck Lively II", "name2": "Mitchell Robinson", "line1": 4.5, "line2": 4.5, "prediction1": 7.96, "prediction2": 6.89, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.56, "kelly": 0.328, "sigma1": "Low", "sigma2": "Med", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 34.8, "l5_2": 0.6, "l15_2": 0.2},
    {"name1": "Tony Bradley", "name2": "Isaac Okoro", "line1": 4.5, "line2": 5.5, "prediction1": 7.06, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.93, "kelly": 0.297, "sigma1": "Low", "sigma2": "High", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Landry Shamet", "name2": "Isaac Okoro", "line1": 9.5, "line2": 5.5, "prediction1": 13.84, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.8, "kelly": 0.29, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Landry Shamet", "name2": "Josh Giddey", "line1": 9.5, "line2": 19.5, "prediction1": 13.84, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.63, "kelly": 0.281, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Tony Bradley", "name2": "Josh Giddey", "line1": 4.5, "line2": 19.5, "prediction1": 7.06, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.54, "kelly": 0.277, "sigma1": "Low", "sigma2": "High", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Tony Bradley", "name2": "Mitchell Robinson", "line1": 4.5, "line2": 4.5, "prediction1": 7.06, "prediction2": 6.89, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.35, "kelly": 0.267, "sigma1": "Low", "sigma2": "Med", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 34.8, "l5_2": 0.6, "l15_2": 0.2},
    {"name1": "Landry Shamet", "name2": "Jerami Grant", "line1": 9.5, "line2": 22.5, "prediction1": 13.84, "prediction2": 17.27, "side1": "over", "side2": "under", "recommendation": 0, "ev": 5.18, "kelly": 0.259, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 91.5, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "Mitchell Robinson", "name2": "Jerami Grant", "line1": 4.5, "line2": 22.5, "prediction1": 6.89, "prediction2": 17.27, "side1": "over", "side2": "under", "recommendation": 0, "ev": 4.89, "kelly": 0.244, "sigma1": "Med", "sigma2": "High", "hitRate1": 34.8, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 91.5, "l5_2": 0.2, "l15_2": 0.13},
];const prizepicksTriosData = [
    {"name1": "Dereck Lively II", "name2": "Josh Giddey", "name3": "Isaac Okoro", "line1": 4.5, "line2": 19.5, "line3": 5.5, "prediction1": 7.96, "prediction2": 25.27, "prediction3": 9.04, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 13.3, "kelly": 0.266, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 87.5, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Tony Bradley", "name2": "Dereck Lively II", "name3": "Isaac Okoro", "line1": 4.5, "line2": 4.5, "line3": 5.5, "prediction1": 7.06, "prediction2": 7.96, "prediction3": 9.04, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 13.01, "kelly": 0.26, "sigma1": "Low", "sigma2": "Low", "sigma3": "High", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 45.8, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 87.5, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Tony Bradley", "name2": "Landry Shamet", "name3": "Josh Giddey", "line1": 4.5, "line2": 9.5, "line3": 19.5, "prediction1": 7.06, "prediction2": 13.84, "prediction3": 25.27, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.21, "kelly": 0.224, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 80.3, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 72.8, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Landry Shamet", "name2": "Mitchell Robinson", "name3": "Jerami Grant", "line1": 9.5, "line2": 4.5, "line3": 22.5, "prediction1": 13.84, "prediction2": 6.89, "prediction3": 17.27, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 10.41, "kelly": 0.208, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 34.8, "l5_2": 0.6, "l15_2": 0.2, "hitRate3": 91.5, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Pelle Larsson", "name2": "Mitchell Robinson", "name3": "Jerami Grant", "line1": 9.5, "line2": 4.5, "line3": 22.5, "prediction1": 13.3, "prediction2": 6.89, "prediction3": 17.27, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 9.76, "kelly": 0.195, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 75.1, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 34.8, "l5_2": 0.6, "l15_2": 0.2, "hitRate3": 91.5, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Pelle Larsson", "name2": "Aaron Gordon", "name3": "Kevin Huerter", "line1": 9.5, "line2": 17.5, "line3": 10.5, "prediction1": 13.3, "prediction2": 22.11, "prediction3": 14.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.64, "kelly": 0.173, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 75.1, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 70.3, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 88.2, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Aaron Gordon", "name2": "Brandon Williams", "name3": "Kevin Huerter", "line1": 17.5, "line2": 14.5, "line3": 10.5, "prediction1": 22.11, "prediction2": 18.68, "prediction3": 14.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.12, "kelly": 0.162, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 70.3, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 39.6, "l5_2": 0.8, "l15_2": 0.33, "hitRate3": 88.2, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Keon Ellis", "name2": "Brandon Williams", "name3": "Ayo Dosunmu", "line1": 6.5, "line2": 14.5, "line3": 11.5, "prediction1": 8.91, "prediction2": 18.68, "prediction3": 14.95, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.22, "kelly": 0.144, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 44.5, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 39.6, "l5_2": 0.8, "l15_2": 0.33, "hitRate3": 86.6, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Keon Ellis", "name2": "Ayo Dosunmu", "name3": "Patrick Williams", "line1": 6.5, "line2": 11.5, "line3": 5.5, "prediction1": 8.91, "prediction2": 14.95, "prediction3": 7.53, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.08, "kelly": 0.142, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 44.5, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 86.6, "l5_2": 1.0, "l15_2": 0.6, "hitRate3": 69.6, "l5_3": 0.8, "l15_3": 0.67},
    {"name1": "Will Richard", "name2": "Zion Williamson", "name3": "Patrick Williams", "line1": 16.5, "line2": 19.5, "line3": 5.5, "prediction1": 12.62, "prediction2": 22.59, "prediction3": 7.53, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.86, "kelly": 0.137, "sigma1": "High", "sigma2": "Med", "sigma3": "Med", "hitRate1": 98.9, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 72.6, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 69.6, "l5_3": 0.8, "l15_3": 0.67},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Dereck Lively II", "name2": "Jerami Grant", "line1": 4.5, "line2": 23.5, "prediction1": 7.96, "prediction2": 17.27, "side1": "over", "side2": "under", "recommendation": 0, "ev": 7.27, "kelly": 0.364, "sigma1": "Low", "sigma2": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 94.4, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "Dereck Lively II", "name2": "Isaac Okoro", "line1": 4.5, "line2": 5.5, "prediction1": 7.96, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.98, "kelly": 0.349, "sigma1": "Low", "sigma2": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Dereck Lively II", "name2": "Josh Giddey", "line1": 4.5, "line2": 19.5, "prediction1": 7.96, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.67, "kelly": 0.333, "sigma1": "Low", "sigma2": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Jerami Grant", "name2": "Isaac Okoro", "line1": 23.5, "line2": 5.5, "prediction1": 17.27, "prediction2": 9.04, "side1": "under", "side2": "over", "recommendation": 0, "ev": 6.38, "kelly": 0.319, "sigma1": "High", "sigma2": "High", "hitRate1": 94.4, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Jerami Grant", "name2": "Josh Giddey", "line1": 23.5, "line2": 19.5, "prediction1": 17.27, "prediction2": 25.27, "side1": "under", "side2": "over", "recommendation": 1, "ev": 6.17, "kelly": 0.308, "sigma1": "High", "sigma2": "High", "hitRate1": 94.4, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Landry Shamet", "name2": "Isaac Okoro", "line1": 9.5, "line2": 5.5, "prediction1": 13.84, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.88, "kelly": 0.294, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Landry Shamet", "name2": "Josh Giddey", "line1": 9.5, "line2": 19.5, "prediction1": 13.84, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.57, "kelly": 0.279, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Aaron Gordon", "name2": "Landry Shamet", "line1": 17.5, "line2": 9.5, "prediction1": 22.11, "prediction2": 13.84, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.83, "kelly": 0.241, "sigma1": "High", "sigma2": "High", "hitRate1": 70.3, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 80.3, "l5_2": 0.8, "l15_2": 0.4},
    {"name1": "Aaron Gordon", "name2": "Patrick Williams", "line1": 17.5, "line2": 5.5, "prediction1": 22.11, "prediction2": 7.53, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.01, "kelly": 0.2, "sigma1": "High", "sigma2": "Med", "hitRate1": 70.3, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 69.6, "l5_2": 0.8, "l15_2": 0.67},
    {"name1": "Aaron Gordon", "name2": "Jeremiah Fears", "line1": 17.5, "line2": 14.5, "prediction1": 22.11, "prediction2": 18.11, "side1": "over", "side2": "over", "recommendation": 0, "ev": 3.89, "kelly": 0.195, "sigma1": "High", "sigma2": "High", "hitRate1": 70.3, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 74.4, "l5_2": 1.0, "l15_2": 0.6},
];const underdogTriosData = [
    {"name1": "Dereck Lively II", "name2": "Jerami Grant", "name3": "Isaac Okoro", "line1": 4.5, "line2": 23.5, "line3": 5.5, "prediction1": 7.96, "prediction2": 17.27, "prediction3": 9.04, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 13.9, "kelly": 0.278, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 94.4, "l5_2": 0.2, "l15_2": 0.13, "hitRate3": 87.5, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Dereck Lively II", "name2": "Jerami Grant", "name3": "Josh Giddey", "line1": 4.5, "line2": 23.5, "line3": 19.5, "prediction1": 7.96, "prediction2": 17.27, "prediction3": 25.27, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 13.65, "kelly": 0.273, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 94.4, "l5_2": 0.2, "l15_2": 0.13, "hitRate3": 72.8, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Landry Shamet", "name2": "Josh Giddey", "name3": "Isaac Okoro", "line1": 9.5, "line2": 19.5, "line3": 5.5, "prediction1": 13.84, "prediction2": 25.27, "prediction3": 9.04, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.7, "kelly": 0.234, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 87.5, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Aaron Gordon", "name2": "Landry Shamet", "name3": "Kevin Huerter", "line1": 17.5, "line2": 9.5, "line3": 10.5, "prediction1": 22.11, "prediction2": 13.84, "prediction3": 14.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 9.26, "kelly": 0.185, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 70.3, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 80.3, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 88.2, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Aaron Gordon", "name2": "Ayo Dosunmu", "name3": "Kevin Huerter", "line1": 17.5, "line2": 11.5, "line3": 10.5, "prediction1": 22.11, "prediction2": 14.95, "prediction3": 14.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.04, "kelly": 0.161, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 70.3, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 86.6, "l5_2": 1.0, "l15_2": 0.6, "hitRate3": 88.2, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Jeremiah Fears", "name2": "Patrick Williams", "name3": "Ayo Dosunmu", "line1": 14.5, "line2": 5.5, "line3": 11.5, "prediction1": 18.11, "prediction2": 7.53, "prediction3": 14.95, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 6.78, "kelly": 0.136, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 74.4, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 69.6, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 86.6, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Jeremiah Fears", "name2": "Patrick Williams", "name3": "Kris Murray", "line1": 14.5, "line2": 5.5, "line3": 8.5, "prediction1": 18.11, "prediction2": 7.53, "prediction3": 5.49, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 6.45, "kelly": 0.129, "sigma1": "High", "sigma2": "Med", "sigma3": "Low", "hitRate1": 74.4, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 69.6, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 97.0, "l5_3": 0.0, "l15_3": 0.13},
    {"name1": "Bennedict Mathurin", "name2": "Kyshawn George", "name3": "Kris Murray", "line1": 20.5, "line2": 14.5, "line3": 8.5, "prediction1": 22.81, "prediction2": 17.97, "prediction3": 5.49, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 5.76, "kelly": 0.115, "sigma1": "Low", "sigma2": "High", "sigma3": "Low", "hitRate1": 53.8, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 57.7, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 97.0, "l5_3": 0.0, "l15_3": 0.13},
    {"name1": "Bennedict Mathurin", "name2": "Kyshawn George", "name3": "Jalen Smith", "line1": 20.5, "line2": 14.5, "line3": 9.5, "prediction1": 22.81, "prediction2": 17.97, "prediction3": 12.19, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.45, "kelly": 0.109, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 53.8, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 57.7, "l5_2": 0.4, "l15_2": 0.47, "hitRate3": 69.5, "l5_3": 0.6, "l15_3": 0.53},
    {"name1": "Sion James", "name2": "Daniel Gafford", "name3": "Jalen Smith", "line1": 6.5, "line2": 11.5, "line3": 9.5, "prediction1": 8.31, "prediction2": 14.36, "prediction3": 12.19, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.16, "kelly": 0.103, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 42.2, "l5_1": 0.4, "l15_1": 0.6, "hitRate2": 38.2, "l5_2": 0.4, "l15_2": 0.27, "hitRate3": 69.5, "l5_3": 0.6, "l15_3": 0.53},
];const prizepicksPointsHitRates = [
    {"name": "Kevin Huerter", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.882, "underPct": 0.118},
    {"name": "Isaac Okoro", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.875, "underPct": 0.125},
    {"name": "Ayo Dosunmu", "line": 11.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.866, "underPct": 0.134},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.83, "underPct": 0.17},
    {"name": "Trey Murphy III", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.813, "underPct": 0.187},
    {"name": "Landry Shamet", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.803, "underPct": 0.197},
    {"name": "Norman Powell", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.766, "underPct": 0.234},
    {"name": "Isaiah Hartenstein", "line": 12.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.763, "underPct": 0.237},
    {"name": "Donovan Mitchell", "line": 27.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.758, "underPct": 0.242},
    {"name": "Pelle Larsson", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.751, "underPct": 0.249},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.744, "underPct": 0.256},
    {"name": "Kon Knueppel", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.738, "underPct": 0.262},
    {"name": "Josh Giddey", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.728, "underPct": 0.272},
    {"name": "Zion Williamson", "line": 19.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.726, "underPct": 0.274},
    {"name": "Jarrett Allen", "line": 12.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.718, "underPct": 0.282},
    {"name": "Tony Bradley", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.713, "underPct": 0.287},
    {"name": "Ajay Mitchell", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.71, "underPct": 0.29},
    {"name": "Aaron Gordon", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.703, "underPct": 0.297},
    {"name": "Bam Adebayo", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.696, "underPct": 0.304},
    {"name": "Patrick Williams", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.696, "underPct": 0.304},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.695, "underPct": 0.305},
    {"name": "Isaiah Joe", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.675, "underPct": 0.325},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.673, "underPct": 0.327},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.664, "underPct": 0.336},
    {"name": "Sandro Mamukelashvili", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.651, "underPct": 0.349},
    {"name": "Reed Sheppard", "line": 12.0, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.648, "underPct": 0.352},
    {"name": "Davion Mitchell", "line": 10.0, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.64, "underPct": 0.36},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.638, "underPct": 0.362},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.618, "underPct": 0.382},
    {"name": "Karl-Anthony Towns", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.607, "underPct": 0.393},
    {"name": "Tre Johnson", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.604, "underPct": 0.396},
    {"name": "Tyrese Maxey", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.583, "underPct": 0.417},
    {"name": "Alperen Sengun", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.578, "underPct": 0.422},
    {"name": "Kyshawn George", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.577, "underPct": 0.423},
    {"name": "Trendon Watford", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.573, "underPct": 0.427},
    {"name": "Coby White", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.568, "underPct": 0.432},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.554, "underPct": 0.446},
    {"name": "Sam Merrill", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.549, "underPct": 0.451},
    {"name": "Lonzo Ball", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "Josh Hart", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.539, "underPct": 0.461},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.538, "underPct": 0.462},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.533, "underPct": 0.467},
    {"name": "Andrew Wiggins", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.52, "underPct": 0.48},
    {"name": "Jalen Brunson", "line": 26.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.518, "underPct": 0.482},
    {"name": "De'Andre Hunter", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.51, "underPct": 0.49},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.51, "underPct": 0.49},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.508, "underPct": 0.492},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.506, "underPct": 0.494},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.499, "underPct": 0.501},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.498, "underPct": 0.502},
    {"name": "Ryan Kalkbrenner", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.497, "underPct": 0.503},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.488, "underPct": 0.512},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.484, "underPct": 0.516},
    {"name": "LaMelo Ball", "line": 22.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.479, "underPct": 0.521},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.467, "underPct": 0.533},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.461, "underPct": 0.539},
    {"name": "Dereck Lively II", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.458, "underPct": 0.542},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.457, "underPct": 0.543},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.456, "underPct": 0.544},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.453, "underPct": 0.547},
    {"name": "Keon Ellis", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.445, "underPct": 0.555},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.444, "underPct": 0.556},
    {"name": "Max Christie", "line": 11.0, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.443, "underPct": 0.557},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.443, "underPct": 0.557},
    {"name": "Isaiah Jackson", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.437, "underPct": 0.563},
    {"name": "Miles Bridges", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.431, "underPct": 0.569},
    {"name": "Sion James", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.422, "underPct": 0.578},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.418, "underPct": 0.582},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.409, "underPct": 0.591},
    {"name": "Jose Alvarado", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.407, "underPct": 0.593},
    {"name": "Gradey Dick", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.405, "underPct": 0.595},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.396, "underPct": 0.604},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.393, "underPct": 0.607},
    {"name": "Miles McBride", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.385, "underPct": 0.615},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.385, "underPct": 0.615},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.384, "underPct": 0.616},
    {"name": "Mike Conley", "line": 6.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.384, "underPct": 0.616},
    {"name": "Josh Okogie", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.383, "underPct": 0.617},
    {"name": "Daniel Gafford", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.382, "underPct": 0.618},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.38, "underPct": 0.62},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.378, "underPct": 0.622},
    {"name": "Pascal Siakam", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.37, "underPct": 0.63},
    {"name": "Mitchell Robinson", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.348, "underPct": 0.652},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.326, "underPct": 0.674},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.326, "underPct": 0.674},
    {"name": "Donte DiVincenzo", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.325, "underPct": 0.675},
    {"name": "Deni Avdija", "line": 29.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.308, "underPct": 0.692},
    {"name": "Khris Middleton", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.306, "underPct": 0.694},
    {"name": "Quentin Grimes", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.297, "underPct": 0.703},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.292, "underPct": 0.708},
    {"name": "Naz Reid", "line": 13.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.279, "underPct": 0.721},
    {"name": "Anthony Edwards", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.271, "underPct": 0.729},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.263, "underPct": 0.737},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.259, "underPct": 0.741},
    {"name": "Dean Wade", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.195, "underPct": 0.805},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.194, "underPct": 0.806},
    {"name": "Jarace Walker", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.192, "underPct": 0.808},
    {"name": "Toumani Camara", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.187, "underPct": 0.813},
    {"name": "Dominick Barlow", "line": 8.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.182, "underPct": 0.818},
    {"name": "Marvin Bagley III", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.176, "underPct": 0.824},
    {"name": "Jaylin Williams", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.148, "underPct": 0.852},
    {"name": "Peyton Watson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.119, "underPct": 0.881},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.116, "underPct": 0.884},
    {"name": "VJ Edgecombe", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.087, "underPct": 0.913},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.085, "underPct": 0.915},
    {"name": "Brandin Podziemski", "line": 15.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.069, "underPct": 0.931},
    {"name": "Cameron Johnson", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.061, "underPct": 0.939},
    {"name": "Moses Moody", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.035, "underPct": 0.965},
    {"name": "Quinten Post", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.034, "underPct": 0.966},
    {"name": "Kris Murray", "line": 8.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.03, "underPct": 0.97},
    {"name": "Will Richard", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.011, "underPct": 0.989},
    {"name": "Caleb Love", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.002, "underPct": 0.998},
];const prizepicksAssistsHitRates = [
    {"name": "Russell Westbrook", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.688, "underPct": 0.312},
    {"name": "Josh Giddey", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.666, "underPct": 0.334},
    {"name": "Josh Hart", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.663, "underPct": 0.337},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.655, "underPct": 0.345},
    {"name": "LaMelo Ball", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.631, "underPct": 0.369},
    {"name": "Mitchell Robinson", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.63, "underPct": 0.37},
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.628, "underPct": 0.372},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.606, "underPct": 0.394},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.596, "underPct": 0.404},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.587, "underPct": 0.413},
    {"name": "Collin Sexton", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.576, "underPct": 0.424},
    {"name": "Alperen Sengun", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.573, "underPct": 0.427},
    {"name": "Lonzo Ball", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.569, "underPct": 0.431},
    {"name": "Jarrett Allen", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.562, "underPct": 0.438},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.555, "underPct": 0.445},
    {"name": "Trendon Watford", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.532, "underPct": 0.468},
    {"name": "Coby White", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.511, "underPct": 0.489},
    {"name": "Zion Williamson", "line": 4.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.477, "underPct": 0.523},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.473, "underPct": 0.527},
    {"name": "Jamal Murray", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.462, "underPct": 0.538},
    {"name": "Pascal Siakam", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.462, "underPct": 0.538},
    {"name": "D'Angelo Russell", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.432, "underPct": 0.568},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.413, "underPct": 0.587},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.399, "underPct": 0.601},
    {"name": "Ben Sheppard", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.386, "underPct": 0.614},
    {"name": "T.J. McConnell", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.376, "underPct": 0.624},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.37, "underPct": 0.63},
    {"name": "Amen Thompson", "line": 5.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.368, "underPct": 0.632},
    {"name": "Tyrese Maxey", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.361, "underPct": 0.639},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.359, "underPct": 0.641},
    {"name": "Shai Gilgeous-Alexander", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.355, "underPct": 0.645},
    {"name": "VJ Edgecombe", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.335, "underPct": 0.665},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.311, "underPct": 0.689},
    {"name": "Deni Avdija", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.295, "underPct": 0.705},
    {"name": "Jarace Walker", "line": 2.5, "l5": 0.0, "l10": 0.4, "l15": 0.53, "overPct": 0.29, "underPct": 0.71},
    {"name": "Andrew Nembhard", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.249, "underPct": 0.751},
];const prizepicksReboundsHitRates = [
    {"name": "Josh Giddey", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.793, "underPct": 0.207},
    {"name": "LaMelo Ball", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.76, "underPct": 0.24},
    {"name": "Trey Murphy III", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.738, "underPct": 0.262},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.731, "underPct": 0.269},
    {"name": "Donovan Mitchell", "line": 4.0, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.715, "underPct": 0.285},
    {"name": "Kon Knueppel", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.693, "underPct": 0.307},
    {"name": "Zion Williamson", "line": 5.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.649, "underPct": 0.351},
    {"name": "Isaiah Hartenstein", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.637, "underPct": 0.363},
    {"name": "Immanuel Quickley", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.636, "underPct": 0.364},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.633, "underPct": 0.367},
    {"name": "VJ Edgecombe", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.626, "underPct": 0.374},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.62, "underPct": 0.38},
    {"name": "Brandon Williams", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.617, "underPct": 0.383},
    {"name": "Alperen Sengun", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.594, "underPct": 0.406},
    {"name": "Brandon Ingram", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.579, "underPct": 0.421},
    {"name": "Mitchell Robinson", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.569, "underPct": 0.431},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.553, "underPct": 0.447},
    {"name": "Aaron Gordon", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.55, "underPct": 0.45},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.533, "underPct": 0.467},
    {"name": "Naz Reid", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.533, "underPct": 0.467},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.532, "underPct": 0.468},
    {"name": "P.J. Washington", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.522, "underPct": 0.478},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.517, "underPct": 0.483},
    {"name": "Scottie Barnes", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.498, "underPct": 0.502},
    {"name": "Jarrett Allen", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.48, "underPct": 0.52},
    {"name": "Jamal Shead", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.477, "underPct": 0.523},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.474, "underPct": 0.526},
    {"name": "T.J. McConnell", "line": 1.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.456, "underPct": 0.544},
    {"name": "Amen Thompson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.452, "underPct": 0.548},
    {"name": "Jaylin Williams", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.45, "underPct": 0.55},
    {"name": "Miles Bridges", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Evan Mobley", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.443, "underPct": 0.557},
    {"name": "Daniel Gafford", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.438, "underPct": 0.562},
    {"name": "Bilal Coulibaly", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.437, "underPct": 0.563},
    {"name": "Bruce Brown", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.429, "underPct": 0.571},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.421, "underPct": 0.579},
    {"name": "Collin Sexton", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.421, "underPct": 0.579},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.42, "underPct": 0.58},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.405, "underPct": 0.595},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.402, "underPct": 0.598},
    {"name": "Derik Queen", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.399, "underPct": 0.601},
    {"name": "De'Andre Hunter", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.387, "underPct": 0.613},
    {"name": "Jakob Poeltl", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.378, "underPct": 0.622},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.369, "underPct": 0.631},
    {"name": "Bam Adebayo", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.369, "underPct": 0.631},
    {"name": "Pelle Larsson", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.361, "underPct": 0.639},
    {"name": "Isaiah Jackson", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.359, "underPct": 0.641},
    {"name": "Kris Murray", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.33, "underPct": 0.67},
    {"name": "Khris Middleton", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.319, "underPct": 0.681},
    {"name": "Corey Kispert", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.318, "underPct": 0.682},
    {"name": "Pascal Siakam", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.313, "underPct": 0.687},
    {"name": "Bennedict Mathurin", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.311, "underPct": 0.689},
    {"name": "Rudy Gobert", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.301, "underPct": 0.699},
    {"name": "Kevin Durant", "line": 5.0, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.279, "underPct": 0.721},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.264, "underPct": 0.736},
    {"name": "Josh Hart", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.254, "underPct": 0.746},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.25, "underPct": 0.75},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.242, "underPct": 0.758},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.228, "underPct": 0.772},
    {"name": "Jarace Walker", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.226, "underPct": 0.774},
    {"name": "Tony Bradley", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.222, "underPct": 0.778},
    {"name": "Dominick Barlow", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.198, "underPct": 0.802},
    {"name": "Caleb Love", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.157, "underPct": 0.843},
    {"name": "Brandin Podziemski", "line": 6.0, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.145, "underPct": 0.855},
    {"name": "Will Richard", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.074, "underPct": 0.926},
];const prizepicksBlocksHitRates = [
    {"name": "Miles Bridges", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.395, "underPct": 0.605},
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.432, "underPct": 0.568},
    {"name": "Kyshawn George", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.462, "underPct": 0.538},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.392, "underPct": 0.608},
    {"name": "Isaiah Hartenstein", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.505, "underPct": 0.495},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.578, "underPct": 0.422},
];const prizepicksStealsHitRates = [
    {"name": "Bennedict Mathurin", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.467, "underPct": 0.533},
    {"name": "Sion James", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.546, "underPct": 0.454},
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.411, "underPct": 0.589},
    {"name": "Sam Merrill", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.395, "underPct": 0.605},
    {"name": "Dominick Barlow", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.409, "underPct": 0.591},
    {"name": "Trendon Watford", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.322, "underPct": 0.678},
    {"name": "Quinten Post", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.369, "underPct": 0.631},
    {"name": "Drew Eubanks", "line": 0.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.282, "underPct": 0.718},
    {"name": "Jaylin Williams", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.455, "underPct": 0.545},
    {"name": "Malik Monk", "line": 0.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.626, "underPct": 0.374},
    {"name": "Zach LaVine", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.485, "underPct": 0.515},
    {"name": "Jordan Clarkson", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.38, "underPct": 0.62},
    {"name": "Donovan Clingan", "line": 0.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.413, "underPct": 0.587},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Davion Mitchell", "line": 19.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 37.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kevin Huerter", "line": 15.5, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Sam Merrill", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pelle Larsson", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 39.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sandro Mamukelashvili", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dean Wade", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Okogie", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 16.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Gordon", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luguentz Dort", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kon Knueppel", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Jackson", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tony Bradley", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 41.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keon Ellis", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 36.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Karl-Anthony Towns", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Max Christie", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaac Okoro", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Landry Shamet", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "D'Angelo Russell", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Shead", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "T.J. McConnell", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "LaMelo Ball", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylin Williams", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dereck Lively II", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Corey Kispert", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 26.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Hart", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Khris Middleton", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Sexton", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sion James", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Robinson-Earl", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Trendon Watford", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Caleb Love", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Miles McBride", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 29.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Coby White", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cameron Johnson", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Donte DiVincenzo", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jarace Walker", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quinten Post", "line": 18.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 24.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandin Podziemski", "line": 27.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 29.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mike Conley", "line": 12.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Deni Avdija", "line": 44.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 24.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kris Murray", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Jakob Poeltl", "line": 21.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Reed Sheppard", "line": 14.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kevin Huerter", "line": 13.5, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Ayo Dosunmu", "line": 14.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Davion Mitchell", "line": 12.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Sam Merrill", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sandro Mamukelashvili", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pelle Larsson", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Robinson-Earl", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 14.5, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Chet Holmgren", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Williams", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Okogie", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jarrett Allen", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Landry Shamet", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Jackson", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 29.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Russell Westbrook", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 16.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaac Okoro", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "T.J. McConnell", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Collin Sexton", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 17.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Derik Queen", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dominick Barlow", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Keon Ellis", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylin Williams", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jose Alvarado", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jarace Walker", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ben Sheppard", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sion James", "line": 9.5, "l5": 0.2, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 27.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "VJ Edgecombe", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Moses Moody", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Caleb Love", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Coby White", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cameron Johnson", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Josh Hart", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandin Podziemski", "line": 22.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 24.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quinten Post", "line": 15.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Marvin Bagley III", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 37.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Murray", "line": 13.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPAHitRates = [
    {"name": "Reed Sheppard", "line": 15.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Davion Mitchell", "line": 16.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 24.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Sam Merrill", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Sandro Mamukelashvili", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 21.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Patrick Williams", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 37.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Kevin Huerter", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jarrett Allen", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Gordon", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tre Johnson", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Kon Knueppel", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Jackson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ryan Kalkbrenner", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Hartenstein", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cooper Flagg", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Max Christie", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Landry Shamet", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaac Okoro", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keon Ellis", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "D'Angelo Russell", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles Bridges", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "VJ Edgecombe", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "T.J. McConnell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jakob Poeltl", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Corey Kispert", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 33.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cameron Johnson", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Karl-Anthony Towns", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylin Williams", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Sexton", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Sion James", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandin Podziemski", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Coby White", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jerami Grant", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jose Alvarado", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Caleb Love", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ben Sheppard", "line": 8.0, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 30.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 13.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quinten Post", "line": 11.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kris Murray", "line": 11.5, "l5": 0.0, "l10": 0.0, "l15": 0.07, "overPct": 0.0, "underPct": 1.0},
];const prizepicksRAHitRates = [
    {"name": "Jamal Murray", "line": 10.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Mitchell", "line": 10.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Alperen Sengun", "line": 17.0, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Robinson-Earl", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 10.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 7.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "P.J. Washington", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "LaMelo Ball", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 8.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Scottie Barnes", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Merrill", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zion Williamson", "line": 9.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 13.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Evan Mobley", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naji Marshall", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jordan Clarkson", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles Bridges", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tony Bradley", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Fears", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Isaiah Hartenstein", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 10.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Chet Holmgren", "line": 9.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ajay Mitchell", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 10.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kon Knueppel", "line": 8.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 7.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jarace Walker", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Coby White", "line": 7.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cameron Johnson", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Davion Mitchell", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trendon Watford", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Tyrese Maxey", "line": 11.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 11.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarrett Allen", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Caleb Love", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Drew Eubanks", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Donte DiVincenzo", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles McBride", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 11.0, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quinten Post", "line": 8.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Malik Monk", "line": 5.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 14.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const prizepicksTurnoversHitRates = [
    {"name": "Jarrett Allen", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 2.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Smith", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Khris Middleton", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Patrick Williams", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lonzo Ball", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ben Sheppard", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Robinson-Earl", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "De'Andre Hunter", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mike Conley", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Anthony Edwards", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Aaron Gordon", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandin Podziemski", "line": 2.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Caleb Love", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Bilal Coulibaly", "line": 1.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "VJ Edgecombe", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Daniel Gafford", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Keon Ellis", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 2.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Will Richard", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Murray", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 1.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const underdogPointsHitRates = [
    {"name": "Kevin Huerter", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.882, "underPct": 0.118},
    {"name": "Isaac Okoro", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.875, "underPct": 0.125},
    {"name": "Ayo Dosunmu", "line": 11.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.866, "underPct": 0.134},
    {"name": "Naji Marshall", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.83, "underPct": 0.17},
    {"name": "Trey Murphy III", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.813, "underPct": 0.187},
    {"name": "Landry Shamet", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.803, "underPct": 0.197},
    {"name": "Donovan Mitchell", "line": 27.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.758, "underPct": 0.242},
    {"name": "Reed Sheppard", "line": 11.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.746, "underPct": 0.254},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.744, "underPct": 0.256},
    {"name": "Kon Knueppel", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.738, "underPct": 0.262},
    {"name": "Josh Giddey", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.728, "underPct": 0.272},
    {"name": "Ajay Mitchell", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.71, "underPct": 0.29},
    {"name": "Aaron Gordon", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.703, "underPct": 0.297},
    {"name": "Norman Powell", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.699, "underPct": 0.301},
    {"name": "Patrick Williams", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.696, "underPct": 0.304},
    {"name": "Bam Adebayo", "line": 18.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.696, "underPct": 0.304},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.695, "underPct": 0.305},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.664, "underPct": 0.336},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.638, "underPct": 0.362},
    {"name": "Corey Kispert", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.628, "underPct": 0.372},
    {"name": "Karl-Anthony Towns", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.607, "underPct": 0.393},
    {"name": "Tyrese Maxey", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.583, "underPct": 0.417},
    {"name": "Alperen Sengun", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.578, "underPct": 0.422},
    {"name": "Kyshawn George", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.577, "underPct": 0.423},
    {"name": "Coby White", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.568, "underPct": 0.432},
    {"name": "Isaiah Joe", "line": 12.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.568, "underPct": 0.432},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.554, "underPct": 0.446},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.545, "underPct": 0.455},
    {"name": "Lonzo Ball", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.538, "underPct": 0.462},
    {"name": "Jalen Brunson", "line": 26.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.518, "underPct": 0.482},
    {"name": "Jordan Clarkson", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.51, "underPct": 0.49},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.499, "underPct": 0.501},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.498, "underPct": 0.502},
    {"name": "Ryan Kalkbrenner", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.497, "underPct": 0.503},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.488, "underPct": 0.512},
    {"name": "Cooper Flagg", "line": 16.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.484, "underPct": 0.516},
    {"name": "LaMelo Ball", "line": 22.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.479, "underPct": 0.521},
    {"name": "Dereck Lively II", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.458, "underPct": 0.542},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.457, "underPct": 0.543},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.456, "underPct": 0.544},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.453, "underPct": 0.547},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.444, "underPct": 0.556},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.443, "underPct": 0.557},
    {"name": "Max Christie", "line": 11.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.443, "underPct": 0.557},
    {"name": "Isaiah Jackson", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.437, "underPct": 0.563},
    {"name": "Andrew Wiggins", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Sion James", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.422, "underPct": 0.578},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.409, "underPct": 0.591},
    {"name": "Jose Alvarado", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.407, "underPct": 0.593},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.393, "underPct": 0.607},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.385, "underPct": 0.615},
    {"name": "Mike Conley", "line": 6.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.384, "underPct": 0.616},
    {"name": "Josh Okogie", "line": 7.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.383, "underPct": 0.617},
    {"name": "Daniel Gafford", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.382, "underPct": 0.618},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.38, "underPct": 0.62},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.378, "underPct": 0.622},
    {"name": "Pascal Siakam", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.37, "underPct": 0.63},
    {"name": "Jamal Shead", "line": 6.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.326, "underPct": 0.674},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.326, "underPct": 0.674},
    {"name": "Deni Avdija", "line": 29.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.308, "underPct": 0.692},
    {"name": "Dominick Barlow", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.294, "underPct": 0.706},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.292, "underPct": 0.708},
    {"name": "Anthony Edwards", "line": 29.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.271, "underPct": 0.729},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.263, "underPct": 0.737},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.259, "underPct": 0.741},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.194, "underPct": 0.806},
    {"name": "Jarace Walker", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.192, "underPct": 0.808},
    {"name": "Jaylin Williams", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.148, "underPct": 0.852},
    {"name": "Toumani Camara", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.123, "underPct": 0.877},
    {"name": "Peyton Watson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.119, "underPct": 0.881},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.116, "underPct": 0.884},
    {"name": "Cameron Johnson", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.061, "underPct": 0.939},
    {"name": "Jerami Grant", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.056, "underPct": 0.944},
    {"name": "Kris Murray", "line": 8.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.03, "underPct": 0.97},
    {"name": "Caleb Love", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.002, "underPct": 0.998},
];const underdogAssistsHitRates = [
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.655, "underPct": 0.345},
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.628, "underPct": 0.372},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.596, "underPct": 0.404},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.587, "underPct": 0.413},
    {"name": "Jamal Shead", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.582, "underPct": 0.418},
    {"name": "Lonzo Ball", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.569, "underPct": 0.431},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.556, "underPct": 0.444},
    {"name": "Julius Randle", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.555, "underPct": 0.445},
    {"name": "Zach LaVine", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Khris Middleton", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.468, "underPct": 0.532},
    {"name": "D'Angelo Russell", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.432, "underPct": 0.568},
    {"name": "Ben Sheppard", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.386, "underPct": 0.614},
    {"name": "T.J. McConnell", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.376, "underPct": 0.624},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.37, "underPct": 0.63},
    {"name": "Amen Thompson", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.368, "underPct": 0.632},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.311, "underPct": 0.689},
    {"name": "Aaron Gordon", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.292, "underPct": 0.708},
    {"name": "Moses Moody", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.18, "underPct": 0.82},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.133, "underPct": 0.867},
];const underdogReboundsHitRates = [
    {"name": "Jeremiah Robinson-Earl", "line": 5.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.797, "underPct": 0.203},
    {"name": "Josh Giddey", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.793, "underPct": 0.207},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.633, "underPct": 0.367},
    {"name": "Brandon Williams", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.617, "underPct": 0.383},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.594, "underPct": 0.406},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.533, "underPct": 0.467},
    {"name": "Naz Reid", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.533, "underPct": 0.467},
    {"name": "P.J. Washington", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.522, "underPct": 0.478},
    {"name": "Zach LaVine", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.514, "underPct": 0.486},
    {"name": "Jamal Shead", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.477, "underPct": 0.523},
    {"name": "T.J. McConnell", "line": 1.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.456, "underPct": 0.544},
    {"name": "Amen Thompson", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.452, "underPct": 0.548},
    {"name": "Miles Bridges", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Daniel Gafford", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.438, "underPct": 0.562},
    {"name": "Collin Sexton", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.421, "underPct": 0.579},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.421, "underPct": 0.579},
    {"name": "Rudy Gobert", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.414, "underPct": 0.586},
    {"name": "Jordan Clarkson", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.382, "underPct": 0.618},
    {"name": "Bam Adebayo", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.369, "underPct": 0.631},
    {"name": "Corey Kispert", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.318, "underPct": 0.682},
    {"name": "Pascal Siakam", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.313, "underPct": 0.687},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.242, "underPct": 0.758},
    {"name": "Tony Bradley", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.222, "underPct": 0.778},
    {"name": "Caleb Love", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.157, "underPct": 0.843},
];const underdogBlocksHitRates = [
    {"name": "Peyton Watson", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.381, "underPct": 0.619},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.392, "underPct": 0.608},
];const underdogStealsHitRates = [
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.411, "underPct": 0.589},
    {"name": "Shai Gilgeous-Alexander", "line": 1.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.437, "underPct": 0.563},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Reed Sheppard", "line": 17.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Mitchell", "line": 37.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Davion Mitchell", "line": 19.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 39.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sam Merrill", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Fears", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Huerter", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Evan Mobley", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dean Wade", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tyrese Maxey", "line": 41.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lonzo Ball", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jarrett Allen", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tony Bradley", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 27.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 36.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jordan Clarkson", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaac Okoro", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Landry Shamet", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andre Drummond", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 23.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Keon Ellis", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Hartenstein", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 29.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bam Adebayo", "line": 30.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "P.J. Washington", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dereck Lively II", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "LaMelo Ball", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 13.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Wiggins", "line": 26.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jaylin Williams", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 39.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naji Marshall", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Sexton", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Drew Eubanks", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jeremiah Robinson-Earl", "line": 13.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Caleb Love", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Toumani Camara", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Coby White", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quentin Grimes", "line": 25.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Donte DiVincenzo", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 15.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Amen Thompson", "line": 29.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarace Walker", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Will Richard", "line": 24.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quinten Post", "line": 18.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mike Conley", "line": 12.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 27.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 44.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Murray", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const underdogPRHitRates = [
    {"name": "Donovan Mitchell", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alperen Sengun", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Chet Holmgren", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Immanuel Quickley", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jarrett Allen", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 24.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Hartenstein", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 30.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 27.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 29.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Karl-Anthony Towns", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 27.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "VJ Edgecombe", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Moses Moody", "line": 22.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "P.J. Washington", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Toumani Camara", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Will Richard", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 37.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogPAHitRates = [
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Alperen Sengun", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Nembhard", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 28.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Aaron Gordon", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Williams", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Wiggins", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 33.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jerami Grant", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Coby White", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 30.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 19.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const underdogRAHitRates = [
    {"name": "Josh Giddey", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kevin Durant", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Scottie Barnes", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Williams", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naji Marshall", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mitchell Robinson", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andre Drummond", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jarace Walker", "line": 7.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.4, "underPct": 0.6},
    {"name": "Coby White", "line": 7.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Will Richard", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Caleb Love", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quinten Post", "line": 8.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogTurnoversHitRates = [
    {"name": "Zion Williamson", "line": 2.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Anthony Edwards", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandin Podziemski", "line": 2.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Zach LaVine", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
];const underdogBlocksStealsHitRates = [
    {"name": "Ryan Kalkbrenner", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Rudy Gobert", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Daniel Gafford", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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
            <th style="width: 16%">Player </th>
            <th style="width: 6%">Line </th>
            <th style="width: 6%">Proj. </th>
            <th style="width: 16%">Player </th>
            <th style="width: 6%">Line </th>
            <th style="width: 6%">Proj. </th>
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
            <th style="width: 13%">Player </th>
            <th style="width: 5%">Line </th>
            <th style="width: 5%">Proj. </th>
            <th style="width: 13%">Player </th>
            <th style="width: 5%">Line </th>
            <th style="width: 5%">Proj. </th>
            <th style="width: 13%">Player </th>
            <th style="width: 5%">Line </th>
            <th style="width: 5%">Proj. </th>
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
                <div class="stat-label">Projection</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Model's projected value given the context of the game and player performance</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Expected Value $</div>
                <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value on a $10 stake (Ex. If EV is $2.00, you can expect to win $2.00 per $10 bet on average)</div>
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

