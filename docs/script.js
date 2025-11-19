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
    {"name1": "Dereck Lively II", "name2": "Josh Giddey", "line1": 4.5, "line2": 19.5, "prediction1": 7.96, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 7.0, "kelly": 0.35, "sigma1": "Low", "sigma2": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Dereck Lively II", "name2": "Isaac Okoro", "line1": 4.5, "line2": 5.5, "prediction1": 7.96, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.82, "kelly": 0.341, "sigma1": "Low", "sigma2": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Zion Williamson", "name2": "Dereck Lively II", "line1": 18.5, "line2": 4.5, "prediction1": 22.59, "prediction2": 7.96, "side1": "over", "side2": "over", "recommendation": 0, "ev": 6.82, "kelly": 0.341, "sigma1": "Med", "sigma2": "Low", "hitRate1": 79.5, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 45.8, "l5_2": 0.2, "l15_2": 0.07},
    {"name1": "Zion Williamson", "name2": "Josh Giddey", "line1": 18.5, "line2": 19.5, "prediction1": 22.59, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.81, "kelly": 0.291, "sigma1": "Med", "sigma2": "High", "hitRate1": 79.5, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Landry Shamet", "name2": "Isaac Okoro", "line1": 9.5, "line2": 5.5, "prediction1": 13.84, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.8, "kelly": 0.29, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Mitchell Robinson", "name2": "Josh Giddey", "line1": 4.5, "line2": 19.5, "prediction1": 6.89, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.69, "kelly": 0.285, "sigma1": "Med", "sigma2": "High", "hitRate1": 34.8, "l5_1": 0.6, "l15_1": 0.2, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Tony Bradley", "name2": "Isaac Okoro", "line1": 4.5, "line2": 5.5, "prediction1": 7.06, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.64, "kelly": 0.282, "sigma1": "Low", "sigma2": "High", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Tony Bradley", "name2": "Zion Williamson", "line1": 4.5, "line2": 18.5, "prediction1": 7.06, "prediction2": 22.59, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.55, "kelly": 0.278, "sigma1": "Low", "sigma2": "Med", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 79.5, "l5_2": 0.8, "l15_2": 0.27},
    {"name1": "Tony Bradley", "name2": "Landry Shamet", "line1": 4.5, "line2": 9.5, "prediction1": 7.06, "prediction2": 13.84, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.46, "kelly": 0.273, "sigma1": "Low", "sigma2": "High", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 80.3, "l5_2": 0.8, "l15_2": 0.4},
    {"name1": "Landry Shamet", "name2": "Jerami Grant", "line1": 9.5, "line2": 22.5, "prediction1": 13.84, "prediction2": 17.27, "side1": "over", "side2": "under", "recommendation": 0, "ev": 5.23, "kelly": 0.262, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 91.5, "l5_2": 0.2, "l15_2": 0.13},
];const prizepicksTriosData = [
    {"name1": "Dereck Lively II", "name2": "Josh Giddey", "name3": "Isaac Okoro", "line1": 4.5, "line2": 19.5, "line3": 5.5, "prediction1": 7.96, "prediction2": 25.27, "prediction3": 9.04, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 13.3, "kelly": 0.266, "sigma1": "Low", "sigma2": "High", "sigma3": "High", "hitRate1": 45.8, "l5_1": 0.2, "l15_1": 0.07, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 87.5, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Zion Williamson", "name2": "Dereck Lively II", "name3": "Isaac Okoro", "line1": 18.5, "line2": 4.5, "line3": 5.5, "prediction1": 22.59, "prediction2": 7.96, "prediction3": 9.04, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 13.09, "kelly": 0.262, "sigma1": "Med", "sigma2": "Low", "sigma3": "High", "hitRate1": 79.5, "l5_1": 0.8, "l15_1": 0.27, "hitRate2": 45.8, "l5_2": 0.2, "l15_2": 0.07, "hitRate3": 87.5, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Tony Bradley", "name2": "Zion Williamson", "name3": "Josh Giddey", "line1": 4.5, "line2": 18.5, "line3": 19.5, "prediction1": 7.06, "prediction2": 22.59, "prediction3": 25.27, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.3, "kelly": 0.226, "sigma1": "Low", "sigma2": "Med", "sigma3": "High", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 79.5, "l5_2": 0.8, "l15_2": 0.27, "hitRate3": 72.8, "l5_3": 0.6, "l15_3": 0.47},
    {"name1": "Tony Bradley", "name2": "Landry Shamet", "name3": "Mitchell Robinson", "line1": 4.5, "line2": 9.5, "line3": 4.5, "prediction1": 7.06, "prediction2": 13.84, "prediction3": 6.89, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 10.7, "kelly": 0.214, "sigma1": "Low", "sigma2": "High", "sigma3": "Med", "hitRate1": 71.3, "l5_1": 0.8, "l15_1": 0.6, "hitRate2": 80.3, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 34.8, "l5_3": 0.6, "l15_3": 0.2},
    {"name1": "Landry Shamet", "name2": "Mitchell Robinson", "name3": "Jerami Grant", "line1": 9.5, "line2": 4.5, "line3": 22.5, "prediction1": 13.84, "prediction2": 6.89, "prediction3": 17.27, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 10.41, "kelly": 0.208, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 34.8, "l5_2": 0.6, "l15_2": 0.2, "hitRate3": 91.5, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Pelle Larsson", "name2": "Aaron Gordon", "name3": "Jerami Grant", "line1": 9.5, "line2": 17.5, "line3": 22.5, "prediction1": 13.3, "prediction2": 22.11, "prediction3": 17.27, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 9.34, "kelly": 0.187, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 75.1, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 70.3, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 91.5, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Pelle Larsson", "name2": "Aaron Gordon", "name3": "Kevin Huerter", "line1": 9.5, "line2": 17.5, "line3": 10.5, "prediction1": 13.3, "prediction2": 22.11, "prediction3": 14.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.64, "kelly": 0.173, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 75.1, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 70.3, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 88.2, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Brandon Williams", "name2": "Ayo Dosunmu", "name3": "Kevin Huerter", "line1": 14.5, "line2": 11.5, "line3": 10.5, "prediction1": 18.68, "prediction2": 14.95, "prediction3": 14.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.55, "kelly": 0.151, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 39.6, "l5_1": 0.8, "l15_1": 0.33, "hitRate2": 86.6, "l5_2": 1.0, "l15_2": 0.6, "hitRate3": 88.2, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Jarace Walker", "name2": "Brandon Williams", "name3": "Ayo Dosunmu", "line1": 9.5, "line2": 14.5, "line3": 11.5, "prediction1": 12.86, "prediction2": 18.68, "prediction3": 14.95, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.24, "kelly": 0.145, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 29.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 39.6, "l5_2": 0.8, "l15_2": 0.33, "hitRate3": 86.6, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Jarace Walker", "name2": "Keon Ellis", "name3": "Patrick Williams", "line1": 9.5, "line2": 6.5, "line3": 5.5, "prediction1": 12.86, "prediction2": 8.91, "prediction3": 7.53, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.05, "kelly": 0.141, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 29.4, "l5_1": 0.4, "l15_1": 0.4, "hitRate2": 44.5, "l5_2": 0.4, "l15_2": 0.4, "hitRate3": 69.6, "l5_3": 0.8, "l15_3": 0.67},
];// Duplicate for Underdog (you can replace with actual Underdog data)
const underdogSinglesData = [...prizepicksSinglesData];
const underdogPairsData = [
    {"name1": "Josh Giddey", "name2": "Jerami Grant", "line1": 19.5, "line2": 23.5, "prediction1": 25.27, "prediction2": 17.27, "side1": "over", "side2": "under", "recommendation": 1, "ev": 6.43, "kelly": 0.321, "sigma1": "High", "sigma2": "High", "hitRate1": 72.8, "l5_1": 0.6, "l15_1": 0.47, "hitRate2": 94.4, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "Jerami Grant", "name2": "Isaac Okoro", "line1": 23.5, "line2": 5.5, "prediction1": 17.27, "prediction2": 9.04, "side1": "under", "side2": "over", "recommendation": 0, "ev": 6.33, "kelly": 0.316, "sigma1": "High", "sigma2": "High", "hitRate1": 94.4, "l5_1": 0.2, "l15_1": 0.13, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Landry Shamet", "name2": "Jerami Grant", "line1": 9.5, "line2": 23.5, "prediction1": 13.84, "prediction2": 17.27, "side1": "over", "side2": "under", "recommendation": 0, "ev": 5.82, "kelly": 0.291, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 94.4, "l5_2": 0.2, "l15_2": 0.13},
    {"name1": "Landry Shamet", "name2": "Isaac Okoro", "line1": 9.5, "line2": 5.5, "prediction1": 13.84, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.7, "kelly": 0.285, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Landry Shamet", "name2": "Josh Giddey", "line1": 9.5, "line2": 19.5, "prediction1": 13.84, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.56, "kelly": 0.278, "sigma1": "High", "sigma2": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Jeremiah Fears", "name2": "Isaac Okoro", "line1": 13.5, "line2": 5.5, "prediction1": 18.11, "prediction2": 9.04, "side1": "over", "side2": "over", "recommendation": 0, "ev": 5.33, "kelly": 0.266, "sigma1": "High", "sigma2": "High", "hitRate1": 81.9, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 87.5, "l5_2": 0.4, "l15_2": 0.47},
    {"name1": "Jeremiah Fears", "name2": "Josh Giddey", "line1": 13.5, "line2": 19.5, "prediction1": 18.11, "prediction2": 25.27, "side1": "over", "side2": "over", "recommendation": 1, "ev": 5.2, "kelly": 0.26, "sigma1": "High", "sigma2": "High", "hitRate1": 81.9, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47},
    {"name1": "Jeremiah Fears", "name2": "Kevin Huerter", "line1": 13.5, "line2": 10.5, "prediction1": 18.11, "prediction2": 14.35, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.52, "kelly": 0.226, "sigma1": "High", "sigma2": "High", "hitRate1": 81.9, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 88.2, "l5_2": 0.8, "l15_2": 0.73},
    {"name1": "Reed Sheppard", "name2": "Aaron Gordon", "line1": 10.5, "line2": 17.5, "prediction1": 13.5, "prediction2": 22.11, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.13, "kelly": 0.207, "sigma1": "Med", "sigma2": "High", "hitRate1": 82.9, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 70.3, "l5_2": 0.8, "l15_2": 0.47},
    {"name1": "Aaron Gordon", "name2": "Patrick Williams", "line1": 17.5, "line2": 5.5, "prediction1": 22.11, "prediction2": 7.53, "side1": "over", "side2": "over", "recommendation": 0, "ev": 4.02, "kelly": 0.201, "sigma1": "High", "sigma2": "Med", "hitRate1": 70.3, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 69.6, "l5_2": 0.8, "l15_2": 0.67},
];const underdogTriosData = [
    {"name1": "Landry Shamet", "name2": "Jerami Grant", "name3": "Isaac Okoro", "line1": 9.5, "line2": 23.5, "line3": 5.5, "prediction1": 13.84, "prediction2": 17.27, "prediction3": 9.04, "side1": "over", "side2": "under", "side3": "over", "recommendation": 0, "ev": 12.26, "kelly": 0.245, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 94.4, "l5_2": 0.2, "l15_2": 0.13, "hitRate3": 87.5, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Landry Shamet", "name2": "Josh Giddey", "name3": "Jerami Grant", "line1": 9.5, "line2": 19.5, "line3": 23.5, "prediction1": 13.84, "prediction2": 25.27, "prediction3": 17.27, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 12.03, "kelly": 0.241, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 80.3, "l5_1": 0.8, "l15_1": 0.4, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 94.4, "l5_3": 0.2, "l15_3": 0.13},
    {"name1": "Jeremiah Fears", "name2": "Josh Giddey", "name3": "Isaac Okoro", "line1": 13.5, "line2": 19.5, "line3": 5.5, "prediction1": 18.11, "prediction2": 25.27, "prediction3": 9.04, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 11.24, "kelly": 0.225, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 81.9, "l5_1": 1.0, "l15_1": 0.6, "hitRate2": 72.8, "l5_2": 0.6, "l15_2": 0.47, "hitRate3": 87.5, "l5_3": 0.4, "l15_3": 0.47},
    {"name1": "Aaron Gordon", "name2": "Jeremiah Fears", "name3": "Kevin Huerter", "line1": 17.5, "line2": 13.5, "line3": 10.5, "prediction1": 22.11, "prediction2": 18.11, "prediction3": 14.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.85, "kelly": 0.177, "sigma1": "High", "sigma2": "High", "sigma3": "High", "hitRate1": 70.3, "l5_1": 0.8, "l15_1": 0.47, "hitRate2": 81.9, "l5_2": 1.0, "l15_2": 0.6, "hitRate3": 88.2, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Reed Sheppard", "name2": "Aaron Gordon", "name3": "Kevin Huerter", "line1": 10.5, "line2": 17.5, "line3": 10.5, "prediction1": 13.5, "prediction2": 22.11, "prediction3": 14.35, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 8.16, "kelly": 0.163, "sigma1": "Med", "sigma2": "High", "sigma3": "High", "hitRate1": 82.9, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 70.3, "l5_2": 0.8, "l15_2": 0.47, "hitRate3": 88.2, "l5_3": 0.8, "l15_3": 0.73},
    {"name1": "Reed Sheppard", "name2": "Patrick Williams", "name3": "Ayo Dosunmu", "line1": 10.5, "line2": 5.5, "line3": 11.5, "prediction1": 13.5, "prediction2": 7.53, "prediction3": 14.95, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.25, "kelly": 0.145, "sigma1": "Med", "sigma2": "Med", "sigma3": "High", "hitRate1": 82.9, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 69.6, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 86.6, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Will Richard", "name2": "Patrick Williams", "name3": "Ayo Dosunmu", "line1": 16.5, "line2": 5.5, "line3": 11.5, "prediction1": 12.62, "prediction2": 7.53, "prediction3": 14.95, "side1": "under", "side2": "over", "side3": "over", "recommendation": 0, "ev": 7.0, "kelly": 0.14, "sigma1": "High", "sigma2": "Med", "sigma3": "High", "hitRate1": 98.9, "l5_1": 0.0, "l15_1": 0.07, "hitRate2": 69.6, "l5_2": 0.8, "l15_2": 0.67, "hitRate3": 86.6, "l5_3": 1.0, "l15_3": 0.6},
    {"name1": "Davion Mitchell", "name2": "Will Richard", "name3": "Kris Murray", "line1": 9.5, "line2": 16.5, "line3": 8.5, "prediction1": 12.4, "prediction2": 12.62, "prediction3": 5.49, "side1": "over", "side2": "under", "side3": "under", "recommendation": 0, "ev": 6.39, "kelly": 0.128, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "hitRate1": 74.7, "l5_1": 1.0, "l15_1": 0.53, "hitRate2": 98.9, "l5_2": 0.0, "l15_2": 0.07, "hitRate3": 97.0, "l5_3": 0.0, "l15_3": 0.13},
    {"name1": "Josh Okogie", "name2": "Davion Mitchell", "name3": "Kris Murray", "line1": 6.5, "line2": 9.5, "line3": 8.5, "prediction1": 8.57, "prediction2": 12.4, "prediction3": 5.49, "side1": "over", "side2": "over", "side3": "under", "recommendation": 0, "ev": 6.11, "kelly": 0.122, "sigma1": "High", "sigma2": "High", "sigma3": "Low", "hitRate1": 53.2, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 74.7, "l5_2": 1.0, "l15_2": 0.53, "hitRate3": 97.0, "l5_3": 0.0, "l15_3": 0.13},
    {"name1": "Josh Okogie", "name2": "Bam Adebayo", "name3": "Jaylen Clark", "line1": 6.5, "line2": 16.5, "line3": 5.5, "prediction1": 8.57, "prediction2": 20.8, "prediction3": 7.24, "side1": "over", "side2": "over", "side3": "over", "recommendation": 0, "ev": 5.9, "kelly": 0.118, "sigma1": "High", "sigma2": "High", "sigma3": "Med", "hitRate1": 53.2, "l5_1": 0.6, "l15_1": 0.6, "hitRate2": 83.6, "l5_2": 0.8, "l15_2": 0.4, "hitRate3": 37.1, "l5_3": 0.2, "l15_3": 0.33},
];const prizepicksPointsHitRates = [
    {"name": "Kevin Huerter", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.882, "underPct": 0.118},
    {"name": "Isaac Okoro", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.875, "underPct": 0.125},
    {"name": "Ayo Dosunmu", "line": 11.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.866, "underPct": 0.134},
    {"name": "Bam Adebayo", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.836, "underPct": 0.164},
    {"name": "Trey Murphy III", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.813, "underPct": 0.187},
    {"name": "Landry Shamet", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.803, "underPct": 0.197},
    {"name": "Zion Williamson", "line": 18.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.795, "underPct": 0.205},
    {"name": "Norman Powell", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.766, "underPct": 0.234},
    {"name": "Isaiah Hartenstein", "line": 12.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.763, "underPct": 0.237},
    {"name": "Donovan Mitchell", "line": 27.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.758, "underPct": 0.242},
    {"name": "Naji Marshall", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.751, "underPct": 0.249},
    {"name": "Pelle Larsson", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.751, "underPct": 0.249},
    {"name": "Jeremiah Fears", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.744, "underPct": 0.256},
    {"name": "Kon Knueppel", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.738, "underPct": 0.262},
    {"name": "Josh Giddey", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.728, "underPct": 0.272},
    {"name": "Tony Bradley", "line": 4.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.713, "underPct": 0.287},
    {"name": "Ajay Mitchell", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.71, "underPct": 0.29},
    {"name": "Aaron Gordon", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.703, "underPct": 0.297},
    {"name": "Patrick Williams", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.696, "underPct": 0.304},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.695, "underPct": 0.305},
    {"name": "Trendon Watford", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.694, "underPct": 0.306},
    {"name": "Immanuel Quickley", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.673, "underPct": 0.327},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.664, "underPct": 0.336},
    {"name": "Sandro Mamukelashvili", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.651, "underPct": 0.349},
    {"name": "Reed Sheppard", "line": 12.0, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.648, "underPct": 0.352},
    {"name": "Jeremiah Robinson-Earl", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.642, "underPct": 0.358},
    {"name": "Davion Mitchell", "line": 10.0, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.64, "underPct": 0.36},
    {"name": "Amen Thompson", "line": 17.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.638, "underPct": 0.362},
    {"name": "Corey Kispert", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.628, "underPct": 0.372},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.622, "underPct": 0.378},
    {"name": "Jarrett Allen", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.62, "underPct": 0.38},
    {"name": "Jakob Poeltl", "line": 11.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.618, "underPct": 0.382},
    {"name": "Karl-Anthony Towns", "line": 23.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.607, "underPct": 0.393},
    {"name": "Tre Johnson", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.604, "underPct": 0.396},
    {"name": "Simone Fontecchio", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.585, "underPct": 0.415},
    {"name": "Tyrese Maxey", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.583, "underPct": 0.417},
    {"name": "Cooper Flagg", "line": 15.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.582, "underPct": 0.418},
    {"name": "Alperen Sengun", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.578, "underPct": 0.422},
    {"name": "Kyshawn George", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.577, "underPct": 0.423},
    {"name": "Jared McCain", "line": 4.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.576, "underPct": 0.424},
    {"name": "Coby White", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.568, "underPct": 0.432},
    {"name": "Isaiah Joe", "line": 12.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.568, "underPct": 0.432},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.554, "underPct": 0.446},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.545, "underPct": 0.455},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "Josh Hart", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.539, "underPct": 0.461},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.538, "underPct": 0.462},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.533, "underPct": 0.467},
    {"name": "Josh Okogie", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.532, "underPct": 0.468},
    {"name": "Jalen Brunson", "line": 26.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.518, "underPct": 0.482},
    {"name": "De'Andre Hunter", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.51, "underPct": 0.49},
    {"name": "Mikal Bridges", "line": 15.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.508, "underPct": 0.492},
    {"name": "Cam Whitmore", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.507, "underPct": 0.493},
    {"name": "Malik Monk", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.506, "underPct": 0.494},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.499, "underPct": 0.501},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.498, "underPct": 0.502},
    {"name": "Ryan Kalkbrenner", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.497, "underPct": 0.503},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.488, "underPct": 0.512},
    {"name": "LaMelo Ball", "line": 22.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.479, "underPct": 0.521},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.467, "underPct": 0.533},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.461, "underPct": 0.539},
    {"name": "Dereck Lively II", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.458, "underPct": 0.542},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.457, "underPct": 0.543},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.456, "underPct": 0.544},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.453, "underPct": 0.547},
    {"name": "Pascal Siakam", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.447, "underPct": 0.553},
    {"name": "Keon Ellis", "line": 6.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.445, "underPct": 0.555},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.444, "underPct": 0.556},
    {"name": "Max Christie", "line": 11.0, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.443, "underPct": 0.557},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.443, "underPct": 0.557},
    {"name": "Isaiah Jackson", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.437, "underPct": 0.563},
    {"name": "Miles Bridges", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.431, "underPct": 0.569},
    {"name": "Donte DiVincenzo", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.427, "underPct": 0.573},
    {"name": "Andrew Wiggins", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Sion James", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.422, "underPct": 0.578},
    {"name": "T.J. McConnell", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.418, "underPct": 0.582},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.409, "underPct": 0.591},
    {"name": "Brandon Williams", "line": 14.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.396, "underPct": 0.604},
    {"name": "Lonzo Ball", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.396, "underPct": 0.604},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.393, "underPct": 0.607},
    {"name": "Quentin Grimes", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.387, "underPct": 0.613},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.385, "underPct": 0.615},
    {"name": "Miles McBride", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.385, "underPct": 0.615},
    {"name": "Cason Wallace", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.384, "underPct": 0.616},
    {"name": "Daniel Gafford", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.382, "underPct": 0.618},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.38, "underPct": 0.62},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.378, "underPct": 0.622},
    {"name": "Mitchell Robinson", "line": 4.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.348, "underPct": 0.652},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.337, "underPct": 0.663},
    {"name": "Klay Thompson", "line": 11.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.327, "underPct": 0.673},
    {"name": "D'Angelo Russell", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.326, "underPct": 0.674},
    {"name": "Deni Avdija", "line": 29.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.308, "underPct": 0.692},
    {"name": "Khris Middleton", "line": 9.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.306, "underPct": 0.694},
    {"name": "Jarace Walker", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.294, "underPct": 0.706},
    {"name": "P.J. Washington", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.292, "underPct": 0.708},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.29, "underPct": 0.71},
    {"name": "Jamal Murray", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.263, "underPct": 0.737},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.259, "underPct": 0.741},
    {"name": "Mike Conley", "line": 7.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.248, "underPct": 0.752},
    {"name": "Jaylen Clark", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.227, "underPct": 0.773},
    {"name": "Dean Wade", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.195, "underPct": 0.805},
    {"name": "Naz Reid", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.195, "underPct": 0.805},
    {"name": "Toumani Camara", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.187, "underPct": 0.813},
    {"name": "Dominick Barlow", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.182, "underPct": 0.818},
    {"name": "Marvin Bagley III", "line": 9.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.176, "underPct": 0.824},
    {"name": "Justin Edwards", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.162, "underPct": 0.838},
    {"name": "Jaylin Williams", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.148, "underPct": 0.852},
    {"name": "Peyton Watson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.119, "underPct": 0.881},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.116, "underPct": 0.884},
    {"name": "VJ Edgecombe", "line": 16.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.087, "underPct": 0.913},
    {"name": "Jerami Grant", "line": 22.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.085, "underPct": 0.915},
    {"name": "Cameron Johnson", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.061, "underPct": 0.939},
    {"name": "Moses Moody", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.059, "underPct": 0.941},
    {"name": "Brandin Podziemski", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.04, "underPct": 0.96},
    {"name": "Kris Murray", "line": 8.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.03, "underPct": 0.97},
    {"name": "Quinten Post", "line": 11.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.015, "underPct": 0.985},
    {"name": "Will Richard", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.011, "underPct": 0.989},
    {"name": "Buddy Hield", "line": 12.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.005, "underPct": 0.995},
    {"name": "Caleb Love", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.002, "underPct": 0.998},
];const prizepicksAssistsHitRates = [
    {"name": "Russell Westbrook", "line": 6.0, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.688, "underPct": 0.312},
    {"name": "Josh Giddey", "line": 8.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.666, "underPct": 0.334},
    {"name": "Josh Hart", "line": 4.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.663, "underPct": 0.337},
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.655, "underPct": 0.345},
    {"name": "LaMelo Ball", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.631, "underPct": 0.369},
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.628, "underPct": 0.372},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.4, "overPct": 0.606, "underPct": 0.394},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.596, "underPct": 0.404},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.587, "underPct": 0.413},
    {"name": "Jamal Shead", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.582, "underPct": 0.418},
    {"name": "Collin Sexton", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.576, "underPct": 0.424},
    {"name": "Alperen Sengun", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.573, "underPct": 0.427},
    {"name": "Lonzo Ball", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.569, "underPct": 0.431},
    {"name": "Jarrett Allen", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.562, "underPct": 0.438},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.556, "underPct": 0.444},
    {"name": "Coby White", "line": 4.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.511, "underPct": 0.489},
    {"name": "Kel'el Ware", "line": 0.5, "l5": 1.0, "l10": 0.6, "l15": 0.53, "overPct": 0.504, "underPct": 0.496},
    {"name": "Zion Williamson", "line": 4.0, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.477, "underPct": 0.523},
    {"name": "Scottie Barnes", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.473, "underPct": 0.527},
    {"name": "Khris Middleton", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.468, "underPct": 0.532},
    {"name": "Pascal Siakam", "line": 4.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.462, "underPct": 0.538},
    {"name": "Jamal Murray", "line": 6.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.462, "underPct": 0.538},
    {"name": "Max Christie", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.461, "underPct": 0.539},
    {"name": "D'Angelo Russell", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.432, "underPct": 0.568},
    {"name": "Anthony Edwards", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.413, "underPct": 0.587},
    {"name": "Immanuel Quickley", "line": 6.0, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Davion Mitchell", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.399, "underPct": 0.601},
    {"name": "T.J. McConnell", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.376, "underPct": 0.624},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.37, "underPct": 0.63},
    {"name": "Amen Thompson", "line": 5.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.368, "underPct": 0.632},
    {"name": "Tyrese Maxey", "line": 7.0, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.361, "underPct": 0.639},
    {"name": "Brandon Ingram", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.359, "underPct": 0.641},
    {"name": "Shai Gilgeous-Alexander", "line": 7.0, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.355, "underPct": 0.645},
    {"name": "VJ Edgecombe", "line": 4.0, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.335, "underPct": 0.665},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.311, "underPct": 0.689},
    {"name": "Deni Avdija", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.295, "underPct": 0.705},
    {"name": "Ryan Kalkbrenner", "line": 0.5, "l5": 0.0, "l10": 0.1, "l15": 0.27, "overPct": 0.269, "underPct": 0.731},
    {"name": "Andrew Nembhard", "line": 7.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.249, "underPct": 0.751},
    {"name": "Moses Moody", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.18, "underPct": 0.82},
];const prizepicksReboundsHitRates = [
    {"name": "Josh Giddey", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.793, "underPct": 0.207},
    {"name": "LaMelo Ball", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.76, "underPct": 0.24},
    {"name": "Trey Murphy III", "line": 5.0, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.738, "underPct": 0.262},
    {"name": "Jamal Murray", "line": 4.0, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.731, "underPct": 0.269},
    {"name": "Donovan Mitchell", "line": 4.0, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.715, "underPct": 0.285},
    {"name": "Alperen Sengun", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.713, "underPct": 0.287},
    {"name": "Kon Knueppel", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.693, "underPct": 0.307},
    {"name": "Zion Williamson", "line": 5.0, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.649, "underPct": 0.351},
    {"name": "Isaiah Hartenstein", "line": 10.0, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.637, "underPct": 0.363},
    {"name": "Immanuel Quickley", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.636, "underPct": 0.364},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.633, "underPct": 0.367},
    {"name": "Isaac Okoro", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.632, "underPct": 0.368},
    {"name": "VJ Edgecombe", "line": 5.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.626, "underPct": 0.374},
    {"name": "Russell Westbrook", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.62, "underPct": 0.38},
    {"name": "Brandon Williams", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.617, "underPct": 0.383},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.594, "underPct": 0.406},
    {"name": "Brandon Ingram", "line": 5.0, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.579, "underPct": 0.421},
    {"name": "Mitchell Robinson", "line": 8.0, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.569, "underPct": 0.431},
    {"name": "Tyrese Maxey", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.553, "underPct": 0.447},
    {"name": "Aaron Gordon", "line": 6.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.55, "underPct": 0.45},
    {"name": "Cooper Flagg", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.536, "underPct": 0.464},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.533, "underPct": 0.467},
    {"name": "Naz Reid", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.533, "underPct": 0.467},
    {"name": "Julius Randle", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.532, "underPct": 0.468},
    {"name": "Andrew Wiggins", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.53, "underPct": 0.47},
    {"name": "P.J. Washington", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.522, "underPct": 0.478},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.517, "underPct": 0.483},
    {"name": "Zach LaVine", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.514, "underPct": 0.486},
    {"name": "Scottie Barnes", "line": 7.0, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.498, "underPct": 0.502},
    {"name": "Jamal Shead", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.477, "underPct": 0.523},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.474, "underPct": 0.526},
    {"name": "Peyton Watson", "line": 4.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.457, "underPct": 0.543},
    {"name": "Jaylin Williams", "line": 5.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.45, "underPct": 0.55},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.449, "underPct": 0.551},
    {"name": "Kevin Durant", "line": 4.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.448, "underPct": 0.552},
    {"name": "Miles Bridges", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Evan Mobley", "line": 8.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.443, "underPct": 0.557},
    {"name": "Daniel Gafford", "line": 6.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.438, "underPct": 0.562},
    {"name": "Bilal Coulibaly", "line": 4.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.437, "underPct": 0.563},
    {"name": "Bruce Brown", "line": 4.0, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.429, "underPct": 0.571},
    {"name": "Collin Sexton", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.421, "underPct": 0.579},
    {"name": "Ryan Kalkbrenner", "line": 7.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.421, "underPct": 0.579},
    {"name": "Donte DiVincenzo", "line": 4.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.42, "underPct": 0.58},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.405, "underPct": 0.595},
    {"name": "Toumani Camara", "line": 5.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.402, "underPct": 0.598},
    {"name": "Derik Queen", "line": 7.0, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.399, "underPct": 0.601},
    {"name": "De'Andre Hunter", "line": 4.0, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.387, "underPct": 0.613},
    {"name": "Jakob Poeltl", "line": 8.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.378, "underPct": 0.622},
    {"name": "Anthony Edwards", "line": 5.0, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.369, "underPct": 0.631},
    {"name": "Steven Adams", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.359, "underPct": 0.641},
    {"name": "Isaiah Jackson", "line": 7.0, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.359, "underPct": 0.641},
    {"name": "Khris Middleton", "line": 4.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.319, "underPct": 0.681},
    {"name": "Moses Moody", "line": 3.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.316, "underPct": 0.684},
    {"name": "Pascal Siakam", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.313, "underPct": 0.687},
    {"name": "Bennedict Mathurin", "line": 5.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.311, "underPct": 0.689},
    {"name": "Rudy Gobert", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.301, "underPct": 0.699},
    {"name": "Andre Drummond", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.264, "underPct": 0.736},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.261, "underPct": 0.739},
    {"name": "Josh Hart", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.254, "underPct": 0.746},
    {"name": "Malik Monk", "line": 2.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.25, "underPct": 0.75},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.242, "underPct": 0.758},
    {"name": "Cameron Johnson", "line": 3.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.231, "underPct": 0.769},
    {"name": "Mikal Bridges", "line": 4.0, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.228, "underPct": 0.772},
    {"name": "Jarace Walker", "line": 5.0, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.226, "underPct": 0.774},
    {"name": "Tony Bradley", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.222, "underPct": 0.778},
    {"name": "Dominick Barlow", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.198, "underPct": 0.802},
    {"name": "Caleb Love", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.157, "underPct": 0.843},
    {"name": "Brandin Podziemski", "line": 6.0, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.145, "underPct": 0.855},
    {"name": "Will Richard", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.074, "underPct": 0.926},
];const prizepicksBlocksHitRates = [
    {"name": "Evan Mobley", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.432, "underPct": 0.568},
    {"name": "Miles Bridges", "line": 0.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.395, "underPct": 0.605},
    {"name": "Kyshawn George", "line": 0.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.462, "underPct": 0.538},
    {"name": "Marvin Bagley III", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.485, "underPct": 0.515},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.392, "underPct": 0.608},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.578, "underPct": 0.422},
];const prizepicksStealsHitRates = [
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.411, "underPct": 0.589},
    {"name": "Dominick Barlow", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.409, "underPct": 0.591},
    {"name": "Trendon Watford", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.322, "underPct": 0.678},
    {"name": "Bennedict Mathurin", "line": 0.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.467, "underPct": 0.533},
    {"name": "Sion James", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.546, "underPct": 0.454},
    {"name": "Quinten Post", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.369, "underPct": 0.631},
    {"name": "Simone Fontecchio", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.395, "underPct": 0.605},
    {"name": "Marvin Bagley III", "line": 0.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.383, "underPct": 0.617},
    {"name": "Jaylin Williams", "line": 0.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.455, "underPct": 0.545},
    {"name": "Malik Monk", "line": 0.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.626, "underPct": 0.374},
    {"name": "Zach LaVine", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.485, "underPct": 0.515},
    {"name": "Jordan Clarkson", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.38, "underPct": 0.62},
    {"name": "Donovan Clingan", "line": 0.5, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.413, "underPct": 0.587},
];// Combo Props - PrizePicks
const prizepicksPRAHitRates = [
    {"name": "Donovan Mitchell", "line": 37.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Reed Sheppard", "line": 17.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Robinson-Earl", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kel'el Ware", "line": 20.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Huerter", "line": 15.5, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pelle Larsson", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sandro Mamukelashvili", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 39.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tre Johnson", "line": 13.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Tony Bradley", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bennedict Mathurin", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Jackson", "line": 17.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Andrew Nembhard", "line": 27.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Simone Fontecchio", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Landry Shamet", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaac Okoro", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jordan Clarkson", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Gordon", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Derik Queen", "line": 21.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 27.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Joe", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 30.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andre Drummond", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 21.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 41.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Brandon Ingram", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lonzo Ball", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 36.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Klay Thompson", "line": 15.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Naz Reid", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mitchell Robinson", "line": 14.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 24.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles McBride", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 19.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "D'Angelo Russell", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jaylin Williams", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 36.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Justin Edwards", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Wiggins", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 25.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Moses Moody", "line": 23.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dereck Lively II", "line": 11.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Corey Kispert", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Edwards", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Malik Monk", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 26.0, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Miles Bridges", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kyshawn George", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "DeMar DeRozan", "line": 24.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Clark", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Naji Marshall", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Quentin Grimes", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Steven Adams", "line": 17.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sion James", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ben Sheppard", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Collin Sexton", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Caleb Love", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Keon Ellis", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jerami Grant", "line": 29.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Coby White", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Toumani Camara", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Cameron Johnson", "line": 19.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Donte DiVincenzo", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 29.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quinten Post", "line": 18.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 24.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandin Podziemski", "line": 27.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jarace Walker", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Marvin Bagley III", "line": 18.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 44.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Murray", "line": 17.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
];const prizepicksPRHitRates = [
    {"name": "Donovan Mitchell", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Reed Sheppard", "line": 14.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jakob Poeltl", "line": 21.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kel'el Ware", "line": 20.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 12.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Ayo Dosunmu", "line": 14.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kevin Huerter", "line": 13.5, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Lonzo Ball", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jeremiah Robinson-Earl", "line": 11.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Sandro Mamukelashvili", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pelle Larsson", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Max Christie", "line": 14.5, "l5": 0.8, "l10": 0.9, "l15": 0.73, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Chet Holmgren", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 32.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Johnson", "line": 12.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Williams", "line": 17.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Norman Powell", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 20.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Simone Fontecchio", "line": 11.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Jackson", "line": 16.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Landry Shamet", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Mitchell Robinson", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 34.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Naz Reid", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 29.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Isaac Okoro", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andre Drummond", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Josh Okogie", "line": 9.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jarrett Allen", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Dean Wade", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 12.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cam Whitmore", "line": 12.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Klay Thompson", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Joe", "line": 15.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donovan Clingan", "line": 20.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Malik Monk", "line": 14.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Donte DiVincenzo", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cooper Flagg", "line": 22.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naji Marshall", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Daniel Gafford", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Shead", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dominick Barlow", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Justin Edwards", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ryan Kalkbrenner", "line": 17.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Collin Sexton", "line": 19.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Yves Missi", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 23.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Clark", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Wiggins", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bilal Coulibaly", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keon Ellis", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylin Williams", "line": 11.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "D'Angelo Russell", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Derik Queen", "line": 19.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "T.J. McConnell", "line": 10.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cameron Johnson", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Khris Middleton", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 27.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trendon Watford", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "VJ Edgecombe", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jarace Walker", "line": 14.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Scottie Barnes", "line": 26.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Steven Adams", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Sion James", "line": 10.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ben Sheppard", "line": 10.5, "l5": 0.2, "l10": 0.3, "l15": 0.47, "overPct": 0.3, "underPct": 0.7},
    {"name": "Caleb Love", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Coby White", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jerami Grant", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Toumani Camara", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 19.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.2, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 23.5, "l5": 0.2, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Buddy Hield", "line": 15.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Quinten Post", "line": 16.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Will Richard", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandin Podziemski", "line": 22.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Amen Thompson", "line": 24.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Marvin Bagley III", "line": 17.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 37.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Murray", "line": 13.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const prizepicksPAHitRates = [
    {"name": "Donovan Mitchell", "line": 34.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Reed Sheppard", "line": 14.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pelle Larsson", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 17.5, "l5": 1.0, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 12.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Shead", "line": 10.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Sandro Mamukelashvili", "line": 9.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 10.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jordan Clarkson", "line": 13.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Shai Gilgeous-Alexander", "line": 37.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 19.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Huerter", "line": 13.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Patrick Williams", "line": 6.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.8, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jarrett Allen", "line": 14.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 21.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Jackson", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kon Knueppel", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Joe", "line": 13.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Simone Fontecchio", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaiah Hartenstein", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tre Johnson", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "De'Andre Hunter", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ayo Dosunmu", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Klay Thompson", "line": 13.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Max Christie", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Williams", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bilal Coulibaly", "line": 13.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Malik Monk", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Cooper Flagg", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rudy Gobert", "line": 12.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Landry Shamet", "line": 11.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Isaac Okoro", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kyshawn George", "line": 18.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 15.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naji Marshall", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Brunson", "line": 33.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "D'Angelo Russell", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Gradey Dick", "line": 8.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Wiggins", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Lonzo Ball", "line": 12.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Corey Kispert", "line": 8.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ajay Mitchell", "line": 18.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Peyton Watson", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jakob Poeltl", "line": 14.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "T.J. McConnell", "line": 12.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles Bridges", "line": 26.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Dominick Barlow", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Moses Moody", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Naz Reid", "line": 16.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "P.J. Washington", "line": 18.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Derik Queen", "line": 14.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylin Williams", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Luguentz Dort", "line": 9.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Miles McBride", "line": 13.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 23.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 11.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylen Clark", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Sion James", "line": 8.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 21.5, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 20.5, "l5": 0.2, "l10": 0.2, "l15": 0.33, "overPct": 0.2, "underPct": 0.8},
    {"name": "Brandin Podziemski", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Caleb Love", "line": 18.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Coby White", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jerami Grant", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donte DiVincenzo", "line": 17.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Bruce Brown", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Khris Middleton", "line": 12.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Ben Sheppard", "line": 8.0, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jarace Walker", "line": 12.5, "l5": 0.0, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "LaMelo Ball", "line": 30.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 19.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quinten Post", "line": 12.5, "l5": 0.0, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Marvin Bagley III", "line": 11.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 18.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
];const prizepicksRAHitRates = [
    {"name": "Alperen Sengun", "line": 16.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Donovan Mitchell", "line": 10.0, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 14.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Robinson-Earl", "line": 6.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 10.0, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Derik Queen", "line": 10.0, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 5.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyshawn George", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "P.J. Washington", "line": 8.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 8.0, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 12.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dean Wade", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 10.0, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Aaron Gordon", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 9.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Trey Murphy III", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "LaMelo Ball", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "VJ Edgecombe", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Naz Reid", "line": 8.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Brandon Williams", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Cooper Flagg", "line": 9.0, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Julius Randle", "line": 13.0, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ayo Dosunmu", "line": 5.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jarrett Allen", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Pelle Larsson", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Andrew Wiggins", "line": 8.0, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaiah Hartenstein", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trendon Watford", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 11.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jordan Clarkson", "line": 5.0, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Mikal Bridges", "line": 8.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cason Wallace", "line": 5.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Josh Hart", "line": 12.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jeremiah Fears", "line": 6.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ajay Mitchell", "line": 7.0, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 11.0, "l5": 0.2, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Nembhard", "line": 10.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Andre Drummond", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jakob Poeltl", "line": 11.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bennedict Mathurin", "line": 7.0, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Davion Mitchell", "line": 10.0, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Sexton", "line": 7.0, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Coby White", "line": 7.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Drew Eubanks", "line": 6.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kon Knueppel", "line": 8.0, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 7.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jarace Walker", "line": 6.5, "l5": 0.2, "l10": 0.5, "l15": 0.6, "overPct": 0.5, "underPct": 0.5},
    {"name": "Cameron Johnson", "line": 6.0, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donte DiVincenzo", "line": 8.0, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Khris Middleton", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Caleb Love", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Miles McBride", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Toumani Camara", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Pascal Siakam", "line": 11.0, "l5": 0.0, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "T.J. McConnell", "line": 5.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Marvin Bagley III", "line": 8.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Malik Monk", "line": 5.0, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Deni Avdija", "line": 14.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const prizepicksTurnoversHitRates = [
    {"name": "Jarrett Allen", "line": 1.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Smith", "line": 0.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Isaac Okoro", "line": 0.5, "l5": 0.8, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Quentin Grimes", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.6, "l10": 0.8, "l15": 0.6, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Immanuel Quickley", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Khris Middleton", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Patrick Williams", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Lonzo Ball", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Reed Sheppard", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ben Sheppard", "line": 0.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jeremiah Robinson-Earl", "line": 0.5, "l5": 0.4, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mikal Bridges", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Aaron Gordon", "line": 1.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zach LaVine", "line": 2.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Caleb Love", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const prizepicksBlocksStealsHitRates = [
    {"name": "Ajay Mitchell", "line": 1.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 1.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zion Williamson", "line": 1.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Gradey Dick", "line": 0.5, "l5": 0.6, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Keon Ellis", "line": 1.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donte DiVincenzo", "line": 1.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 1.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jeremiah Robinson-Earl", "line": 0.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Julius Randle", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Karl-Anthony Towns", "line": 1.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandin Podziemski", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Will Richard", "line": 1.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Andrew Nembhard", "line": 1.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Kris Murray", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 1.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
];const underdogPointsHitRates = [
    {"name": "Kevin Huerter", "line": 10.5, "l5": 0.8, "l10": 0.8, "l15": 0.73, "overPct": 0.882, "underPct": 0.118},
    {"name": "Isaac Okoro", "line": 5.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.875, "underPct": 0.125},
    {"name": "Ayo Dosunmu", "line": 11.5, "l5": 1.0, "l10": 0.8, "l15": 0.6, "overPct": 0.866, "underPct": 0.134},
    {"name": "Bam Adebayo", "line": 16.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.836, "underPct": 0.164},
    {"name": "Reed Sheppard", "line": 10.5, "l5": 1.0, "l10": 0.8, "l15": 0.53, "overPct": 0.829, "underPct": 0.171},
    {"name": "Jeremiah Fears", "line": 13.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.819, "underPct": 0.181},
    {"name": "Trey Murphy III", "line": 18.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.813, "underPct": 0.187},
    {"name": "Landry Shamet", "line": 9.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.803, "underPct": 0.197},
    {"name": "Donovan Mitchell", "line": 27.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.758, "underPct": 0.242},
    {"name": "Davion Mitchell", "line": 9.5, "l5": 1.0, "l10": 0.7, "l15": 0.53, "overPct": 0.747, "underPct": 0.253},
    {"name": "Josh Giddey", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.728, "underPct": 0.272},
    {"name": "Amen Thompson", "line": 16.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.722, "underPct": 0.278},
    {"name": "Ajay Mitchell", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.6, "overPct": 0.71, "underPct": 0.29},
    {"name": "Aaron Gordon", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.703, "underPct": 0.297},
    {"name": "Norman Powell", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.699, "underPct": 0.301},
    {"name": "Patrick Williams", "line": 5.5, "l5": 0.8, "l10": 0.8, "l15": 0.67, "overPct": 0.696, "underPct": 0.304},
    {"name": "Jalen Smith", "line": 9.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.695, "underPct": 0.305},
    {"name": "Trendon Watford", "line": 8.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.694, "underPct": 0.306},
    {"name": "Karl-Anthony Towns", "line": 22.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.684, "underPct": 0.316},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.664, "underPct": 0.336},
    {"name": "Kon Knueppel", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.658, "underPct": 0.342},
    {"name": "Corey Kispert", "line": 7.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.628, "underPct": 0.372},
    {"name": "Jordan Clarkson", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.622, "underPct": 0.378},
    {"name": "Tyrese Maxey", "line": 30.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.583, "underPct": 0.417},
    {"name": "Alperen Sengun", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.578, "underPct": 0.422},
    {"name": "Kyshawn George", "line": 14.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.577, "underPct": 0.423},
    {"name": "Isaiah Joe", "line": 12.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.568, "underPct": 0.432},
    {"name": "Coby White", "line": 20.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.568, "underPct": 0.432},
    {"name": "Andrew Nembhard", "line": 16.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.554, "underPct": 0.446},
    {"name": "Lonzo Ball", "line": 6.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "Gradey Dick", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.545, "underPct": 0.455},
    {"name": "Derik Queen", "line": 11.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.545, "underPct": 0.455},
    {"name": "Bennedict Mathurin", "line": 20.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.538, "underPct": 0.462},
    {"name": "Chet Holmgren", "line": 17.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.533, "underPct": 0.467},
    {"name": "Josh Okogie", "line": 6.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.532, "underPct": 0.468},
    {"name": "Jalen Brunson", "line": 26.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.518, "underPct": 0.482},
    {"name": "Shai Gilgeous-Alexander", "line": 30.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.499, "underPct": 0.501},
    {"name": "Julius Randle", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.6, "overPct": 0.498, "underPct": 0.502},
    {"name": "Ryan Kalkbrenner", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.497, "underPct": 0.503},
    {"name": "DeMar DeRozan", "line": 17.5, "l5": 0.6, "l10": 0.8, "l15": 0.73, "overPct": 0.488, "underPct": 0.512},
    {"name": "LaMelo Ball", "line": 22.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.479, "underPct": 0.521},
    {"name": "Drew Eubanks", "line": 8.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.467, "underPct": 0.533},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.461, "underPct": 0.539},
    {"name": "Evan Mobley", "line": 18.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.457, "underPct": 0.543},
    {"name": "Scottie Barnes", "line": 18.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.456, "underPct": 0.544},
    {"name": "Kevin Durant", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.453, "underPct": 0.547},
    {"name": "Pascal Siakam", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.447, "underPct": 0.553},
    {"name": "Matas Buzelis", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.444, "underPct": 0.556},
    {"name": "Luguentz Dort", "line": 8.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.443, "underPct": 0.557},
    {"name": "Isaiah Jackson", "line": 9.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.437, "underPct": 0.563},
    {"name": "Andrew Wiggins", "line": 17.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.424, "underPct": 0.576},
    {"name": "Sion James", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.422, "underPct": 0.578},
    {"name": "Brandon Ingram", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.409, "underPct": 0.591},
    {"name": "Andre Drummond", "line": 11.5, "l5": 0.8, "l10": 0.5, "l15": 0.33, "overPct": 0.393, "underPct": 0.607},
    {"name": "Quentin Grimes", "line": 16.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.387, "underPct": 0.613},
    {"name": "Miles McBride", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.385, "underPct": 0.615},
    {"name": "Ben Sheppard", "line": 6.5, "l5": 0.0, "l10": 0.1, "l15": 0.2, "overPct": 0.385, "underPct": 0.615},
    {"name": "Mike Conley", "line": 6.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.384, "underPct": 0.616},
    {"name": "Zach LaVine", "line": 19.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.38, "underPct": 0.62},
    {"name": "Bilal Coulibaly", "line": 10.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.378, "underPct": 0.622},
    {"name": "Jaylen Clark", "line": 5.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.371, "underPct": 0.629},
    {"name": "Anthony Edwards", "line": 28.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.337, "underPct": 0.663},
    {"name": "Jamal Murray", "line": 22.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.336, "underPct": 0.664},
    {"name": "Deni Avdija", "line": 29.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.308, "underPct": 0.692},
    {"name": "Donovan Clingan", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.29, "underPct": 0.71},
    {"name": "Collin Sexton", "line": 16.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.259, "underPct": 0.741},
    {"name": "Dean Wade", "line": 6.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.195, "underPct": 0.805},
    {"name": "Jarace Walker", "line": 10.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.192, "underPct": 0.808},
    {"name": "Justin Edwards", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.162, "underPct": 0.838},
    {"name": "Rob Dillingham", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.157, "underPct": 0.843},
    {"name": "Jaylin Williams", "line": 6.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.148, "underPct": 0.852},
    {"name": "Toumani Camara", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.123, "underPct": 0.877},
    {"name": "Peyton Watson", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.119, "underPct": 0.881},
    {"name": "Bruce Brown", "line": 8.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.116, "underPct": 0.884},
    {"name": "Cameron Johnson", "line": 12.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.061, "underPct": 0.939},
    {"name": "Moses Moody", "line": 17.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.059, "underPct": 0.941},
    {"name": "Jerami Grant", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.056, "underPct": 0.944},
    {"name": "Quinten Post", "line": 10.5, "l5": 0.0, "l10": 0.3, "l15": 0.27, "overPct": 0.034, "underPct": 0.966},
    {"name": "Kris Murray", "line": 8.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.03, "underPct": 0.97},
    {"name": "Will Richard", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.011, "underPct": 0.989},
    {"name": "Buddy Hield", "line": 12.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.005, "underPct": 0.995},
    {"name": "Caleb Love", "line": 13.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.002, "underPct": 0.998},
];const underdogAssistsHitRates = [
    {"name": "Kevin Durant", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.655, "underPct": 0.345},
    {"name": "Miles Bridges", "line": 3.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.628, "underPct": 0.372},
    {"name": "Jalen Brunson", "line": 6.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.596, "underPct": 0.404},
    {"name": "Kon Knueppel", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.593, "underPct": 0.407},
    {"name": "Donovan Mitchell", "line": 5.5, "l5": 0.8, "l10": 0.6, "l15": 0.4, "overPct": 0.587, "underPct": 0.413},
    {"name": "Jamal Shead", "line": 4.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.582, "underPct": 0.418},
    {"name": "Collin Sexton", "line": 4.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.576, "underPct": 0.424},
    {"name": "Lonzo Ball", "line": 4.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.569, "underPct": 0.431},
    {"name": "Ajay Mitchell", "line": 3.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.556, "underPct": 0.444},
    {"name": "Zach LaVine", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.533, "underPct": 0.467},
    {"name": "Quentin Grimes", "line": 3.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.492, "underPct": 0.508},
    {"name": "Justin Edwards", "line": 1.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.485, "underPct": 0.515},
    {"name": "Khris Middleton", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.468, "underPct": 0.532},
    {"name": "Miles McBride", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.423, "underPct": 0.577},
    {"name": "Ben Sheppard", "line": 1.5, "l5": 0.2, "l10": 0.4, "l15": 0.53, "overPct": 0.386, "underPct": 0.614},
    {"name": "T.J. McConnell", "line": 3.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.376, "underPct": 0.624},
    {"name": "Mike Conley", "line": 3.5, "l5": 0.0, "l10": 0.4, "l15": 0.4, "overPct": 0.375, "underPct": 0.625},
    {"name": "Cameron Johnson", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.37, "underPct": 0.63},
    {"name": "Amen Thompson", "line": 5.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.368, "underPct": 0.632},
    {"name": "Donte DiVincenzo", "line": 3.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.311, "underPct": 0.689},
    {"name": "Rob Dillingham", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.296, "underPct": 0.704},
    {"name": "Aaron Gordon", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.292, "underPct": 0.708},
    {"name": "Buddy Hield", "line": 2.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.163, "underPct": 0.837},
    {"name": "Donovan Clingan", "line": 1.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.133, "underPct": 0.867},
    {"name": "Caleb Love", "line": 4.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.038, "underPct": 0.962},
];const underdogReboundsHitRates = [
    {"name": "Josh Giddey", "line": 8.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.793, "underPct": 0.207},
    {"name": "Kel'el Ware", "line": 9.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.753, "underPct": 0.247},
    {"name": "Kon Knueppel", "line": 5.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.693, "underPct": 0.307},
    {"name": "Karl-Anthony Towns", "line": 11.5, "l5": 0.8, "l10": 0.7, "l15": 0.6, "overPct": 0.633, "underPct": 0.367},
    {"name": "Trendon Watford", "line": 4.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.562, "underPct": 0.438},
    {"name": "Naz Reid", "line": 6.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.533, "underPct": 0.467},
    {"name": "Cason Wallace", "line": 2.5, "l5": 0.2, "l10": 0.4, "l15": 0.47, "overPct": 0.533, "underPct": 0.467},
    {"name": "Reed Sheppard", "line": 2.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.517, "underPct": 0.483},
    {"name": "Zach LaVine", "line": 3.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.514, "underPct": 0.486},
    {"name": "Isaiah Jackson", "line": 6.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.507, "underPct": 0.493},
    {"name": "Scottie Barnes", "line": 7.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.498, "underPct": 0.502},
    {"name": "Jamal Shead", "line": 1.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.477, "underPct": 0.523},
    {"name": "Donovan Clingan", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.474, "underPct": 0.526},
    {"name": "T.J. McConnell", "line": 1.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.456, "underPct": 0.544},
    {"name": "Miles Bridges", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.446, "underPct": 0.554},
    {"name": "Bilal Coulibaly", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.437, "underPct": 0.563},
    {"name": "Collin Sexton", "line": 2.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.421, "underPct": 0.579},
    {"name": "Jarace Walker", "line": 4.5, "l5": 0.2, "l10": 0.5, "l15": 0.53, "overPct": 0.384, "underPct": 0.616},
    {"name": "Jordan Clarkson", "line": 2.5, "l5": 0.6, "l10": 0.3, "l15": 0.33, "overPct": 0.382, "underPct": 0.618},
    {"name": "Steven Adams", "line": 9.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.359, "underPct": 0.641},
    {"name": "Dominick Barlow", "line": 4.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.35, "underPct": 0.65},
    {"name": "Pascal Siakam", "line": 6.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.313, "underPct": 0.687},
    {"name": "Kyshawn George", "line": 5.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.261, "underPct": 0.739},
    {"name": "DeMar DeRozan", "line": 3.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.242, "underPct": 0.758},
    {"name": "Tony Bradley", "line": 4.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.222, "underPct": 0.778},
    {"name": "Caleb Love", "line": 4.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.157, "underPct": 0.843},
];const underdogBlocksHitRates = [
    {"name": "Chet Holmgren", "line": 1.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.392, "underPct": 0.608},
];const underdogStealsHitRates = [
    {"name": "Amen Thompson", "line": 1.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.411, "underPct": 0.589},
];// Combo Props - Underdog
const underdogPRAHitRates = [
    {"name": "Donovan Mitchell", "line": 37.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Reed Sheppard", "line": 16.5, "l5": 1.0, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kel'el Ware", "line": 20.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Huerter", "line": 15.5, "l5": 1.0, "l10": 0.9, "l15": 0.8, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Jeremiah Robinson-Earl", "line": 12.5, "l5": 1.0, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 39.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bam Adebayo", "line": 26.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jamal Murray", "line": 33.5, "l5": 0.8, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Trey Murphy III", "line": 27.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jeremiah Fears", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Davion Mitchell", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 26.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Ayo Dosunmu", "line": 17.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 27.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Smith", "line": 15.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Shai Gilgeous-Alexander", "line": 42.5, "l5": 0.8, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 28.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andre Drummond", "line": 26.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bennedict Mathurin", "line": 28.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jarrett Allen", "line": 22.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kevin Durant", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "De'Andre Hunter", "line": 23.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Lonzo Ball", "line": 15.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Dean Wade", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Okogie", "line": 10.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Gradey Dick", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tyrese Maxey", "line": 41.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Isaiah Joe", "line": 17.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jordan Clarkson", "line": 17.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Karl-Anthony Towns", "line": 38.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaac Okoro", "line": 10.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Landry Shamet", "line": 14.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jalen Brunson", "line": 36.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kyshawn George", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Luguentz Dort", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Isaiah Hartenstein", "line": 25.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Julius Randle", "line": 35.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Jakob Poeltl", "line": 23.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Simone Fontecchio", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.67, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 24.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Norman Powell", "line": 28.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Dru Smith", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 26.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Tony Bradley", "line": 10.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Hart", "line": 23.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Donovan Clingan", "line": 22.5, "l5": 0.4, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Matas Buzelis", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jaylin Williams", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 35.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jamal Shead", "line": 12.5, "l5": 0.4, "l10": 0.5, "l15": 0.53, "overPct": 0.5, "underPct": 0.5},
    {"name": "Nae'Qwan Tomlin", "line": 15.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Justin Edwards", "line": 14.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles Bridges", "line": 33.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Ajay Mitchell", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.6, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylen Clark", "line": 9.5, "l5": 0.4, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Pascal Siakam", "line": 36.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 24.5, "l5": 0.4, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Bilal Coulibaly", "line": 18.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Anthony Edwards", "line": 38.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andrew Wiggins", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Drew Eubanks", "line": 14.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Scottie Barnes", "line": 31.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 24.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trendon Watford", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Collin Sexton", "line": 23.5, "l5": 0.2, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Caleb Love", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Yves Missi", "line": 13.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Peyton Watson", "line": 16.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Ben Sheppard", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.4, "overPct": 0.2, "underPct": 0.8},
    {"name": "Bruce Brown", "line": 14.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Jose Alvarado", "line": 15.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Donte DiVincenzo", "line": 22.5, "l5": 0.2, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Rob Dillingham", "line": 12.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Toumani Camara", "line": 23.5, "l5": 0.2, "l10": 0.2, "l15": 0.27, "overPct": 0.2, "underPct": 0.8},
    {"name": "Keon Ellis", "line": 10.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Miles McBride", "line": 16.5, "l5": 0.2, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Coby White", "line": 28.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Cason Wallace", "line": 14.5, "l5": 0.2, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Amen Thompson", "line": 29.5, "l5": 0.0, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "T.J. McConnell", "line": 14.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Jarace Walker", "line": 18.5, "l5": 0.0, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Mike Conley", "line": 12.5, "l5": 0.0, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quinten Post", "line": 18.5, "l5": 0.0, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Buddy Hield", "line": 18.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Will Richard", "line": 24.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandin Podziemski", "line": 27.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Marvin Bagley III", "line": 18.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 44.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Kris Murray", "line": 16.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
];const underdogPRHitRates = [
    {"name": "Donovan Mitchell", "line": 32.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Kel'el Ware", "line": 20.5, "l5": 1.0, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Tyrese Maxey", "line": 34.5, "l5": 0.8, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Immanuel Quickley", "line": 20.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Alperen Sengun", "line": 31.5, "l5": 0.8, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zion Williamson", "line": 24.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Trey Murphy III", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bam Adebayo", "line": 23.5, "l5": 0.8, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Chet Holmgren", "line": 25.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Andrew Nembhard", "line": 19.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Miles Bridges", "line": 29.5, "l5": 0.6, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Aaron Gordon", "line": 23.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kevin Durant", "line": 31.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jarrett Allen", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 25.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Bennedict Mathurin", "line": 25.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Evan Mobley", "line": 26.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Rudy Gobert", "line": 22.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 35.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Matas Buzelis", "line": 19.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Jalen Brunson", "line": 29.5, "l5": 0.6, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Isaiah Hartenstein", "line": 22.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 26.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Julius Randle", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.67, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Naz Reid", "line": 21.5, "l5": 0.6, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kon Knueppel", "line": 23.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "DeMar DeRozan", "line": 21.5, "l5": 0.4, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Kyshawn George", "line": 19.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Scottie Barnes", "line": 25.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "De'Andre Hunter", "line": 20.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Pascal Siakam", "line": 32.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Collin Sexton", "line": 18.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Shai Gilgeous-Alexander", "line": 35.5, "l5": 0.4, "l10": 0.3, "l15": 0.4, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jamal Murray", "line": 27.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Zach LaVine", "line": 22.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Moses Moody", "line": 21.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "LaMelo Ball", "line": 27.5, "l5": 0.2, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "VJ Edgecombe", "line": 21.5, "l5": 0.2, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Coby White", "line": 23.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Anthony Edwards", "line": 34.5, "l5": 0.2, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jerami Grant", "line": 27.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Toumani Camara", "line": 20.5, "l5": 0.2, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Will Richard", "line": 20.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Brandin Podziemski", "line": 22.5, "l5": 0.0, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 37.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogPAHitRates = [
    {"name": "Donovan Mitchell", "line": 33.5, "l5": 1.0, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Trey Murphy III", "line": 21.5, "l5": 0.8, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Alperen Sengun", "line": 29.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 28.5, "l5": 0.8, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Zion Williamson", "line": 23.5, "l5": 0.8, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kevin Durant", "line": 30.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 37.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "De'Andre Hunter", "line": 19.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandon Ingram", "line": 24.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Norman Powell", "line": 25.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Kon Knueppel", "line": 20.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bennedict Mathurin", "line": 22.5, "l5": 0.6, "l10": 0.3, "l15": 0.2, "overPct": 0.3, "underPct": 0.7},
    {"name": "Julius Randle", "line": 27.5, "l5": 0.6, "l10": 0.8, "l15": 0.67, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Aaron Gordon", "line": 19.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Immanuel Quickley", "line": 22.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Nembhard", "line": 24.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Pascal Siakam", "line": 29.5, "l5": 0.6, "l10": 0.7, "l15": 0.6, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Mikal Bridges", "line": 20.5, "l5": 0.6, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Josh Giddey", "line": 28.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Russell Westbrook", "line": 20.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Zach LaVine", "line": 21.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Quentin Grimes", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Anthony Edwards", "line": 33.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Moses Moody", "line": 20.5, "l5": 0.4, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Andrew Wiggins", "line": 20.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Karl-Anthony Towns", "line": 26.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Jalen Brunson", "line": 33.5, "l5": 0.4, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Brandin Podziemski", "line": 21.5, "l5": 0.2, "l10": 0.2, "l15": 0.2, "overPct": 0.2, "underPct": 0.8},
    {"name": "Coby White", "line": 24.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Jerami Grant", "line": 26.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Deni Avdija", "line": 35.5, "l5": 0.2, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Will Richard", "line": 19.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "LaMelo Ball", "line": 30.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogRAHitRates = [
    {"name": "Alperen Sengun", "line": 16.5, "l5": 1.0, "l10": 0.9, "l15": 0.67, "overPct": 0.9, "underPct": 0.09999999999999998},
    {"name": "Josh Giddey", "line": 16.5, "l5": 0.8, "l10": 0.8, "l15": 0.53, "overPct": 0.8, "underPct": 0.19999999999999996},
    {"name": "Scottie Barnes", "line": 12.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Kevin Durant", "line": 8.5, "l5": 0.6, "l10": 0.6, "l15": 0.47, "overPct": 0.6, "underPct": 0.4},
    {"name": "Rudy Gobert", "line": 11.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Russell Westbrook", "line": 13.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Aaron Gordon", "line": 7.5, "l5": 0.6, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "LaMelo Ball", "line": 13.5, "l5": 0.6, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Mitchell Robinson", "line": 8.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Brandon Williams", "line": 7.5, "l5": 0.6, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Jalen Brunson", "line": 9.5, "l5": 0.6, "l10": 0.4, "l15": 0.27, "overPct": 0.4, "underPct": 0.6},
    {"name": "Kyshawn George", "line": 9.5, "l5": 0.6, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Jaylin Williams", "line": 7.5, "l5": 0.4, "l10": 0.5, "l15": 0.4, "overPct": 0.5, "underPct": 0.5},
    {"name": "Ryan Kalkbrenner", "line": 8.5, "l5": 0.4, "l10": 0.3, "l15": 0.27, "overPct": 0.3, "underPct": 0.7},
    {"name": "Quentin Grimes", "line": 7.5, "l5": 0.4, "l10": 0.4, "l15": 0.4, "overPct": 0.4, "underPct": 0.6},
    {"name": "Amen Thompson", "line": 11.5, "l5": 0.4, "l10": 0.6, "l15": 0.4, "overPct": 0.6, "underPct": 0.4},
    {"name": "Bilal Coulibaly", "line": 7.5, "l5": 0.4, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
    {"name": "Donovan Clingan", "line": 11.5, "l5": 0.4, "l10": 0.3, "l15": 0.33, "overPct": 0.3, "underPct": 0.7},
    {"name": "Isaiah Hartenstein", "line": 13.5, "l5": 0.4, "l10": 0.5, "l15": 0.47, "overPct": 0.5, "underPct": 0.5},
    {"name": "Andre Drummond", "line": 14.5, "l5": 0.2, "l10": 0.1, "l15": 0.13, "overPct": 0.1, "underPct": 0.9},
    {"name": "Coby White", "line": 7.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Caleb Love", "line": 9.5, "l5": 0.2, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Marvin Bagley III", "line": 8.5, "l5": 0.0, "l10": 0.1, "l15": 0.07, "overPct": 0.1, "underPct": 0.9},
    {"name": "Quinten Post", "line": 8.5, "l5": 0.0, "l10": 0.2, "l15": 0.13, "overPct": 0.2, "underPct": 0.8},
];const underdogTurnoversHitRates = [
    {"name": "Zion Williamson", "line": 2.5, "l5": 1.0, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
    {"name": "Jamal Murray", "line": 2.5, "l5": 0.8, "l10": 0.4, "l15": 0.33, "overPct": 0.4, "underPct": 0.6},
    {"name": "Tyrese Maxey", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.47, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Quentin Grimes", "line": 2.5, "l5": 0.6, "l10": 0.7, "l15": 0.53, "overPct": 0.7, "underPct": 0.30000000000000004},
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.4, "l10": 0.6, "l15": 0.53, "overPct": 0.6, "underPct": 0.4},
    {"name": "Brandon Ingram", "line": 2.5, "l5": 0.4, "l10": 0.4, "l15": 0.47, "overPct": 0.4, "underPct": 0.6},
    {"name": "Anthony Edwards", "line": 2.5, "l5": 0.4, "l10": 0.5, "l15": 0.33, "overPct": 0.5, "underPct": 0.5},
];const underdogBlocksStealsHitRates = [
    {"name": "Evan Mobley", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
    {"name": "Rudy Gobert", "line": 2.5, "l5": 0.0, "l10": 0.0, "l15": 0.0, "overPct": 0.0, "underPct": 1.0},
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

